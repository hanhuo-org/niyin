/**
 * 汉火古音 TTS V12 Studio 浏览器推理 Web Worker (hanfire_worker.mjs)
 * - 统一 8-bit 权重量化 (MatMulNBits-8, asym acc=1, cos=0.99995 vs FP32) 双栖推理引擎：
 *   * 检测到支持 WebGPU 的显卡时自动启用 GPU 加速 (fm_decoder + vocos_spec)；
 *   * 无显卡 / 移动端 / 旧浏览器自动无缝回落多核 WASM SIMD；
 * - 分片压缩多线程并发下载 (5 路并发 .gzbin) + 浏览器原生 DecompressionStream("gzip") 流式解压 + IndexedDB 分片持久化缓存与断点续传
 * - 100% 复刻 Python `sample_clean_zero_carrier` + 三槽位/零声母首字补偿对齐 + `[字=规格]` 句内覆写 + 零削波后处理管线
 */
import {
  SR,
  FEAT_SCALE,
  istftVocos1024,
  sweetSpotMelEnvelopeInPlace,
  bandpass75_9500ZeroPhase,
  safeNormalizeAndFade,
  weightsToExactFrames,
  makeSphericalInitialNoise,
  getSwayTimeSteps,
  encodeWavPCM16,
  trimEdgeSilence,
} from "./hanfire_dsp.mjs?v=20261119d";
import {
  fetchManifest,
  inspectModelCache,
  loadChunkedModel,
  clearModelCache,
} from "./model_cache.mjs?v=20261119d";

const _isWorker = typeof self !== "undefined" && typeof self.postMessage === "function";
const ASSET_VER = new URL(import.meta.url).searchParams.get("v") || "20261119d";
const ortDistUrl = new URL("./vendor/ort/", import.meta.url).href;
const NUM_WASM_THREADS = _isWorker && self.crossOriginIsolated
  ? Math.min(8, Math.max(2, navigator.hardwareConcurrency || 4))
  : 1;

let ort = null;
async function ensureOrtModule() {
  if (ort) return ort;
  const mod = await import("./vendor/ort/ort.webgpu.bundle.min.mjs");
  ort = mod;
  ort.env.wasm.wasmPaths = ortDistUrl;
  ort.env.wasm.simd = true;
  ort.env.wasm.numThreads = NUM_WASM_THREADS;
  return ort;
}

let lexicon = null;
let promptBank = null;
let activeBackend = null;
let activeBackendPref = "wasm"; // 默认纯处理器推理，检测到可用显卡且显存充足时提示用户一键开启显卡加速
let gpuHardwareAvailable = false;
let gpuVramOk = false;
let gpuVramReason = "";
let activeGpuName = "";
let activeGpuDiag = "";
let lastFallbackReason = "";
let lastCacheInfo = null;
let _ortRuntimePrepared = false;
let _cachedGpuProbe = null;
let sessions = { te: null, fm: null, voc: null };
let idleGpuReleaseTimer = null;
const IDLE_GPU_RELEASE_MS = 180000; // 空闲 3 分钟自动释放 WebGPU 显存会话，下次合成自动从本地缓存毫秒重载
const MIN_GPU_BUFFER_MB = 512;      // 安全运行模型所需的最低 maxBufferSize 上限 (512 MB)
const VRAM_PROBE_CHUNK_MB = 256;    // 瞬时探测 2 × 256 MB = 512 MB 可用显存后立刻销毁释放

function cancelIdleGpuRelease() {
  if (idleGpuReleaseTimer !== null) {
    clearTimeout(idleGpuReleaseTimer);
    idleGpuReleaseTimer = null;
  }
}

function scheduleIdleGpuRelease() {
  cancelIdleGpuRelease();
  if (activeBackend !== "webgpu" || !sessions.fm) return;
  idleGpuReleaseTimer = setTimeout(async () => {
    idleGpuReleaseTimer = null;
    if (sessions.fm && activeBackend === "webgpu") {
      try {
        await sessions.fm.release();
      } catch (_) {}
      sessions.fm = null;
      console.log("[Hanfire Worker] Released idle WebGPU fm session to free VRAM (will auto-reload from IndexedDB on next run).");
    }
  }, IDLE_GPU_RELEASE_MS);
}

async function gunzipArrayBuffer(buf) {
  const u8 = new Uint8Array(buf);
  if (u8.length < 2 || u8[0] !== 0x1f || u8[1] !== 0x8b) {
    return buf;
  }
  if (typeof DecompressionStream === "undefined") {
    throw new Error("浏览器不支持 DecompressionStream('gzip')");
  }
  const ds = new DecompressionStream("gzip");
  return await new Response(new Blob([buf]).stream().pipeThrough(ds)).arrayBuffer();
}

async function detectWebGPU(forceReprobe = false) {
  if (_cachedGpuProbe && !forceReprobe) return _cachedGpuProbe;
  if (typeof navigator === "undefined" || !navigator.gpu) {
    _cachedGpuProbe = {
      available: false,
      vramOk: false,
      vramReason: "浏览器未启用显卡加速接口",
      maxBufferMB: 0,
      adapterName: "",
      diag: "navigator.gpu 不可用（请确认使用 HTTPS 或新版 Chrome/Edge）",
    };
    return _cachedGpuProbe;
  }
  try {
    let adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      _cachedGpuProbe = {
        available: false,
        vramOk: false,
        vramReason: "未检测到可用显卡适配器",
        maxBufferMB: 0,
        adapterName: "",
        diag: "navigator.gpu.requestAdapter() 返回 null（Linux Chrome 需同时启用 #enable-vulkan + #enable-unsafe-webgpu + #ignore-gpu-blocklist 且系统具备 Vulkan 驱动）",
      };
      return _cachedGpuProbe;
    }
    const info = adapter.info || {};
    const name = [info.vendor, info.architecture, info.description, info.device]
      .filter(Boolean)
      .join(" ")
      .trim() || "WebGPU GPU";
    const isSoft = Boolean(adapter.isFallbackAdapter) || /swiftshader|llvmpipe|software|basic render/i.test(name);
    if (isSoft) {
      _cachedGpuProbe = {
        available: false,
        vramOk: false,
        vramReason: "检测到软渲染回退，已自动使用处理器",
        maxBufferMB: 0,
        adapterName: name,
        diag: `检测到软渲染回退适配器 (${name})，已自动改用更快的多核 WASM 处理器推理`,
      };
      return _cachedGpuProbe;
    }

    const limits = adapter.limits || {};
    const maxBufferMB = Math.round((limits.maxBufferSize || 0) / (1024 * 1024));
    const maxBindingMB = Math.round((limits.maxStorageBufferBindingSize || 0) / (1024 * 1024));
    const devMemGB = typeof navigator.deviceMemory === "number" ? navigator.deviceMemory : null;

    if (maxBufferMB > 0 && maxBufferMB < MIN_GPU_BUFFER_MB) {
      _cachedGpuProbe = {
        available: false,
        vramOk: false,
        vramReason: `显卡缓冲区上限仅 ${maxBufferMB} MB（低于安全阈值 ${MIN_GPU_BUFFER_MB} MB），已自动使用纯处理器推理`,
        maxBufferMB,
        adapterName: name,
        diag: `VRAM/limit too small: maxBufferSize=${maxBufferMB}MB < ${MIN_GPU_BUFFER_MB}MB (${name})`,
      };
      return _cachedGpuProbe;
    }
    if (maxBindingMB > 0 && maxBindingMB < 128) {
      _cachedGpuProbe = {
        available: false,
        vramOk: false,
        vramReason: `显卡单缓冲绑定上限仅 ${maxBindingMB} MB，已自动使用纯处理器推理`,
        maxBufferMB,
        adapterName: name,
        diag: `VRAM/binding limit too small: maxStorageBufferBindingSize=${maxBindingMB}MB (${name})`,
      };
      return _cachedGpuProbe;
    }
    if (devMemGB !== null && devMemGB <= 2) {
      _cachedGpuProbe = {
        available: false,
        vramOk: false,
        vramReason: `设备内存偏小（≤${devMemGB} GB），为防显存溢出已自动使用纯处理器推理`,
        maxBufferMB,
        adapterName: name,
        diag: `Low deviceMemory=${devMemGB}GB, forcing safe CPU mode (${name})`,
      };
      return _cachedGpuProbe;
    }

    // 真实瞬时显存容量探测：尝试分配 2 × 256 MB (= 512 MB) GPUBuffer 并立刻销毁释放，
    // 确保显存不足或已被其它程序占满的显卡不会在推理中 OOM，自动稳妥回落纯处理器。
    let probeOk = true;
    let probeErr = "";
    let probeDevice = null;
    const probeBufs = [];
    try {
      const reqBytes = Math.min(VRAM_PROBE_CHUNK_MB * 1024 * 1024, limits.maxBufferSize || VRAM_PROBE_CHUNK_MB * 1024 * 1024);
      probeDevice = await adapter.requestDevice({
        requiredLimits: limits.maxBufferSize ? { maxBufferSize: Math.min(limits.maxBufferSize, 1024 * 1024 * 1024) } : undefined,
      });
      probeDevice.pushErrorScope("out-of-memory");
      probeDevice.pushErrorScope("validation");
      for (let i = 0; i < 2; i++) {
        probeBufs.push(probeDevice.createBuffer({
          size: reqBytes,
          usage: GPUBufferUsage.STORAGE,
        }));
      }
      const valErr = await probeDevice.popErrorScope();
      const oomErr = await probeDevice.popErrorScope();
      if (oomErr) {
        probeOk = false;
        probeErr = `可用显存不足 512 MB (${oomErr.message || "out-of-memory"})`;
      } else if (valErr) {
        // validation error on limit shouldn't falsely block if maxBufferMB >= 512
        console.warn("[Hanfire Worker] VRAM probe validation notice:", valErr.message);
      }
    } catch (e) {
      probeOk = false;
      probeErr = String(e?.message || e);
    } finally {
      for (const b of probeBufs) {
        try { b.destroy(); } catch (_) {}
      }
      if (probeDevice) {
        try { probeDevice.destroy(); } catch (_) {}
      }
    }

    if (!probeOk) {
      _cachedGpuProbe = {
        available: false,
        vramOk: false,
        vramReason: "检测到当前可用显存偏小（不足 512 MB），已自动使用纯处理器推理",
        maxBufferMB,
        adapterName: name,
        diag: `VRAM probe failed (${probeErr}) on ${name}`,
      };
      return _cachedGpuProbe;
    }

    _cachedGpuProbe = {
      available: true,
      vramOk: true,
      vramReason: "",
      maxBufferMB,
      adapterName: name,
      diag: `hardware adapter ready (${name}, maxBuffer=${maxBufferMB}MB, 512MB VRAM probe PASS)`,
    };
    return _cachedGpuProbe;
  } catch (err) {
    _cachedGpuProbe = {
      available: false,
      vramOk: false,
      vramReason: "显卡检测异常，已使用纯处理器推理",
      maxBufferMB: 0,
      adapterName: "",
      diag: `requestAdapter 异常: ${err?.message || err}`,
    };
    return _cachedGpuProbe;
  }
}

async function prepareOrtWasmBlobRuntime() {
  await ensureOrtModule();
  if (_ortRuntimePrepared || !_isWorker || typeof URL.createObjectURL !== "function") return;
  try {
    const mjsUrl = new URL("./vendor/ort/ort-wasm-simd-threaded.asyncify.mjs", import.meta.url).href;
    const wasmGzUrl = new URL("./vendor/ort/ort-wasm-simd-threaded.asyncify.wasm.gzbin", import.meta.url).href;
    const wasmPlainUrl = new URL("./vendor/ort/ort-wasm-simd-threaded.asyncify.wasm", import.meta.url).href;
    const [mjsResp, wasmResp] = await Promise.all([
      fetch(mjsUrl, { credentials: "same-origin", cache: "force-cache" }),
      fetch(wasmGzUrl, { credentials: "same-origin", cache: "force-cache" }).then((r) =>
        r.ok ? r : fetch(wasmPlainUrl, { credentials: "same-origin", cache: "force-cache" })
      ),
    ]);
    if (!mjsResp.ok || !wasmResp.ok) {
      throw new Error(`ORT runtime fetch failed (${mjsResp.status}/${wasmResp.status})`);
    }
    const [mjsText, wasmRawOrGz] = await Promise.all([mjsResp.text(), wasmResp.arrayBuffer()]);
    const wasmBuf = await gunzipArrayBuffer(wasmRawOrGz);
    // 将 Emscripten pthread 脚本转为同源内存 Blob URL：
    // 1) 彻底绕过 Chromium 在自签名 HTTPS 下并发创建嵌套 pthread Worker 触发的 ERR_ABORTED 死锁；
    // 2) 消除多线程初始化时对同一 .mjs 的 N 次重复网络请求；
    // 3) asyncify.wasm (28.7 MB) 以 7.2 MB .gzbin 传输后在内存中极速解压并注入 wasmBinary，兼容 Cloudflare 25 MiB 限制。
    const mjsBlobUrl = URL.createObjectURL(new Blob([mjsText], { type: "text/javascript; charset=utf-8" }));
    ort.env.wasm.wasmPaths = { mjs: mjsBlobUrl, wasm: wasmPlainUrl };
    ort.env.wasm.wasmBinary = wasmBuf;
    _ortRuntimePrepared = true;
    console.log("[Hanfire Worker] ORT WebGPU+WASM Blob runtime ready:", {
      numThreads: ort.env.wasm.numThreads,
      wasmBytes: wasmBuf.byteLength,
    });
  } catch (err) {
    console.warn("[Hanfire Worker] Blob runtime prep failed, falling back to URL path:", err);
  }
}

async function fetchArrayBufferWithRetryTimeout(url, { retries = 2, timeoutMs = 25000, cache = "force-cache" } = {}) {
  let lastErr = null;
  for (let attempt = 0; attempt <= retries; attempt++) {
    const ctrl = typeof AbortController !== "undefined" ? new AbortController() : null;
    const timer = ctrl ? setTimeout(() => ctrl.abort(), timeoutMs) : null;
    try {
      const resp = await fetch(url, { cache, signal: ctrl ? ctrl.signal : undefined });
      if (!resp.ok) {
        if (timer) clearTimeout(timer);
        lastErr = new Error(`HTTP ${resp.status} (${url})`);
      } else {
        // 超时计时覆盖完整响应体读取，防止 HTTP 200 头返回后正文卡死
        const buf = await resp.arrayBuffer();
        if (timer) clearTimeout(timer);
        return buf;
      }
    } catch (err) {
      if (timer) clearTimeout(timer);
      lastErr = err;
    }
    if (attempt < retries) {
      await new Promise((r) => setTimeout(r, 350 * (attempt + 1)));
    }
  }
  throw lastErr || new Error(`拉取失败: ${url}`);
}

async function fetchJsonWithGzipFallback(gzUrl, jsonUrl) {
  try {
    const buf = await fetchArrayBufferWithRetryTimeout(gzUrl, { retries: 2, timeoutMs: 25000, cache: "force-cache" });
    const u8 = new Uint8Array(buf);
    if (u8.length >= 2 && u8[0] === 0x1f && u8[1] === 0x8b) {
      if (typeof DecompressionStream !== "undefined") {
        const ds = new DecompressionStream("gzip");
        const text = await new Response(new Blob([buf]).stream().pipeThrough(ds)).text();
        return JSON.parse(text);
      }
    } else {
      return JSON.parse(new TextDecoder("utf-8").decode(u8));
    }
  } catch (_) {}
  const buf = await fetchArrayBufferWithRetryTimeout(jsonUrl, { retries: 1, timeoutMs: 30000, cache: "force-cache" });
  return JSON.parse(new TextDecoder("utf-8").decode(new Uint8Array(buf)));
}

async function ensureLexiconLoaded() {
  if (!lexicon) {
    self.postMessage({ type: "status", stage: "正在加载三代古音词典..." });
    lexicon = await fetchJsonWithGzipFallback(
      `./assets/lexicon_3era.json.gz?v=${ASSET_VER}`,
      `./assets/lexicon_3era.json?v=${ASSET_VER}`
    );
  }
}

async function ensurePromptBankLoaded() {
  if (!promptBank) {
    self.postMessage({ type: "status", stage: "正在加载参考人声..." });
    promptBank = await fetchJsonWithGzipFallback(
      `./assets/voice_prompts.json.gz?v=${ASSET_VER}`,
      `./assets/voice_prompts.json?v=${ASSET_VER}`
    );
  }
}

async function ensureAssetsLoaded() {
  await ensureLexiconLoaded();
  await ensurePromptBankLoaded();
}

async function disposeSessions(onlyKey = null) {
  cancelIdleGpuRelease();
  const keys = onlyKey ? [onlyKey] : ["te", "fm", "voc"];
  for (const k of keys) {
    if (sessions[k]) {
      try { await sessions[k].release(); } catch (_) {}
      sessions[k] = null;
    }
  }
  if (!onlyKey) activeBackend = null;
}

async function createHybridSession(key, bufOrPath, useWebGPU, wasmOpts, webgpuOpts) {
  // text_encoder (~8ms) 与 vocos_spec (~48ms) 每句仅跑一次，固定走 WASM：
  // 1) 省去额外 WebGPU 会话、着色器编译与显存池占用；
  // 2) 将显卡资源 100% 留给每句执行 16 步的 fm_decoder 主干。
  if (key === "te" || key === "voc" || !useWebGPU) {
    const s = await ort.InferenceSession.create(bufOrPath, wasmOpts);
    return { session: s, usedGpu: false, errMsg: "" };
  }
  try {
    const s = await ort.InferenceSession.create(bufOrPath, webgpuOpts);
    return { session: s, usedGpu: true, errMsg: "" };
  } catch (err) {
    const errMsg = String(err?.message || err);
    console.warn(`[Hanfire Worker] WebGPU session init failed for '${key}', falling back to WASM:`, err);
    const s = await ort.InferenceSession.create(bufOrPath, wasmOpts);
    return { session: s, usedGpu: false, errMsg };
  }
}

async function loadBackendSessions(preferredMode = activeBackendPref || "wasm") {
  cancelIdleGpuRelease();
  // 默认纯处理器推理 ("wasm")；仅当用户主动切换至 "webgpu" 且显存探测通过时才启用显卡加速
  const pref = preferredMode === "webgpu" ? "webgpu" : "wasm";
  const wantGpuChanged = activeBackendPref !== pref;
  if (!wantGpuChanged && activeBackend && sessions.te && sessions.fm && sessions.voc) {
    return {
      backend: activeBackend,
      backendPref: activeBackendPref,
      gpuHardwareAvailable,
      gpuVramOk,
      gpuVramReason,
      gpuAdapter: activeGpuName,
      gpuDiag: activeGpuDiag,
      fallbackReason: lastFallbackReason,
      cacheInfo: lastCacheInfo,
    };
  }

  activeBackendPref = pref;
  // 若 te 和 voc 已就绪（例如仅切换 GPU/CPU 或空闲释放后重载 fm），仅重拉 fm，避免重复加载另外两个模块
  const reloadFmOnly = Boolean(sessions.te && sessions.voc);
  if (reloadFmOnly) {
    await disposeSessions("fm");
  } else {
    await disposeSessions();
  }
  // 流水线并行：拉取/解压 ORT WASM 与读取 manifest + 第一个模型并行执行，缩短冷启动耗时
  const ortPrepPromise = prepareOrtWasmBlobRuntime();
  const gpu = await detectWebGPU();
  gpuHardwareAvailable = gpu.available;
  gpuVramOk = gpu.vramOk;
  gpuVramReason = gpu.vramReason || "";
  const allowWebGPU = pref === "webgpu" && gpu.available && gpu.vramOk;
  const wasmOpts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
  const webgpuOpts = { executionProviders: ["webgpu"], graphOptimizationLevel: "all" };
  let fmUsedGpu = false;
  let vocUsedGpu = false;
  let gpuErrMsg = "";
  const tLoad0 = performance.now();

  let manifest = null;
  try {
    manifest = await fetchManifest("./models_chunks/manifest.json");
  } catch (_) {
    manifest = null;
  }

  if (manifest && manifest.models) {
    const fullOrder = [
      ["te", "音韵词典模块 (1/3)"],
      ["fm", "古音发声核心 (2/3)"],
      ["voc", "高保真声音模块 (3/3)"],
    ];
    const order = reloadFmOnly ? [["fm", "古音发声核心"]] : fullOrder;
    const totalAllGz = order.reduce((acc, [k]) => acc + (manifest.models[k]?.totalGzSize || 0), 0);
    const totalAllChunks = order.reduce((acc, [k]) => acc + (manifest.models[k]?.chunks?.length || 0), 0);
    let doneGzBase = 0;
    let doneCachedChunks = 0;
    let doneNetChunks = 0;
    let opfsHits = 0;
    let allFromCache = true;

    for (const [key, label] of order) {
      const mInfo = manifest.models[key];
      let lastReportAt = 0;
      const res = await loadChunkedModel(key, mInfo, "./models_chunks/", {
        concurrency: 5,
        retries: 3,
        onProgress: (p) => {
          const now = performance.now();
          if (p.phase !== "done" && now - lastReportAt < 80) return;
          lastReportAt = now;
          const curLoaded = doneGzBase + p.loadedBytes;
          const curCached = doneCachedChunks + p.cachedChunks;
          const curNet = doneNetChunks + p.downloadedChunks;
          const pct = Math.min(99, Math.round((curLoaded / Math.max(1, totalAllGz)) * 99));
          const srcTag = p.phase === "cache" ? "正在读取本地语音库" : "正在准备语音库";
          self.postMessage({
            type: "download_progress",
            pct,
            loadedBytes: curLoaded,
            totalBytes: totalAllGz,
            cachedChunks: curCached,
            downloadedChunks: curNet,
            totalChunks: totalAllChunks,
            speedMBps: p.speedMBps,
            phase: p.phase,
            modelKey: key,
            stage: `${srcTag} · ${label}`,
          });
        },
      });
      doneGzBase += mInfo.totalGzSize || 0;
      doneCachedChunks += res.cachedChunks;
      doneNetChunks += res.downloadedChunks;
      if (res.fromOpfs) opfsHits += 1;
      if (!res.fromCacheOnly) allFromCache = false;

      await ortPrepPromise;
      self.postMessage({
        type: "status",
        stage: `正在初始化语音引擎 · ${label}...`,
      });
      // 传入拼装好的 Uint8Array 并立刻释放 JS 缓冲引用，降低峰值内存
      let buf = res.bytes;
      const created = await createHybridSession(key, buf, allowWebGPU, wasmOpts, webgpuOpts);
      sessions[key] = created.session;
      if (key === "fm") { fmUsedGpu = created.usedGpu; if (created.errMsg) gpuErrMsg = created.errMsg; }
      if (key === "voc") vocUsedGpu = created.usedGpu;
      buf = null;
    }

    if (!reloadFmOnly || !lastCacheInfo) {
      lastCacheInfo = {
        fromCacheOnly: allFromCache,
        opfsHits,
        cachedChunks: doneCachedChunks,
        downloadedChunks: doneNetChunks,
        totalChunks: totalAllChunks,
        totalRawMB: +(manifest.totalRawBytes / 1e6).toFixed(1),
        totalGzMB: +(manifest.totalGzBytes / 1e6).toFixed(1),
        buildId: manifest.buildId || "",
        elapsedSec: +((performance.now() - tLoad0) / 1000).toFixed(2),
      };
    }
  } else {
    await ortPrepPromise;
    if (!reloadFmOnly) {
      self.postMessage({ type: "status", stage: "正在加载语音引擎 (1/3)..." });
      sessions.te = (await createHybridSession("te", "./models/text_encoder_int8.onnx", false, wasmOpts, webgpuOpts)).session;
    }
    self.postMessage({ type: "status", stage: reloadFmOnly ? "正在切换计算引擎..." : "正在加载语音引擎 (2/3)..." });
    const fmC = await createHybridSession("fm", "./models/fm_decoder_int8.onnx", allowWebGPU, wasmOpts, webgpuOpts);
    sessions.fm = fmC.session;
    fmUsedGpu = fmC.usedGpu;
    if (!reloadFmOnly) {
      self.postMessage({ type: "status", stage: "正在加载语音引擎 (3/3)..." });
      const vocC = await createHybridSession("voc", "./models/vocos_spec_int8.onnx", false, wasmOpts, webgpuOpts);
      sessions.voc = vocC.session;
      vocUsedGpu = vocC.usedGpu;
    }
    lastCacheInfo = null;
  }

  activeBackend = fmUsedGpu ? "webgpu" : "wasm";
  activeGpuName = gpu.adapterName || "";
  activeGpuDiag = fmUsedGpu
    ? gpu.diag
    : pref === "wasm"
      ? (gpu.available && gpu.vramOk
        ? `默认纯处理器运行（检测到可用显卡：${gpu.adapterName}，可点击切换显卡加速）`
        : gpu.diag)
      : (gpuErrMsg ? `WebGPU session error: ${gpuErrMsg}` : gpu.diag);
  console.log("[Hanfire Worker] Runtime initialized:", {
    backend: activeBackend,
    backendPref: activeBackendPref,
    gpuHardwareAvailable,
    gpuVramOk,
    gpuVramReason: gpuVramReason || null,
    gpuAdapter: activeGpuName || null,
    gpuDiag: activeGpuDiag,
    fmOnWebGPU: fmUsedGpu,
    vocOnWebGPU: vocUsedGpu,
    quantization: "Weight-Only INT8 (MatMulNBits-8 bs=32 asym, FFN DP4A + Attn FP32)",
    numThreads: ort.env.wasm.numThreads,
    crossOriginIsolated: !!self.crossOriginIsolated,
    cacheInfo: lastCacheInfo,
  });
  lastFallbackReason = activeBackend === "webgpu"
    ? ""
    : self.crossOriginIsolated
      ? ""
      : "当前处于纯处理器单核模式；若在局域网访问，改用 HTTPS 加密地址即可开启多核加速或显卡加速。";
  scheduleIdleGpuRelease();
  return {
    backend: activeBackend,
    backendPref: activeBackendPref,
    gpuHardwareAvailable,
    gpuVramOk,
    gpuVramReason,
    gpuAdapter: activeGpuName,
    gpuDiag: activeGpuDiag,
    fallbackReason: lastFallbackReason,
    cacheInfo: lastCacheInfo,
  };
}

const ONSET_TO_ZV = {
  "": "", "∅": "", "ʔ": "",
  "p": "b0", "pʰ": "p0", "b": "OC_b", "m": "m0", "m̥": "OC_hm", "f": "f0", "v": "EM_ʋ", "ʋ": "EM_ʋ",
  "t": "d0", "tʰ": "t0", "d": "OC_d", "n": "n0", "n̥": "OC_hn", "l": "l0", "l̥": "OC_hl", "r": "l0", "ɻ": "r0",
  "ʈ": "zh0", "ʈʰ": "ch0", "ɖ": "OC_dʒ", "ɳ": "n0",
  "ts": "z0", "t͡s": "z0", "tsʰ": "c0", "t͡sʰ": "c0", "dz": "OC_dz", "d͡z": "OC_dz", "s": "s0", "z": "OC_z",
  "tɕ": "j0", "t͡ɕ": "j0", "tɕʰ": "q0", "t͡ɕʰ": "q0", "dʑ": "OC_dʑ", "d͡ʑ": "OC_dʑ", "ɕ": "x0", "ʑ": "OC_ʑ", "ɲ": "OC_ɲ",
  "tʃ": "zh0", "t͡ʃ": "zh0", "tʃʰ": "ch0", "t͡ʃʰ": "ch0", "dʒ": "OC_dʒ", "d͡ʒ": "OC_dʒ", "ʃ": "sh0", "ʒ": "OC_ʒ",
  "tʂ": "zh0", "t͡ʂ": "zh0", "tʂʰ": "ch0", "t͡ʂʰ": "ch0", "dʐ": "OC_dʒ", "d͡ʐ": "OC_dʒ", "ʂ": "sh0", "ʐ": "r0",
  "kʷ": "g0", "kʷʰ": "k0", "gʷ": "OC_g", "ɡʷ": "OC_g",
  "k": "g0", "kʰ": "k0", "g": "OC_g", "ɡ": "OC_g", "ŋ": "OC_ŋ", "ŋ̊": "OC_hŋ",
  "x": "h0", "h": "h0", "χ": "h0", "ɣ": "OC_ɣ", "ɦ": "OC_ɣ", "q": "g0", "qʰ": "k0", "ɢ": "OC_g",
  "j": "y0", "w": "w0", "ɥ": "y0",
};

const IPA_VOWELS = new Set([
  "a", "ɑ", "æ", "ɐ", "ɒ", "e", "ɛ", "ə", "ɜ", "ø", "œ", "ɤ", "i", "ɪ", "ɨ", "ɿ", "ʅ",
  "o", "ɔ", "ʌ", "u", "ʊ", "ʉ", "ɯ", "y",
]);

function resolveFinalTokensJS(t2i, medial, nucleus, coda, tone, checked) {
  const m = medial === "j" || medial === "w" || medial === "ɥ" ? medial : "";
  const v = nucleus || "a";
  const c = coda || "";
  const low = "aɑæɐɔɒ".includes(v);
  const backRound = "uʊʉo".includes(v);

  if (c === "m") {
    let base;
    if (m === "j" || m === "ɥ") base = "aɑæeɛ".includes(v) ? `OC_iam${tone}` : `OC_iəm${tone}`;
    else if (m === "w") base = `uan${tone}`;
    else base = "aɑæ".includes(v) ? `OC_am${tone}` : `OC_əm${tone}`;
    return [base, "OC_coda_m"];
  }
  if (checked || c === "p" || c === "t" || c === "k") {
    if (c === "p") return ["aɑæ".includes(v) ? "OC_ap_chk" : "OC_əp_chk", null];
    if (c === "t") {
      if (m === "j" || m === "ɥ" || "eɛiɪ".includes(v)) return ["OC_iet_chk", null];
      return ["aɑæ".includes(v) ? "OC_at_chk" : "OC_ət_chk", null];
    }
    if (m === "j" || m === "ɥ") {
      if (low || "eɛ".includes(v)) return ["OC_iak_chk", null];
      if (backRound) return ["OC_iuk_chk", null];
      return ["OC_iək_chk", null];
    }
    if (backRound) return ["OC_uk_chk", null];
    if (low) return ["OC_ak_chk", null];
    return ["OC_ək_chk", null];
  }
  if (c === "n" || c === "r") {
    if (m === "ɥ") return ["aɑæɐeɛɜ".includes(v) ? `uan${tone}` : `un${tone}`, null];
    if (m === "w") return ["aɑæɐeɛɔɒ".includes(v) ? `uan${tone}` : `un${tone}`, null];
    if (m === "j") {
      if ("uʊʉoɔ".includes(v)) return [`un${tone}`, null];
      return ["aɑæɐeɛɜ".includes(v) ? `ian${tone}` : `in${tone}`, null];
    }
    if (v === "e") return [`ian${tone}`, null];
    if ("iɪ".includes(v)) return [`in${tone}`, null];
    if ("uʊʉyo".includes(v)) return [`un${tone}`, null];
    return ["aɑæɐɛɔɒ".includes(v) ? `an${tone}` : `en${tone}`, null];
  }
  if (c === "ŋ") {
    if (m === "w" || m === "ɥ") {
      if (m === "ɥ" && "æɐeɛi".includes(v)) return [`iong${tone}` in t2i ? `iong${tone}` : "iong3", null];
      return ["aɑæɐeɛɔɒ".includes(v) ? `uang${tone}` : `ong${tone}`, null];
    }
    if (m === "j") {
      if ("aɑ".includes(v)) return [`iang${tone}`, null];
      if ("uoɔ".includes(v)) return [`iong${tone}` in t2i ? `iong${tone}` : "iong3", null];
      return [`ing${tone}`, null];
    }
    if ("iɪ".includes(v)) return [`ing${tone}`, null];
    if ("aɑæɔɒ".includes(v)) return [`ang${tone}`, null];
    if ("ouʊ".includes(v)) return [`ong${tone}`, null];
    return [`eng${tone}`, null];
  }
  if (c === "j" || c === "i") {
    if (m === "w" || m === "ɥ") return ["aɑæ".includes(v) ? `OC_iwai${tone}` : `OC_iwei${tone}`, null];
    if (m === "j") return ["aɑæ".includes(v) ? `OC_iwai${tone}` : `OC_iei${tone}`, null];
    if ("eɛ".includes(v)) return [`OC_iei${tone}`, null];
    if (v === "ɒ") return [`MC_ɒi${tone}`, null];
    if ("uʊoɔ".includes(v)) return [`ui${tone}`, null];
    return ["aɑæɐ".includes(v) ? `ai${tone}` : `ei${tone}`, null];
  }
  if (c === "w" || c === "u") {
    if (m === "j" || m === "ɥ") {
      return ["aɑæɐeɛ".includes(v) ? `iao${tone}` : `OC_iu${tone}`, null];
    }
    if ("eɛ".includes(v)) return [`OC_eu${tone}`, null];
    return ["aɑæɐʌɒ".includes(v) ? `ao${tone}` : `ou${tone}`, null];
  }
  if (m === "j") {
    if ("aɑæɐ".includes(v)) return [`ia${tone}`, null];
    if ("eɛ".includes(v)) return [`ie${tone}`, null];
    if ("uʊ".includes(v)) return [`OC_iu${tone}`, null];
    if ("oɔ".includes(v)) return [`iao${tone}`, null];
    if ("iɪ".includes(v)) return [`i${tone}`, null];
    return [`OC_iə${tone}`, null];
  }
  if (m === "ɥ") {
    if ("aɑæɐ".includes(v)) return [`ua${tone}`, null];
    if ("eɛ".includes(v)) return [`ue${tone}`, null];
    if ("iɪy".includes(v)) return [`v${tone}` in t2i ? `v${tone}` : "v2", null];
    return [`OC_wə${tone}`, null];
  }
  if (m === "w") {
    if ("aɑæɐ".includes(v)) return [`ua${tone}`, null];
    if ("oɔə".includes(v)) return [`uo${tone}`, null];
    if ("eɛ".includes(v)) return [`OC_iwei${tone}`, null];
    return [`u${tone}`, null];
  }
  if ("aɑæɐ".includes(v)) return [`a${tone}`, null];
  if ("eɛ".includes(v)) return [`ie${tone}`, null];
  if ("iɪɿʅ".includes(v)) return [`i${tone}`, null];
  if ("oɔɒ".includes(v)) return [`uo${tone}`, null];
  if ("uʊʉ".includes(v)) return [`u${tone}`, null];
  if (v === "y") return [`v${tone}` in t2i ? `v${tone}` : "v2", null];
  return [`e${tone}`, null];
}

/** 浏览器端实时 5 槽 IPA / 拼音+声调编译器（与 Python _compile_ipa_override 对齐） */
function compileIpaSpecLive(lex, ipaSpec) {
  if (!lex.t2i) return null;
  let s = ipaSpec.trim().replace(/^[/\[]|[/\]]$/g, "").replace(/\*/g, "");
  if (!s || /[\u4e00-\u9fff]/.test(s)) return null;
  let tone = "1";
  if ("1234".includes(s[s.length - 1])) {
    tone = s[s.length - 1];
    s = s.slice(0, -1);
  } else if (s.endsWith("h") || s.endsWith("s")) {
    tone = "4";
    s = s.slice(0, -1);
  } else if (s.endsWith("ʔ") || s.endsWith("?")) {
    tone = "3";
    s = s.slice(0, -1);
  }
  // 常见拼音/拉丁便捷写法归一化
  s = s
    .replace(/ng/g, "ŋ")
    .replace(/^zh/, "tʂ")
    .replace(/^ch/, "tʂʰ")
    .replace(/^sh/, "ʂ")
    .replace(/^hu(?=[aɑeo])/, "xw")
    .replace(/(ts|tɕ|tʃ|tʂ|p|t|k|q)h(?=[aeiouɑæɐəɛɜɪɨɔœøɒʌʊʉɯɤjwɥr])/g, "$1ʰ");

  // 1) 最长匹配声母
  const onsetKeys = Object.keys(ONSET_TO_ZV)
    .filter((k) => k.length > 0 && k !== "j" && k !== "w" && k !== "ɥ")
    .sort((a, b) => b.length - a.length);
  let onset = "";
  for (const ok of onsetKeys) {
    if (s.startsWith(ok)) {
      onset = ok;
      s = s.slice(ok.length);
      break;
    }
  }
  // 2) 介音
  let medial = "";
  if (onset.endsWith("ʷ")) medial = "w";
  if (s.startsWith("j") || s.startsWith("w") || s.startsWith("ɥ")) {
    medial = s[0];
    s = s.slice(1);
  } else if (s.length >= 2 && s[0] === "i" && IPA_VOWELS.has(s[1])) {
    medial = "j";
    s = s.slice(1);
  } else if (s.length >= 2 && s[0] === "u" && IPA_VOWELS.has(s[1])) {
    medial = "w";
    s = s.slice(1);
  } else if (s.length >= 2 && s[0] === "y" && IPA_VOWELS.has(s[1])) {
    medial = "ɥ";
    s = s.slice(1);
  }
  // 3) 韵尾
  let coda = "";
  if (s.length >= 2 && "ptkmnŋrjwiu".includes(s[s.length - 1])) {
    const last = s[s.length - 1];
    // 保留单核如 u/i 本身不切空
    if (!(s.length === 1)) {
      coda = last;
      s = s.slice(0, -1);
    }
  }
  const nucleus = s[0] || "a";
  const checked = coda === "p" || coda === "t" || coda === "k";
  if (checked) tone = "4";

  let initTok = ONSET_TO_ZV[onset] || "";
  if (!initTok) {
    if (medial === "j" || medial === "ɥ") initTok = "y0";
    else if (medial === "w") initTok = "w0";
  }
  const [finTok, codaTok] = resolveFinalTokensJS(lex.t2i, medial, nucleus, coda, tone, checked);
  if (!(finTok in lex.t2i)) return null;

  const total = checked ? 92 : 100;
  const flat = [];
  const dbg = [];
  if (initTok && initTok in lex.t2i) {
    const wI = checked ? 26 : 30;
    if (codaTok && codaTok in lex.t2i && !checked) {
      flat.push(lex.t2i[initTok], wI, lex.t2i[finTok], total - wI - 20, lex.t2i[codaTok], 20);
      dbg.push(initTok, finTok, codaTok);
    } else {
      flat.push(lex.t2i[initTok], wI, lex.t2i[finTok], total - wI);
      dbg.push(initTok, finTok);
    }
  } else {
    flat.push(lex.t2i[finTok], Math.min(total, checked ? 76 : 72));
    dbg.push(finTok);
  }
  const flags = (initTok ? 0 : 1) | (checked ? 4 : 0);
  return [flat, ipaSpec.trim(), `ipa:${dbg.join("+")}`, flags];
}

function getPolyphonesForChar(lex, mode, ch) {
  const poly = lex.polyphones || {};
  const prefix = `${mode}|${ch}|`;
  const out = [];
  const seen = new Set();
  for (const [k, entry] of Object.entries(poly)) {
    if (!k.startsWith(prefix)) continue;
    const spec = k.slice(prefix.length);
    // 优先展示完整地位描述（如 羣開三B支去），避免与单字调类（去）重复展示同一读音
    const sig = `${entry[1]}|${entry[2]}`;
    if (spec.length === 1 && seen.has(sig)) continue;
    seen.add(sig);
    out.push({ spec, ipa: entry[1], tokens: entry[2] });
  }
  // 将完整描述排在前面
  out.sort((a, b) => b.spec.length - a.spec.length);
  return out;
}

function resolveCharSpec(lex, mode, ch, spec, modeLex) {
  if (!spec) return modeLex[ch] || null;
  const s = spec.trim();
  const poly = lex.polyphones || {};
  if (poly[`${mode}|${ch}|${s}`]) return poly[`${mode}|${ch}|${s}`];
  if (s.startsWith("同") && s.length >= 2) {
    return modeLex[s[1]] || modeLex[ch] || null;
  }
  if ((s.includes(":") || s.includes("+")) && lex.t2i) {
    const flat = [];
    const names = [];
    for (const part of s.split("+")) {
      const p = part.trim();
      if (!p) continue;
      const [tok, wStr] = p.split(":");
      if (tok in lex.t2i) {
        flat.push(lex.t2i[tok], parseInt(wStr || "50", 10));
        names.push(tok);
      }
    }
    if (flat.length > 0) return [flat, s, names.join("+"), 0];
  }
  if (mode === "mc" && s.includes("_") && lex.t2i) {
    let candFin = s.startsWith("Q_") ? s : `Q_${s}`;
    if (!(candFin in lex.t2i)) {
      const idx = s.lastIndexOf("_");
      if (idx > 0) {
        const yl = s.slice(0, idx).replace(/^Q_/, "");
        const tc = s.slice(idx + 1);
        const alt = `Q_${yl}_${tc.includes("入") ? "入" : "舒"}_${tc}`;
        if (alt in lex.t2i) candFin = alt;
      }
    }
    if (candFin in lex.t2i) {
      const checked = candFin.includes("_入_");
      const total = checked ? 92 : 100;
      const base = modeLex[ch];
      if (base && base[0] && base[0].length >= 2) {
        const iniId = base[0][0];
        const wI = checked ? 26 : 30;
        return [[iniId, wI, lex.t2i[candFin], total - wI], s, candFin, checked ? 4 : 0];
      }
      return [[lex.t2i[candFin], total], s, candFin, checked ? 4 : 0];
    }
  }
  // 5) 纯拉丁/IPA 实时编译（不含汉字，避免误判切韵重纽描述 A/B）
  if (!/[\u4e00-\u9fff]/.test(s) && /[a-zA-Z\u0250-\u02af]/.test(s)) {
    const live = compileIpaSpecLive(lex, s);
    if (live) return live;
  }
  return modeLex[ch] || null;
}

/**
 * 与 Python QieyunMCCompiler / ThreeEraHanfireCompiler 100% 对齐的分句编译器
 */
export function compileTextToClauses(lex, text, mode) {
  const modeLex = lex.modes[mode] || lex.modes.mc;
  const rawParts = text
    .replace(/\r\n/g, "\n")
    .split(/([，,。！？!?;；\n]+)/)
    .filter((s) => s.length > 0);

  const clauses = [];
  let charGroups = [];
  let curGloss = [];
  let prevEndsOpen = false;
  let lastFlags = 0;
  const hiatus = lex.hiatus_pause || null;
  const toneOfId = lex.tone_of_id || {};
  const sandhi3to2 = lex.sandhi_3to2 || {};
  const holdW = lex.rhyme_hold_w ?? 16;

  const grpNucleusTone = (grp) => {
    for (let i = grp.length - 1; i >= 0; i--) {
      const t = toneOfId[String(grp[i][0])];
      if (t) return [i, t];
    }
    return [-1, ""];
  };

  const flushClause = (punctChar) => {
    if (charGroups.length === 0) return;
    // 上下文连上变调防气泡音 (em: 3+3->2+3; oc: _chk+3->2 及 3+3+3->2+2+3)
    if ((mode === "oc" || mode === "em" || mode === "em_zyyy") && charGroups.length >= 2) {
      for (let gi = 0; gi < charGroups.length - 1; gi++) {
        const [iCur, tCur] = grpNucleusTone(charGroups[gi]);
        if (iCur < 0 || tCur !== "3") continue;
        const [, tNxt] = grpNucleusTone(charGroups[gi + 1]);
        const [, tNxt2] = gi + 2 < charGroups.length ? grpNucleusTone(charGroups[gi + 2]) : [-1, ""];
        const [, tPrv] = gi > 0 ? grpNucleusTone(charGroups[gi - 1]) : [-1, ""];
        const needSandhi =
          mode === "em" || mode === "em_zyyy"
            ? tNxt === "3"
            : tPrv === "chk" || (tNxt === "3" && (tNxt2 === "3" || tPrv === "2"));
        if (needSandhi) {
          const origId = String(charGroups[gi][iCur][0]);
          if (origId in sandhi3to2) {
            charGroups[gi][iCur] = [sandhi3to2[origId], charGroups[gi][iCur][1]];
          }
        }
      }
    }

    // 上古音 (oc) 连续高前元音防粘连：前字开元音 i 族收尾（如“伊”）且后字为鼻音声母 + i 族韵母（如“人”）时加厚鼻音声母至 38w
    if (mode === "oc" && charGroups.length >= 2 && lex.t2i) {
      const id2tok = {};
      for (const [k, v] of Object.entries(lex.t2i)) id2tok[v] = k;
      const nasalInis = new Set(["n0", "m0", "OC_ɲ", "OC_ŋ"]);
      for (let gi = 1; gi < charGroups.length; gi++) {
        const prevGrp = charGroups[gi - 1];
        const curGrp = charGroups[gi];
        if (prevGrp.length === 0 || curGrp.length < 2) continue;
        const prevFin = id2tok[prevGrp[prevGrp.length - 1][0]] || "";
        const cIni = id2tok[curGrp[0][0]] || "";
        const cFin = id2tok[curGrp[1][0]] || "";
        const prevStripped = prevFin.replace(/[0-5]+$/, "");
        const prevIsOpenI =
          (prevFin.startsWith("i") || prevFin.startsWith("OC_i")) &&
          !prevStripped.endsWith("n") &&
          !prevStripped.endsWith("ng") &&
          !prevStripped.endsWith("m") &&
          !prevFin.endsWith("_chk");
        const curIsI = cFin.startsWith("i") || cFin.startsWith("OC_i");
        if (prevIsOpenI && nasalInis.has(cIni) && curIsI && curGrp[0][1] < 38) {
          const delta = 38 - curGrp[0][1];
          curGrp[0] = [curGrp[0][0], 38];
          curGrp[1] = [curGrp[1][0], Math.max(36, curGrp[1][1] - delta)];
        }
      }
    }

    const flatItems = [];
    for (const grp of charGroups) {
      for (const pair of grp) flatItems.push([pair[0], pair[1]]);
    }

    // 句末韵脚稳态延音（MC 入声字与 OC/EM 开口零声母字不额外叠加，严格遵循 AGENTS.md）
    const lastZeroOpen = (lastFlags & 1) !== 0;
    const lastIsMcChecked = mode === "mc" && (lastFlags & 4) !== 0;
    if (holdW > 0 && flatItems.length > 0 && !lastZeroOpen && !lastIsMcChecked) {
      flatItems[flatItems.length - 1][1] += holdW;
    }

    const pKey = "。！？!?".includes(punctChar) ? "." : ",";
    clauses.push({ items: flatItems, gloss: curGloss, endPunct: pKey });
    charGroups = [];
    curGloss = [];
    prevEndsOpen = false;
    lastFlags = 0;
  };

  const tagRe = /\[([^\[\]=]+)=([^\[\]]+)\]/y;
  const modeBigrams = lex.bigrams?.[mode] || {};
  const bigramKeys = Object.keys(modeBigrams).sort((a, b) => b.length - a.length);
  const id2tokMap = {};
  if (lex.t2i) {
    for (const [k, v] of Object.entries(lex.t2i)) id2tokMap[v] = k;
  }
  const formatTokSpec = (flatPairs) => {
    const segs = [];
    for (let i = 0; i < flatPairs.length; i += 2) {
      const nm = id2tokMap[flatPairs[i]] || String(flatPairs[i]);
      segs.push(`${nm}:${flatPairs[i + 1]}`);
    }
    return segs.join("+");
  };

  let globalOffset = 0;
  for (const part of rawParts) {
    const partBase = globalOffset;
    globalOffset += part.length;
    if (/^[，,。！？!?;；\n]+$/.test(part)) {
      flushClause(part[0]);
      continue;
    }
    let pos = 0;
    while (pos < part.length) {
      tagRe.lastIndex = pos;
      const m = tagRe.exec(part);
      if (m) {
        const matchStart = partBase + pos;
        const matchEnd = partBase + tagRe.lastIndex;
        const chPart = m[1].trim();
        const spec = m[2].trim();
        for (const ch of chPart) {
          const entry = resolveCharSpec(lex, mode, ch, spec, modeLex);
          if (!entry) continue;
          const [flatPairs, ipaStr, dbgStr, flags = 0] = entry;
          const zeroOpen = (flags & 1) !== 0;
          const endsOpen = (flags & 2) !== 0;
          const grp = [];
          if (mode === "oc" && hiatus && prevEndsOpen && zeroOpen) grp.push([hiatus[0], hiatus[1]]);
          for (let i = 0; i < flatPairs.length; i += 2) grp.push([flatPairs[i], flatPairs[i + 1]]);
          charGroups.push(grp);
          curGloss.push({
            char: ch,
            ipa: ipaStr,
            tokens: dbgStr,
            tokSpec: formatTokSpec(flatPairs),
            spec,
            isCustom: true,
            startPos: matchStart,
            endPos: matchEnd,
            clauseIdx: clauses.length,
            polyphones: getPolyphonesForChar(lex, mode, ch),
          });
          prevEndsOpen = endsOpen;
          lastFlags = flags;
        }
        pos = tagRe.lastIndex;
        continue;
      }
      let matchedBg = null;
      for (const bg of bigramKeys) {
        if (part.startsWith(bg, pos)) { matchedBg = bg; break; }
      }
      if (matchedBg) {
        const entries = modeBigrams[matchedBg];
        for (let bi = 0; bi < matchedBg.length; bi++) {
          const ch = matchedBg[bi];
          const entry = entries[bi];
          if (!entry) continue;
          const [flatPairs, ipaStr, dbgStr, flags = 0] = entry;
          const zeroOpen = (flags & 1) !== 0;
          const endsOpen = (flags & 2) !== 0;
          const grp = [];
          if (mode === "oc" && hiatus && prevEndsOpen && zeroOpen) grp.push([hiatus[0], hiatus[1]]);
          for (let i = 0; i < flatPairs.length; i += 2) grp.push([flatPairs[i], flatPairs[i + 1]]);
          charGroups.push(grp);
          curGloss.push({
            char: ch,
            ipa: ipaStr,
            tokens: dbgStr,
            tokSpec: formatTokSpec(flatPairs),
            spec: "",
            isCustom: false,
            startPos: partBase + pos + bi,
            endPos: partBase + pos + bi + 1,
            clauseIdx: clauses.length,
            polyphones: getPolyphonesForChar(lex, mode, ch),
          });
          prevEndsOpen = endsOpen;
          lastFlags = flags;
        }
        pos += matchedBg.length;
        continue;
      }
      const charStart = partBase + pos;
      const ch = part[pos];
      pos += 1;
      const entry = modeLex[ch] || lex.modes.mc?.[ch];
      if (!entry) continue;
      const [flatPairs, ipaStr, dbgStr, flags = 0] = entry;
      const zeroOpen = (flags & 1) !== 0;
      const endsOpen = (flags & 2) !== 0;
      const grp = [];
      if (mode === "oc" && hiatus && prevEndsOpen && zeroOpen) grp.push([hiatus[0], hiatus[1]]);
      for (let i = 0; i < flatPairs.length; i += 2) grp.push([flatPairs[i], flatPairs[i + 1]]);
      charGroups.push(grp);
      curGloss.push({
        char: ch,
        ipa: ipaStr,
        tokens: dbgStr,
        tokSpec: formatTokSpec(flatPairs),
        spec: "",
        isCustom: false,
        startPos: charStart,
        endPos: charStart + 1,
        clauseIdx: clauses.length,
        polyphones: getPolyphonesForChar(lex, mode, ch),
      });
      prevEndsOpen = endsOpen;
      lastFlags = flags;
      if (curGloss.length >= 14) flushClause(",");
    }
  }
  if (charGroups.length > 0) flushClause(".");
  return clauses;
}

async function synthesize({
  text,
  era = "mc",
  voiceKey = "male_128Hz",
  backend = "wasm",
  numStep = 16,
  guidanceScale = 1.0,
  speed = 1.0,
  customPrompt = null,
  requestMeta = null,
}) {
  cancelIdleGpuRelease();
  await ensureAssetsLoaded();
  const { backend: usedBackend, fallbackReason } = await loadBackendSessions(backend || activeBackendPref || "wasm");
  cancelIdleGpuRelease();

  const clauses = compileTextToClauses(lexicon, text, era);
  if (clauses.length === 0) {
    scheduleIdleGpuRelease();
    throw new Error("未在输入中找到有效汉字，请输入古诗文汉字。");
  }

  const allGloss = clauses.flatMap((c) => c.gloss);
  self.postMessage({
    type: "compiled",
    gloss: allGloss,
    requestMeta,
    backend: usedBackend,
    threads: ort.env.wasm.numThreads,
    fallbackReason,
  });

  const vInfo = customPrompt || promptBank.voices[voiceKey] || promptBank.voices.male_128Hz;
  const pmtItems = vInfo.prompt_items;
  const pmtFrames = vInfo.mel_shape[0];
  const pmtMelFlat =
    vInfo.mel_flat instanceof Float32Array ? vInfo.mel_flat : new Float32Array(vInfo.mel_flat);

  // 字速严丝合缝匹配参考音真实语速：优先使用与 Python 生产渲染逐一对应的 音色×时代 帧预算表
  const baseEraRate = (lexicon.era_rate && lexicon.era_rate[era]) || 24.5;
  const voiceScale = (vInfo.frames_per_100w || 24.5) / 24.5;
  const eraFp = vInfo.era_fp100 && vInfo.era_fp100[era === "em_zyyy" ? "em" : era];
  const framesPer100w = (eraFp || baseEraRate * voiceScale) / Math.max(0.65, Math.min(1.5, speed));

  const pmtIds = pmtItems.map((x) => x[0]);
  const pmtWs = pmtItems.map((x) => x[1]);
  const pmtDurs = weightsToExactFrames(pmtWs, pmtFrames);

  const leadBlock = lexicon.lead_block || [3, 26];
  const tailBlock = lexicon.tail_block || [4, 38];
  const y0Id = lexicon.y0_id ?? -1;
  const w0Id = lexicon.w0_id ?? -1;
  const char0BoostW = lexicon.char0_boost_w ?? 44;
  const id2tok = {};
  if (lexicon.t2i) {
    for (const [k, v] of Object.entries(lexicon.t2i)) id2tok[v] = k;
  }
  const isInitialTok = (tok) => {
    if (!tok) return false;
    if (tok.length >= 2 && tok.endsWith("0")) return true;
    if (tok.startsWith("Q_")) return !tok.includes("_舒_") && !tok.includes("_入_");
    if (tok.startsWith("OC_") || tok.startsWith("EM_") || tok.startsWith("MC_")) {
      const last = tok[tok.length - 1];
      return !(last >= "0" && last <= "9") && !tok.endsWith("_chk") && !tok.includes("coda");
    }
    return false;
  };

  const tStartAll = performance.now();
  const wavChunks = [];
  const totalStepsAll = clauses.length * numStep;
  let completedSteps = 0;

  try {
    for (let cIdx = 0; cIdx < clauses.length; cIdx++) {
      // 严格复刻 sample_clean_zero_carrier: 仅对多槽声母起音首字施加补偿（单槽零声母首字如“安/庵/翁”严禁加权防裂变“安安”），
      // 且三槽位首字补偿加在主元音上，绝不吹胀 y0/w0 介音！
      const tgt = clauses[cIdx].items.map((p) => [p[0], p[1]]);
      if (char0BoostW > 0 && tgt.length >= 2 && isInitialTok(id2tok[tgt[0][0]] || "")) {
        const bI = Math.floor(char0BoostW / 4);
        const bF = char0BoostW - bI;
        tgt[0][1] += bI;
        const finSlot = tgt.length >= 3 && (tgt[1][0] === y0Id || tgt[1][0] === w0Id) ? 2 : 1;
        tgt[finSlot][1] += bF;
      }

      const genAll = [leadBlock, ...tgt, tailBlock];
      const catIds = pmtIds.concat(genAll.map((x) => x[0]));

      const leadFrames = Math.max(6, Math.round((leadBlock[1] * framesPer100w) / 100.0));
      const wSum = tgt.reduce((a, b) => a + b[1], 0);
      const spFrames = Math.max(Math.round((wSum * framesPer100w) / 100.0), tgt.length + 4);
      const tailFrames = Math.max(8, Math.round((tailBlock[1] * framesPer100w) / 100.0));
      const totalFrames = pmtFrames + leadFrames + spFrames + tailFrames;

      const catDurs = pmtDurs.concat(
        [leadFrames],
        weightsToExactFrames(tgt.map((x) => x[1]), spFrames),
        [tailFrames]
      );

      const featDim = 100;
      const textCondData = new Float32Array(totalFrames * featDim);
      const tokBigInt = BigInt64Array.from(catIds, (v) => BigInt(v));
      const tokTensor = new ort.Tensor("int64", tokBigInt, [1, catIds.length]);
      let teOut = null;
      try {
        teOut = await sessions.te.run({ tokens: tokTensor });
        const catEmb = teOut.embed.data;
        let frameCursor = 0;
        for (let s = 0; s < catDurs.length; s++) {
          const dur = catDurs[s];
          const srcOff = s * featDim;
          for (let f = 0; f < dur; f++) {
            textCondData.set(catEmb.subarray(srcOff, srcOff + featDim), frameCursor * featDim);
            frameCursor++;
          }
        }
      } finally {
        tokTensor.dispose?.();
        teOut?.embed?.dispose?.();
      }

      const speechCondData = new Float32Array(totalFrames * featDim);
      speechCondData.set(pmtMelFlat.subarray(0, pmtFrames * featDim), 0);

      const textCondTensor = new ort.Tensor("float32", textCondData, [1, totalFrames, featDim]);
      const speechCondTensor = new ort.Tensor("float32", speechCondData, [1, totalFrames, featDim]);
      const gsTensor = new ort.Tensor("float32", new Float32Array([guidanceScale]), [1]);

      // 跨平台统一确定性帧级零均值球面初值 + F5-TTS Sway(s=-0.8) 调度（与 Python sample_clean_zero_carrier 逐位一致）
      const xData = makeSphericalInitialNoise(totalFrames, featDim, 42, 0.667, 2.5);
      const ts = getSwayTimeSteps(numStep, -0.8);

      try {
        for (let step = 0; step < numStep; step++) {
          const tTensor = new ort.Tensor("float32", new Float32Array([ts[step]]), [1]);
          const xTensor = new ort.Tensor("float32", xData, [1, totalFrames, featDim]);
          let fmOut = null;
          try {
            fmOut = await sessions.fm.run({
              t: tTensor,
              x: xTensor,
              text_condition: textCondTensor,
              speech_condition: speechCondTensor,
              guidance_scale: gsTensor,
            });
            const vData = fmOut.v.data;
            const dt = ts[step + 1] - ts[step];
            for (let i = 0; i < xData.length; i++) {
              xData[i] += vData[i] * dt;
            }
          } finally {
            tTensor.dispose?.();
            xTensor.dispose?.();
            fmOut?.v?.dispose?.();
          }
          completedSteps++;
          self.postMessage({
            type: "progress",
            done: completedSteps,
            total: totalStepsAll,
            coupletIdx: cIdx + 1,
            coupletTotal: clauses.length,
            backend: usedBackend,
          });
          // WebGPU 步间留出 6ms 微间隙给桌面合成器与浏览器界面渲染，避免单显卡设备出现画面掉帧或卡顿
          if (usedBackend === "webgpu" && step + 1 < numStep) {
            await new Promise((r) => setTimeout(r, 6));
          }
        }
      } finally {
        textCondTensor.dispose?.();
        speechCondTensor.dispose?.();
        gsTensor.dispose?.();
      }

      const s0 = pmtFrames + Math.max(0, leadFrames - 2);
      const s1 = totalFrames;
      const sliceFrames = s1 - s0;

      const vocMel = new Float32Array(featDim * sliceFrames);
      const invScale = 1.0 / FEAT_SCALE;
      for (let t = 0; t < sliceFrames; t++) {
        const rowOff = (s0 + t) * featDim;
        for (let c = 0; c < featDim; c++) {
          vocMel[c * sliceFrames + t] = xData[rowOff + c] * invScale;
        }
      }

      sweetSpotMelEnvelopeInPlace(vocMel, sliceFrames);

      const melTensor = new ort.Tensor("float32", vocMel, [1, featDim, sliceFrames]);
      let vocOut = null;
      let wav;
      try {
        vocOut = await sessions.voc.run({ mel: melTensor });
        wav = istftVocos1024(vocOut.spec_real.data, vocOut.spec_imag.data, sliceFrames);
      } finally {
        melTensor.dispose?.();
        vocOut?.spec_real?.dispose?.();
        vocOut?.spec_imag?.dispose?.();
      }
      wav = bandpass75_9500ZeroPhase(wav);
      wav = trimEdgeSilence(wav, SR, 52.0, 35, 65, -58.0);
      wav = safeNormalizeAndFade(wav, 0.055, 0.72, SR);

      wavChunks.push(wav);
      if (cIdx + 1 < clauses.length) {
        const gap = cIdx % 2 === 0 ? 0.26 : 0.42;
        wavChunks.push(new Float32Array(Math.floor(gap * SR)));
      }
    }
  } finally {
    scheduleIdleGpuRelease();
  }

  const totalSamples = wavChunks.reduce((acc, c) => acc + c.length, 0);
  const fullPcm = new Float32Array(totalSamples);
  let off = 0;
  for (const c of wavChunks) {
    fullPcm.set(c, off);
    off += c.length;
  }

  const elapsedMs = performance.now() - tStartAll;
  const audioDurSec = totalSamples / SR;
  const rtf = elapsedMs / 1000.0 / Math.max(0.01, audioDurSec);
  const wavBuf = encodeWavPCM16(fullPcm, SR);

  self.postMessage(
    {
      type: "result",
      wavBuffer: wavBuf,
      pcmPreview: fullPcm.filter((_, idx) => idx % 48 === 0),
      sampleRate: SR,
      durationSec: audioDurSec,
      elapsedMs,
      rtf,
      backend: usedBackend,
      threads: ort.env.wasm.numThreads,
      fallbackReason,
      requestMeta,
      gloss: allGloss,
      buildId: lastCacheInfo?.buildId || "",
    },
    [wavBuf]
  );
}

if (_isWorker) {
  self.onmessage = async (ev) => {
    const msg = ev.data;
    try {
      if (msg.cmd === "inspect_env") {
        // 极简并行首屏探测：词典加载、WebGPU 显存探测（2s 超时保护）、本地 OPFS/IDB 缓存扫描三路并发，零模型/零 ORT 下载！
        const lexPromise = ensureLexiconLoaded().then(() => {
          self.postMessage({ type: "lexicon_ready" });
        });
        const gpuPromise = Promise.race([
          detectWebGPU(),
          new Promise((resolve) =>
            setTimeout(
              () =>
                resolve({
                  available: false,
                  vramOk: false,
                  vramReason: "显卡探测超时，已使用纯处理器推理",
                  maxBufferMB: 0,
                  adapterName: "",
                  diag: "WebGPU probe timeout (2000ms)",
                }),
              2000
            )
          ),
        ]);
        const cachePromise = (async () => {
          try {
            const manifest = await fetchManifest("./models_chunks/manifest.json");
            return await inspectModelCache(manifest);
          } catch (_) {
            return { totalGzBytes: 0, cachedGzBytes: 0, missingGzBytes: 0, totalChunks: 0, cachedChunks: 0, allCached: false, opfsModels: 0 };
          }
        })();

        const [, gpu, cacheProbe] = await Promise.all([lexPromise, gpuPromise, cachePromise]);
        gpuHardwareAvailable = gpu.available;
        gpuVramOk = gpu.vramOk;
        gpuVramReason = gpu.vramReason || "";
        activeGpuName = gpu.adapterName || "";
        activeGpuDiag = gpu.diag || "";
        self.postMessage({
          type: "env_inspected",
          gpuHardwareAvailable: gpu.available,
          gpuVramOk: gpu.vramOk,
          gpuVramReason: gpu.vramReason || "",
          gpuAdapter: gpu.adapterName || "",
          gpuDiag: gpu.diag || "",
          threads: NUM_WASM_THREADS,
          crossOriginIsolated: !!self.crossOriginIsolated,
          cacheProbe,
          melFilterbank: promptBank ? promptBank.mel_filterbank : null,
          defaultPromptItems: promptBank ? promptBank.voices.male_128Hz.prompt_items : null,
        });
      } else if (msg.cmd === "ensure_prompts") {
        await ensurePromptBankLoaded();
        self.postMessage({
          type: "prompts_ready",
          melFilterbank: promptBank.mel_filterbank,
          defaultPromptItems: promptBank.voices.male_128Hz.prompt_items,
        });
      } else if (msg.cmd === "preload" || msg.cmd === "set_backend") {
        await ensureAssetsLoaded();
        const info = await loadBackendSessions(msg.backendPref || "wasm");
        self.postMessage({
          type: "ready",
          backend: info.backend,
          backendPref: info.backendPref,
          gpuHardwareAvailable: info.gpuHardwareAvailable,
          gpuVramOk: info.gpuVramOk,
          gpuVramReason: info.gpuVramReason || "",
          gpuAdapter: info.gpuAdapter || "",
          gpuDiag: info.gpuDiag || "",
          fallbackReason: info.fallbackReason,
          cacheInfo: info.cacheInfo,
          threads: NUM_WASM_THREADS,
          crossOriginIsolated: !!self.crossOriginIsolated,
          melFilterbank: promptBank.mel_filterbank,
          defaultPromptItems: promptBank.voices.male_128Hz.prompt_items,
        });
      } else if (msg.cmd === "clear_cache") {
        await disposeSessions();
        await clearModelCache();
        lastCacheInfo = null;
        self.postMessage({ type: "cache_cleared" });
      } else if (msg.cmd === "preview_compile") {
        await ensureLexiconLoaded();
        const clauses = compileTextToClauses(lexicon, msg.text || "", msg.era || "mc");
        const gloss = clauses.flatMap((c) => c.gloss);
        self.postMessage({
          type: "preview_result",
          reqId: msg.reqId || 0,
          era: msg.era || "mc",
          gloss,
          clauseCount: clauses.length,
        });
      } else if (msg.cmd === "eval_char_spec") {
        await ensureLexiconLoaded();
        const mode = msg.era || "mc";
        const modeLex = lexicon.modes[mode] || lexicon.modes.mc;
        const entry = resolveCharSpec(lexicon, mode, msg.char || "", msg.spec || "", modeLex);
        const id2tok = {};
        if (lexicon.t2i) {
          for (const [k, v] of Object.entries(lexicon.t2i)) id2tok[v] = k;
        }
        let tokSpec = "";
        if (entry && entry[0]) {
          const segs = [];
          for (let i = 0; i < entry[0].length; i += 2) {
            segs.push(`${id2tok[entry[0][i]] || entry[0][i]}:${entry[0][i + 1]}`);
          }
          tokSpec = segs.join("+");
        }
        self.postMessage({
          type: "eval_char_result",
          reqId: msg.reqId || 0,
          ok: Boolean(entry),
          ipa: entry ? entry[1] : "",
          tokens: entry ? entry[2] : "",
          tokSpec,
        });
      } else if (msg.cmd === "synthesize") {
        await synthesize(msg.payload);
      }
    } catch (err) {
      self.postMessage({
        type: "error",
        message: err?.stack || err?.message || String(err),
      });
    }
  };
}
