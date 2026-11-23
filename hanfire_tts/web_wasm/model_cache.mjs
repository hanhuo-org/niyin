/**
 * 汉火古音 TTS —— 浏览器端分片多线程下载 + 原生 Gzip 流式解压 + IndexedDB 分片持久化缓存引擎
 * (model_cache.mjs)
 *
 * 设计亮点：
 * 1. 先按 16 MiB 切原始 ONNX 再逐块 Gzip (.gzbin)，所有单文件 < 15 MiB，天然兼容 Cloudflare Workers Static Assets / R2；
 * 2. 4~6 路并发 fetch + 单片指数退避重试 + 实时字节级进度与速率回报；
 * 3. 使用浏览器原生 DecompressionStream("gzip") 逐片就地解压、SHA-256 校验后直接写入预分配 Uint8Array 与 IndexedDB；
 * 4. IndexedDB 按 16 MiB 解压分片存储（规避 iOS Safari 单条 >100MB 写入 OOM），天然支持断点续传、二次 0.3s 秒开、以及单模型版本增量热更新！
 */

const DB_NAME = "hanfire_tts_model_cache_v1";
const DB_VERSION = 2;
const STORE_CHUNKS = "chunks";
const STORE_META = "meta";
const STORE_HISTORY = "history";
const MAX_HISTORY_ITEMS = 20;
const OPFS_DIR_NAME = "hanfire_models_v1";

// ================================================================
// OPFS（Origin Private File System）完整单文件缓存层
// 优势：零序列化拷贝、不触发 iOS IndexedDB 单条 >100MB OOM、二次直接整包零合并秒读
// ================================================================
async function getOpfsModelsDir(create = false) {
  try {
    if (typeof navigator === "undefined" || !navigator.storage || typeof navigator.storage.getDirectory !== "function") {
      return null;
    }
    const root = await navigator.storage.getDirectory();
    return await root.getDirectoryHandle(OPFS_DIR_NAME, { create });
  } catch (_) {
    return null;
  }
}

function opfsModelFileName(modelKey, version) {
  return `${modelKey}.${version}.onnx`;
}

async function opfsCheckModelFile(modelKey, version, expectedSize) {
  try {
    const dir = await getOpfsModelsDir(false);
    if (!dir) return false;
    const fh = await dir.getFileHandle(opfsModelFileName(modelKey, version), { create: false });
    const file = await fh.getFile();
    return file.size === expectedSize;
  } catch (_) {
    return false;
  }
}

async function opfsReadCompleteModel(modelKey, version, expectedSize) {
  try {
    const dir = await getOpfsModelsDir(false);
    if (!dir) return null;
    const fh = await dir.getFileHandle(opfsModelFileName(modelKey, version), { create: false });
    // Worker 中优先使用 createSyncAccessHandle 直接零拷贝读入预分配缓冲
    if (typeof fh.createSyncAccessHandle === "function") {
      const access = await fh.createSyncAccessHandle();
      try {
        const size = access.getSize();
        if (size !== expectedSize) return null;
        const out = new Uint8Array(size);
        let offset = 0;
        while (offset < size) {
          const n = access.read(out.subarray(offset), { at: offset });
          if (n <= 0) return null;
          offset += n;
        }
        return out;
      } finally {
        try { access.close(); } catch (_) {}
      }
    }
    const file = await fh.getFile();
    if (file.size !== expectedSize) return null;
    const ab = await file.arrayBuffer();
    return new Uint8Array(ab);
  } catch (_) {
    return null;
  }
}

async function opfsWriteCompleteModel(modelKey, version, bytes) {
  try {
    const dir = await getOpfsModelsDir(true);
    if (!dir) return false;
    const targetName = opfsModelFileName(modelKey, version);
    const tmpName = `${targetName}.tmp`;
    let wrote = false;
    const fh = await dir.getFileHandle(targetName, { create: true });
    if (typeof fh.createSyncAccessHandle === "function") {
      const access = await fh.createSyncAccessHandle();
      try {
        access.truncate(0);
        let offset = 0;
        while (offset < bytes.byteLength) {
          const n = access.write(bytes.subarray(offset), { at: offset });
          if (n <= 0) break;
          offset += n;
        }
        access.flush();
        wrote = access.getSize() === bytes.byteLength;
      } finally {
        try { access.close(); } catch (_) {}
      }
    } else if (typeof fh.createWritable === "function") {
      const w = await fh.createWritable();
      try {
        await w.write(bytes);
        await w.close();
        const f = await fh.getFile();
        wrote = f.size === bytes.byteLength;
      } catch (_) {
        try { await w.abort(); } catch (__) {}
        wrote = false;
      }
    }
    if (!wrote) {
      try { await dir.removeEntry(targetName); } catch (_) {}
      try { await dir.removeEntry(tmpName); } catch (_) {}
      return false;
    }
    // 清理该模型的旧版本 OPFS 文件
    try {
      const prefix = `${modelKey}.`;
      for await (const [name] of dir.entries()) {
        if (name.startsWith(prefix) && name !== targetName) {
          try { await dir.removeEntry(name); } catch (_) {}
        }
      }
    } catch (_) {}
    return true;
  } catch (_) {
    return false;
  }
}

async function opfsClearAllModels() {
  try {
    if (typeof navigator === "undefined" || !navigator.storage || typeof navigator.storage.getDirectory !== "function") {
      return;
    }
    const root = await navigator.storage.getDirectory();
    await root.removeEntry(OPFS_DIR_NAME, { recursive: true });
  } catch (_) {}
}

function openCacheDB() {
  if (typeof indexedDB === "undefined") {
    return Promise.resolve(null);
  }
  return new Promise((resolve) => {
    try {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(STORE_CHUNKS)) {
          const store = db.createObjectStore(STORE_CHUNKS, { keyPath: "key" });
          store.createIndex("by_model", "modelKey", { unique: false });
        }
        if (!db.objectStoreNames.contains(STORE_META)) {
          db.createObjectStore(STORE_META, { keyPath: "modelKey" });
        }
        if (!db.objectStoreNames.contains(STORE_HISTORY)) {
          const hStore = db.createObjectStore(STORE_HISTORY, { keyPath: "id" });
          hStore.createIndex("by_created", "createdAt", { unique: false });
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
    } catch (_) {
      resolve(null);
    }
  });
}

function idbGet(db, storeName, key) {
  if (!db) return Promise.resolve(null);
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(storeName, "readonly");
      const req = tx.objectStore(storeName).get(key);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => resolve(null);
    } catch (_) {
      resolve(null);
    }
  });
}

function idbPut(db, storeName, value) {
  if (!db) return Promise.resolve(false);
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(storeName, "readwrite");
      tx.objectStore(storeName).put(value);
      tx.oncomplete = () => resolve(true);
      tx.onerror = () => resolve(false);
    } catch (_) {
      resolve(false);
    }
  });
}

function idbListKeysForModel(db, modelKey) {
  if (!db) return Promise.resolve([]);
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE_CHUNKS, "readonly");
      const idx = tx.objectStore(STORE_CHUNKS).index("by_model");
      const req = idx.getAllKeys(modelKey);
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => resolve([]);
    } catch (_) {
      resolve([]);
    }
  });
}

function idbDeleteKeys(db, keys) {
  if (!db || !keys || keys.length === 0) return Promise.resolve();
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE_CHUNKS, "readwrite");
      const store = tx.objectStore(STORE_CHUNKS);
      for (const k of keys) store.delete(k);
      tx.oncomplete = () => resolve();
      tx.onerror = () => resolve();
    } catch (_) {
      resolve();
    }
  });
}

function idbGetMany(db, storeName, keys) {
  if (!db || !keys || keys.length === 0) return Promise.resolve(new Map());
  return new Promise((resolve) => {
    const out = new Map();
    try {
      // 单次只读事务并发发起全部读取请求，比串行开 N 次事务快 3~5 倍
      const tx = db.transaction(storeName, "readonly");
      const store = tx.objectStore(storeName);
      for (const k of keys) {
        const req = store.get(k);
        req.onsuccess = () => {
          if (req.result) out.set(k, req.result);
        };
      }
      tx.oncomplete = () => resolve(out);
      tx.onerror = () => resolve(out);
      tx.onabort = () => resolve(out);
    } catch (_) {
      resolve(out);
    }
  });
}

export async function clearModelCache() {
  await opfsClearAllModels();
  const db = await openCacheDB();
  if (!db) return true;
  return new Promise((resolve) => {
    try {
      const tx = db.transaction([STORE_CHUNKS, STORE_META], "readwrite");
      tx.objectStore(STORE_CHUNKS).clear();
      tx.objectStore(STORE_META).clear();
      tx.oncomplete = () => {
        db.close();
        resolve(true);
      };
      tx.onerror = () => {
        db.close();
        resolve(false);
      };
    } catch (_) {
      try { db.close(); } catch (__) {}
      resolve(false);
    }
  });
}

async function gunzipUint8(gzBytes) {
  // 若边缘/代理意外已自动解压，直接返回避免二次解压崩溃
  if (gzBytes.length < 2 || gzBytes[0] !== 0x1f || gzBytes[1] !== 0x8b) {
    return gzBytes;
  }
  if (typeof DecompressionStream === "undefined") {
    throw new Error("当前浏览器不支持原生 DecompressionStream('gzip')，请升级浏览器");
  }
  const ds = new DecompressionStream("gzip");
  const stream = new Blob([gzBytes]).stream().pipeThrough(ds);
  const buf = await new Response(stream).arrayBuffer();
  return new Uint8Array(buf);
}

async function sha256Hex(u8) {
  if (typeof crypto === "undefined" || !crypto.subtle) return null;
  const hashBuf = await crypto.subtle.digest("SHA-256", u8);
  const arr = new Uint8Array(hashBuf);
  let hex = "";
  for (let i = 0; i < arr.length; i++) {
    hex += arr[i].toString(16).padStart(2, "0");
  }
  return hex;
}

async function fetchChunkWithProgress(url, expectedGzSize, onDelta, retries = 3) {
  let lastErr = null;
  for (let attempt = 0; attempt <= retries; attempt++) {
    let reported = 0;
    try {
      const resp = await fetch(url, { cache: "force-cache" });
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status} (${url})`);
      }
      if (!resp.body || typeof resp.body.getReader !== "function") {
        const ab = await resp.arrayBuffer();
        const u8 = new Uint8Array(ab);
        onDelta(u8.byteLength);
        return u8;
      }
      const reader = resp.body.getReader();
      const parts = [];
      let total = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value && value.byteLength > 0) {
          parts.push(value);
          total += value.byteLength;
          reported += value.byteLength;
          onDelta(value.byteLength);
        }
      }
      const merged = new Uint8Array(total);
      let off = 0;
      for (const p of parts) {
        merged.set(p, off);
        off += p.byteLength;
      }
      return merged;
    } catch (err) {
      lastErr = err;
      if (reported > 0) onDelta(-reported);
      if (attempt < retries) {
        await new Promise((r) => setTimeout(r, 400 * Math.pow(2, attempt)));
      }
    }
  }
  throw lastErr || new Error(`分片下载失败: ${url}`);
}

export async function fetchManifest(manifestUrl = "./models_chunks/manifest.json") {
  const resp = await fetch(manifestUrl, { cache: "no-cache" });
  if (!resp.ok) {
    throw new Error(`无法加载分片清单 ${manifestUrl} (HTTP ${resp.status})`);
  }
  return await resp.json();
}

/**
 * 仅检查 OPFS 完整模型与 IndexedDB 本地分片缓存命中情况（零模型下载流量），用于手机端流量提醒与缓存状态展示。
 */
export async function inspectModelCache(manifest) {
  if (!manifest || !manifest.models) {
    return { totalGzBytes: 0, cachedGzBytes: 0, missingGzBytes: 0, totalChunks: 0, cachedChunks: 0, allCached: false, opfsModels: 0 };
  }
  const db = await openCacheDB();
  let totalGzBytes = 0;
  let cachedGzBytes = 0;
  let totalChunks = 0;
  let cachedChunks = 0;
  let opfsModels = 0;
  try {
    for (const modelKey of ["te", "fm", "voc"]) {
      const mInfo = manifest.models[modelKey];
      if (!mInfo || !Array.isArray(mInfo.chunks)) continue;
      const modelGz = mInfo.chunks.reduce((a, c) => a + (c.gzSize || 0), 0);
      totalGzBytes += modelGz;
      totalChunks += mInfo.chunks.length;
      // 优先检查 OPFS 完整整包（存在即整模型 100% 已就绪）
      if (await opfsCheckModelFile(modelKey, mInfo.version, mInfo.totalSize)) {
        cachedGzBytes += modelGz;
        cachedChunks += mInfo.chunks.length;
        opfsModels += 1;
        continue;
      }
      const existingKeys = new Set(await idbListKeysForModel(db, modelKey));
      for (const ch of mInfo.chunks) {
        if (existingKeys.has(`${modelKey}:${mInfo.version}:${ch.index}`)) {
          cachedGzBytes += ch.gzSize || 0;
          cachedChunks += 1;
        }
      }
    }
  } finally {
    if (db) db.close();
  }
  const missingGzBytes = Math.max(0, totalGzBytes - cachedGzBytes);
  return {
    totalGzBytes,
    cachedGzBytes,
    missingGzBytes,
    totalChunks,
    cachedChunks,
    allCached: totalChunks > 0 && cachedChunks === totalChunks,
    opfsModels,
  };
}

/**
 * 加载单个分片模型：
 * 1) 优先从 OPFS 整包零合并极速直读（一次 I/O，零解压、零合并）；
 * 2) 若 OPFS 未命中，则单事务并发读取 IndexedDB 已缓存分片（支持断点续传），缺失分片 5 并发下载 -> gzip 解压 -> SHA-256 校验；
 * 3) 全部拼装完成后自动落盘为 OPFS 完整单文件并清理冗余 IndexedDB 分片（仅占 1 份磁盘空间，下次真正零合并秒读）。
 */
export async function loadChunkedModel(
  modelKey,
  modelInfo,
  baseUrl = "./models_chunks/",
  { concurrency = 5, retries = 3, onProgress = null } = {}
) {
  const version = modelInfo.version;
  const chunks = modelInfo.chunks;
  const totalGzBytes = modelInfo.totalGzSize || chunks.reduce((a, c) => a + c.gzSize, 0);
  const t0 = performance.now();

  // 0) 极速通道：OPFS 完整单文件直读（零合并、零解压、零事务开销）
  const opfsBytes = await opfsReadCompleteModel(modelKey, version, modelInfo.totalSize);
  if (opfsBytes) {
    if (onProgress) {
      onProgress({
        modelKey,
        phase: "cache",
        loadedBytes: totalGzBytes,
        totalBytes: totalGzBytes,
        cachedChunks: chunks.length,
        downloadedChunks: 0,
        totalChunks: chunks.length,
        speedMBps: 0,
      });
      onProgress({
        modelKey,
        phase: "done",
        loadedBytes: totalGzBytes,
        totalBytes: totalGzBytes,
        cachedChunks: chunks.length,
        downloadedChunks: 0,
        totalChunks: chunks.length,
        speedMBps: 0,
      });
    }
    return {
      bytes: opfsBytes,
      fromCacheOnly: true,
      fromOpfs: true,
      cachedChunks: chunks.length,
      downloadedChunks: 0,
      totalChunks: chunks.length,
      elapsedMs: performance.now() - t0,
    };
  }

  const db = await openCacheDB();
  const chunkKey = (idx) => `${modelKey}:${version}:${idx}`;

  // 1) 清理该模型旧版本的过期分片（不动其它模型的缓存，实现单模型热更新）
  const existingKeys = new Set(await idbListKeysForModel(db, modelKey));
  const staleKeys = [];
  const validPrefix = `${modelKey}:${version}:`;
  for (const k of existingKeys) {
    if (!k.startsWith(validPrefix)) staleKeys.push(k);
  }
  if (staleKeys.length > 0) {
    await idbDeleteKeys(db, staleKeys);
    for (const k of staleKeys) existingKeys.delete(k);
  }

  // 2) 预分配最终合并缓冲（单次分配，峰值内存最优）
  const assembled = new Uint8Array(modelInfo.totalSize);
  const missing = [];
  let cachedChunks = 0;
  let networkChunks = 0;
  let loadedGzBytes = 0;
  let downloadedNetBytes = 0;

  const emitProgress = (phase) => {
    if (!onProgress) return;
    const elapsedSec = Math.max(0.05, (performance.now() - t0) / 1000);
    const speedMBps = downloadedNetBytes > 0 ? downloadedNetBytes / (1024 * 1024) / elapsedSec : 0;
    onProgress({
      modelKey,
      phase,
      loadedBytes: Math.min(totalGzBytes, Math.max(0, loadedGzBytes)),
      totalBytes: totalGzBytes,
      cachedChunks,
      downloadedChunks: networkChunks,
      totalChunks: chunks.length,
      speedMBps,
    });
  };

  // 3) 单次只读事务并发恢复全部已缓存 IndexedDB 分片（天然断点续传，比串行读快 3~5 倍）
  const keysToRead = chunks.map((ch) => chunkKey(ch.index)).filter((k) => existingKeys.has(k));
  const idbMap = keysToRead.length > 0 ? await idbGetMany(db, STORE_CHUNKS, keysToRead) : new Map();
  for (const ch of chunks) {
    const k = chunkKey(ch.index);
    const rec = idbMap.get(k);
    if (rec && rec.data && rec.data.byteLength === ch.rawSize) {
      assembled.set(rec.data, ch.offset);
      cachedChunks++;
      loadedGzBytes += ch.gzSize;
      continue;
    }
    missing.push(ch);
  }
  if (cachedChunks > 0) emitProgress("cache");

  // 4) 对缺失分片执行多线程并发下载 -> 原生 gzip 解压 -> SHA-256 校验 -> 写回 IndexedDB
  if (missing.length > 0) {
    let cursor = 0;
    const workers = [];
    const nWorkers = Math.min(concurrency, missing.length);

    const runWorker = async () => {
      while (cursor < missing.length) {
        const ch = missing[cursor++];
        const url = new URL(ch.file, new URL(baseUrl, self.location.href)).href;
        const gzBytes = await fetchChunkWithProgress(
          url,
          ch.gzSize,
          (delta) => {
            loadedGzBytes += delta;
            downloadedNetBytes += delta;
            emitProgress("download");
          },
          retries
        );
        const raw = await gunzipUint8(gzBytes);
        if (raw.byteLength !== ch.rawSize) {
          throw new Error(
            `分片 ${ch.file} 解压长度异常: got=${raw.byteLength}, expected=${ch.rawSize}`
          );
        }
        if (ch.rawSha256) {
          const gotHex = await sha256Hex(raw);
          if (gotHex && gotHex !== ch.rawSha256) {
            throw new Error(`分片 ${ch.file} SHA-256 校验失败，请重试`);
          }
        }
        assembled.set(raw, ch.offset);
        networkChunks++;
        // 立刻落盘 IndexedDB，确保中途关闭页面也能断点续传
        await idbPut(db, STORE_CHUNKS, {
          key: chunkKey(ch.index),
          modelKey,
          version,
          index: ch.index,
          rawSize: ch.rawSize,
          data: raw,
        });
        emitProgress("download");
      }
    };

    for (let i = 0; i < nWorkers; i++) workers.push(runWorker());
    await Promise.all(workers);
  }

  // 5) 全部拼装完成后，合并写入 OPFS 完整单文件，并自动删除 IndexedDB 里的分散分片（省 50% 空间 + 下次零合并直读）
  const savedToOpfs = await opfsWriteCompleteModel(modelKey, version, assembled);
  if (savedToOpfs) {
    const allChunkKeys = chunks.map((ch) => chunkKey(ch.index));
    await idbDeleteKeys(db, allChunkKeys);
  }

  await idbPut(db, STORE_META, {
    modelKey,
    version,
    totalSize: modelInfo.totalSize,
    storage: savedToOpfs ? "opfs" : "idb_chunks",
    updatedAt: Date.now(),
  });
  if (db) db.close();

  emitProgress("done");
  return {
    bytes: assembled,
    fromCacheOnly: missing.length === 0,
    fromOpfs: false,
    migratedToOpfs: savedToOpfs,
    cachedChunks,
    downloadedChunks: networkChunks,
    totalChunks: chunks.length,
    elapsedMs: performance.now() - t0,
  };
}

/** 持久化保存合成历史记录（含完整 WAV ArrayBuffer，上限 20 条自动滚动清理，刷新不丢） */
export async function saveHistoryItem(item) {
  const db = await openCacheDB();
  if (!db) return false;
  try {
    await idbPut(db, STORE_HISTORY, item);
    // 超过 MAX_HISTORY_ITEMS 时删除最旧的记录
    const all = await new Promise((resolve) => {
      try {
        const tx = db.transaction(STORE_HISTORY, "readonly");
        const req = tx.objectStore(STORE_HISTORY).getAll();
        req.onsuccess = () => resolve(req.result || []);
        req.onerror = () => resolve([]);
      } catch (_) {
        resolve([]);
      }
    });
    if (all.length > MAX_HISTORY_ITEMS) {
      all.sort((a, b) => (b.createdAt || 0) - (a.createdAt || 0));
      const toDel = all.slice(MAX_HISTORY_ITEMS).map((x) => x.id);
      await new Promise((resolve) => {
        try {
          const tx = db.transaction(STORE_HISTORY, "readwrite");
          const st = tx.objectStore(STORE_HISTORY);
          for (const id of toDel) st.delete(id);
          tx.oncomplete = () => resolve();
          tx.onerror = () => resolve();
        } catch (_) {
          resolve();
        }
      });
    }
    db.close();
    return true;
  } catch (_) {
    try { db.close(); } catch (__) {}
    return false;
  }
}

export async function listHistoryItems() {
  const db = await openCacheDB();
  if (!db) return [];
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE_HISTORY, "readonly");
      const req = tx.objectStore(STORE_HISTORY).getAll();
      req.onsuccess = () => {
        const arr = (req.result || []).sort((a, b) => (b.createdAt || 0) - (a.createdAt || 0));
        db.close();
        resolve(arr);
      };
      req.onerror = () => { db.close(); resolve([]); };
    } catch (_) {
      try { db.close(); } catch (__) {}
      resolve([]);
    }
  });
}

export async function deleteHistoryItem(id) {
  const db = await openCacheDB();
  if (!db) return false;
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE_HISTORY, "readwrite");
      tx.objectStore(STORE_HISTORY).delete(id);
      tx.oncomplete = () => { db.close(); resolve(true); };
      tx.onerror = () => { db.close(); resolve(false); };
    } catch (_) {
      try { db.close(); } catch (__) {}
      resolve(false);
    }
  });
}

export async function clearHistoryItems() {
  const db = await openCacheDB();
  if (!db) return false;
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE_HISTORY, "readwrite");
      tx.objectStore(STORE_HISTORY).clear();
      tx.oncomplete = () => { db.close(); resolve(true); };
      tx.onerror = () => { db.close(); resolve(false); };
    } catch (_) {
      try { db.close(); } catch (__) {}
      resolve(false);
    }
  });
}
