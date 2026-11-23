/**
 * 汉火古音 TTS 浏览器 WASM SIMD (QAT-INT8) 真实端到端推理与声学回归测试 (test_wasm_e2e.mjs)
 * 直接加载浏览器同款 onnxruntime-web WASM 二进制引擎 + hanfire_dsp.mjs + compileTextToClauses，
 * 在纯 WebAssembly (wasm) 后端下合成《静夜思》《江雪》《桃花庵歌》《蒹葭》完整 WAV 并供 audit_regression 核验！
 */
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import { fileURLToPath, pathToFileURL } from "node:url";
import * as ort from "./node_modules/onnxruntime-web/dist/ort.wasm.bundle.min.mjs";
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
} from "./hanfire_dsp.mjs";
import { compileTextToClauses } from "./hanfire_worker.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
// Node 的 fetch 不支持 file://，直接注入浏览器同款 ort-wasm-simd-threaded.wasm 二进制（与浏览器 WASM 后端完全一致）
ort.env.wasm.wasmBinary = fs.readFileSync(path.join(__dirname, "node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm"));
ort.env.wasm.simd = true;
ort.env.wasm.numThreads = 1;
const ONLY = process.argv.slice(2);

const outDir = path.join(__dirname, "../v10_outputs/audio_v12_wasm_int8");
fs.mkdirSync(outDir, { recursive: true });

console.log("1. 加载 V12 Studio 压缩前端词表 (.gz) 与 0.45x 甜区参考音锚...");
const lexGz = fs.readFileSync(path.join(__dirname, "assets/lexicon_3era.json.gz"));
const lexicon = JSON.parse(zlib.gunzipSync(lexGz).toString("utf-8"));
const promptBank = JSON.parse(fs.readFileSync(path.join(__dirname, "assets/voice_prompts.json"), "utf-8"));

console.log("2. 初始化 onnxruntime-web 纯 WASM SIMD QAT-INT8 三大会话...");
const wasmOpts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
const teBuf = fs.readFileSync(path.join(__dirname, "models/text_encoder_int8.onnx"));
const fmBuf = fs.readFileSync(path.join(__dirname, "models/fm_decoder_int8.onnx"));
const vocBuf = fs.readFileSync(path.join(__dirname, "models/vocos_spec_int8.onnx"));

const sessions = {
  te: await ort.InferenceSession.create(teBuf, wasmOpts),
  fm: await ort.InferenceSession.create(fmBuf, wasmOpts),
  voc: await ort.InferenceSession.create(vocBuf, wasmOpts),
};
console.log("  ✓ WASM QAT-INT8 会话初始化成功!");

async function synthPoemWasm(fname, era, voiceKey, text, numStep = 16) {
  if (ONLY.length && !ONLY.some((k) => fname.includes(k))) return;
  const t0 = performance.now();
  const clauses = compileTextToClauses(lexicon, text, era);
  const vInfo = promptBank.voices[voiceKey];
  const pmtItems = vInfo.prompt_items;
  const pmtFrames = vInfo.mel_shape[0];
  const pmtMelFlat = new Float32Array(vInfo.mel_flat);

  const baseEraRate = lexicon.era_rate[era] || 24.5;
  const voiceScale = (vInfo.frames_per_100w || 24.5) / 24.5;
  const framesPer100w = (vInfo.era_fp100 && vInfo.era_fp100[era]) || baseEraRate * voiceScale;

  const pmtIds = pmtItems.map((x) => x[0]);
  const pmtWs = pmtItems.map((x) => x[1]);
  const pmtDurs = weightsToExactFrames(pmtWs, pmtFrames);

  const leadBlock = lexicon.lead_block;
  const tailBlock = lexicon.tail_block;
  const y0Id = lexicon.y0_id;
  const w0Id = lexicon.w0_id;
  const char0BoostW = lexicon.char0_boost_w ?? 44;
  const featDim = 100;
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

  const wavChunks = [];
  for (let cIdx = 0; cIdx < clauses.length; cIdx++) {
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

    const tokBigInt = BigInt64Array.from(catIds, (v) => BigInt(v));
    const teOut = await sessions.te.run({ tokens: new ort.Tensor("int64", tokBigInt, [1, catIds.length]) });
    const catEmb = teOut.embed.data;

    const textCondData = new Float32Array(totalFrames * featDim);
    let cur = 0;
    for (let s = 0; s < catDurs.length; s++) {
      const dur = catDurs[s];
      const srcOff = s * featDim;
      for (let f = 0; f < dur; f++) {
        textCondData.set(catEmb.subarray(srcOff, srcOff + featDim), cur * featDim);
        cur++;
      }
    }

    const speechCondData = new Float32Array(totalFrames * featDim);
    speechCondData.set(pmtMelFlat.subarray(0, pmtFrames * featDim), 0);

    const textCondTensor = new ort.Tensor("float32", textCondData, [1, totalFrames, featDim]);
    const speechCondTensor = new ort.Tensor("float32", speechCondData, [1, totalFrames, featDim]);
    const gsTensor = new ort.Tensor("float32", new Float32Array([1.0]), [1]);

    const xData = makeSphericalInitialNoise(totalFrames, featDim, 42, 0.667, 2.5);
    const ts = getSwayTimeSteps(numStep, -0.8);

    for (let step = 0; step < numStep; step++) {
      const fmOut = await sessions.fm.run({
        t: new ort.Tensor("float32", new Float32Array([ts[step]]), [1]),
        x: new ort.Tensor("float32", xData, [1, totalFrames, featDim]),
        text_condition: textCondTensor,
        speech_condition: speechCondTensor,
        guidance_scale: gsTensor,
      });
      const vData = fmOut.v.data;
      const dt = ts[step + 1] - ts[step];
      for (let i = 0; i < xData.length; i++) xData[i] += vData[i] * dt;
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
    const vocOut = await sessions.voc.run({
      mel: new ort.Tensor("float32", vocMel, [1, featDim, sliceFrames]),
    });
    let wav = istftVocos1024(vocOut.spec_real.data, vocOut.spec_imag.data, sliceFrames);
    wav = bandpass75_9500ZeroPhase(wav);
    wav = trimEdgeSilence(wav, SR, 52.0, 35, 65, -58.0);
    wav = safeNormalizeAndFade(wav, 0.055, 0.72, SR);

    wavChunks.push(wav);
    if (cIdx + 1 < clauses.length) {
      wavChunks.push(new Float32Array(Math.floor((cIdx % 2 === 0 ? 0.26 : 0.42) * SR)));
    }
  }

  const totalSamples = wavChunks.reduce((a, c) => a + c.length, 0);
  const fullPcm = new Float32Array(totalSamples);
  let off = 0;
  for (const c of wavChunks) {
    fullPcm.set(c, off);
    off += c.length;
  }
  const wavBuf = encodeWavPCM16(fullPcm, SR);
  const outPath = path.join(outDir, fname);
  fs.writeFileSync(outPath, Buffer.from(wavBuf));
  const sec = totalSamples / SR;
  const elapsed = (performance.now() - t0) / 1000.0;
  console.log(`  ✓ [WASM-INT8] ${fname.padEnd(36)} | dur=${sec.toFixed(2)}s | time=${elapsed.toFixed(2)}s | RTF=${(elapsed / sec).toFixed(2)}x`);
}

await synthPoemWasm("WASM_MC_静夜思_male.wav", "mc", "male_128Hz", "床前明月光，疑是地上霜。舉頭望明月，低頭思故鄉。", 16);
await synthPoemWasm("WASM_MC_春晓_male.wav", "mc", "male_128Hz", "春眠不覺曉，[處=昌開三魚去]處聞啼鳥。夜來風雨聲，花落知多少。", 16);
await synthPoemWasm("WASM_MC_江雪_male.wav", "mc", "male_128Hz", "千山鳥飛絕，萬徑人蹤滅。孤舟蓑笠翁，獨釣寒江雪。", 16);
await synthPoemWasm("WASM_EM_桃花庵歌_male.wav", "em", "male_128Hz", "桃花塢裏桃花庵，桃花庵下桃花仙。桃花仙人[種=tʂuŋh]桃樹，又摘桃花換酒錢。", 16);
await synthPoemWasm("WASM_OC_蒹葭_male.wav", "oc", "male_128Hz", "蒹葭蒼蒼，白露為霜。所謂伊人，在水一方。", 16);
console.log("✅ 全部 5 首三代代表诗已在纯 WebAssembly (QAT-INT8) 引擎下合成完毕!");
