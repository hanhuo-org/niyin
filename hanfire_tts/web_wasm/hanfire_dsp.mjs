/**
 * 汉火古音 TTS V12 Studio 浏览器纯前端数字信号处理引擎 (hanfire_dsp.mjs)
 * 严格对齐 Python render_v10_studio.py 与 AGENTS.md 声码器红线：
 * 1. 零内部硬削波 Overlap-Add ISTFT (严禁 .clamp(-0.96, 0.96))；
 * 2. Vocos 前级局部过冲高斯平滑包络归拢 (sweetSpotMelEnvelope: maxAbs=3.78, maxRelP90=0.48)；
 * 3. 四阶 Butterworth [75Hz, 9500Hz] 零相位双向滤波 (bandpass75_9500ZeroPhase == scipy.signal.sosfiltfilt)；
 * 4. -58dB 下限静音裁剪 + 无损峰值保护归一化 (targetRms=0.055, maxPeak=0.72) + 25ms 余弦淡入淡出。
 */

export const SR = 24000;
export const N_FFT = 1024;
export const HOP = 256;
export const FEAT_SCALE = 0.1;

const HANN_1024 = new Float32Array(N_FFT);
const HANN_SQ_1024 = new Float32Array(N_FFT);
for (let i = 0; i < N_FFT; i++) {
  const w = 0.5 - 0.5 * Math.cos((2.0 * Math.PI * i) / N_FFT);
  HANN_1024[i] = w;
  HANN_SQ_1024[i] = w * w;
}

const BIT_REV_1024 = new Uint16Array(N_FFT);
for (let i = 0; i < N_FFT; i++) {
  let x = i;
  let y = 0;
  for (let b = 0; b < 10; b++) {
    y = (y << 1) | (x & 1);
    x >>= 1;
  }
  BIT_REV_1024[i] = y;
}

// 与 scipy.signal.butter(4, [75/12000, 9500/12000], btype="band", output="sos") IEEE-754 精确一致的二阶节矩阵
const SOS_BP_75_9500 = [
  [0.4060782270838145, 0.812156454167629, 0.4060782270838145, 1.0, 1.0115334602929664, 0.27843572472255085],
  [1.0, 2.0, 1.0, 1.0, 1.28752301433687, 0.6236204338068799],
  [1.0, -2.0, 1.0, 1.0, -1.9638195984142006, 0.9642025657391778],
  [1.0, -2.0, 1.0, 1.0, -1.9847716943897666, 0.985155091858249],
];

export function fft1024InPlace(re, im, inverse = false) {
  for (let i = 0; i < N_FFT; i++) {
    const j = BIT_REV_1024[i];
    if (i < j) {
      const tr = re[i]; re[i] = re[j]; re[j] = tr;
      const ti = im[i]; im[i] = im[j]; im[j] = ti;
    }
  }
  const sign = inverse ? 1.0 : -1.0;
  for (let len = 2; len <= N_FFT; len <<= 1) {
    const half = len >> 1;
    const ang = (sign * 2.0 * Math.PI) / len;
    const wLenRe = Math.cos(ang);
    const wLenIm = Math.sin(ang);
    for (let i = 0; i < N_FFT; i += len) {
      let wRe = 1.0;
      let wIm = 0.0;
      for (let k = 0; k < half; k++) {
        const uRe = re[i + k];
        const uIm = im[i + k];
        const vRe = re[i + k + half] * wRe - im[i + k + half] * wIm;
        const vIm = re[i + k + half] * wIm + im[i + k + half] * wRe;
        re[i + k] = uRe + vRe;
        im[i + k] = uIm + vIm;
        re[i + k + half] = uRe - vRe;
        im[i + k + half] = uIm - vIm;
        const nextWRe = wRe * wLenRe - wIm * wLenIm;
        wIm = wRe * wLenIm + wIm * wLenRe;
        wRe = nextWRe;
      }
    }
  }
  if (inverse) {
    const invN = 1.0 / N_FFT;
    for (let i = 0; i < N_FFT; i++) {
      re[i] *= invN;
      im[i] *= invN;
    }
  }
}

/**
 * 纯实数 ISTFT 波形重建（零内部硬削波，严格遵守 AGENTS.md）
 */
export function istftVocos1024(specReal, specImag, nFrames) {
  const outLen = N_FFT + (nFrames - 1) * HOP;
  const y = new Float32Array(outLen);
  const wsum = new Float32Array(outLen);
  const re = new Float64Array(N_FFT);
  const im = new Float64Array(N_FFT);
  const nBins = 513;

  for (let t = 0; t < nFrames; t++) {
    for (let k = 0; k < nBins; k++) {
      const idx = k * nFrames + t;
      re[k] = specReal[idx];
      im[k] = specImag[idx];
    }
    im[0] = 0.0;
    im[512] = 0.0;
    for (let k = 1; k < 512; k++) {
      re[N_FFT - k] = re[k];
      im[N_FFT - k] = -im[k];
    }
    fft1024InPlace(re, im, true);
    const pos = t * HOP;
    for (let n = 0; n < N_FFT; n++) {
      y[pos + n] += re[n] * HANN_1024[n];
      wsum[pos + n] += HANN_SQ_1024[n];
    }
  }

  const pad = N_FFT >> 1;
  const validLen = Math.max(0, outLen - 2 * pad);
  const wav = new Float32Array(validLen);
  for (let i = 0; i < validLen; i++) {
    const denom = wsum[i + pad] > 1e-8 ? wsum[i + pad] : 1e-8;
    wav[i] = y[i + pad] / denom;
  }
  return wav;
}

function gaussianFilter1dReflect(arr, sigma) {
  const radius = Math.max(1, Math.round(4.0 * sigma));
  const kernel = new Float64Array(2 * radius + 1);
  let kSum = 0.0;
  const inv2s2 = 1.0 / (2.0 * sigma * sigma);
  for (let k = -radius; k <= radius; k++) {
    const v = Math.exp(-(k * k) * inv2s2);
    kernel[k + radius] = v;
    kSum += v;
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= kSum;

  const n = arr.length;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let acc = 0.0;
    for (let k = -radius; k <= radius; k++) {
      let idx = i + k;
      while (idx < 0 || idx >= n) {
        if (idx < 0) idx = -idx - 1;
        if (idx >= n) idx = 2 * n - idx - 1;
      }
      acc += arr[idx] * kernel[k + radius];
    }
    out[i] = acc;
  }
  return out;
}

/**
 * 与 Python PureVocosDecoder._sweet_spot_mel_envelope 完全一致的局部过冲包络归拢
 * @param {Float32Array} vocMel [100, nFrames] 行优先 (c * nFrames + t)
 */
export function sweetSpotMelEnvelopeInPlace(vocMel, nFrames, maxAbs = 3.78, maxRelP90 = 0.48) {
  if (nFrames < 2) return vocMel;
  const fMax = new Float32Array(nFrames);
  const fP90 = new Float32Array(nFrames);
  const col = new Float32Array(100);
  let maxP90All = -1e9;

  for (let t = 0; t < nFrames; t++) {
    let mx = -1e9;
    for (let c = 0; c < 100; c++) {
      const v = vocMel[c * nFrames + t];
      col[c] = v;
      if (v > mx) mx = v;
    }
    col.sort();
    // numpy.percentile(col, 90) on length 100 -> linear interp at 89.1
    const p90 = col[89] * 0.9 + col[90] * 0.1;
    fMax[t] = mx;
    fP90[t] = p90;
    if (p90 > maxP90All) maxP90All = p90;
  }

  const voicedVals = [];
  const vThr = maxP90All - 1.8;
  for (let t = 0; t < nFrames; t++) {
    if (fP90[t] > vThr) voicedVals.push(fP90[t]);
  }
  let medP90 = maxP90All;
  if (voicedVals.length >= 4) {
    voicedVals.sort((a, b) => a - b);
    const mid = voicedVals.length >> 1;
    medP90 = voicedVals.length % 2 === 1 ? voicedVals[mid] : 0.5 * (voicedVals[mid - 1] + voicedVals[mid]);
  }

  const excess = new Float32Array(nFrames);
  let maxEx = 0.0;
  for (let t = 0; t < nFrames; t++) {
    const ex = Math.max(0.0, fMax[t] - maxAbs, fP90[t] - (medP90 + maxRelP90));
    excess[t] = ex;
    if (ex > maxEx) maxEx = ex;
  }

  if (maxEx > 1e-4) {
    const sm3 = gaussianFilter1dReflect(excess, 3.0);
    const sm13 = gaussianFilter1dReflect(excess, 1.3);
    for (let t = 0; t < nFrames; t++) {
      const delta = Math.max(sm3[t], sm13[t] * 0.85);
      for (let c = 0; c < 100; c++) {
        vocMel[c * nFrames + t] -= delta;
      }
    }
  }
  return vocMel;
}

function sosFilterPass(x) {
  const n = x.length;
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) y[i] = x[i];
  for (let s = 0; s < SOS_BP_75_9500.length; s++) {
    const [b0, b1, b2, , a1, a2] = SOS_BP_75_9500[s];
    let z1 = 0.0;
    let z2 = 0.0;
    for (let i = 0; i < n; i++) {
      const xi = y[i];
      const yi = b0 * xi + z1;
      z1 = b1 * xi - a1 * yi + z2;
      z2 = b2 * xi - a2 * yi;
      y[i] = yi;
    }
  }
  return y;
}

/**
 * 四阶 Butterworth [75Hz, 9500Hz] 零相位双向带通滤波（等价于 scipy.signal.sosfiltfilt）
 */
export function bandpass75_9500ZeroPhase(wav) {
  const n = wav.length;
  if (n < 64) return wav;
  const pad = 32;
  const ext = new Float64Array(n + 2 * pad);
  for (let i = 0; i < pad; i++) {
    ext[pad - 1 - i] = 2.0 * wav[0] - wav[Math.min(n - 1, i + 1)];
    ext[pad + n + i] = 2.0 * wav[n - 1] - wav[Math.max(0, n - 2 - i)];
  }
  for (let i = 0; i < n; i++) ext[pad + i] = wav[i];

  const fwd = sosFilterPass(ext);
  fwd.reverse();
  const bwd = sosFilterPass(fwd);
  bwd.reverse();

  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = bwd[pad + i];
  return out;
}

/**
 * 无损峰值保护归一化 + 25ms 余弦淡入淡出（与 render_v10_studio.py 100% 一致）
 */
export function safeNormalizeAndFade(wav, targetRms = 0.055, maxPeak = 0.72, sr = SR) {
  const n = wav.length;
  if (n === 0) return wav;
  let sq = 0.0;
  let pk = 1e-8;
  for (let i = 0; i < n; i++) {
    const v = wav[i];
    sq += v * v;
    const av = Math.abs(v);
    if (av > pk) pk = av;
  }
  const rms = Math.max(Math.sqrt(sq / n), 1e-8);
  const gain = Math.min(targetRms / rms, maxPeak / pk);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = wav[i] * gain;

  const fade = Math.floor(0.025 * sr);
  if (n > fade * 2) {
    for (let i = 0; i < fade; i++) {
      const u = (Math.PI * i) / (fade - 1);
      out[i] *= 0.5 * (1.0 - Math.cos(u));
      out[n - fade + i] *= 0.5 * (1.0 + Math.cos(u));
    }
  }
  return out;
}

/**
 * 从浏览器解码的 24kHz 单声道 PCM 实时提取 Vocos 100 维缩放 Mel 特征（自动对齐 0.045 RMS 甜区增益）
 */
export function extractVocosMel24k(pcm24k, melFilterbank, sweetSpotNormalize = true) {
  let src = pcm24k;
  if (sweetSpotNormalize && pcm24k.length > 0) {
    let sq = 0.0;
    for (let i = 0; i < pcm24k.length; i++) sq += pcm24k[i] * pcm24k[i];
    const rms = Math.sqrt(sq / pcm24k.length);
    if (rms > 1e-5) {
      const g = Math.min(0.045 / rms, 0.45 / Math.max(rms, 1e-4));
      src = new Float32Array(pcm24k.length);
      for (let i = 0; i < pcm24k.length; i++) src[i] = pcm24k[i] * g;
    }
  }

  const pad = N_FFT >> 1;
  const padded = new Float32Array(src.length + 2 * pad);
  for (let i = 0; i < pad; i++) {
    padded[pad - 1 - i] = src[Math.min(src.length - 1, i + 1)];
    padded[pad + src.length + i] = src[Math.max(0, src.length - 2 - i)];
  }
  padded.set(src, pad);

  const rawFrames = Math.floor((padded.length - N_FFT) / HOP) + 1;
  const expectedFrames = Math.max(1, Math.round(src.length / HOP));
  const numFrames = expectedFrames;
  const melFlat = new Float32Array(numFrames * 100);
  const re = new Float64Array(N_FFT);
  const im = new Float64Array(N_FFT);
  const mag = new Float32Array(513);

  const computeFrames = Math.min(rawFrames, expectedFrames);
  for (let t = 0; t < computeFrames; t++) {
    const base = t * HOP;
    for (let n = 0; n < N_FFT; n++) {
      re[n] = padded[base + n] * HANN_1024[n];
      im[n] = 0.0;
    }
    fft1024InPlace(re, im, false);
    for (let k = 0; k < 513; k++) {
      mag[k] = Math.hypot(re[k], im[k]);
    }
    const outBase = t * 100;
    for (let m = 0; m < 100; m++) {
      let sum = 0.0;
      for (let k = 0; k < 513; k++) {
        sum += mag[k] * melFilterbank[k][m];
      }
      melFlat[outBase + m] = Math.log(Math.max(sum, 1e-7)) * FEAT_SCALE;
    }
  }
  for (let t = computeFrames; t < expectedFrames; t++) {
    melFlat.set(melFlat.subarray((computeFrames - 1) * 100, computeFrames * 100), t * 100);
  }
  return { melFlat, numFrames };
}

export function weightsToExactFrames(weights, targetTotalFrames) {
  const n = weights.length;
  if (n === 0) return [];
  const total = Math.max(targetTotalFrames, n);
  let wSum = 0;
  for (let i = 0; i < n; i++) wSum += weights[i];
  const raw = new Float64Array(n);
  const durs = new Int32Array(n);
  let dSum = 0;
  for (let i = 0; i < n; i++) {
    raw[i] = (weights[i] / wSum) * total;
    durs[i] = Math.max(1, Math.floor(raw[i]));
    dSum += durs[i];
  }
  let diff = total - dSum;
  if (diff > 0) {
    const order = Array.from({ length: n }, (_, i) => i).sort(
      (a, b) => (raw[b] - Math.floor(raw[b])) - (raw[a] - Math.floor(raw[a]))
    );
    for (let i = 0; i < diff; i++) durs[order[i % n]] += 1;
  } else if (diff < 0) {
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => durs[b] - durs[a]);
    let idx = 0;
    while (diff < 0 && idx < n * 4) {
      const k = order[idx % n];
      if (durs[k] > 1) {
        durs[k] -= 1;
        diff += 1;
      }
      idx++;
    }
  }
  return Array.from(durs);
}

export function randnFloat32(length, seed = 42) {
  let state = seed >>> 0;
  const nextRand = () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296.0;
  };
  const out = new Float32Array(length);
  for (let i = 0; i < length; i += 2) {
    const u1 = Math.max(nextRand(), 1e-12);
    const u2 = nextRand();
    const r = Math.sqrt(-2.0 * Math.log(u1));
    const theta = 2.0 * Math.PI * u2;
    out[i] = r * Math.cos(theta);
    if (i + 1 < length) out[i + 1] = r * Math.sin(theta);
  }
  return out;
}

/**
 * 跨平台统一确定性帧级零均值球面初值噪声（与 Python hanfire_initial_noise 逐位一致）：
 * 1. 先经 Mulberry32 + Box-Muller 生成标准高斯噪声；
 * 2. 截尾 [-clipSigma, +clipSigma]（默认 ±2.5σ），消除 4~5σ 离群尖峰导致的爆破毛刺；
 * 3. 每帧强制去均值（mean=0）并归一化至精确球壳半径（std=tau，默认 0.667），
 *    从数学上根除帧级负均值能量空洞诱发的句中随机断句（如 seed=102 现象）。
 */
export function makeSphericalInitialNoise(
  totalFrames,
  featDim = 100,
  seed = 42,
  tau = 0.667,
  clipSigma = 2.5
) {
  const z = randnFloat32(totalFrames * featDim, seed);
  for (let t = 0; t < totalFrames; t++) {
    const base = t * featDim;
    let sum = 0.0;
    for (let c = 0; c < featDim; c++) {
      let v = z[base + c];
      if (v > clipSigma) v = clipSigma;
      else if (v < -clipSigma) v = -clipSigma;
      z[base + c] = v;
      sum += v;
    }
    const mean = sum / featDim;
    let sq = 0.0;
    for (let c = 0; c < featDim; c++) {
      const d = z[base + c] - mean;
      z[base + c] = d;
      sq += d * d;
    }
    const scale = tau / (Math.sqrt(sq / featDim) + 1e-8);
    for (let c = 0; c < featDim; c++) {
      z[base + c] *= scale;
    }
  }
  return z;
}

export function getTimeSteps(numStep = 12, tShift = 0.5) {
  const ts = new Float32Array(numStep + 1);
  for (let i = 0; i <= numStep; i++) {
    const u = i / numStep;
    ts[i] = (tShift * u) / (1.0 + (tShift - 1.0) * u);
  }
  return ts;
}

/**
 * F5-TTS Sway Sampling 时间步调度（与 Python get_sway_time_steps 逐位一致）：
 * 在 t→0 低信噪比关键分岔区（对齐与元音共振峰形成期）加密步长，零额外 NFE 开销。
 */
export function getSwayTimeSteps(numStep = 16, swayS = -0.8) {
  const ts = new Float32Array(numStep + 1);
  for (let i = 0; i <= numStep; i++) {
    const u = i / numStep;
    ts[i] = u + swayS * (Math.cos((Math.PI * 0.5) * u) - 1.0 + u);
  }
  return ts;
}

/**
 * 按能量切除首尾静音（下限 -58dB，relDb=52dB，前留 35ms，后留 65ms，防止误切轻清送气声母如“春”）
 */
export function trimEdgeSilence(wav, sr = SR, relDb = 52.0, preMs = 35, postMs = 65, floorDb = -58.0) {
  const hop = Math.floor(0.01 * sr);
  const n = Math.floor(wav.length / hop);
  if (n < 3) return wav;
  const db = new Float32Array(n);
  let maxDb = -1e9;
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let k = i * hop; k < (i + 1) * hop; k++) s += wav[k] * wav[k];
    db[i] = 20 * Math.log10(Math.sqrt(s / hop) + 1e-9);
    if (db[i] > maxDb) maxDb = db[i];
  }
  const thr = Math.max(floorDb, maxDb - relDb);
  let a = -1, b = -1;
  for (let i = 0; i < n; i++) if (db[i] > thr) { if (a < 0) a = i; b = i; }
  if (a < 0) return wav;
  const s0 = Math.max(0, a * hop - Math.floor((preMs * sr) / 1000));
  const s1 = Math.min(wav.length, (b + 1) * hop + Math.floor((postMs * sr) / 1000));
  return wav.slice(s0, s1);
}

export function encodeWavPCM16(samples, sampleRate = SR) {
  const numSamples = samples.length;
  const buffer = new ArrayBuffer(44 + numSamples * 2);
  const view = new DataView(buffer);
  const writeStr = (off, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + numSamples * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, numSamples * 2, true);
  let off = 44;
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1.0, Math.min(1.0, samples[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    off += 2;
  }
  return buffer;
}
