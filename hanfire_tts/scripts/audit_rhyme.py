#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯声学韵脚押韵检测器 (audit_rhyme.py) —— 无需依赖计划帧对齐，男女声通用：
1. 自动按静音切出各分句，定位每句最后一个浊音音节核；
2. 用倒谱平滑 (Cepstral Liftering, 去除基频谐波) 提取声道共振峰包络 (MFCC 1..12)，避免高基频女声 LPC 误抓谐波；
3. 计算尾部鼻音比 (nasal_tail: <450Hz 能量占比) 与急截斜率 (区分舒声 vs 入声 -p/-t/-k)；
4. 用“押韵句对距离 vs 非押韵句对距离”的相对对比度判断全诗是否押韵，并定位失韵句。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
from scipy import fftpack, signal


def split_clauses(w: np.ndarray, sr: int = 24000, min_sil_ms: int = 120) -> List[np.ndarray]:
    hop = int(0.01 * sr)
    n = len(w) // hop
    db = 20.0 * np.log10(np.sqrt(np.mean(w[: n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-9)
    act = db > max(-48.0, float(db.max()) - 38.0)
    segs, s, gap = [], None, 0
    min_gap = min_sil_ms // 10
    for i, a in enumerate(act):
        if a:
            if s is None:
                s = i
            gap = 0
            e = i
        elif s is not None:
            gap += 1
            if gap >= min_gap:
                if e - s >= 8:
                    segs.append(w[s * hop : (e + 1) * hop])
                s = None
    if s is not None and e - s >= 8:
        segs.append(w[s * hop : (e + 1) * hop])
    return segs


def extract_rhyme_signature(clause_wav: np.ndarray, sr: int = 24000) -> dict:
    """提取分句末尾最后一个音节的：倒谱声道包络 (cep12)、鼻音尾比 (nasal_tail)、收尾衰减 (decay_db)。"""
    y = signal.resample_poly(clause_wav.astype(np.float64), 2, 3)  # 16 kHz
    y = y / (np.max(np.abs(y)) + 1e-8)
    hop, win = 160, 480
    yb = signal.sosfiltfilt(signal.butter(4, [300 / 8000, 3400 / 8000], btype="band", output="sos"), y)
    n = max(1, (len(y) - win) // hop)
    env = np.array([np.sqrt(np.mean(yb[i * hop : i * hop + win] ** 2)) for i in range(n)])
    env = signal.sosfiltfilt(signal.butter(3, 12 / 50, output="sos"), env) if len(env) > 12 else env
    env = np.maximum(env, 0.0) / (env.max() + 1e-8)
    pks, _ = signal.find_peaks(env, distance=10, prominence=0.10, height=0.10)
    pk_i = int(pks[-1]) if len(pks) else int(np.argmax(env))
    # 若 [300..3400]Hz 带通漏检了句末高元音（如末字落在距句尾 >270ms 之前，说明误抓了倒数第二字），则在末字 260ms 窗口用全频带重定位（四言/五言/七言通用）
    tail_guard = max(int(len(env) * 0.65), len(env) - 27)
    if pk_i < tail_guard and len(env) >= 20:
        yb_low = signal.sosfiltfilt(signal.butter(4, [150 / 8000, 3400 / 8000], btype="band", output="sos"), y)
        env_low = np.array([np.sqrt(np.mean(yb_low[i * hop : i * hop + win] ** 2)) for i in range(n)])
        s_tail = max(int(len(env_low) * 0.68), len(env_low) - 25)
        e_tail = max(s_tail + 2, len(env_low) - 3)
        pk_i = s_tail + int(np.argmax(env_low[s_tail:e_tail]))

    # 取峰值后韵腹/韵尾稳态段 (center + 12ms ± 20ms，避开三等 -i- 韵头过渡段) 做倒谱声道包络
    c_mid = min(max(320, len(y) - 320), pk_i * hop + win // 2 + 192)
    seg = y[max(0, c_mid - 320) : min(len(y), c_mid + 320)]
    if len(seg) < 512:
        seg = np.pad(seg, (0, 512 - len(seg)))
    spec = np.log(np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=1024)) + 1e-6)
    cep = fftpack.dct(spec, type=2, norm="ortho")[1:13]
    cep = cep / (np.linalg.norm(cep) + 1e-8)

    # 尾部鼻音比：峰值后有效浊音尾段 (env >= 0.15，最多 100ms) 内 <450Hz 能量占比
    act_tail = np.where(env[pk_i : min(len(env), pk_i + 12)] >= 0.15)[0]
    end_off = int(act_tail[-1]) + 1 if len(act_tail) else 4
    t_s = min(len(y) - 320, (pk_i + 2) * hop)
    t_e = min(len(y), max(t_s + 320, (pk_i + max(4, end_off)) * hop))
    tail = y[max(0, t_s):t_e]
    mag = np.abs(np.fft.rfft(tail * np.hanning(len(tail))))
    fr = np.fft.rfftfreq(len(tail), 1 / 16000)
    nasal_tail = float(mag[fr < 450].sum() / (mag.sum() + 1e-9))

    # 收尾衰减斜率：峰值后 80ms 能量下降 dB（对零点振铃设 -36dB 下限，防止 log10(0)=-120dB 制造伪跳变）
    post = 20.0 * np.log10(np.maximum(env[pk_i : min(len(env), pk_i + 9)], 0.016))
    decay_db = float(np.clip(post[0] - post[-1], 0.0, 36.0)) if len(post) >= 3 else 0.0
    return {"cep": cep, "nasal_tail": nasal_tail, "decay_db": decay_db}


def rhyme_dist(a: dict, b: dict) -> float:
    v_dist = float(np.linalg.norm(a["cep"] - b["cep"]))
    n_dist = abs(a["nasal_tail"] - b["nasal_tail"]) * 1.5
    d_dist = abs(a["decay_db"] - b["decay_db"]) * 0.02
    return v_dist + n_dist + d_dist


def audit_poem(wav_path: Path, rhyme_indices: Tuple[int, ...], labels: List[str]) -> None:
    w, sr = sf.read(str(wav_path), dtype="float32")
    clauses = split_clauses(w, sr)
    sigs = [extract_rhyme_signature(c, sr) for c in clauses]
    r_idx = [i for i in rhyme_indices if i < len(sigs)]
    nr_idx = [i for i in range(len(sigs)) if i not in r_idx]
    r_dists = [rhyme_dist(sigs[i], sigs[j]) for idx, i in enumerate(r_idx) for j in r_idx[idx + 1 :]]
    nr_dists = [rhyme_dist(sigs[i], sigs[j]) for i in r_idx for j in nr_idx]
    r_mean = float(np.mean(r_dists)) if r_dists else 0.0
    nr_mean = float(np.mean(nr_dists)) if nr_dists else 0.0
    contrast = nr_mean - r_mean
    # 每个韵脚到其余韵脚的平均距离（抓孤立失韵字）
    per_r = {labels[i]: round(float(np.mean([rhyme_dist(sigs[i], sigs[j]) for j in r_idx if j != i])), 3) for i in r_idx}
    tails = {labels[i]: round(sigs[i]["nasal_tail"], 2) for i in range(min(len(labels), len(sigs)))}
    print(f"[{wav_path.name}] 韵脚距离={r_mean:.3f} vs 非韵脚距离={nr_mean:.3f} | 押韵对比度={contrast:+.3f} | 各韵脚均距={per_r} | 鼻音尾比={tails}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", type=Path)
    ap.add_argument("--rhyme", default="0,1,3", help="押韵分句索引，逗号分隔")
    ap.add_argument("--labels", default="句1,句2,句3,句4")
    args = ap.parse_args()
    audit_poem(args.wav, tuple(int(x) for x in args.rhyme.split(",")), args.labels.split(","))
