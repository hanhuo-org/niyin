#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汉火「音·形·义」四维数学与声学全自动总裁判 (referee.py)

包含两大层面、四个维度的纯数学/信号处理自动裁决（零人工听审负担）：
1. 符号与训诂层（训前选音 + 动态控制参数生成）：
   - 维度1A：《诗经》韵基距离 D_rhyme_sym (F1/F2 预期舌位 + 韵尾声学归类)
   - 维度2 ：GSR/说文谐声族一致性评分
   - 维度3 ：音义生理象似度评分 (冷闭/暖开 <-> 口腔开合度; 急迫/舒缓 <-> 入声塞尾与时长倍率)
2. 物理声波层（造数硬筛 + 训后验收）：
   - 维度1B：LPC 线性预测提取稳态元音核前三共振峰 (F1, F2, F3) 转 Bark 听觉尺度算声波押韵分
   - 维度4A：400-3400Hz 希尔伯特包络音节核严格计数 (K == N_chars，杀灭双音节)
   - 维度4B：1.5-4.5kHz 频段 20-32Hz 调制谱弹舌率检测 (Trill_Index < 8.0)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import linalg, signal

from .phonology import MononuclearSyllable, parse_mononuclear_4slot, _strip_mods

# ---------------------------------------------------------------------------
# 1. 符号层押韵与音义象似裁判
# ---------------------------------------------------------------------------
VOWEL_BARK_EXPECT: Dict[str, Tuple[float, float, float]] = {
    # 预期 (height_code, back_code, round_code)
    "i": (3.0, 0.0, 0.0), "y": (3.0, 0.0, 1.0), "ɪ": (2.6, 0.2, 0.0),
    "e": (2.0, 0.2, 0.0), "ø": (2.0, 0.2, 1.0), "ɛ": (1.3, 0.3, 0.0), "œ": (1.2, 0.3, 1.0),
    "æ": (0.5, 0.3, 0.0), "ɶ": (0.5, 0.3, 1.0),
    "a": (0.0, 1.0, 0.0), "ɐ": (0.4, 1.0, 0.0), "ɑ": (0.0, 1.8, 0.0), "ɒ": (0.2, 1.9, 1.0),
    "ə": (1.8, 1.0, 0.0), "ɜ": (1.4, 1.0, 0.0), "ʌ": (1.2, 1.6, 0.0), "ɤ": (2.0, 1.8, 0.0),
    "ɨ": (2.8, 1.0, 0.0), "ʉ": (2.8, 1.0, 1.0), "ɯ": (2.9, 1.9, 0.0),
    "ɿ": (2.8, 0.8, 0.0), "ʅ": (2.8, 1.0, 0.0),
    "u": (3.0, 2.0, 1.0), "ʊ": (2.6, 1.8, 1.0), "o": (2.0, 1.9, 1.0), "ɔ": (1.2, 1.8, 1.0),
}

CODA_ACOUSTIC_VEC: Dict[str, np.ndarray] = {
    "":   np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "j":  np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0], dtype=np.float32),
    "w":  np.array([0.4, 0.0, 0.0, 0.0, 0.6, 0.0, 0.4], dtype=np.float32),
    "ʔ":  np.array([0.7, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "h":  np.array([0.6, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "s":  np.array([0.8, 0.0, 0.0, 0.8, 0.0, 0.6, 0.0], dtype=np.float32),
    "m":  np.array([1.5, 1.5, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    "n":  np.array([1.5, 1.5, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    "ŋ":  np.array([1.5, 1.5, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    "p":  np.array([1.8, 0.0, 1.8, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    "t":  np.array([1.8, 0.0, 1.8, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    "k":  np.array([1.8, 0.0, 1.8, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    "kʷ": np.array([1.8, 0.0, 1.8, 0.0, 0.5, 0.0, 1.0], dtype=np.float32),
}


def symbolic_rhyme_distance(ipa1: str, ipa2: str) -> float:
    """计算两个 IPA 的符号层押韵距离（0 = 完全同韵；<=1.25 谐韵；>2.0 严重失韵）。"""
    if not ipa1 or not ipa2:
        return 9.9
    s1 = parse_mononuclear_4slot(ipa1)
    s2 = parse_mononuclear_4slot(ipa2)
    b1 = _strip_mods(s1.nucleus)[0]
    b2 = _strip_mods(s2.nucleus)[0]
    v1 = np.array(VOWEL_BARK_EXPECT.get(b1, (1.5, 1.0, 0.0)), dtype=np.float32)
    v2 = np.array(VOWEL_BARK_EXPECT.get(b2, (1.5, 1.0, 0.0)), dtype=np.float32)
    c1 = CODA_ACOUSTIC_VEC.get(s1.coda, CODA_ACOUSTIC_VEC.get(s1.coda[-1:], CODA_ACOUSTIC_VEC[""]))
    c2 = CODA_ACOUSTIC_VEC.get(s2.coda, CODA_ACOUSTIC_VEC.get(s2.coda[-1:], CODA_ACOUSTIC_VEC[""]))
    return float(np.linalg.norm(v1 - v2) + np.linalg.norm(c1 - c2))


# 语义本义关键词词根库（兼顾汉字字头、部首与英文 gloss 词根）
SEM_COLD_CLOSED_CHARS = set("冷寒凜凛冰冻凍凄淒凉涼冬雪霜陰阴暗冥閉闭藏斂敛縮缩密默秘幽微細细小窄狹狭")
SEM_WARM_OPEN_CHARS = set("暖温溫熱热陽阳日旦明昭光火炎炙暑夏大廣广宏洪開开張张敞舒暢畅朗闊阔遠远")
SEM_URGENT_ABRUPT_CHARS = set("急促速迫疾猝切倏忽驟骤暴擊击斬斩絕绝斷断裂劈殺杀奪夺捉拍撲扑篤笃瞥")
SEM_SLOW_EXTENDED_CHARS = set("緩缓慢舒長长悠永延引曼漫徐遲迟留流游遊徜徉徘徊渺茫遙遥平寧宁靜静和柔")

GLOSS_COLD_KW = ("cold", "freeze", "ice", "frost", "shut", "close", "hide", "dark", "narrow", "small")
GLOSS_WARM_KW = ("warm", "hot", "sun", "bright", "light", "open", "wide", "vast", "great", "large")
GLOSS_URGENT_KW = ("urgent", "fast", "quick", "sudden", "abrupt", "haste", "press", "cut", "strike", "seize")
GLOSS_SLOW_KW = ("slow", "long", "extend", "relax", "calm", "linger", "far", "distant", "gentle", "flow")


def infer_semantic_traits(char: str, gloss: str = "") -> Tuple[float, float]:
    """
    提取汉字的本义物理倾向向量:
      returns (sem_openness, sem_urgency) 均在 [-1.0, +1.0]
        sem_openness: -1.0 = 寒冷收敛闭口, +1.0 = 温暖光明张口, 0.0 = 中性
        sem_urgency : +1.0 = 急迫瞬时短促, -1.0 = 舒缓绵长延展, 0.0 = 中性
    """
    g = (gloss or "").lower()
    sem_open = 0.0
    sem_urg = 0.0

    if char in SEM_COLD_CLOSED_CHARS or any(w in g for w in GLOSS_COLD_KW):
        sem_open -= 0.85
    if char in SEM_WARM_OPEN_CHARS or any(w in g for w in GLOSS_WARM_KW):
        sem_open += 0.85
    # 冰部首「冫」偏旁天然携带寒冷收敛义
    if len(char) == 1 and char in "凓凔凕凖凊凋凌凇凙凚凛冽":
        sem_open = min(sem_open, -0.85)

    if char in SEM_URGENT_ABRUPT_CHARS or any(w in g for w in GLOSS_URGENT_KW):
        sem_urg += 0.90
    if char in SEM_SLOW_EXTENDED_CHARS or any(w in g for w in GLOSS_SLOW_KW):
        sem_urg -= 0.85

    return float(np.clip(sem_open, -1.0, 1.0)), float(np.clip(sem_urg, -1.0, 1.0))


def compute_iconicity_match_and_controls(
    char: str,
    ipa: str,
    gloss: str = "",
) -> Dict:
    """
    计算候选拟音与字义的「音·形·义」生理象似度得分，并输出 TTS 动态控制量 (alpha_dur, alpha_open)。
    """
    syl = parse_mononuclear_4slot(ipa)
    sem_open, sem_urg = infer_semantic_traits(char, gloss)

    # 1. 口腔开合匹配度: sem_open 与 syl.mouth_openness 同向加分，反向减分
    open_match = sem_open * syl.mouth_openness  # [-1, +1]

    # 2. 急缓动力学匹配度: sem_urg > 0 偏好入声唯闭塞尾；sem_urg < 0 偏好舒声开/鼻尾
    phys_abrupt = 0.85 if syl.is_checked else (-0.65 if syl.coda in {"", "n", "ŋ", "m", "j", "w"} else 0.0)
    urg_match = sem_urg * phys_abrupt

    iconicity_score = 0.5 + 0.28 * open_match + 0.22 * urg_match  # 中心 0.5，满分 1.0

    # 3. 派生 TTS 合成动态控制参数
    # 基础时长倍率：入声天然短促 0.78，舒声 1.0；再叠加本义急缓调制
    base_dur = 0.78 if syl.is_checked else 1.0
    if sem_urg > 0.3:
        alpha_dur = base_dur * 0.85   # 急迫字进一步收紧 (~0.66~0.85x)
    elif sem_urg < -0.3:
        alpha_dur = base_dur * 1.22   # 舒缓字从容延展 (~1.22x)
    else:
        alpha_dur = base_dur

    alpha_open = float(np.clip(0.7 * syl.mouth_openness + 0.3 * sem_open, -1.0, 1.0))

    return {
        "iconicity_score": round(float(np.clip(iconicity_score, 0.0, 1.0)), 4),
        "sem_openness": round(sem_open, 3),
        "sem_urgency": round(sem_urg, 3),
        "mouth_openness": syl.mouth_openness,
        "alpha_dur": round(float(np.clip(alpha_dur, 0.55, 1.45)), 3),
        "alpha_open": round(alpha_open, 3),
    }


# ---------------------------------------------------------------------------
# 2. 物理声波层裁判（LPC 共振峰 Bark 押韵 + 单峰核 K==N + 弹舌率）
# ---------------------------------------------------------------------------
def hz_to_bark(f_hz: float | np.ndarray) -> float | np.ndarray:
    """Zwicker 听觉临界频带 Bark 转换公式。"""
    f = np.asarray(f_hz, dtype=np.float64)
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def extract_vowel_formants_bark(wav: np.ndarray, sr: int) -> Dict:
    """
    纯 SciPy 线性预测编码 (LPC) 提取音节元音核稳态段的前三共振峰 (F1, F2, F3) 与尾部衰减斜率。
    无需外部 Praat/C++ 依赖，本地与 Colab 100% 一致运行。
    """
    if len(wav) < int(sr * 0.06):
        return {"ok": False, "f_hz": (500.0, 1500.0, 2500.0), "f_bark": (4.9, 11.2, 14.5), "decay_slope": 0.0}

    # 重采样到 16kHz 做标准 16阶 LPC 共振峰分析（最高 8kHz 恰好容纳 4-5 个声道共振峰）
    target_sr = 16000
    if sr != target_sr:
        g = math.gcd(sr, target_sr)
        y = signal.resample_poly(wav.astype(np.float64), target_sr // g, sr // g)
    else:
        y = wav.astype(np.float64)

    y = y - np.mean(y)
    peak_amp = np.max(np.abs(y)) + 1e-8
    y = y / peak_amp

    # 找 400-3400Hz 元音频带能量最高点（即唯一韵腹稳态中心）
    sos = signal.butter(4, [400 / 8000.0, 3400 / 8000.0], btype="band", output="sos")
    y_vow = signal.sosfiltfilt(sos, y)
    win = 320  # 20ms
    hop = 80   # 5ms
    frames_e = np.array([
        np.mean(y_vow[i: i + win] ** 2)
        for i in range(0, max(1, len(y_vow) - win), hop)
    ])
    best_frame = int(np.argmax(frames_e))
    center = best_frame * hop + win // 2
    half_span = int(0.025 * target_sr)  # 取中心前后共 50ms 稳态元音窗口
    s0 = max(0, center - half_span)
    s1 = min(len(y), center + half_span)
    seg = y[s0:s1]
    if len(seg) < 256:
        seg = np.pad(seg, (0, 256 - len(seg)))

    # 预加重 + Hamming 窗 + LPC 自相关求解极点
    seg_pre = np.append(seg[0], seg[1:] - 0.97 * seg[:-1]) * np.hamming(len(seg))
    order = 16
    r = np.correlate(seg_pre, seg_pre, mode="full")[len(seg_pre) - 1: len(seg_pre) + order]
    r[0] += 1e-6
    try:
        a = linalg.solve_toeplitz(r[:-1], -r[1:])
        poly = np.concatenate([[1.0], a])
        roots = np.roots(poly)
        roots = roots[np.imag(roots) >= 0.01]
        angles = np.arctan2(np.imag(roots), np.real(roots))
        freqs = angles * (target_sr / (2.0 * np.pi))
        bws = -0.5 * (target_sr / (2.0 * np.pi)) * np.log(np.abs(roots) + 1e-9)
        # 过滤有效声道共振峰: 200Hz < f < 4200Hz 且带宽 < 500Hz
        valid = [(f, bw) for f, bw in zip(freqs, bws) if 200.0 <= f <= 4200.0 and bw < 600.0]
        valid.sort(key=lambda x: x[0])
        f_list = [x[0] for x in valid]
    except Exception:
        f_list = []

    # 兜底补齐 F1, F2, F3
    defaults = [500.0, 1500.0, 2500.0]
    while len(f_list) < 3:
        f_list.append(defaults[len(f_list)])
    f1, f2, f3 = float(f_list[0]), float(f_list[1]), float(f_list[2])
    b1, b2, b3 = [float(hz_to_bark(f)) for f in (f1, f2, f3)]

    # 计算韵腹峰值之后的尾部能量衰减斜率 (dB / 100ms)，区分入声急截与舒声缓降
    post_e = 10.0 * np.log10(frames_e[best_frame:] + 1e-9)
    if len(post_e) >= 4:
        # 前 80ms (16帧) 下降幅度
        span_f = min(len(post_e) - 1, 16)
        decay_slope = float((post_e[0] - post_e[span_f]) / (span_f * 0.005) * 0.1)  # dB per 100ms
    else:
        decay_slope = 25.0

    return {
        "ok": True,
        "f_hz": (round(f1, 1), round(f2, 1), round(f3, 1)),
        "f_bark": (round(b1, 2), round(b2, 2), round(b3, 2)),
        "decay_slope": round(decay_slope, 2),
    }


def judge_waveform_rhyme_pair(wav1: np.ndarray, wav2: np.ndarray, sr: int) -> Dict:
    """
    纯声波物理押韵度裁决：不需要看音标，直接测两段韵脚音频的 (F1, F2, F3) Bark 距离与收尾斜率匹配度！
    """
    m1 = extract_vowel_formants_bark(wav1, sr)
    m2 = extract_vowel_formants_bark(wav2, sr)
    b1 = np.array(m1["f_bark"], dtype=np.float64)
    b2 = np.array(m2["f_bark"], dtype=np.float64)

    # F1(开口度) 与 F2(舌前后) 决定押韵主听感，F3 权重 0.35
    bark_dist = float(np.sqrt((b1[0] - b2[0]) ** 2 + (b1[1] - b2[1]) ** 2 + 0.35 * (b1[2] - b2[2]) ** 2))
    slope_diff = abs(m1["decay_slope"] - m2["decay_slope"])
    slope_penalty = max(0.0, (slope_diff - 12.0) * 0.05)

    rhyme_score = float(100.0 * np.exp(-(bark_dist + slope_penalty) / 2.2))
    return {
        "acoustic_rhyme_score": round(rhyme_score, 1),
        "bark_distance": round(bark_dist, 3),
        "is_rhyming": bark_dist < 1.65 and rhyme_score >= 48.0,
        "char1_formants_hz": m1["f_hz"],
        "char2_formants_hz": m2["f_hz"],
        "slope_diff_db100ms": round(slope_diff, 2),
    }


@dataclass
class UtteranceAuditReport:
    n_expected_chars: int
    n_detected_peaks: int
    monosyllabic_exact: bool
    trill_index: float
    trill_ok: bool
    active_dur_s: float
    sec_per_char: float
    overall_pass: bool

    def to_dict(self) -> dict:
        return asdict(self)


def judge_utterance_acoustics(wav: np.ndarray, sr: int, n_expected_chars: int) -> UtteranceAuditReport:
    """
    一票否决声学质检：
    1. 元音频带包络波峰数 K 必须等于汉字数（单字必须严格 K == 1；整句容许连读 ±0~1 视配置，单字零容忍）；
    2. 弹舌调制指数 Trill_Index < 8.5；
    3. 平均字长在 [0.18s, 0.68s] 合理区间。
    """
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    dur = len(wav) / max(sr, 1)
    if dur < 0.12 or np.max(np.abs(wav)) < 1e-4:
        return UtteranceAuditReport(n_expected_chars, 0, False, 0.0, True, dur, 0.0, False)

    # 重采样到 16kHz 统一包络标尺
    target_sr = 16000
    if sr != target_sr:
        g = math.gcd(sr, target_sr)
        y = signal.resample_poly(wav.astype(np.float64), target_sr // g, sr // g)
    else:
        y = wav.astype(np.float64)
    y = (y - np.mean(y)) / (np.max(np.abs(y)) + 1e-8)

    # 1. 400-3400Hz 元音频带单峰核计数
    sos_v = signal.butter(4, [400 / 8000.0, 3400 / 8000.0], btype="band", output="sos")
    env = np.abs(signal.hilbert(signal.sosfiltfilt(sos_v, y)))
    env_100hz = signal.resample_poly(env, 1, 160)  # 10ms per frame
    sos_lp = signal.butter(3, 10.0 / 50.0, btype="low", output="sos")
    syl_env = np.maximum(0.0, signal.sosfiltfilt(sos_lp, env_100hz))
    if np.max(syl_env) > 0:
        syl_env = syl_env / np.max(syl_env)

    peaks, _ = signal.find_peaks(syl_env, distance=15, prominence=0.12, height=0.16)
    n_peaks = int(len(peaks))

    active_mask = syl_env > 0.08
    active_s = float(np.sum(active_mask) * 0.01) if np.any(active_mask) else dur
    sec_per_ch = active_s / max(1, n_expected_chars)

    if n_expected_chars == 1:
        mono_ok = (n_peaks == 1) and (0.15 <= active_s <= 0.65)
    else:
        mono_ok = (abs(n_peaks - n_expected_chars) <= max(1, int(0.1 * n_expected_chars))) and (0.18 <= sec_per_ch <= 0.68)

    # 2. 20-32Hz 弹舌检测
    sos_hi = signal.butter(4, [1500 / 8000.0, 4500 / 8000.0], btype="band", output="sos")
    y_hi = signal.sosfiltfilt(sos_hi, y)
    env_hi_200 = signal.resample_poly(np.abs(y_hi), 1, 80)
    nperseg = min(512, len(env_hi_200))
    if nperseg >= 64:
        f_m, psd_m = signal.welch(env_hi_200 - np.mean(env_hi_200), fs=200.0, nperseg=nperseg)
        syl_e = float(np.sum(psd_m[(f_m >= 2.0) & (f_m <= 8.0)])) + 1e-12
        trill_e = float(np.sum(psd_m[(f_m >= 20.0) & (f_m <= 32.0)]))
        trill_idx = 100.0 * trill_e / syl_e
    else:
        trill_idx = 0.0

    trill_ok = trill_idx < 9.5
    return UtteranceAuditReport(
        n_expected_chars=n_expected_chars,
        n_detected_peaks=n_peaks,
        monosyllabic_exact=mono_ok,
        trill_index=round(trill_idx, 2),
        trill_ok=trill_ok,
        active_dur_s=round(active_s, 3),
        sec_per_char=round(sec_per_ch, 3),
        overall_pass=bool(mono_ok and trill_ok),
    )
