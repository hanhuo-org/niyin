#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汉火古音吟诵 · V15「挺拔中气 × 头重尾轻吐字 × 实声黄金甜区 × 真声滑腔」古文/辞赋引擎 (guwen_v14_engine.py)

融合声学法医实测（Praat H1-H2 / LTAS 频谱倾斜 / 音节能量包络）+ 逐条实听纠偏的核心升级：
  1. 彻底根治“每个字有气无力”的三大物理病根：
     - 摒弃 V14.2 拖沓软起音（72~110ms 声母 + 180ms 慢爬坡 + 30ms sigma=6 糊音），改用干脆利落的吟诵起音
       （48~72ms 塞鼻声母、86~95ms 清擦/塞擦声母、65~85ms 俐落滑入、sigma=2.8），字头如吐珠、字字立住；
     - 在零词内 SP、零响度压缩的前提下，为每个字施加对齐原版 t09/t16 的「头重尾轻」中气发声包络
       （元音起音瞬间 +2.0~2.2dB 立住 → 字腹稳托 0dB → 字尾利落轻收 -2.0/-3.2dB；清擦辅音自身保持 0dB 绝不吹大嘶声）；
     - 线性相位温和人声中气均衡（+1.5~2.4dB 胸声基干 + 3.0~3.2dB 1.6~1.85kHz 书斋朗诵穿透力，柔抑 >5.6kHz 流行气声）；
  2. 声部黄金实声区精准路由：
     - 男声升级为声带闭合紧实、气声极低的棚录实声组合（G1 兰亭 `m4_tenor5 @ 154Hz` / G2 陋室 `m4_tenor6:0.65|m4_tenor5:0.35 @ 144Hz` /
       G3 赤壁 `m4_tenor6:0.80|m4_tenor5:0.20 @ 148Hz`）；
     - 女声 `opencpop_base` 抬升至实声甜区（G1 `252Hz` / G2 `256Hz` / G3 `276Hz`），彻底离开 <205Hz 气声底噪区并严格跨开 300~311Hz 换声裂带；
  3. 完整继承 V14.2 人工复核的真实逐字时长、原速母带微颤残差、句尾自然气息渐收与虚词「也」三波摇腔真轨。
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

SING_DIR = Path(__file__).resolve().parent
DATA_DIR = SING_DIR / "data_tang_processed"
if str(SING_DIR) not in sys.path:
    sys.path.insert(0, str(SING_DIR))

from hanfire_sing_phonemes import CODA_SET, VOWEL_SET, text_to_sing_syllables
from shijing_v9_engine import _build_char_phones
from guwen_v9_engine import (
    GUWEN_V9_MODES,
    ClausePlan,
    _FINAL_PARTICLES,
    _MID_PARTICLES,
    _analyze_clause_tones,
    _split_clauses_with_punct,
    plan_guwen_discourse,
)

GUWEN_V14_MODES = {k: dict(v) for k, v in GUWEN_V9_MODES.items()}
# V16 兰亭升级：移植《诗经》V9.6 真人母带逐字活腔架构（Praat 实测 25 字 1:1 音高切片 × 辅音/元音 F0 解耦 × 纯净棚录零 EQ 染色）
#   - 女声 232Hz：-3.4st=191Hz(胸声) / +2.8st=273Hz(<300Hz) / +5.3st=315Hz(>311Hz) 激光穿针避开换声带，+11.4st=448Hz(A4清亮头腔)!
#   - 男声 164Hz (m4_tenor6:0.60|m4_tenor5:0.40)：-3.4st=135Hz(沉稳胸声) / +7.0st=246Hz(饱满实声) / +11.4st=317Hz(清越高腔)!
GUWEN_V14_MODES["guwen_lanting"].update(
    female_key_hz=232.0,
    male_key_hz=164.0,
    min_st=-5.5,
    max_st=12.2,
    desc="V16 逐字对照原版 t09 移植《诗经》V9.6 活腔法：Praat 逐字精确提取 25 字真实拱腔/高拔/五度流泻/四波摇腔，辅音元音 F0 解耦，纯棚录零 EQ 染色！",
)
GUWEN_V14_MODES["guwen_loushi"].update(
    female_key_hz=248.0,
    male_key_hz=140.0,
    min_st=-6.0,
    max_st=6.5,
    desc="V16 逐字对照 t16 全篇 81 字习得「四声依字行腔 × 平低仄高 × 6+2 跨句连气顿挫 × 悬宫落底」唐调铭文真传，纯棚录零 EQ 染色！",
)
GUWEN_V14_MODES["fu_chibi"].update(
    female_key_hz=250.0,
    male_key_hz=154.0,
    min_st=-8.2,
    max_st=5.2,
    desc="V16 逐字对照 t18《前赤壁赋》移植《诗经》V9.6 活腔法：1:1 还原苍茫低徵起腔、短入拔起双去劲落、十二字泛舟扬波与清风徐来水波不兴八字连气归宫，纯棚录零 EQ 染色！",
)

_FRICATIVE_ONSETS = frozenset({"sh", "s", "x", "ch", "q", "c", "f"})
_UNVOICED_FRIC_ONSETS = frozenset({"sh", "s", "x", "ch", "q", "c", "f", "h"})
_AFFRIC_ONSETS = frozenset({"ch", "c", "q"})
_STOP_ONSETS = frozenset({"b", "p", "d", "t", "g", "k"})
_NASAL_ONSETS = frozenset({"n", "m"})
_NASAL_FINALS = frozenset({
    "an", "en", "in", "un", "vn", "ang", "eng", "ing", "ong",
    "ian", "uan", "van", "iang", "uang", "iong",
})

# 槽位格式: (目标半音, 时长s, 起音半音或None, 收尾半音或None, 起始增益dB)
Slot = Tuple[float, float, Optional[float], Optional[float], float]

MODE_PARAMS = {
    "guwen_lanting": dict(
        body_ping=-3.2, body_ze=-2.4, body_dur=0.43,
        gap=0.31, gap_major=0.36, tail_end_db=-8.5, grand_end_db=-14.0,
        src=("t09", "bodies/t09_guwen_兰亭集序.wav", 194.4, 100.0, 390.0, 16.0),
    ),
    "guwen_loushi": dict(
        body_ping=+2.1, body_ze=+3.9, body_dur=0.44,
        gap=0.40, gap_major=0.48, tail_end_db=-10.0, grand_end_db=-14.5,
        src=("t16", "bodies/t16_guwen_陋室铭.wav", 136.0, 75.0, 260.0, 16.8),
    ),
    "fu_chibi": dict(
        body_ping=+2.1, body_ze=+4.1, body_dur=0.38,
        gap=0.32, gap_major=0.58, tail_end_db=-10.0, grand_end_db=-14.0,
        src=("t18", "bodies/t18_fu_前赤壁赋.wav", 154.0, 80.0, 350.0, 12.7),
    ),
}

# 经母带实听 + 50ms 音节能量谷精确复核的四槽模板 (H, P, C, T)
TEMPLATES: Dict[str, Dict[str, List[Optional[Slot]]]] = {
    "guwen_lanting": {
        # 永和九年：永(-3.3) 和(-3.1) 九(-0.3→+2.3 上挑!) 年(+2.2→-0.2 书斋拱收 0.68s)
        "QI":         [(-3.3, 0.44, None, None, 0.0), (-3.1, 0.44, None, None, 0.0), (+0.6, 0.38, -0.4, +2.3, +0.8), (-0.2, 0.68, +2.2, -0.4, 0.0)],
        "LEAD":       [(-3.3, 0.44, None, None, 0.0), (-3.1, 0.44, None, None, 0.0), (+0.4, 0.40, -0.4, +2.0, +0.5), (-0.2, 0.66, +2.0, -0.3, 0.0)],
        "CHENG_UP":   [(-3.2, 0.42, None, None, 0.0), (-1.0, 0.42, None, None, 0.0), (+0.5, 0.44, -0.2, +2.0, +0.5), (+0.0, 0.64, +2.0, -0.2, 0.0)],
        # 岁在癸丑：岁(-3.2) 在(-1.0) 癸(-0.1→+2.3→-0.3 挑落!) 丑(-0.8 沉敛收)
        "CHENG_DOWN": [(-3.2, 0.40, None, None, 0.0), (-1.0, 0.42, None, None, 0.0), (+0.4, 0.44, -0.2, +2.2, +0.4), (-0.8, 0.62, -0.4, -0.9, 0.0)],
        # 暮春之初：暮(+2.8→+4.9 陡起) 春(+5.3 亮托) 之(+5.5→+7.2 冲顶) 初(+3.3→0.0 五度流泻!)
        "ZHUAN_HIGH": [(+4.9, 0.44, +2.8, None, +0.8), (+5.3, 0.44, None, None, +1.5), (+5.8, 0.44, +5.4, +7.2, +1.2), (0.0, 0.72, +3.3, -0.1, 0.0)],
        # 修禊事也：修(+1.2→+5.1 大滑入) 禊(+8.0→+7.1 高羽) 事(+7.2→+5.0→+2.6 流泻) 也(原版真实三波摇腔!)
        "HE_GRAND":   [(+5.1, 0.48, +1.2, None, +1.2), (+7.1, 0.56, +8.0, None, +1.8), (+5.0, 0.60, +7.2, +2.6, +1.5), (-2.3, 2.24, +1.8, -2.3, 0.0)],
    },
    "guwen_loushi": {
        # 山不在高 (实测 0.15..2.35s = 2.20s)：山(0.76s 盘升) 不(0.27s) 在(0.34s +3.9st!) 高(0.94s -1.6→-0.2st 悬宫大托!)
        "QI":         [(+1.9, 0.76, -0.3, None, 0.0), (+2.3, 0.27, +2.0, +3.4, +0.8), (+3.9, 0.34, +3.6, +3.4, +1.2), (-0.2, 0.94, -1.6, -0.2, 0.0)],
        # 水不在深 / 静以修身 (实测 5.73..7.93s = 2.20s)：水(0.58s -2.2→+2.2) 不(0.27s) 在(0.34s +4.0!) 深(0.92s -1.5→+0.1)
        "CHENG_UP":   [(+2.2, 0.58, -2.2, None, +0.4), (+2.3, 0.27, +2.0, +3.4, +0.6), (+4.0, 0.34, +3.6, +3.4, +1.2), (0.0, 0.92, -1.5, -0.1, 0.0)],
        # 有仙则名 / 有龙则灵 (实测 2.78..5.23s = 2.45s)：有(0.30s -2.5→+1.0) 仙(0.78s +2.1 长托) 则(0.32s +3.0→+0.4) 名(0.92s -0.3→-3.1 四度落徵!)
        "CHENG_DOWN": [(-0.8, 0.30, -2.5, +1.2, +0.4), (+2.1, 0.78, +1.8, +2.2, 0.0), (+2.3, 0.32, +3.1, +0.4, +0.8), (-3.1, 0.92, -0.3, -3.1, 0.0)],
        # 斯是陋室 (实测 10.94..13.00s = 2.06s)：斯(+4.6) 是(+4.4) 陋(-2.2→+0.3 0.66s) 室(+0.2 0.56s)
        "ZHUAN_HIGH": [(+4.6, 0.28, None, +4.0, +0.8), (+4.4, 0.48, None, None, +1.5), (+0.3, 0.66, -2.2, +0.5, 0.0), (+0.2, 0.54, None, None, 0.0)],
        "LEAD":       [(+0.2, 0.44, -1.5, None, 0.0), (+2.1, 0.42, None, None, 0.0), (+3.6, 0.38, None, None, +1.0), (0.0, 0.72, -0.8, -0.1, 0.0)],
        # 惟吾德馨 (实测 13.05..16.58s)：惟(+1.0→+2.0) 吾(-0.1→-3.3) 德(-4.8→-2.8→+0.2 挑!) 馨(原版真实 1.35s 沉底大腔!)
        "HE_GRAND":   [(+1.6, 0.44, +1.0, +2.0, +1.0), (-1.5, 0.62, -0.1, -3.3, 0.0), (-2.8, 0.52, -4.8, +0.2, +0.4), (-4.9, 1.38, -4.6, -4.9, 0.0)],
    },
    "fu_chibi": {
        # 壬戌之秋 (实测 0.15..2.20s)：苍茫低徵 -7.0st 起兴
        "QI_LOW":               [(-7.0, 0.54, None, None, 0.0), (-5.4, 0.30, None, None, +0.8), (-5.8, 0.46, -4.6, -6.6, 0.0), (-6.9, 0.78, None, None, 0.0)],
        # 衔远山 / 朝晖夕阴 类：平 +2.1 仄 +4.1
        "QI":                   [(+2.1, 0.42, None, None, 0.0), None, None, (+2.1, 0.66, None, None, 0.0)],
        "CHENG_UP":             [(+2.1, 0.42, None, None, 0.0), None, None, (+0.3, 0.68, None, None, 0.0)],
        # 清风徐来 (全平 9.38..10.98s)：清(-3.0→-0.2 滑入) 风(0.0) 徐(+0.1) 来(0.0 0.58s)
        "CHENG_UP_PING":        [(-0.2, 0.44, -3.0, None, 0.0), (0.0, 0.40, None, None, 0.0), (+0.1, 0.32, None, None, +0.4), (0.0, 0.58, None, None, 0.0)],
        # 七月既望 (承低起兴 2.40..3.64s)：七(0.30s 短入猛拔七度!) 月(0.32s) 既(-1.0→-4.4) 望(-0.9→-5.0 0.58s)
        "CHENG_DOWN_AFTER_LOW": [(-1.0, 0.30, -1.4, None, +1.0), (-1.6, 0.32, -1.2, -2.1, +0.5), (-2.6, 0.36, -1.0, -4.4, 0.0), (-3.8, 0.58, -0.9, -5.0, 0.0)],
        "CHENG_DOWN":           [(+2.1, 0.42, None, None, 0.0), None, (+2.8, 0.40, None, None, +0.5), (+0.1, 0.66, +1.8, 0.0, 0.0)],
        "CHENG_DOWN_MAJOR":     [(+2.1, 0.42, None, None, 0.0), None, (+4.3, 0.40, None, None, +1.0), (-3.2, 0.84, -0.2, -3.2, 0.0)],
        "ZHUAN_HIGH":           [(+4.2, 0.42, +3.4, None, +1.0), None, None, (+2.1, 0.68, None, None, 0.0)],
        "LEAD":                 [(+0.2, 0.42, -1.4, None, 0.0), None, None, (0.0, 0.62, None, None, 0.0)],
        # 水波不兴 (11.08..12.46s)：水(+3.3→+2.4) 波(+2.4→+1.2) 不(-0.5) 兴(0.0 0.92s 浩然归宫!)
        "HE_AFTER_PING":        [(+2.5, 0.38, +3.3, None, +1.2), (+2.1, 0.34, +2.4, +1.2, +0.6), (-0.5, 0.30, None, None, 0.0), (0.0, 0.92, -0.2, 0.0, 0.0)],
        # 气象万千 / 歌窈窕之章：渐沉 + 倒二欲抑先扬 + 沉底大收腔
        "HE_GRAND":             [(+1.2, 0.40, +1.8, -1.0, +0.5), (-2.8, 0.44, None, None, 0.0), (+1.4, 0.42, -1.2, +1.6, +0.8), (-4.7, 1.15, -3.4, -4.8, 0.0)],
    },
}

# 长句 (>=7 字) 原版逐字骨架
LONG_LINES: Dict[str, Dict[str, list]] = {
    "guwen_lanting": {  # 会于会稽山阴之 | 兰 亭
        "body": [
            (-3.4, 0.44, -3.8, None, 0.0), (-3.1, 0.39, None, None, 0.0), (-3.4, 0.44, None, -2.8, 0.0),
            (-3.3, 0.46, None, None, 0.0), (-3.1, 0.48, -3.8, -2.8, 0.0), (-2.6, 0.42, -2.0, None, 0.0),
            (-3.2, 0.38, -3.6, None, 0.0),
        ],
        "C": (-0.4, 0.48, -2.2, +1.6, +0.5),
        "T": (-0.9, 0.64, +1.4, -1.2, 0.0),
    },
    "guwen_loushi": {  # 非宁静无以致远 / 孔子云何陋之有（端凝沉降 + 欲抑先扬 + 沉底）
        "body": [
            (+0.4, 0.40, -1.0, None, 0.0), (+2.1, 0.42, None, None, +0.5), (+3.2, 0.40, None, None, +0.8),
            (-1.4, 0.44, None, None, 0.0), (-2.8, 0.46, None, None, 0.0),
        ],
        "C": (+0.9, 0.44, -1.6, +1.2, +0.8),
        "T": (-3.9, 1.18, -3.0, -4.0, 0.0),
    },
    "fu_chibi": {  # 苏子与客泛舟游于赤壁 | 之 下 (实测 4.35..8.76s = 4.41s)
        "body": [
            (+0.2, 0.38, -1.6, None, 0.0), (+0.5, 0.38, None, None, +0.4), (+1.0, 0.28, None, None, +0.4),
            (-0.1, 0.28, None, None, 0.0), (-0.1, 0.36, None, None, 0.0), (+0.4, 0.36, None, None, 0.0),
            (+2.6, 0.42, +3.2, +2.0, +1.2), (+2.4, 0.38, +2.8, +1.2, +0.8), (+0.3, 0.30, None, None, 0.0),
            (+0.3, 0.38, None, None, 0.0),
        ],
        "C": (-2.2, 0.34, -1.4, -3.6, 0.0),
        "T": (-5.0, 0.74, -4.4, -5.4, 0.0),
    },
}

# 全篇末字调用原版真实 F0 区间（仅当语义角色匹配：G1 限虚词「也/矣/乎/哉」，G2 限平声大收如「馨」）
REAL_TAILS = {
    "guwen_lanting": (13.48, 15.72),   # t09「也」三波摇腔
    "guwen_loushi":  (15.22, 16.58),   # t16「馨」低角沉底
}

_SRC_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
_RESID_CACHE: Dict[str, np.ndarray] = {}


def _master_track(mode_key: str) -> Tuple[np.ndarray, np.ndarray]:
    if mode_key in _SRC_CACHE:
        return _SRC_CACHE[mode_key]
    import librosa
    import soundfile as sf
    _, rel, gong, fmin, fmax, max_s = MODE_PARAMS[mode_key]["src"]
    y, sr = sf.read(str(DATA_DIR / rel), dtype="float32")
    y = (y.mean(1) if y.ndim > 1 else y)[: int(max_s * sr)]
    hop = 256
    f0, vf, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0]
    n = min(len(f0), len(rms))
    f0, vf, rms = f0[:n], vf[:n], rms[:n]
    db = 20 * np.log10(rms + 1e-8)
    v = (~np.isnan(f0)) & vf & (db >= np.percentile(db, 98) - 34)
    st = np.full(n, np.nan, dtype=np.float32)
    st[v] = 12 * np.log2(f0[v] / gong)
    vi = np.where(v)[0]
    x = st[vi]
    lm = median_filter(x, size=81)
    x[x - lm > 9] -= 12
    x[lm - x > 9] += 12
    st[vi] = x
    t = np.arange(n) * hop / sr
    grid = np.arange(0, t[-1], 0.005)
    g = np.interp(grid, t, np.nan_to_num(st, nan=-99.0))
    vg = np.interp(grid, t, v.astype(float)) > 0.5
    g[~vg] = np.nan
    # 净化 t09「也」14.4s 气声突刺
    if mode_key == "guwen_lanting":
        m_ye = (grid >= 13.45) & (grid <= 15.75) & (~np.isnan(g))
        g[m_ye] = np.clip(g[m_ye], -2.5, 2.3)
    _SRC_CACHE[mode_key] = (grid, g.astype(np.float32))
    return _SRC_CACHE[mode_key]


def _micro_residual(mode_key: str) -> np.ndarray:
    """原版母带原速真实微颤残差（去掉 60ms 以上慢变化与大跳变，顺序取用绝不拉伸）。"""
    if mode_key in _RESID_CACHE:
        return _RESID_CACHE[mode_key]
    _, g = _master_track(mode_key)
    pieces = []
    v = ~np.isnan(g)
    i = 0
    while i < len(g):
        if not v[i]:
            i += 1
            continue
        j = i
        while j < len(g) and v[j]:
            j += 1
        if j - i >= 50:
            seg = g[i:j].astype(np.float64)
            r = seg - gaussian_filter1d(seg, sigma=12)
            slope = np.abs(np.gradient(gaussian_filter1d(seg, 2))) / 0.005
            r[slope > 18] = 0.0
            r = np.clip(r, -0.85, 0.85)
            pieces.append(r[6:-6])
        i = j
    res = np.concatenate(pieces).astype(np.float32) if pieces else np.zeros(400, np.float32)
    _RESID_CACHE[mode_key] = res
    return res


def _pick_template(mode_key: str, plan: ClausePlan, tones, prev_end_st: float) -> Tuple[List[Optional[Slot]], str]:
    T = TEMPLATES[mode_key]
    role = plan.role
    if mode_key == "fu_chibi":
        all_ping = all(t.sheng == "ping" for t in tones)
        n = len(tones)
        if role == "QI" and n == 4 and tones[-1].sheng == "ping" and plan.punc == "，" and not plan.tight_next:
            return T["QI_LOW"], "起·苍茫低徵"
        if role == "CHENG_UP" and all_ping:
            return T["CHENG_UP_PING"], "承·江风平腔"
        if role == "CHENG_DOWN" and prev_end_st <= -5.5:
            return T["CHENG_DOWN_AFTER_LOW"], "承·低腔拔起"
        if role == "CHENG_DOWN" and plan.is_major_end:
            return T["CHENG_DOWN_MAJOR"], "承·句号落徵"
        if role == "HE_GRAND" and n == 4 and abs(prev_end_st) <= 0.8 and tones[0].sheng != "ping":
            return T["HE_AFTER_PING"], "合·仄扬归宫"
    if role == "PU_LONG":
        return T.get("CHENG_UP"), "铺·长句"
    return T.get(role, T.get("CHENG_UP")), plan.role_desc


def _assign_slots(mode_key: str, plan: ClausePlan, sylls, tones, prev_end_st: float) -> Tuple[List[Slot], str]:
    mp = MODE_PARAMS[mode_key]
    n = len(sylls)
    last_t = tones[-1].sheng
    has_part = (sylls[-1].char in _FINAL_PARTICLES)
    tpl, tag = _pick_template(mode_key, plan, tones, prev_end_st)
    H, P, C, Tt = tpl

    def body(k: int) -> Slot:
        sh = tones[k].sheng
        if sh == "ru":
            return (mp["body_ze"] if mode_key == "fu_chibi" else mp["body_ping"] + 0.5, 0.30, None, None, 0.0)
        return (mp["body_ze"] if sh != "ping" else mp["body_ping"], mp["body_dur"] if sh == "ping" else 0.38, None, None, 0.0)

    def tone_slot(base: Optional[Slot], k: int) -> Slot:
        return body(k) if base is None else base

    # 长句 (>=7 字)
    if n >= 7:
        # 1) 高腔转折长句（如《爱莲说》「晋陶渊明独爱菊」、《诫子书》「非淡泊无以明志」）：保持明亮高音区！
        if plan.role == "ZHUAN_HIGH":
            out: List[Slot] = []
            hi_base = +5.1 if mode_key == "guwen_lanting" else (+2.4 if mode_key == "guwen_loushi" else +2.2)
            hi_ze = +6.4 if mode_key == "guwen_lanting" else (+4.6 if mode_key == "guwen_loushi" else +4.4)
            for k in range(n - 1):
                sh = tones[k].sheng
                if k == 0:
                    out.append((hi_base, 0.38, hi_base - 2.0, None, +0.6))
                elif k == n - 2 and sh != "ping":
                    out.append((hi_ze, 0.40, None, hi_ze + 0.4 if mode_key == "guwen_lanting" else None, +1.0))
                elif sh == "ru":
                    out.append((hi_base if mode_key == "guwen_lanting" else hi_ze - 0.3, 0.30, None, None, 0.0))
                elif sh != "ping":
                    out.append((hi_ze, 0.38, None, None, +0.8))
                else:
                    out.append((hi_base, 0.40, None, None, +0.4))
            if mode_key == "guwen_lanting":
                out.append((+0.4 if last_t == "ru" else 0.0, 0.54 if last_t == "ru" else 0.72, +3.0, +0.2, 0.0))
            elif mode_key == "guwen_loushi":
                out.append((+3.2 if last_t != "ping" else +0.1, 0.72 if last_t != "ping" else 0.84, None, None, 0.0))
            else:
                out.append((+2.1, 0.68, None, None, 0.0))
            return out, f"{tag}·高腔长句{n}字"

        # 2) G1 长压轴句（如 11 字「予独爱莲之出淤泥而不染」）：中商领起 → 低羽铺陈 → 欲抑先扬 → 端庄叹咏归底！
        if plan.role == "HE_GRAND" and mode_key == "guwen_lanting":
            split_k = min(4, n // 2)
            out = []
            for k in range(n - 1):
                sh = tones[k].sheng
                if k < split_k:
                    out.append((+3.6 if (sh != "ping" and k >= 1) else +2.0, 0.36 if sh != "ping" else 0.40, None, None, +0.5))
                elif k == n - 2:
                    out.append((+0.8, 0.38, -1.6, +1.4, +0.6))  # 「而[不]染」温和欲抑先扬
                elif sh == "ru":
                    out.append((-2.2, 0.30, None, None, 0.0))
                else:
                    out.append((-2.8, 0.38, None, None, 0.0))
            if has_part:
                out.append((-2.3, 2.20, +1.8, -2.3, 0.0))
            else:
                out.append((-2.4, 1.24, +1.2, -2.5, 0.0))  # 实词端庄长落腔，绝不三波大颤！
            return out, f"{tag}·长句{n}字"

        # 3) 其它长句（如「会于会稽山阴之兰亭」「苏子与客泛舟游于赤壁之下」「非宁静无以致远」）
        L = LONG_LINES[mode_key]
        bl = L["body"]
        out = []
        for k in range(n - 2):
            src = bl[min(len(bl) - 1, int(round(k * (len(bl) - 1) / max(1, n - 3))))]
            out.append(src)
        out += [L["C"], L["T"]]
        return out, f"{tag}·长句{n}字"

    # 4 字标准句
    if n == 4:
        out = [H, tone_slot(P, 1), tone_slot(C, 2), Tt]
        if mode_key == "guwen_loushi":
            if plan.role == "CHENG_UP":
                # 如「静以修身」(上上平平) vs 「水不在深」(上入上平)：按实际四声调整 1/2/3 槽！
                if tones[1].sheng in ("shang", "qu"):
                    out[1] = (+3.8, 0.38, None, None, +0.8)
                elif tones[1].sheng == "ping":
                    out[1] = (+2.1, 0.46, None, None, 0.0)
                if tones[2].sheng == "ping":
                    out[2] = (+2.1, 0.46, None, None, 0.0)
            elif plan.role == "CHENG_DOWN":
                if tones[1].sheng != "ping":
                    out[1] = (+2.2, 0.42, None, None, 0.0)
                if last_t == "ru":
                    out[3] = (-3.0, 0.54, -0.2, -3.1, 0.0)  # 「俭以养德。」干脆落徵收句
        elif mode_key == "guwen_lanting":
            if plan.role == "CHENG_DOWN" and last_t == "ping":
                out[3] = (-2.2, 0.80, -0.4, -2.4, 0.0)
            if plan.role == "HE_GRAND" and not has_part:
                out[3] = (-2.4, 1.24, +1.2, -2.5, 0.0)
        return out, tag

    # 3 字短句（如「衔远山」「吞长江」）
    if n == 3:
        if mode_key == "fu_chibi":
            if plan.role == "QI":
                return [
                    (+2.1, 0.42, None, None, 0.0),
                    (+4.1 if tones[1].sheng != "ping" else +3.2, 0.40, None, None, +0.8),
                    (+2.1, 0.68, None, None, 0.0),
                ], tag
            if plan.role == "CHENG_DOWN":
                return [
                    (+2.1, 0.40, None, None, 0.0),
                    (+3.6 if tones[1].sheng == "ping" else +4.1, 0.42, None, None, +0.6),
                    (+0.1 if not plan.is_major_end else -3.1, 0.70, +1.8, 0.0 if not plan.is_major_end else -3.2, 0.0),
                ], tag
        return [tone_slot(H, 0), tone_slot(C, 1), Tt], tag

    if n <= 2:
        return ([Tt] if n == 1 else [tone_slot(H, 0), Tt]), tag

    # 5~6 字句（如「水陆草木之花」「可爱者甚蕃」「夫君子之行」）
    out = [tone_slot(H, 0), tone_slot(P, 1)]
    for k in range(2, n - 2):
        out.append(body(k))
    c_sl = tone_slot(C, n - 2)
    if mode_key == "guwen_lanting" and tones[n - 2].sheng == "ping" and plan.role == "QI":
        c_sl = (-1.0, 0.36, -2.4, +0.8, 0.0)  # 如「水陆草木[之]花」虚词温润过渡
    t_sl = Tt
    if mode_key == "guwen_lanting" and plan.role == "CHENG_DOWN" and last_t == "ping":
        t_sl = (-2.2, 0.84, -0.4, -2.4, 0.0)  # 如「可爱者甚[蕃]。」
    out += [c_sl, t_sl]
    return out, tag


_T09_PRAAT_CACHE: Optional[Tuple[np.ndarray, np.ndarray]] = None


def _get_t09_praat_f0() -> Tuple[np.ndarray, np.ndarray]:
    """抽取 t09《兰亭集序》经 Praat 自相关精确验证的 5ms 真人活腔轨迹（完整保留 +8.6st 之 / +11.4st 禊事 / 也四波摇腔，仅净化 14.92s 气声单帧干裂）。"""
    global _T09_PRAAT_CACHE
    if _T09_PRAAT_CACHE is not None:
        return _T09_PRAAT_CACHE
    import parselmouth
    import soundfile as sf

    _, rel, gong, _, _, max_s = MODE_PARAMS["guwen_lanting"]["src"]
    y, sr = sf.read(str(DATA_DIR / rel), dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    y = y[: int((max_s + 0.2) * sr)]
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=float(sr))
    pitch = snd.to_pitch_ac(
        time_step=0.005, pitch_floor=90.0, pitch_ceiling=480.0,
        silence_threshold=0.02, voicing_threshold=0.35,
    )
    t_grid = np.arange(0.0, max_s, 0.005)
    f0_raw = np.array([pitch.get_value_at_time(float(t)) for t in t_grid], dtype=np.float64)
    valid = (~np.isnan(f0_raw)) & (f0_raw > 0)
    t_v = t_grid[valid].astype(np.float32)
    st_v = (12.0 * np.log2(f0_raw[valid] / gong)).astype(np.float32)
    # 仅剔除孤立单帧跳变毛刺 (>5st)，100% 保留全部拱腔、高拔与四波摇腔！
    if len(st_v) >= 5:
        med5 = median_filter(st_v, size=5)
        spike = np.abs(st_v - med5) > 5.0
        st_v[spike] = med5[spike]
    # 仅净化「也」第3波落第4波处 14.88..15.00s 的 -40dB 单帧声门干裂 (-5.44st -> 顺滑衔入 -2.25st 尾波)
    creak_mask = (t_v >= 14.84) & (t_v <= 15.04)
    st_v[creak_mask] = np.maximum(st_v[creak_mask], -2.65)
    _T09_PRAAT_CACHE = (t_v, st_v)
    return _T09_PRAAT_CACHE


# V16 兰亭 t09 25字逐字声学切片原型库: (f0_s, f0_e, tpl_tone, target_dur_s, shift_st)
# 全部坐标经 20ms 质心/ZCR/Praat 谐波实测逐字复核，F0 严格只切纯净有声元音核，与字长/声母时长完全解耦！
_LANTING_SLICE_BANK = {
    "QI": [
        (0.15, 0.53, "shang", 0.43, 0.0),   # 永 (-3.3st 稳起)
        (0.66, 0.99, "ping",  0.44, 0.0),   # 和 (-3.1st 温托)
        (1.12, 1.44, "shang", 0.42, 0.0),   # 九 (-0.4 -> +3.2 -> -0.5st 上拱!)
        (1.46, 2.02, "ping",  0.60, 0.0),   # 年 (-1.0 -> +2.7 -> -0.4st 拱收!)
    ],
    "CHENG_DOWN": [
        (2.41, 2.72, "qu",    0.45, 0.0),   # 岁 (-3.3st)
        (2.82, 3.15, "qu",    0.42, 0.0),   # 在 (-1.1st 台阶上移)
        (3.30, 3.64, "shang", 0.47, 0.0),   # 癸 (-0.4 -> +2.9 -> -0.2st 上拱!)
        (3.81, 4.24, "shang", 0.58, 0.0),   # 丑 (-0.8st 双微颤沉敛收)
    ],
    "ZHUAN_HIGH": [
        (4.65, 4.98, "qu",    0.39, 0.0),   # 暮 (+1.8 -> +5.3st 陡起高徵!)
        (5.10, 5.46, "ping",  0.45, 0.0),   # 春 (+5.2 -> +6.0st 高亮稳托!)
        (5.56, 5.88, "ping",  0.42, 0.0),   # 之 (+5.2 -> +8.6 -> +5.2st 冲霄峰!)
        (6.06, 6.56, "ping",  0.65, 0.0),   # 初 (+3.3 悬停 -> 0.0st 五度流泻!)
    ],
    "PU_LONG_BODY": [
        (7.00, 7.34, "qu",    0.42, 0.0),   # 会 (-3.4st)
        (7.38, 7.76, "ping",  0.42, 0.0),   # 于 (-3.1st)
        (7.90, 8.24, "qu",    0.47, 0.0),   # 会(kuai) (-3.7 -> -2.4st)
        (8.37, 8.72, "ping",  0.46, 0.0),   # 稽(ji) (-3.3st)
        (8.84, 9.26, "ping",  0.52, 0.0),   # 山 (-4.9st 抑扬深钩 -> -2.8st!)
        (9.29, 9.69, "ping",  0.42, 0.0),   # 阴 (-2.9 -> -2.5st)
        (9.80, 10.11,"ping",  0.41, 0.0),   # 之 (-3.4st)
    ],
    "PU_LONG_CT": [
        (10.19, 10.58, "ping", 0.46, 0.0),  # 兰 (-2.3 -> +2.3 -> -0.5st 上拱!)
        (10.69, 11.19, "ping", 0.60, 0.0),  # 亭 (-1.1 -> +2.8 -> -1.3st 拱收!)
    ],
    "HE_GRAND": [
        (11.58, 11.98, "ping",  0.54, 0.0),  # 修 (-1.0 -> +5.5st 六半音大滑升!)
        (12.14, 12.58, "qu",    0.59, 0.0),  # 禊 (+11.4 -> +6.8st 高亢劲落!)
        (12.76, 13.19, "qu",    0.59, 0.0),  # 事 (+11.1 -> +5.5 -> +2.0st 流泻!)
        (13.22, 15.64, "shang", 2.42, 0.0),  # 也 (完整四波荡气回肠书斋大摇腔!)
    ],
}

_VOICED_ONSETS = frozenset({"y", "w", "m", "n", "l", "r"})


def _lanting_onset_dur(first_mp: str, ch: str, cdur: float, is_long_tail: bool = False) -> float:
    """经 t09 20ms 实测 + M4Singer 声学盲测标定的吟诵声母时长（杜绝 ch 糊成 sh、z 糊成 d）。"""
    if first_mp in ("x", "sh", "s", "f"):
        base = 0.114 if cdur >= 0.56 else (0.102 if cdur >= 0.50 else 0.092)
    elif first_mp == "z":
        base = 0.098  # 齿音塞擦 z [ts] 给足 98ms 清晰嘶擦，杜绝糊成 d!
    elif first_mp in ("ch", "c", "q"):
        base = 0.075  # 送气塞擦 ch/c/q 紧致 75ms 干脆爆破，绝不拖长糊成 sh/x!
    elif first_mp in ("zh", "j"):
        base = 0.080
    elif first_mp in ("k", "p", "t"):
        base = 0.090 if first_mp == "k" else 0.080
    elif first_mp in ("g", "d", "b"):
        base = 0.082
    elif first_mp == "h":
        base = 0.082
    elif first_mp == "n":
        base = 0.072
    elif first_mp == "l":
        base = 0.062
    elif first_mp == "m":
        base = 0.052
    elif first_mp in ("y", "w"):
        base = 0.075 if is_long_tail else 0.045
    else:
        base = 0.065
    return min(base, max(0.042, cdur - 0.16), cdur * 0.32)


def _assign_lanting_live_slices(plan: ClausePlan, sylls, tones) -> Tuple[List[Tuple[float, float, str, float, float]], str, bool]:
    """按句群起承转合角色，为任意长度句分配 100% 源自 t09 母带的逐字实录 F0 切片。"""
    n = len(sylls)
    last_t = tones[-1].sheng
    has_part = (sylls[-1].char in _FINAL_PARTICLES)
    role = plan.role
    B = _LANTING_SLICE_BANK
    use_ye_fade = False

    if role == "HE_GRAND":
        H = B["HE_GRAND"]
        if n <= 6:
            if n == 4:
                out = list(H)
            elif n == 3:
                out = [H[0], H[2], H[3]]
            elif n <= 2:
                out = [H[2], H[3]][-n:]
            else:
                out = [H[0], H[1]] + [H[1]] * (n - 4) + [H[2], H[3]]
            if has_part:
                use_ye_fade = True
            else:
                # 实词结尾不拉 2.4s 虚词四波，改用端庄沉稳 1.25s 拱落归底
                out[-1] = (10.69, 11.19, "ping", 1.22, -1.0)
            return out, f"{plan.role_desc}·t09实录压轴", use_ye_fade
        # 长压轴句 (>=7字，如 11字《爱莲说》末句)：中商领起 -> 低羽铺陈 -> 兰亭拱收归底
        split_k = min(4, n // 2)
        body_low = B["PU_LONG_BODY"]
        out = []
        for k in range(n - 2):
            if k == 0:
                out.append((11.58, 11.98, "ping", 0.44, -2.5))  # 温润滑入中商
            elif k < split_k:
                sh = tones[k].sheng
                out.append((5.10, 5.46, "ping", 0.36 if sh != "ping" else 0.40, -1.8 if sh != "ping" else -3.0))
            else:
                src = body_low[(k - split_k) % len(body_low)]
                d = 0.32 if tones[k].sheng == "ru" else min(0.40, src[3])
                out.append((src[0], src[1], src[2], d, +0.4))
        out.append((10.19, 10.58, "ping", 0.42, 0.0))  # 欲抑先扬挑拱
        if has_part:
            out.append(H[3])
            use_ye_fade = True
        else:
            out.append((10.69, 11.19, "ping", 1.24, -1.0))  # 实词端庄归底 (-2.3st)
        return out, f"{plan.role_desc}·t09长句压轴{n}字", use_ye_fade

    if role == "ZHUAN_HIGH":
        Z = B["ZHUAN_HIGH"]
        if n == 4:
            out = list(Z)
        elif n < 4:
            out = [Z[0], Z[2], Z[3]][-n:]
        else:
            out = [Z[0]]
            for k in range(1, n - 2):
                sh = tones[k].sheng
                d = 0.34 if sh == "ru" else 0.40
                out.append((5.10, 5.46, "ping", d, +0.8 if sh in ("shang", "qu") else 0.0))
            out.append((5.56, 5.88, "ping", 0.40, -0.6 if n >= 7 else 0.0))
            out.append((6.06, 6.56, "ping", 0.54 if last_t == "ru" else 0.65, +0.4 if last_t == "ru" else 0.0))
        return out, f"{plan.role_desc}·t09实录高转", False

    if role == "PU_LONG" or n >= 7:
        bl = B["PU_LONG_BODY"]
        ct = B["PU_LONG_CT"]
        out = []
        for k in range(n - 2):
            src = bl[min(len(bl) - 1, int(round(k * (len(bl) - 1) / max(1, n - 3))))]
            out.append(src)
        out.extend(ct)
        return out, f"{plan.role_desc}·t09实录长铺{n}字", False

    if role == "CHENG_DOWN":
        CD = B["CHENG_DOWN"]
        if n == 4:
            out = list(CD)
        elif n < 4:
            out = [CD[0], CD[2], CD[3]][-n:]
        else:
            out = [CD[0]] + [CD[1]] * (n - 3) + [CD[2], CD[3]]
        # 泛化偶句平声句号收尾（如《爱莲说》「可爱者甚蕃。」）：柔和落低羽 -2.2st
        if last_t == "ping" and plan.is_major_end:
            f0s, f0e, tt, td, sh = out[-1]
            out[-1] = (f0s, f0e, tt, max(0.76, td), sh - 1.4)
        return out, f"{plan.role_desc}·t09实录承抑", False

    # QI / LEAD / CHENG_UP
    Q = B["QI"]
    if n == 4:
        out = list(Q)
    elif n < 4:
        out = [Q[0], Q[2], Q[3]][-n:]
    else:
        bl = B["PU_LONG_BODY"]
        out = [Q[0], Q[1]]
        for k in range(2, n - 2):
            out.append(bl[k % len(bl)])
        # 若倒数第2字为「之」等轻虚词（如「水陆草木之花」），用温润过渡而非陡峰
        if sylls[n - 2].char in _MID_PARTICLES:
            out.append((10.19, 10.58, "ping", 0.36, -0.5))
        else:
            out.append(Q[2])
        out.append(Q[3])
    return out, f"{plan.role_desc}·t09实录起兴", False


def _compile_lanting_live_ds(
    text: str,
    era: str = "zh",
    key_hz: Optional[float] = None,
    speaker: str = "opencpop_base",
    voice_gender: str = "female",
    tempo_scale: float = 1.0,
) -> List[dict]:
    """
    V16《兰亭集序》调式 G1 专用编译器（完整移植《诗经》V9.6 真人母带活腔架构）：
      1. 100% 真人 Praat 实录 F0 逐字直切（零分段直线近似、零截顶、零词内 SP）；
      2. 声母/元音 F0 物理分离（清擦/塞擦声母保持元音起音高，彻底杜绝高拔起腔被无声辅音吃掉）；
      3. 极轻 sigma=1.4 (7ms) 平滑保留 100% 拱腔活力 + 纯净棚录零 EQ 染色，仅在末句虚词「也」尾波施加自然气息渐收！
    """
    cfg = GUWEN_V14_MODES["guwen_lanting"]
    use_key_hz = float(key_hz) if key_hz is not None else float(
        cfg["male_key_hz"] if voice_gender == "male" else cfg["female_key_hz"]
    )
    min_st, max_st = float(cfg["min_st"]), float(cfg["max_st"])
    speed = float(cfg["speed_scale"]) * float(tempo_scale)
    t_v, st_v = _get_t09_praat_f0()

    raw = _split_clauses_with_punct(text)
    if not raw:
        return []
    plans = plan_guwen_discourse(raw, mode_key="guwen_lanting")
    tone_era = "em" if era.lower() == "em" else "mc"
    f0_step = 0.005
    items: List[dict] = []
    offset = 0.0

    for plan in plans:
        sylls = [s for s in text_to_sing_syllables(plan.text, era=era) if s.char != " "]
        n = len(sylls)
        if n == 0:
            continue
        tones = _analyze_clause_tones(plan.text, sylls, tone_era=tone_era)
        slices, tag, use_ye_fade = _assign_lanting_live_slices(plan, sylls, tones)

        lead_sp = 0.06
        ph_seq: List[str] = ["SP"]
        ph_dur: List[float] = [lead_sp]
        f0_chunks: List[np.ndarray] = []
        dips: List[Tuple[float, float, float]] = []
        prev_nuc_d = 0.0

        for k, (sy, tp, (f0_s, f0_e, _, t_dur, sh_st)) in enumerate(zip(sylls, tones, slices)):
            cdur = max(0.25, t_dur * speed)
            pairs, _, has_m = _build_char_phones(sy)
            first_mp = pairs[0][1]
            is_cons = (pairs[0][0] not in VOWEL_SET and pairs[0][0] not in CODA_SET)
            is_long_tail = (k == n - 1 and cdur >= 1.50)

            actual_on = _lanting_onset_dur(first_mp, sy.char, cdur, is_long_tail=is_long_tail) if is_cons else 0.0
            m_d = min(0.048, cdur * 0.14) if has_m else 0.0
            actual_nuc = max(0.16, cdur - actual_on - m_d)

            # 仅对长元音 (>=0.45s) 后的鼻音 n/m 与塞音 b/p/d/t/g/k 施加诗经 V9.6 成阻微谷；清塞擦 z/ch/zh 绝不加谷防变浊！
            dip_amp = 0.0
            if is_cons and k > 0 and prev_nuc_d >= 0.45:
                if first_mp in _NASAL_ONSETS:
                    dip_amp = 1.35
                elif first_mp in _STOP_ONSETS:
                    dip_amp = 1.55
            prev_nuc_d = actual_nuc

            n_on = int(round(actual_on / f0_step)) if actual_on > 0 else 0
            n_voiced = max(6, int(round((actual_nuc + m_d) / f0_step)))
            sl_nuc = np.interp(np.linspace(f0_s, f0_e, n_voiced, dtype=np.float32), t_v, st_v).astype(np.float32)
            if abs(sh_st) > 1e-3:
                sl_nuc = sl_nuc + float(sh_st)

            if n_on > 0:
                if first_mp in _VOICED_ONSETS and k > 0 and f0_chunks:
                    prev_v = float(f0_chunks[-1][-1])
                    sl_on = np.linspace(prev_v, float(sl_nuc[0]), n_on, dtype=np.float32)
                else:
                    sl_on = np.full(n_on, float(sl_nuc[0]), dtype=np.float32)
                sl_full = np.concatenate([sl_on, sl_nuc])
            else:
                sl_full = sl_nuc

            char_t0 = sum(ph_dur)
            if dip_amp > 0.0 and actual_on > 0.0:
                dips.append((char_t0, char_t0 + actual_on, dip_amp))

            for rp, mp in pairs:
                if rp == "coda_m":
                    d = round(m_d, 4)
                elif rp in VOWEL_SET:
                    d = round(actual_nuc, 4)
                else:
                    d = round(actual_on if actual_on > 0 else 0.048, 4)
                ph_seq.append(mp)
                ph_dur.append(d)
            f0_chunks.append(sl_full)

        # 精确对齐 t09 原版句间呼吸时长 (C1:0.24s, C2:0.34s, C3:0.33s, C4:0.25s, 终句:0.30s)
        if plan.is_passage_end:
            pause_tot = 0.30
        elif plan.is_major_end or plan.role == "ZHUAN_HIGH":
            pause_tot = 0.33
        else:
            pause_tot = 0.24
        tail_sp = round(min(0.18, max(0.12, pause_tot * 0.58)), 4)
        gap_after = max(0.03, pause_tot - tail_sp - lead_sp)
        ph_seq.append("SP")
        ph_dur.append(tail_sp)

        total_dur = sum(ph_dur)
        n_tot = max(12, int(round(total_dur / f0_step)))
        f0_st = np.zeros(n_tot, dtype=np.float32)
        first_v = float(f0_chunks[0][0]) if f0_chunks else 0.0
        cur_i = int(round(lead_sp / f0_step))
        f0_st[:cur_i] = first_v
        last_v = first_v
        for arr in f0_chunks:
            nx = min(n_tot, cur_i + len(arr))
            if nx > cur_i:
                f0_st[cur_i:nx] = arr[: nx - cur_i]
                last_v = float(f0_st[nx - 1])
            cur_i = nx
        if cur_i < n_tot:
            f0_st[cur_i:] = last_v

        # 诗经 V9.6 极轻平滑 (sigma=1.4 = 7ms)：零台阶又 100% 保留每个字的拱腔与微颤！
        f0_st = gaussian_filter1d(f0_st, sigma=1.4).astype(np.float32)
        for d_t0, d_t1, d_amp in dips:
            i0, i1 = int(round(d_t0 / f0_step)), min(n_tot, int(round(d_t1 / f0_step)))
            if i1 > i0 + 2:
                f0_st[i0:i1] -= d_amp * np.hanning(i1 - i0).astype(np.float32)

        f0_st = np.clip(f0_st, min_st, max_st)
        f0_hz = use_key_hz * (2.0 ** (f0_st / 12.0))

        # 仅当压轴末字为虚词大摇腔「也」时，记录柔和自然的气息渐收锚点（配合 DiffSinger 原生尾音衰减，1:1 对齐 t09 第3/4波动态）
        tail_fade = None
        if use_ye_fade:
            end_voiced = offset + sum(ph_dur[:-1])
            tail_fade = [round(max(0.0, end_voiced - 1.15), 4), round(max(0.0, end_voiced - 0.65), 4), round(end_voiced + 0.05, 4)]

        items.append({
            "text": plan.text + plan.punc,
            "era": era,
            "genre": "guwen_lanting_v16",
            "archetype": tag,
            "pz": "".join(t.pz for t in tones),
            "offset": round(offset, 4),
            "ph_seq": " ".join(ph_seq),
            "ph_dur": " ".join(f"{x:.4f}" for x in ph_dur),
            "f0_seq": " ".join(f"{float(x):.2f}" for x in f0_hz),
            "f0_timestep": str(f0_step),
            "spk_name": speaker,
            "x_pure_studio": True,
            **({"x_tail_fade": tail_fade} if tail_fade is not None else {}),
        })
        offset += total_dur + gap_after
    return items


_T16_PRAAT_CACHE: Optional[Tuple[np.ndarray, np.ndarray]] = None


def _get_t16_praat_f0() -> Tuple[np.ndarray, np.ndarray]:
    """抽取 t16《陋室铭》全篇 56.4s (81字) 经 Praat 自相关精确验证的 5ms 真人活腔轨迹 (gong=136.0Hz)。"""
    global _T16_PRAAT_CACHE
    if _T16_PRAAT_CACHE is not None:
        return _T16_PRAAT_CACHE
    import parselmouth
    import soundfile as sf

    y, sr = sf.read(str(DATA_DIR / "bodies/t16_guwen_陋室铭.wav"), dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=float(sr))
    pitch = snd.to_pitch_ac(
        time_step=0.005, pitch_floor=65.0, pitch_ceiling=260.0,
        silence_threshold=0.02, voicing_threshold=0.35,
    )
    t_grid = np.arange(0.0, len(y) / sr, 0.005)
    f0_raw = np.array([pitch.get_value_at_time(float(t)) for t in t_grid], dtype=np.float64)
    valid = (~np.isnan(f0_raw)) & (f0_raw > 0)
    t_v = t_grid[valid].astype(np.float32)
    st_v = (12.0 * np.log2(f0_raw[valid] / 136.0)).astype(np.float32)
    if len(st_v) >= 7:
        m7 = median_filter(st_v, size=7)
        spike = np.abs(st_v - m7) > 4.0
        st_v[spike] = m7[spike]
    _T16_PRAAT_CACHE = (t_v, st_v)
    return _T16_PRAAT_CACHE


# V16 陋室铭 t16 全篇 81 字声学法医实测活腔切片库: (f0_s, f0_e, target_dur_s, shift_st)
# 纠正了旧版把「不/在/高」「不/在/深」整体错移一字、漏掉「斯是陋室惟吾 —— 德馨」6+2 顿挫气口与「馨」悬宫落底的根本偏差！
_LOUSHI_SLICE_BANK = {
    # 1) 开篇 24 字逐字 1:1 实录原型
    "c1_shan": (0.16, 0.94, 0.82, 0.0),    # 山(平): -0.3 -> +1.9st 长平立商
    "c1_bu":   (1.06, 1.23, 0.28, 0.0),    # 不(入): +3.9st 高角短促入声!
    "c1_zai":  (1.35, 1.94, 0.58, 0.0),    # 在(上/去): -2.6 -> -0.6st 低起盘升长枢轴!
    "c1_gao":  (2.04, 2.32, 0.54, 0.0),    # 高(平): 0.0st 稳落主宫悬韵
    "c2_you":  (2.79, 3.40, 0.60, 0.0),    # 有(上): -4.4 -> +2.0st 下联上声深钩仰挑!
    "c2_xian": (3.55, 4.09, 0.62, 0.0),    # 仙(平): +2.0st 饱满立商
    "c2_ze":   (4.24, 4.41, 0.28, 0.0),    # 则(入): +0.2st 踏宫入声短垫!
    "c2_ming": (4.44, 5.20, 0.80, 0.0),    # 名(平): -2.6 -> -3.2st 四度落低徵长收
    "c3_shui": (5.74, 6.48, 0.80, 0.0),    # 水(上): -2.8 -> +2.4st 上声低起仰挑长腔!
    "c3_bu":   (6.60, 6.81, 0.28, 0.0),    # 不(入): +4.1st 高角短促入声!
    "c3_zai":  (6.93, 7.49, 0.58, 0.0),    # 在(上/去): -2.2 -> +0.2st 低起盘升长枢轴!
    "c3_shen": (7.64, 7.90, 0.54, 0.0),    # 深(平): +0.1st 稳落主宫悬韵
    "c4_you":  (8.27, 8.85, 0.60, 0.0),    # 有(上): -4.4 -> +1.8st 下联上声深钩仰挑!
    "c4_long": (8.88, 9.49, 0.62, 0.0),    # 龙(平): +1.9st 饱满立商
    "c4_ze":   (9.62, 9.81, 0.28, 0.0),    # 则(入): +0.3st 踏宫入声短垫!
    "c4_ling": (9.84, 10.40, 0.78, 0.0),   # 灵(平): -1.2 -> -2.8st 四度落低徵长收
    "c5_si":   (10.98, 11.06, 0.28, 0.0),  # 斯(平): +4.7st 高角破阵起首
    "c5_shi":  (11.21, 11.63, 0.52, 0.0),  # 是(上/去): +4.4st 重音高擎!
    "c5_lou":  (11.70, 12.29, 0.60, 0.0),  # 陋(去): -2.5 -> +0.5st 深钩回立主宫
    "c5_shi2": (12.48, 13.01, 0.52, 0.0),  # 室(入): +0.1st 主宫稳承连下句
    "c6_wei":  (13.06, 13.45, 0.42, 0.0),  # 惟(平): +0.5 -> +2.2st 扬入商音
    "c6_wu":   (13.48, 14.06, 0.64, 0.0),  # 吾(平): 0.0 -> -3.4st 缓沉低徵蓄势!
    "c6_de":   (14.52, 15.00, 0.50, 0.0),  # 德(入): -5.3 -> -2.5st 气口后沉雄挑起!
    "c6_xin":  (15.16, 16.46, 1.42, 0.0),  # 馨(平): +0.4st(0.3s) ↘ -5.0st(1.1s) 悬宫落底大腔!
    # 2) 全篇 81 字提炼的四声 × 平仄 × 句位通用活腔构件（严守依字行腔、零倒字！）
    "head_ping":       (0.16, 0.94, 0.50, -0.3),   # 平声领起 (-0.6 -> +1.6st)
    "head_shang_up":   (5.74, 6.48, 0.56, 0.0),    # 上联上声起首 (-2.8 -> +2.4st: 静)
    "head_shang_dn":   (8.27, 8.85, 0.56, 0.0),    # 下联上声深钩 (-4.4 -> +1.8st: 俭)
    "head_long_rise":  (37.78, 38.30, 0.44, 0.0),  # 长联平声起首 (-3.0 -> +2.8st: 无/非)
    "body_ping":       (8.88, 9.49, 0.46, 0.0),    # 句中平声稳托商音 (+1.9st: 修/君/宁/明)
    "body_ping_gong":  (38.88, 39.40, 0.37, 0.0),  # 句中虚词/转折踏宫 (+0.2st: 之/无)
    "body_shang":      (6.60, 6.81, 0.38, -0.2),   # 句中上声清挑角音 (+3.8st: 子/以)
    "body_qu":         (11.21, 11.63, 0.42, 0.0),  # 句中去声高起劲落 (+4.8 -> +3.8st: 淡)
    "body_ru_hi":      (6.60, 6.81, 0.30, -0.2),   # 句中入声短促高挑 (+3.9st: 泊/竹)
    "body_ru_gong":    (9.62, 9.81, 0.29, 0.0),    # 句中入声踏宫短垫 (+0.3st: 则/牍)
    "piv_ze_lift":     (47.30, 47.48, 0.40, -0.5), # 下联倒二欲抑先扬 (+3.6st: 养)
    "piv_grand_arch":  (40.54, 40.82, 0.42, +0.5), # 压轴倒二欲抑先扬 (+0.6 -> +1.7st: 劳/致)
    "cad_up_ping":     (7.64, 7.90, 0.56, 0.0),    # 上联平声悬宫收句 (+0.1st: 行/身)
    "cad_up_shang":    (37.06, 37.42, 0.62, 0.0),  # 上联上声高悬收句 (+1.6 -> +3.0st: 耳)
    "cad_up_qu":       (36.82, 37.42, 0.66, -0.4), # 上联去声高悬收句 (+4.6 -> +2.6st: 志)
    "cad_dn_ping":     (9.84, 10.40, 0.78, 0.0),   # 下联平声四度落低徵 (-1.2 -> -2.9st)
    "cad_dn_ru":       (19.70, 20.30, 0.54, 0.0),  # 下联入声劲落低徵 (+1.5 -> -3.2st: 绿/德)
    "cad_dn_ze":       (47.94, 48.36, 0.68, 0.0),  # 下联上去沉敛落韵 (+0.5 -> -2.2st)
    "cad_grand_ping":  (15.16, 16.46, 1.42, 0.0),  # 压轴平声悬宫落底 (+0.4 -> -5.0st: 馨/形)
    "cad_grand_shang": (53.80, 55.46, 1.28, 0.0),  # 压轴上声悬宫落底 (+1.1 -> -4.1st: 有/远)
}


def _assign_loushi_phrase_slices(
    plan: ClausePlan, ci: int, sylls, tones
) -> Tuple[List[Tuple[float, float, float, float]], str, bool]:
    """依据【起承转合角色 × 中古四声 × 平低仄高对仗律】指派 t16 100% 真人实录活腔切片。"""
    S = _LOUSHI_SLICE_BANK
    n = len(sylls)
    shs = [t.sheng for t in tones]
    role = plan.role
    last_t = shs[-1]

    if n == 4:
        if role in ("QI", "CHENG_UP"):
            if shs[1] == "ru" and shs[2] != "ping" and last_t == "ping":
                if shs[0] == "ping":
                    return [S["c1_shan"], S["c1_bu"], S["c1_zai"], S["c1_gao"]], f"{plan.role_desc}·平起悬宫", False
                return [S["c3_shui"], S["c3_bu"], S["c3_zai"], S["c3_shen"]], f"{plan.role_desc}·上起悬宫", False
            h = S["c1_shan"] if shs[0] == "ping" else (S["head_shang_up"] if shs[0] == "shang" else S["body_qu"])
            p = S["body_ru_hi"] if shs[1] == "ru" else (S["body_ping"] if shs[1] == "ping" else S["body_shang"])
            c = (8.88, 9.49, 0.50, 0.0) if shs[2] == "ping" else (S["body_ru_hi"] if shs[2] == "ru" else S["body_shang"])
            t = S["cad_up_ping"] if last_t == "ping" else (
                S["cad_up_shang"] if last_t == "shang" else (S["cad_up_qu"] if last_t == "qu" else S["c5_shi2"])
            )
            return [h, p, c, t], f"{plan.role_desc}·平低仄高悬宫", False

        if role == "CHENG_DOWN":
            if shs[0] != "ping" and shs[1] == "ping" and shs[2] == "ru" and last_t == "ping":
                if ci <= 2:
                    return [S["c2_you"], S["c2_xian"], S["c2_ze"], S["c2_ming"]], f"{plan.role_desc}·四度落徵", False
                return [S["c4_you"], S["c4_long"], S["c4_ze"], S["c4_ling"]], f"{plan.role_desc}·四度落徵", False
            h = S["head_shang_dn"] if shs[0] != "ping" else (0.16, 0.94, 0.48, -1.5)
            p = (8.88, 9.49, 0.54, 0.0) if shs[1] == "ping" else (8.88, 9.49, 0.40, +0.3)
            c = S["c4_ze"] if shs[2] == "ru" else (S["piv_ze_lift"] if shs[2] != "ping" else S["body_ping"])
            t = S["cad_dn_ping"] if last_t == "ping" else (S["cad_dn_ru"] if last_t == "ru" else S["cad_dn_ze"])
            return [h, p, c, t], f"{plan.role_desc}·依声落徵", False

        if role == "ZHUAN_HIGH":
            t = S["c5_shi2"] if last_t == "ru" else (S["cad_up_ping"] if last_t == "ping" else S["cad_up_qu"])
            return [S["c5_si"], S["c5_shi"], S["c5_lou"], t], f"{plan.role_desc}·高角转折", False

        if role == "HE_GRAND":
            if shs[2] == "ru" and last_t == "ping":
                return [S["c6_wei"], S["c6_wu"], (14.52, 15.00, 0.46, +1.2), S["c6_xin"]], f"{plan.role_desc}·悬宫落底", True
            t = S["cad_grand_ping"] if last_t == "ping" else (S["cad_grand_shang"] if last_t == "shang" else S["cad_dn_ru"])
            return [S["c6_wei"], S["c6_wu"], S["piv_grand_arch"], t], f"{plan.role_desc}·悬宫落底", (last_t in ("ping", "shang"))

    if role == "LEAD":
        out = []
        for k in range(n - 1):
            if k == 0:
                out.append((0.16, 0.94, 0.46, -0.4) if shs[0] == "ping" else S["head_shang_up"])
            elif sylls[k].char in _MID_PARTICLES:
                out.append(S["body_ping_gong"])
            elif shs[k] == "ru":
                out.append(S["body_ru_hi"])
            elif shs[k] != "ping":
                out.append(S["body_shang"])
            else:
                out.append(S["body_ping"])
        out.append(S["cad_up_ping"] if last_t == "ping" else S["cad_up_qu"])
        return out, f"{plan.role_desc}·端凝立调", False

    # 长对仗上句 (如 7 字「非淡泊无以明志」/ 6 字「无丝竹之乱耳」)：3+4 连贯乐句拱弧，杜绝逐字锯齿跳变！
    if role in ("ZHUAN_HIGH", "CHENG_UP", "QI"):
        out = []
        mid = max(2, n // 2)
        for k in range(n - 1):
            if k == 0:
                out.append(S["head_long_rise"] if shs[0] == "ping" else S["head_shang_up"])
            elif k < mid:
                if shs[k] == "ru":
                    out.append(S["body_ru_hi"])
                elif shs[k] == "qu":
                    out.append(S["body_qu"])
                elif shs[k] == "shang":
                    out.append(S["body_shang"])
                else:
                    out.append(S["body_ping"])
            else:
                # 后半桥接段温润承托在商/次角 (+1.8~+2.6st)，把最高悬扬张力留给句末仄韵（如「志」「耳」）！
                if shs[k] == "ru":
                    out.append((6.60, 6.81, 0.30, -0.8))
                elif shs[k] != "ping":
                    out.append((8.88, 9.49, 0.38, +0.6))
                else:
                    out.append((8.88, 9.49, 0.40, +0.0))
        t = S["cad_up_ping"] if last_t == "ping" else (
            S["cad_up_shang"] if last_t == "shang" else (S["cad_up_qu"] if last_t == "qu" else S["c5_shi2"])
        )
        out.append(t)
        return out, f"{plan.role_desc}·长联仄悬{n}字", False

    # 长对仗下句 / 压轴长句 (如 7 字「非宁静无以致远。」/ 6 字「无案牍之劳形。」)：前半呼应 -> 后半沉蓄 -> 欲抑先扬 -> 悬宫落底！
    out = []
    mid = max(2, n // 2)
    for k in range(n - 2):
        if k == 0:
            out.append(S["head_long_rise"] if shs[0] == "ping" else S["head_shang_dn"])
        elif k < mid:
            if shs[k] == "ru":
                out.append(S["body_ru_hi"])
            elif shs[k] != "ping":
                out.append((11.21, 11.63, 0.42, -0.2))
            else:
                out.append((8.88, 9.49, 0.42, +0.2))
        else:
            if shs[k] == "ru":
                out.append(S["body_ru_gong"])
            elif shs[k] != "ping":
                out.append((40.08, 40.48, 0.38, +0.5))
            else:
                out.append(S["body_ping_gong"])
    out.append(S["piv_grand_arch"] if role == "HE_GRAND" else S["piv_ze_lift"])
    if role == "HE_GRAND":
        t = S["cad_grand_ping"] if last_t == "ping" else (S["cad_grand_shang"] if last_t == "shang" else S["cad_dn_ru"])
        out.append(t)
        return out, f"{plan.role_desc}·欲抑先扬落底{n}字", (last_t in ("ping", "shang"))
    t = S["cad_dn_ping"] if last_t == "ping" else (S["cad_dn_ru"] if last_t == "ru" else S["cad_dn_ze"])
    out.append(t)
    return out, f"{plan.role_desc}·长联落徵{n}字", False


def _build_loushi_item(
    clause_txt: str,
    punc: str,
    tag: str,
    era: str,
    sylls,
    tones,
    slices: List[Tuple[float, float, float, float]],
    t_v: np.ndarray,
    st_v: np.ndarray,
    use_key_hz: float,
    min_st: float,
    max_st: float,
    speed: float,
    speaker: str,
    offset: float,
    lead_sp: float,
    tail_sp: float,
    use_fade: bool,
) -> Tuple[dict, float]:
    f0_step = 0.005
    ph_seq: List[str] = ["SP"]
    ph_dur: List[float] = [lead_sp]
    f0_chunks: List[np.ndarray] = []
    dips: List[Tuple[float, float, float]] = []
    prev_nuc_d = 0.0

    for k, (sy, (f0_s, f0_e, t_dur, sh_st)) in enumerate(zip(sylls, slices)):
        cdur = max(0.25, t_dur * speed)
        pairs, _, has_m = _build_char_phones(sy)
        first_mp = pairs[0][1]
        is_cons = (pairs[0][0] not in VOWEL_SET and pairs[0][0] not in CODA_SET)
        if is_cons:
            if first_mp in ("x", "sh", "s", "f"):
                actual_on = 0.100 if cdur >= 0.50 else 0.088
            elif first_mp == "z":
                actual_on = 0.095
            elif first_mp in ("ch", "c", "q"):
                actual_on = 0.075
            elif first_mp in ("zh", "j"):
                actual_on = 0.080
            elif first_mp in ("p", "t", "k"):
                actual_on = 0.082
            elif first_mp in ("b", "d", "g"):
                actual_on = 0.076
            elif first_mp in ("n", "m"):
                actual_on = 0.082 if (k > 0 and prev_nuc_d >= 0.48) else 0.064
            elif first_mp == "l":
                actual_on = 0.064
            elif first_mp in ("y", "w"):
                actual_on = 0.055
            else:
                actual_on = 0.072
            actual_on = min(actual_on, max(0.042, cdur - 0.15), cdur * 0.30)
        else:
            actual_on = 0.0

        m_d = min(0.048, cdur * 0.14) if has_m else 0.0
        actual_nuc = max(0.15, cdur - actual_on - m_d)

        dip_amp = 0.0
        if is_cons and k > 0 and prev_nuc_d >= 0.48:
            if first_mp in _NASAL_ONSETS:
                dip_amp = 1.25
            elif first_mp in ("b", "d", "p", "t"):
                dip_amp = 1.20
            elif first_mp in ("g", "k"):
                dip_amp = 0.55  # 轻成阻微谷保「高」字 g 爆发利落不化为熬!
        prev_nuc_d = actual_nuc

        n_on = int(round(actual_on / f0_step)) if actual_on > 0 else 0
        n_voiced = max(6, int(round((actual_nuc + m_d) / f0_step)))
        sl_nuc = np.interp(np.linspace(f0_s, f0_e, n_voiced, dtype=np.float32), t_v, st_v).astype(np.float32) + float(sh_st)

        if n_on > 0:
            if first_mp in _VOICED_ONSETS and k > 0 and f0_chunks:
                sl_on = np.linspace(float(f0_chunks[-1][-1]), float(sl_nuc[0]), n_on, dtype=np.float32)
            else:
                sl_on = np.full(n_on, float(sl_nuc[0]), dtype=np.float32)
            sl_full = np.concatenate([sl_on, sl_nuc])
        else:
            sl_full = sl_nuc

        char_t0 = sum(ph_dur)
        if dip_amp > 0.0 and actual_on > 0.0:
            dips.append((char_t0, char_t0 + actual_on, dip_amp))

        for rp, mp in pairs:
            if rp == "coda_m":
                d = round(m_d, 4)
            elif rp in VOWEL_SET:
                d = round(actual_nuc, 4)
            else:
                d = round(actual_on if actual_on > 0 else 0.048, 4)
            ph_seq.append(mp)
            ph_dur.append(d)
        f0_chunks.append(sl_full)

    ph_seq.append("SP")
    ph_dur.append(round(tail_sp, 4))

    total_dur = sum(ph_dur)
    n_tot = max(12, int(round(total_dur / f0_step)))
    f0_st = np.zeros(n_tot, dtype=np.float32)
    first_v = float(f0_chunks[0][0]) if f0_chunks else 0.0
    cur_i = int(round(lead_sp / f0_step))
    f0_st[:cur_i] = first_v
    last_v = first_v
    for arr in f0_chunks:
        nx = min(n_tot, cur_i + len(arr))
        if nx > cur_i:
            f0_st[cur_i:nx] = arr[: nx - cur_i]
            last_v = float(f0_st[nx - 1])
        cur_i = nx
    if cur_i < n_tot:
        f0_st[cur_i:] = last_v

    f0_st = gaussian_filter1d(f0_st, sigma=1.5).astype(np.float32)
    for d_t0, d_t1, d_amp in dips:
        i0, i1 = int(round(d_t0 / f0_step)), min(n_tot, int(round(d_t1 / f0_step)))
        if i1 > i0 + 2:
            f0_st[i0:i1] -= d_amp * np.hanning(i1 - i0).astype(np.float32)

    f0_st = np.clip(f0_st, min_st, max_st)
    f0_hz = use_key_hz * (2.0 ** (f0_st / 12.0))

    tail_fade = None
    if use_fade:
        end_voiced = offset + sum(ph_dur[:-1])
        tail_fade = [round(max(0.0, end_voiced - 0.85), 4), round(max(0.0, end_voiced - 0.45), 4), round(end_voiced + 0.04, 4)]

    item = {
        "text": clause_txt + punc,
        "era": era,
        "genre": "guwen_loushi_v16",
        "archetype": tag,
        "pz": "".join(t.pz for t in tones),
        "offset": round(offset, 4),
        "ph_seq": " ".join(ph_seq),
        "ph_dur": " ".join(f"{x:.4f}" for x in ph_dur),
        "f0_seq": " ".join(f"{float(x):.2f}" for x in f0_hz),
        "f0_timestep": str(f0_step),
        "spk_name": speaker,
        "x_pure_studio": True,
        **({"x_tail_fade": tail_fade} if tail_fade is not None else {}),
    }
    return item, total_dur


def _compile_loushi_live_ds(
    text: str,
    era: str = "zh",
    key_hz: Optional[float] = None,
    speaker: str = "opencpop_base",
    voice_gender: str = "female",
    tempo_scale: float = 1.0,
) -> List[dict]:
    """
    V16《陋室铭》调式 G2 专用编译器（t16 全篇 81 字实录活腔 × 中古四声平低仄高 × 6+2 跨句连气顿挫 × 悬宫落底）：
      - 彻底纠正旧版「不/在/高」「不/在/深」错移一字的声学误判，1:1 还原入声高挑(+4.0st)、上声深钩与平韵稳落；
      - 自动识别唐调四言铭结句（如「斯是陋室，惟吾德馨。」）的「6+2 跨句连气蓄势 + 0.32s 顿挫气口 + 悬宫落底」真传句法；
      - 泛化新文（如《诫子书》）严格按《广韵》平上去入四声与 3+4 乐句拱弧依字行腔，零词内 SP、零人工 EQ 染色！
    """
    cfg = GUWEN_V14_MODES["guwen_loushi"]
    use_key_hz = float(key_hz) if key_hz is not None else float(
        cfg["male_key_hz"] if voice_gender == "male" else cfg["female_key_hz"]
    )
    min_st, max_st = float(cfg["min_st"]), float(cfg["max_st"])
    speed = float(cfg["speed_scale"]) * float(tempo_scale)
    t_v, st_v = _get_t16_praat_f0()

    raw = _split_clauses_with_punct(text)
    if not raw:
        return []
    plans = plan_guwen_discourse(raw, mode_key="guwen_loushi")
    tone_era = "em" if era.lower() == "em" else "mc"
    S = _LOUSHI_SLICE_BANK
    items: List[dict] = []
    offset = 0.0
    ci = 0

    while ci < len(plans):
        plan = plans[ci]
        # 唐调铭训压轴 4+4 跨句连气顿挫律（1:1 还原 t16 10.93..16.46s「斯是陋室惟吾 —— 德馨。」）
        if (
            plan.tight_next
            and ci + 1 == len(plans) - 1
            and plans[ci + 1].role == "HE_GRAND"
        ):
            p_nx = plans[ci + 1]
            syl_a = [s for s in text_to_sing_syllables(plan.text, era=era) if s.char != " "]
            syl_b = [s for s in text_to_sing_syllables(p_nx.text, era=era) if s.char != " "]
            ton_a = _analyze_clause_tones(plan.text, syl_a, tone_era=tone_era)
            ton_b = _analyze_clause_tones(p_nx.text, syl_b, tone_era=tone_era)
            if len(syl_a) == 4 and len(syl_b) == 4 and ton_b[0].sheng == "ping" and ton_b[1].sheng == "ping":
                sl_a, _, _ = _assign_loushi_phrase_slices(plan, ci, syl_a, ton_a)
                sl_6 = list(sl_a) + [S["c6_wei"], S["c6_wu"]]
                it_6, dur_6 = _build_loushi_item(
                    plan.text + p_nx.text[:2], "，", "转合连气·六字蓄势", era,
                    syl_a + syl_b[:2], ton_a + ton_b[:2], sl_6,
                    t_v, st_v, use_key_hz, min_st, max_st, speed, speaker,
                    offset, lead_sp=0.06, tail_sp=0.20, use_fade=False,
                )
                items.append(it_6)
                offset += dur_6 + 0.12  # 合共 0.38s 庄重顿挫气口!

                last_b = ton_b[-1].sheng
                c_tail = S["c6_de"] if ton_b[2].sheng == "ru" else S["piv_grand_arch"]
                t_tail = S["c6_xin"] if last_b == "ping" else (S["cad_grand_shang"] if last_b == "shang" else S["cad_dn_ru"])
                it_2, dur_2 = _build_loushi_item(
                    p_nx.text[2:], p_nx.punc, "合·顿挫悬宫落底", era,
                    syl_b[2:], ton_b[2:], [c_tail, t_tail],
                    t_v, st_v, use_key_hz, min_st, max_st, speed, speaker,
                    offset, lead_sp=0.06, tail_sp=0.22, use_fade=(last_b in ("ping", "shang")),
                )
                items.append(it_2)
                offset += dur_2 + 0.10
                ci += 2
                continue

        sylls = [s for s in text_to_sing_syllables(plan.text, era=era) if s.char != " "]
        if not sylls:
            ci += 1
            continue
        tones = _analyze_clause_tones(plan.text, sylls, tone_era=tone_era)
        slices, tag, use_fade = _assign_loushi_phrase_slices(plan, ci, sylls, tones)

        if plan.is_passage_end:
            tail_sp, gap_after = 0.22, 0.10
        elif plan.is_major_end:
            tail_sp, gap_after = 0.22, 0.18
        else:
            tail_sp, gap_after = 0.20, 0.15

        it, dur = _build_loushi_item(
            plan.text, plan.punc, tag, era, sylls, tones, slices,
            t_v, st_v, use_key_hz, min_st, max_st, speed, speaker,
            offset, lead_sp=0.06, tail_sp=tail_sp, use_fade=use_fade,
        )
        items.append(it)
        offset += dur + gap_after
        ci += 1

    return items


_T18_PRAAT_CACHE: Optional[Tuple[np.ndarray, np.ndarray]] = None


def _get_t18_praat_f0() -> Tuple[np.ndarray, np.ndarray]:
    """抽取 t18《前赤壁赋》前 44.5s 经 Praat 自相关与声强门限净化的 5ms 真人活腔轨迹 (gong=154.0Hz)。"""
    global _T18_PRAAT_CACHE
    if _T18_PRAAT_CACHE is not None:
        return _T18_PRAAT_CACHE
    import parselmouth
    import soundfile as sf

    y, sr = sf.read(str(DATA_DIR / "bodies/t18_fu_前赤壁赋.wav"), dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    y = y[: int(44.5 * sr)]
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=float(sr))
    pitch = snd.to_pitch_ac(
        time_step=0.005, pitch_floor=86.0, pitch_ceiling=280.0,
        silence_threshold=0.02, voicing_threshold=0.35,
    )
    inten = snd.to_intensity(minimum_pitch=86.0, time_step=0.005)
    t_grid = np.arange(0.0, len(y) / sr, 0.005)
    f0_raw = np.array([pitch.get_value_at_time(float(t)) for t in t_grid], dtype=np.float64)
    db_raw = np.array([inten.get_value(float(t)) for t in t_grid], dtype=np.float64)
    p98 = float(np.nanpercentile(db_raw, 98))
    valid = (~np.isnan(f0_raw)) & (f0_raw > 0) & (~np.isnan(db_raw)) & (db_raw >= p98 - 25.0)
    t_v = t_grid[valid].astype(np.float32)
    st_v = (12.0 * np.log2(f0_raw[valid] / 154.0)).astype(np.float32)
    if len(st_v) >= 7:
        m7 = median_filter(st_v, size=7)
        spike = np.abs(st_v - m7) > 3.2
        st_v[spike] = m7[spike]
    _T18_PRAAT_CACHE = (t_v, st_v)
    return _T18_PRAAT_CACHE


# V16 前赤壁赋 t18 声学法医实测活腔切片库: (f0_s, f0_e, target_dur_s, shift_st)
# 纠正了旧版把「泛舟游于」扬波主峰后移一字、「既望」漏掉双去声连珠劲落、「清风徐来/水波不兴」中间硬插断句的偏差！
_CHIBI_SLICE_BANK = {
    # 1) 开篇 28 字逐字 1:1 实录原型
    "c1_ren":  (0.16, 0.66, 0.52, 0.0),    # 壬(平): -6.6 -> -7.8 -> -7.0st 苍茫低徵起腔
    "c1_xu":   (0.80, 0.96, 0.30, 0.0),    # 戌(入): -5.3st 入声清脆挑起 (+1.8st)
    "c1_zhi":  (1.10, 1.48, 0.44, 0.0),    # 之(平): -4.3 -> -5.3 -> -7.5 -> -6.8st 低回摇曳
    "c1_qiu":  (1.66, 2.08, 0.58, 0.0),    # 秋(平): -6.7 -> -6.5 -> -7.6 -> -6.8st 苍茫收韵
    "c2_qi":   (2.44, 2.62, 0.30, 0.0),    # 七(入): -1.6 -> -0.8st 短入猛拔六度!
    "c2_yue":  (2.66, 2.91, 0.31, 0.0),    # 月(入): -1.4 -> -2.3 -> -1.7st 入声承接
    "c2_ji":   (3.04, 3.32, 0.36, 0.0),    # 既(去): -1.1 -> -4.6st 去声三度半劲落!
    "c2_wang": (3.42, 3.60, 0.38, 0.0),    # 望(去): -0.8 -> -5.3st 双去声连珠高起劲落!
    "c3_su":   (4.38, 4.68, 0.40, 0.0),    # 苏(平): -1.9 -> +0.9 -> +0.2st 沉稳入宫
    "c3_zi":   (4.80, 5.12, 0.39, 0.0),    # 子(上): +0.3st 稳立主宫
    "c3_yu":   (5.18, 5.33, 0.28, 0.0),    # 与(上): -3.5 -> +1.97st 上声深钩仰挑!
    "c3_ke":   (5.44, 5.58, 0.33, 0.0),    # 客(入): 0.0st 干脆入声踏宫
    "c3_fan":  (5.68, 6.00, 0.38, 0.0),    # 泛(去): -0.8 -> +0.2st 蓄势推舟
    "c3_zhou": (6.10, 6.42, 0.39, 0.0),    # 舟(平): +3.4 -> +1.6 -> +2.1st 拔入商音扬波!
    "c3_you":  (6.46, 6.68, 0.35, 0.0),    # 游(平): +2.0 -> +3.0 -> +1.1st 江心泛波主峰!
    "c3_yu2":  (6.72, 7.18, 0.45, 0.0),    # 于(平): -0.2 -> +0.4 -> +0.1st 回托主宫
    "c3_chi":  (7.34, 7.41, 0.35, 0.0),    # 赤(入): +2.5 -> +0.2st 入声清挑!
    "c3_bi":   (7.46, 7.82, 0.38, 0.0),    # 壁(入): +0.3st 稳立主宫
    "c3_zhi2": (7.94, 8.36, 0.45, 0.0),    # 之(平): -3.2 -> -1.4 -> -5.0st 小拱流泻落阶!
    "c3_xia":  (8.50, 8.72, 0.56, 0.0),    # 下(上): -3.3 -> -5.5st 沉落低徵收句!
    "c4_qing": (9.40, 9.74, 0.48, 0.0),    # 清(平): -4.0 -> +0.5 -> 0.0st 清朗滑入主宫
    "c4_feng": (9.88, 10.20, 0.44, 0.0),   # 风(平): -1.2 -> +0.4 -> 0.0st 微澜稳托
    "c4_xu":   (10.36, 10.55, 0.32, 0.0),  # 徐(平): +0.2 -> +0.5 -> -0.4st 轻柔过渡
    "c4_lai":  (10.58, 10.94, 0.44, 0.0),  # 来(平): -0.3 -> +0.5 -> -0.1st 无缝连气入「水」!
    "c5_shui": (11.10, 11.40, 0.44, 0.0),  # 水(上): +3.4 -> +2.2st 上声扬波跃商!
    "c5_bo":   (11.50, 11.92, 0.46, 0.0),  # 波(平): +2.6 -> -0.6st 由商柔落主宫!
    "c5_bu":   (12.02, 12.10, 0.27, 0.0),  # 不(入): +0.1 -> -0.5st 干脆入声踏宫
    "c5_xing": (17.22, 17.90, 0.88, 0.0),  # 兴(平): 0.0~+0.3st 原速主宫澄平悠然长收!
    # 2) 全篇提炼的辞赋「四声 × 平仄 × 三言/四言排比连气对仗」通用活腔构件（与 t18/t19/t10 三版大师实测律 100% 同源！）
    "fu_shang_hook":   (13.06, 13.38, 0.42, 0.0),    # 上声/起首低钩入宫 (-4.0 -> +0.3st: 浩1/衔/举)
    "fu_head_gong":    (40.48, 40.82, 0.42, 0.0),    # 平声低起滑入主宫 (-3.6 -> +0.6 -> 0.0st: t18「横」!)
    "fu_head_qu_low":  (4.38,  4.68,  0.40, 0.0),    # 压轴去声沉蓄回宫 (-1.9 -> +0.5 -> +0.2st: 顺滑衔接无断崖!)
    "fu_head_shang":   (6.10,  6.42,  0.44, 0.0),    # 高商破阵起首 (+3.4 -> +1.8 -> +2.1st: 朝/舟)
    "fu_ping_gong":    (4.80,  5.12,  0.41, 0.0),    # 句中平/叠字稳立主宫 (+0.2st: 浩2)
    "fu_ping_poise":   (4.80,  5.12,  0.50, 0.0),    # 排比上句尾字从容立宫承下 (+0.2st: 阴)
    "fu_ping_shang":   (15.92, 16.14, 0.40, 0.0),    # 句中平声饱满立商 (+2.1st: 无/吞)
    "fu_wave_hi":      (11.10, 11.40, 0.44, 0.0),    # 扬波跃商 (+3.4 -> +2.2st: 远/汤1/象/水)
    "fu_wave_fall":    (11.50, 11.92, 0.48, 0.0),    # 由商柔落主宫 (+2.6 -> -0.3st: 山/长/汤2/波)
    "fu_wave_bright":  (13.96, 14.22, 0.44, +0.4),   # 朗耀由商落宫 (+3.3 -> +0.4st: 晖)
    "fu_qu_crest":     (42.08, 42.30, 0.45, +0.5),   # 去声倒二冲顶劲落 (+4.5 -> +0.5st: 际/万!)
    "fu_ru_tap":       (5.44,  5.58,  0.28, 0.0),    # 入声极脆短点踏宫 (+0.1st: 夕/客)
    "fu_ru_peak":      (42.08, 42.22, 0.29, +0.2),   # 句末/高转入声清挑 (+4.1 -> +1.5st)
    "fu_cad_gong_mid": (4.80,  5.12,  0.54, 0.0),    # 排比下联逗号稳立主宫 (+0.2st: 江)
    "fu_cad_gong_maj": (17.22, 17.64, 0.62, 0.0),    # 排比下联句号浩然立宫 (+0.2 -> 0.0st: 涯)
    "fu_cad_gong_end": (17.22, 17.90, 0.92, 0.0),    # 排比全篇压轴浩然立宫大收 (+0.2 -> 0.0st: 千/兴)
    "fu_cad_up_ze":    (13.52, 13.84, 0.50, +0.5),   # 上联仄声高悬 (+2.7st)
    "fu_cad_dn_major": (19.82, 20.48, 0.82, +1.5),   # 散句/哀咏句号落低徵 (-1.4 -> -5.5st)
    "fu_he_piv_lift":  (13.96, 14.12, 0.42, -0.8),   # 抒情长收倒二欲抑先扬 (+2.1 -> -0.2st)
    "fu_he_grand":     (19.82, 20.60, 1.15, +0.3),   # 抒情叹咏全篇沉底大腔 (-2.5 -> -6.8st: 章)
}


def _assign_chibi_phrase_slices(
    plan: ClausePlan, sylls, tones, prev_end_st: float, in_couplet: bool = False
) -> Tuple[List[Tuple[float, float, float, float]], str, bool]:
    """
    依据【唐调辞赋三版实测律（t18/t19/t10 同源）：排比连气 × 上声低钩 × 去声冲顶劲落 × 入声脆点 × 浩然立宫】
    指派 100% 源自 t18 母带的真人实录活腔切片。
    """
    S = _CHIBI_SLICE_BANK
    n = len(sylls)
    shs = [t.sheng for t in tones]
    role = plan.role
    last_t = shs[-1]

    if n <= 2:
        if n == 1:
            return [S["fu_cad_gong_maj"] if last_t == "ping" else S["fu_cad_dn_major"]], plan.role_desc, False
        return [
            S["fu_head_shang"] if shs[0] == "ping" else S["fu_wave_hi"],
            S["fu_cad_gong_maj"] if last_t == "ping" else S["fu_cad_dn_major"],
        ], plan.role_desc, False

    # 1) 三言短对仗句（如「衔远山，吞长江，」：上句低钩扬商落宫 -> 下句紧承推波立宫！）
    if n == 3:
        if role in ("QI", "CHENG_UP", "ZHUAN_HIGH"):
            h = S["fu_shang_hook"] if shs[0] == "ping" else S["fu_wave_hi"]
            m = S["fu_ru_tap"] if shs[1] == "ru" else (S["fu_wave_hi"] if shs[1] != "ping" else S["fu_ping_shang"])
            t = S["fu_wave_fall"] if last_t == "ping" else S["fu_cad_up_ze"]
            return [h, m, t], f"{plan.role_desc}·三言钩起扬商", False
        h = (13.52, 13.84, 0.35, -0.3) if shs[0] == "ping" else S["fu_qu_crest"]
        m = S["fu_wave_fall"] if shs[1] == "ping" else (S["fu_ru_tap"] if shs[1] == "ru" else S["fu_wave_hi"])
        if role == "HE_GRAND":
            return [h, m, S["fu_cad_gong_end"] if last_t == "ping" else S["fu_cad_dn_major"]], f"{plan.role_desc}·三言立宫收", True
        t = S["fu_cad_gong_maj"] if plan.is_major_end else S["fu_cad_gong_mid"]
        return [h, m, t], f"{plan.role_desc}·三言推波立宫", False

    # 2) 四言标准赋句
    if n == 4:
        # 2a) 苍茫低徵起兴句（如「壬戌之秋，」平入平平 逗号非紧连）
        if role == "QI" and shs == ["ping", "ru", "ping", "ping"] and plan.punc == "，" and not plan.tight_next:
            return [S["c1_ren"], S["c1_xu"], S["c1_zhi"], S["c1_qiu"]], "起·苍茫低徵实录", False
        # 2b) 承低拔起双去劲落句（如「七月既望，」入入去去）
        if role == "CHENG_DOWN" and prev_end_st <= -5.2 and shs[0] == "ru" and shs[2] == "qu" and last_t == "qu":
            return [S["c2_qi"], S["c2_yue"], S["c2_ji"], S["c2_wang"]], "承·短入拔起双去劲落", False
        # 2c) 全平江风句（如「清风徐来，」）
        if role in ("CHENG_UP", "QI") and all(x == "ping" for x in shs):
            return [S["c4_qing"], S["c4_feng"], S["c4_xu"], S["c4_lai"]], "承·江风平腔悬宫", False

        # 2d) 排比转折上句（如「浩浩汤汤，」：叠字上声先低钩蓄势再立宫，把商音扬波留给后二字奔涌！）
        if role == "ZHUAN_HIGH":
            if shs[0] == "shang" and shs[1] == "shang":
                h, p = S["fu_shang_hook"], S["fu_ping_gong"]
                c = S["fu_wave_hi"]
                t = S["fu_wave_fall"] if last_t == "ping" else S["fu_cad_up_ze"]
                return [h, p, c, t], f"{plan.role_desc}·上声低钩商波奔涌", False
            h = S["fu_head_shang"] if shs[0] == "ping" else S["fu_shang_hook"]
            p = S["fu_ru_tap"] if shs[1] == "ru" else (S["fu_wave_hi"] if shs[1] != "ping" else S["fu_wave_bright"])
            c = S["fu_ru_tap"] if shs[2] == "ru" else (S["fu_qu_crest"] if shs[2] == "qu" else S["fu_wave_hi"])
            t = S["fu_wave_fall"] if last_t == "ping" else S["fu_cad_up_ze"]
            return [h, p, c, t], f"{plan.role_desc}·高商扬波悬韵", False

        # 2e) 排比承扬上句（如「朝晖夕阴，」「白露横江，」：高商破阵 -> 朗耀落宫 -> 入声极脆点拍 -> 从容立宫直推下句！）
        if role in ("QI", "CHENG_UP", "LEAD"):
            h = S["fu_head_shang"] if shs[0] == "ping" else (S["fu_ru_tap"] if shs[0] == "ru" else S["fu_shang_hook"])
            p = S["fu_wave_bright"] if shs[1] == "ping" else (S["fu_ru_tap"] if shs[1] == "ru" else S["fu_wave_hi"])
            c = S["fu_ru_tap"] if shs[2] == "ru" else (S["fu_qu_crest"] if shs[2] == "qu" else S["fu_ping_gong"])
            t = S["fu_ping_poise"] if (in_couplet and last_t == "ping") else (S["fu_cad_gong_mid"] if last_t == "ping" else S["fu_cad_up_ze"])
            return [h, p, c, t], f"{plan.role_desc}·高商破阵脆入承宫", False

        # 2f) 排比承抑下句（如「横无际涯。」：平声低起滑入主宫 -> 阔展立商 -> 去声倒二冲顶劲落(+4.5↘+0.5st) -> 浩然立宫！）
        if role == "CHENG_DOWN":
            h = S["fu_head_gong"] if shs[0] == "ping" else S["fu_head_qu_low"]
            p = S["fu_ping_shang"] if shs[1] == "ping" else (S["fu_ru_tap"] if shs[1] == "ru" else S["fu_wave_hi"])
            c = S["fu_qu_crest"] if shs[2] == "qu" else (S["fu_ru_tap"] if shs[2] == "ru" else S["fu_wave_hi"])
            if last_t == "ping":
                t = S["fu_cad_gong_maj"] if plan.is_major_end else S["fu_cad_gong_mid"]
            else:
                t = S["fu_cad_dn_major"]
            return [h, p, c, t], f"{plan.role_desc}·去声冲顶浩然立宫", False

        # 2g) 全篇压轴四言合句（如「水波不兴。」/「气象万千。」）
        if role == "HE_GRAND":
            if abs(prev_end_st) <= 0.6 and shs == ["shang", "ping", "ru", "ping"]:
                return [S["c5_shui"], S["c5_bo"], S["c5_bu"], S["c5_xing"]], "合·扬波澄平归宫", True
            # 排比宏阔四言收束（如「气象万千。」）：首字沉蓄回宫 -> 次字拔高立商 -> 倒二去声全篇极峰冲顶劲落 -> 末字浩然立宫长收！
            if in_couplet and last_t == "ping":
                h = S["fu_head_qu_low"] if shs[0] != "ping" else S["fu_head_gong"]
                p = S["fu_wave_hi"] if shs[1] != "ping" else S["fu_ping_shang"]
                c = S["fu_qu_crest"] if shs[2] == "qu" else (S["fu_ru_tap"] if shs[2] == "ru" else S["fu_wave_hi"])
                return [h, p, c, S["fu_cad_gong_end"]], "合·沉蓄冲顶浩然立宫", True
            h = S["fu_head_qu_low"] if shs[0] != "ping" else S["fu_head_gong"]
            p = S["fu_ping_gong"]
            c = S["fu_he_piv_lift"]
            t = S["fu_he_grand"] if last_t == "ping" else S["fu_cad_dn_major"]
            return [h, p, c, t], "合·欲抑先扬苍茫落底", (last_t == "ping")

    if n == 12 and shs[:4] == ["ping", "shang", "shang", "ru"] and shs[-4:] == ["ru", "ru", "ping", "shang"]:
        return [
            S["c3_su"], S["c3_zi"], S["c3_yu"], S["c3_ke"],
            S["c3_fan"], S["c3_zhou"], S["c3_you"], S["c3_yu2"],
            S["c3_chi"], S["c3_bi"], S["c3_zhi2"], S["c3_xia"],
        ], "铺·t18十二字江上泛舟实录", False

    out = []
    peak_i = max(1, n // 2)
    for k in range(n - 2):
        sh = shs[k]
        if k == 0:
            out.append(S["c3_su"] if sh == "ping" else S["fu_shang_hook"])
        elif k in (peak_i - 1, peak_i):
            out.append(S["c3_zhou"] if k == peak_i - 1 else S["c3_you"])
        elif sylls[k].char in _MID_PARTICLES:
            out.append(S["c3_yu"] if sh == "shang" else S["c3_yu2"])
        elif sh == "ru":
            out.append(S["c3_chi"] if k >= peak_i else S["c3_ke"])
        elif sh != "ping":
            out.append(S["c3_fan"])
        else:
            out.append(S["c3_zi"])
    if role == "HE_GRAND":
        out.append(S["fu_he_piv_lift"])
        out.append(S["fu_he_grand"] if last_t == "ping" else S["c3_xia"])
        return out, f"合·长赋压轴落底{n}字", (last_t == "ping")
    if plan.is_major_end or role == "CHENG_DOWN":
        out.append(S["c3_zhi2"] if sylls[n - 2].char in _MID_PARTICLES else S["fu_qu_crest"])
        out.append(S["c3_xia"] if last_t != "ping" else S["fu_cad_dn_major"])
        return out, f"铺·长赋扬波落徵{n}字", False
    out.append(S["fu_ping_shang"] if shs[n - 2] == "ping" else S["fu_ru_peak"])
    out.append(S["fu_cad_gong_mid"])
    return out, f"铺·长赋扬波悬宫{n}字", False


def _build_chibi_item(
    clause_txt: str,
    punc: str,
    tag: str,
    era: str,
    sylls,
    tones,
    slices: List[Tuple[float, float, float, float]],
    t_v: np.ndarray,
    st_v: np.ndarray,
    use_key_hz: float,
    min_st: float,
    max_st: float,
    speed: float,
    speaker: str,
    voice_gender: str,
    offset: float,
    lead_sp: float,
    tail_sp: float,
    use_fade: bool,
) -> Tuple[dict, float, float]:
    f0_step = 0.005
    ph_seq: List[str] = ["SP"]
    ph_dur: List[float] = [lead_sp]
    f0_chunks: List[np.ndarray] = []
    dips: List[Tuple[float, float, float]] = []
    prev_nuc_d = 0.0

    for k, (sy, (f0_s, f0_e, t_dur, sh_st)) in enumerate(zip(sylls, slices)):
        cdur = max(0.25, t_dur * speed)
        pairs, _, has_m = _build_char_phones(sy)
        first_mp = pairs[0][1]
        is_cons = (pairs[0][0] not in VOWEL_SET and pairs[0][0] not in CODA_SET)
        if is_cons:
            # 实证声学定律：
            #   1) 送气塞音/塞擦音 (k/p/t, ch/c/q) 与清擦音 (sh/x/s/f) 给足 112~122ms 零微谷无声送气/摩擦，杜绝客->何、赤->日；
            #   2) 不送气舌面塞擦 j 与不送气塞音/鼻音 (b/d/g, n/m) 在元音后给足 100~118ms 成阻并加 Hanning 基频微谷，杜绝际->衣！
            if first_mp in ("sh", "x", "s", "f"):
                actual_on = 0.120 if cdur >= 0.36 else 0.104
            elif first_mp == "ch":
                actual_on = 0.122
            elif first_mp in ("k", "p", "t"):
                actual_on = 0.118
            elif first_mp == "j":
                actual_on = 0.118
            elif first_mp == "z":
                actual_on = 0.096
            elif first_mp == "zh":
                actual_on = 0.094
            elif first_mp in ("c", "q"):
                actual_on = 0.104
            elif first_mp == "h":
                actual_on = 0.092
            elif first_mp in ("b", "d", "g"):
                actual_on = 0.082
            elif first_mp in ("n", "m"):
                actual_on = 0.082 if (k > 0 and prev_nuc_d >= 0.44) else 0.064
            elif first_mp in ("l", "r"):
                actual_on = 0.064
            elif first_mp in ("y", "w"):
                actual_on = 0.052
            else:
                actual_on = 0.072
            cap_r = 0.37 if first_mp in ("ch", "c", "q", "k", "p", "t", "j", "sh", "x", "s", "f", "h", "zh", "z") else 0.30
            actual_on = min(actual_on, max(0.044, cdur - 0.15), cdur * cap_r)
        else:
            actual_on = 0.0

        m_d = min(0.048, cdur * 0.14) if has_m else 0.0
        actual_nuc = max(0.15, cdur - actual_on - m_d)

        dip_amp = 0.0
        if is_cons and k > 0:
            if first_mp == "j" and prev_nuc_d >= 0.30:
                dip_amp = 1.65  # 不送气舌面塞擦成阻微谷，保「无[际]涯」干脆爆发不滑为 yi!
            elif prev_nuc_d >= 0.44:
                if first_mp in _NASAL_ONSETS:
                    dip_amp = 1.25
                elif first_mp in ("b", "d"):
                    dip_amp = 1.25
                elif first_mp == "g":
                    dip_amp = 0.60
        prev_nuc_d = actual_nuc

        n_on = int(round(actual_on / f0_step)) if actual_on > 0 else 0
        n_voiced = max(6, int(round((actual_nuc + m_d) / f0_step)))
        sl_nuc = np.interp(np.linspace(f0_s, f0_e, n_voiced, dtype=np.float32), t_v, st_v).astype(np.float32) + float(sh_st)

        # 女声低音实声胸腔软膝：将 < -4.4st 收敛 45%（-7.1st -> -5.9st = 178Hz @ 250Hz），离开气声底区且完整保留微腔起伏！
        if voice_gender == "female":
            low_m = sl_nuc < -4.4
            if np.any(low_m):
                sl_nuc[low_m] = -4.4 + 0.55 * (sl_nuc[low_m] + 4.4)

        if n_on > 0:
            if first_mp in _VOICED_ONSETS and k > 0 and f0_chunks:
                sl_on = np.linspace(float(f0_chunks[-1][-1]), float(sl_nuc[0]), n_on, dtype=np.float32)
            else:
                sl_on = np.full(n_on, float(sl_nuc[0]), dtype=np.float32)
            sl_full = np.concatenate([sl_on, sl_nuc])
        else:
            sl_full = sl_nuc

        char_t0 = sum(ph_dur)
        if dip_amp > 0.0 and actual_on > 0.0:
            dips.append((char_t0, char_t0 + actual_on, dip_amp))

        for rp, mp in pairs:
            if rp == "coda_m":
                d = round(m_d, 4)
            elif rp in VOWEL_SET:
                d = round(actual_nuc, 4)
            else:
                d = round(actual_on if actual_on > 0 else 0.048, 4)
            ph_seq.append(mp)
            ph_dur.append(d)
        f0_chunks.append(sl_full)

    ph_seq.append("SP")
    ph_dur.append(round(tail_sp, 4))

    total_dur = sum(ph_dur)
    n_tot = max(12, int(round(total_dur / f0_step)))
    f0_st = np.zeros(n_tot, dtype=np.float32)
    first_v = float(f0_chunks[0][0]) if f0_chunks else 0.0
    cur_i = int(round(lead_sp / f0_step))
    f0_st[:cur_i] = first_v
    last_v = first_v
    for arr in f0_chunks:
        nx = min(n_tot, cur_i + len(arr))
        if nx > cur_i:
            f0_st[cur_i:nx] = arr[: nx - cur_i]
            last_v = float(f0_st[nx - 1])
        cur_i = nx
    if cur_i < n_tot:
        f0_st[cur_i:] = last_v

    f0_st = gaussian_filter1d(f0_st, sigma=1.4).astype(np.float32)
    for d_t0, d_t1, d_amp in dips:
        i0, i1 = int(round(d_t0 / f0_step)), min(n_tot, int(round(d_t1 / f0_step)))
        if i1 > i0 + 2:
            f0_st[i0:i1] -= d_amp * np.hanning(i1 - i0).astype(np.float32)

    f0_st = np.clip(f0_st, min_st, max_st)
    end_st = float(np.median(f0_chunks[-1][-min(12, len(f0_chunks[-1])):]) if f0_chunks else 0.0)
    f0_hz = use_key_hz * (2.0 ** (f0_st / 12.0))

    tail_fade = None
    if use_fade:
        end_voiced = offset + sum(ph_dur[:-1])
        tail_fade = [round(max(0.0, end_voiced - 0.72), 4), round(max(0.0, end_voiced - 0.36), 4), round(end_voiced + 0.04, 4)]

    item = {
        "text": clause_txt + punc,
        "era": era,
        "genre": "fu_chibi_v16",
        "archetype": tag,
        "pz": "".join(t.pz for t in tones),
        "offset": round(offset, 4),
        "ph_seq": " ".join(ph_seq),
        "ph_dur": " ".join(f"{x:.4f}" for x in ph_dur),
        "f0_seq": " ".join(f"{float(x):.2f}" for x in f0_hz),
        "f0_timestep": str(f0_step),
        "spk_name": speaker,
        "x_pure_studio": True,
        **({"x_tail_fade": tail_fade} if tail_fade is not None else {}),
    }
    return item, total_dur, end_st


def _compile_chibi_live_ds(
    text: str,
    era: str = "zh",
    key_hz: Optional[float] = None,
    speaker: str = "opencpop_base",
    voice_gender: str = "female",
    tempo_scale: float = 1.0,
) -> List[dict]:
    """
    V16《前赤壁赋》调式 G3 专用编译器（t18 实录活腔 × 中古四声平低仄高 × 3+3/4+4/长句对仗 × 江风八字连气归宫）：
      - 逐字对齐 t18 开篇 28 字：苍茫低徵起兴(壬戌之秋) -> 短入拔起双去劲落(七月既望) -> 十二字泛舟扬波落徵(苏子..之下)
        -> 八字无停顿连气澄平归宫(清风徐来水波不兴)；
      - 跨篇泛化（如《岳阳楼记》）严守唐调辞赋「三言上扬下沉 · 四言叠字递升 · 入声短促挑角 · 压轴欲抑先扬沉底」四声律，零词内 SP、零人工 EQ！
    """
    cfg = GUWEN_V14_MODES["fu_chibi"]
    use_key_hz = float(key_hz) if key_hz is not None else float(
        cfg["male_key_hz"] if voice_gender == "male" else cfg["female_key_hz"]
    )
    min_st = -6.4 if voice_gender == "female" else float(cfg["min_st"])
    max_st = float(cfg["max_st"])
    speed = float(cfg["speed_scale"]) * float(tempo_scale)
    t_v, st_v = _get_t18_praat_f0()

    raw = _split_clauses_with_punct(text)
    if not raw:
        return []
    plans = plan_guwen_discourse(raw, mode_key="fu_chibi")
    tone_era = "em" if era.lower() == "em" else "mc"
    S = _CHIBI_SLICE_BANK
    items: List[dict] = []
    offset = 0.0
    prev_end_st = 0.0
    ci = 0

    while ci < len(plans):
        plan = plans[ci]
        # 唐调辞赋排比偶句连气滚诵律（t18「清风徐来水波不兴」× t19/t10「衔远山吞长江 / 浩浩汤汤横无际涯 / 朝晖夕阴气象万千」）：
        # 成对 3+3 或 4+4 排比偶句（排除低徵慢起兴如平入平平「壬戌之秋」）合为一气无断诵出，句群之间再自然换气！
        if ci + 1 < len(plans) and not plan.is_major_end:
            p_nx = plans[ci + 1]
            syl_a = [s for s in text_to_sing_syllables(plan.text, era=era) if s.char != " "]
            syl_b = [s for s in text_to_sing_syllables(p_nx.text, era=era) if s.char != " "]
            ton_a = _analyze_clause_tones(plan.text, syl_a, tone_era=tone_era)
            ton_b = _analyze_clause_tones(p_nx.text, syl_b, tone_era=tone_era)
            is_low_qi = (plan.role == "QI" and len(syl_a) == 4 and [t.sheng for t in ton_a] == ["ping", "ru", "ping", "ping"])
            is_parallel_pair = (
                len(syl_a) == len(syl_b)
                and len(syl_a) in (3, 4)
                and plan.role in ("QI", "CHENG_UP", "ZHUAN_HIGH")
                and p_nx.role in ("CHENG_DOWN", "HE_GRAND")
                and not is_low_qi
            )
            if is_parallel_pair:
                # 1) 全平江风连气原版特例（「清风徐来水波不兴。」）
                if len(syl_a) == 4 and all(t.sheng == "ping" for t in ton_a) and [t.sheng for t in ton_b] == ["shang", "ping", "ru", "ping"]:
                    sl_a = [S["c4_qing"], S["c4_feng"], S["c4_xu"], S["c4_lai"]]
                    sl_b = [S["c5_shui"], S["c5_bo"], S["c5_bu"], S["c5_xing"]]
                    c_tag, c_fade = "承合连气·江风扬波归宫", True
                else:
                    sl_a, tag_a, _ = _assign_chibi_phrase_slices(plan, syl_a, ton_a, prev_end_st, in_couplet=True)
                    sl_b, tag_b, c_fade = _assign_chibi_phrase_slices(p_nx, syl_b, ton_b, 0.0, in_couplet=True)
                    c_tag = f"{len(syl_a)}+{len(syl_b)}连气·{tag_b.split('·')[-1]}"
                if p_nx.is_passage_end:
                    tail_sp, gap_after = 0.22, 0.10
                elif p_nx.is_major_end:
                    tail_sp, gap_after = 0.20, 0.22  # 排比大句群间 0.48s 宏阔换气
                else:
                    tail_sp, gap_after = 0.16, 0.12  # 排比偶句间 0.34s 紧凑气口
                it_pair, dur_pair, prev_end_st = _build_chibi_item(
                    plan.text + p_nx.text, p_nx.punc, c_tag, era,
                    syl_a + syl_b, ton_a + ton_b, list(sl_a) + list(sl_b),
                    t_v, st_v, use_key_hz, min_st, max_st, speed, speaker, voice_gender,
                    offset, lead_sp=0.06, tail_sp=tail_sp, use_fade=c_fade,
                )
                items.append(it_pair)
                offset += dur_pair + gap_after
                ci += 2
                continue

        sylls = [s for s in text_to_sing_syllables(plan.text, era=era) if s.char != " "]
        if not sylls:
            ci += 1
            continue
        tones = _analyze_clause_tones(plan.text, sylls, tone_era=tone_era)
        slices, tag, use_fade = _assign_chibi_phrase_slices(plan, sylls, tones, prev_end_st)

        next_is_long = (ci + 1 < len(plans) and len(plans[ci + 1].text) >= 8)
        if plan.is_passage_end:
            tail_sp, gap_after = 0.22, 0.10
        elif next_is_long or plan.is_major_end:
            tail_sp, gap_after = 0.22, 0.28
        else:
            tail_sp, gap_after = 0.16, 0.04

        it, dur, prev_end_st = _build_chibi_item(
            plan.text, plan.punc, tag, era, sylls, tones, slices,
            t_v, st_v, use_key_hz, min_st, max_st, speed, speaker, voice_gender,
            offset, lead_sp=0.06, tail_sp=tail_sp, use_fade=use_fade,
        )
        items.append(it)
        offset += dur + gap_after
        ci += 1

    return items


def compile_guwen_v14_ds(
    text: str,
    mode_key: str = "guwen_lanting",
    era: str = "zh",
    key_hz: Optional[float] = None,
    speaker: str = "opencpop_base",
    voice_gender: str = "female",
    tempo_scale: float = 1.0,
) -> List[dict]:
    if mode_key == "guwen_lanting":
        return _compile_lanting_live_ds(
            text=text, era=era, key_hz=key_hz, speaker=speaker,
            voice_gender=voice_gender, tempo_scale=tempo_scale,
        )
    if mode_key == "guwen_loushi":
        return _compile_loushi_live_ds(
            text=text, era=era, key_hz=key_hz, speaker=speaker,
            voice_gender=voice_gender, tempo_scale=tempo_scale,
        )
    if mode_key == "fu_chibi":
        return _compile_chibi_live_ds(
            text=text, era=era, key_hz=key_hz, speaker=speaker,
            voice_gender=voice_gender, tempo_scale=tempo_scale,
        )
    cfg = GUWEN_V14_MODES[mode_key]
    mp = MODE_PARAMS[mode_key]
    use_key_hz = float(key_hz) if key_hz is not None else float(cfg["male_key_hz"] if voice_gender == "male" else cfg["female_key_hz"])
    min_st, max_st = float(cfg["min_st"]), float(cfg["max_st"])
    speed = float(cfg["speed_scale"]) * float(tempo_scale)

    raw = _split_clauses_with_punct(text)
    if not raw:
        return []
    plans = plan_guwen_discourse(raw, mode_key=mode_key)
    tone_era = "em" if era.lower() == "em" else "mc"
    resid = _micro_residual(mode_key)
    r_ptr = 0
    f0_step = 0.005
    items: List[dict] = []
    offset = 0.0
    prev_end = 0.0

    for plan in plans:
        sylls = [s for s in text_to_sing_syllables(plan.text, era=era) if s.char != " "]
        n = len(sylls)
        if n == 0:
            continue
        tones = _analyze_clause_tones(plan.text, sylls, tone_era=tone_era)
        slots, tag = _assign_slots(mode_key, plan, sylls, tones, prev_end)
        has_final_particle = (sylls[-1].char in _FINAL_PARTICLES)
        last_sheng = tones[-1].sheng

        lead = 0.06
        ph_seq, ph_dur = ["SP"], [lead]
        spans = []
        dips = []
        prev_nuc = 0.0
        for k, (sy, tp, sl) in enumerate(zip(sylls, tones, slots)):
            cdur = max(0.25, sl[1] * speed)
            pairs, _, has_m = _build_char_phones(sy)
            first = pairs[0][1]
            is_cons = pairs[0][0] not in VOWEL_SET and pairs[0][0] not in CODA_SET
            is_fric = 1 if (is_cons and first in _UNVOICED_FRIC_ONSETS) else 0
            dip = None
            if is_cons:
                # V15 干脆利落的吟诵声母时长（清擦/塞擦给足 86~95ms 保清晰，塞/鼻声母紧凑有力不拖沓！）
                if first in _FRICATIVE_ONSETS:
                    on = 0.095 if cdur >= 0.50 else 0.086
                    if first in _AFFRIC_ONSETS and prev_nuc >= 0.30:
                        dip = (1.25, 0.55)  # 塞擦前半成阻微谷，确保「赤/清」爆发清晰不滑成 sh/x
                elif first in _NASAL_ONSETS and prev_nuc >= 0.50:
                    on, dip = 0.092, (1.35, 1.0)
                elif first in _STOP_ONSETS and prev_nuc >= 0.50:
                    on, dip = 0.088, (1.50, 1.0)
                elif first in ("p", "t", "k"):
                    on = 0.072
                elif first in _STOP_ONSETS:
                    on = 0.056  # 紧实塞音爆发，绝不软绵绵！
                else:
                    on = 0.060
                cap_ratio = 0.30 if first in _FRICATIVE_ONSETS else 0.26
                on = min(on, max(0.044, cdur - 0.16), cdur * cap_ratio)
            else:
                on = 0.0
            m_d = min(0.045, cdur * 0.14) if has_m else 0.0
            nuc = max(0.16, cdur - on - m_d)
            prev_nuc = nuc
            a = sum(ph_dur)
            for rp, mpp in pairs:
                d = m_d if rp == "coda_m" else (nuc if rp in VOWEL_SET else (on if on > 0 else 0.048))
                ph_seq.append(mpp)
                ph_dur.append(round(d, 4))
            b = sum(ph_dur)
            if dip is not None:
                d_amp, d_frac = dip
                dips.append((a, a + on * d_frac, d_amp))
            nasal = pairs[-1][1] in _NASAL_FINALS or has_m
            spans.append((a, b, sl, nasal, on, is_fric))

        # 句间静音保持足够 tail_sp（>=0.16s），杜绝尾音渗漏进间隙制造 -47dB 杂音！
        if plan.tight_next:
            tail_sp, gap_after = 0.08, 0.02
        else:
            total_gap = mp["gap_major"] if plan.is_major_end else mp["gap"]
            tail_sp = 0.18 if plan.is_major_end else 0.16
            gap_after = max(0.04, total_gap - tail_sp - lead)

        ph_seq.append("SP")
        ph_dur.append(tail_sp)

        tot = sum(ph_dur)
        nf = int(round(tot / f0_step)) + 1
        tt = np.arange(nf) * f0_step
        pts_t, pts_v = [], []
        real_mask = np.zeros(nf, bool)
        real_vals = np.zeros(nf, np.float32)

        use_real_tail = (
            plan.is_passage_end
            and (
                (mode_key == "guwen_lanting" and has_final_particle)
                or (mode_key == "guwen_loushi" and last_sheng == "ping" and n <= 5)
            )
        )

        for k, (a, b, sl, nasal, on, _) in enumerate(spans):
            tgt, _, beg, end, _ = sl
            d = b - a
            v0 = a + on * 0.50
            # V15 利落起音滑入 (65~85ms, 替代拖沓 180ms 慢爬坡)，字头立住掷地有声！
            if beg is not None:
                g = min(0.085, 0.24 * d)
                pts_t += [a, v0, v0 + g]
                pts_v += [beg, beg, tgt]
            else:
                pts_t += [a, v0]
                pts_v += [tgt, tgt]
            if end is not None:
                pts_t += [b - 0.36 * d, b]
                pts_v += [tgt, end]
            else:
                pts_t += [b]
                pts_v += [tgt]

            if use_real_tail and k == n - 1 and mode_key in REAL_TAILS:
                gs, gv = _master_track(mode_key)
                r0, r1 = REAL_TAILS[mode_key]
                m = (gs >= r0) & (gs <= r1)
                src = gv[m]
                ok = ~np.isnan(src)
                if ok.sum() > 20:
                    src = np.interp(np.arange(len(src)), np.where(ok)[0], src[ok])
                    i0, i1 = int(round(v0 / f0_step)), min(nf, int(round(b / f0_step)))
                    L = i1 - i0
                    if L > 10:
                        real_vals[i0:i1] = np.interp(np.linspace(0, len(src) - 1, L), np.arange(len(src)), src)
                        real_mask[i0:i1] = True

        f0 = np.interp(tt, pts_t, pts_v).astype(np.float32)
        f0[: int(round(lead / f0_step))] = pts_v[0]
        # V15 紧致 14ms 高斯平滑 (sigma=2.8)：既顺滑无按键台阶，又保住每个字的干脆顿挫！
        f0 = gaussian_filter1d(f0, sigma=2.8).astype(np.float32)

        # 叠加原速真实人类微颤残差（绝不拉伸变形！）
        if r_ptr + nf > len(resid):
            r_ptr = 0
        mic = resid[r_ptr: r_ptr + nf] if len(resid) >= nf else np.resize(resid, nf)
        r_ptr += nf
        f0 = f0 + 0.85 * mic

        # 替换匹配的全篇招牌真声落腔（如「也————」「馨————」）
        if real_mask.any():
            idx = np.where(real_mask)[0]
            w = np.ones(len(idx), np.float32)
            ramp = min(12, len(idx) // 4)
            w[:ramp] = np.linspace(0, 1, ramp)
            f0[idx] = (1.0 - w) * f0[idx] + w * real_vals[idx]

        # 平滑后仅在声母闭塞帧加物理跨声门压微谷
        for d0, d1, amp in dips:
            j0, j1 = int(round(d0 / f0_step)), min(nf, int(round(d1 / f0_step)))
            if j1 > j0 + 2:
                f0[j0:j1] -= amp * np.hanning(j1 - j0).astype(np.float32)

        f0 = np.clip(f0, min_st, max_st)
        prev_end = float(np.median(f0[int(round(spans[-1][0] / f0_step)): int(round(spans[-1][1] / f0_step))]))
        hz = use_key_hz * 2.0 ** (f0 / 12.0)

        # V15 每个字的「头重尾轻」中气发声包络（对齐原版 t09/t16 吐字立住 + 句尾自然叹咏渐收，零压缩！）
        vital = []
        for k, (a, b, sl, nasal, on, is_fric) in enumerate(spans):
            base_g = float(sl[4])
            d = b - a
            is_last = (k == n - 1)
            if is_last:
                gh = base_g + 1.8
                gb = base_g - 0.3
                gt = mp["grand_end_db"] if plan.is_passage_end else (mp["tail_end_db"] if not plan.tight_next else -4.0)
            elif d < 0.39:
                # 流畅连字：立住字头但轻收字尾 (-2.0dB)，保证长句行云流水无机械点跳
                gh = base_g + 2.0
                gb = base_g + 0.0
                gt = base_g - 2.0
            else:
                # 中长吟诵字：字头挺拔立住 (+2.2dB) → 字腹饱满 (0dB) → 字尾干脆收气 (-3.2dB)
                gh = base_g + 2.2
                gb = base_g + 0.0
                gt = base_g - 3.2
            vital.append([offset + a, offset + a + on, offset + b, gh, gb, gt, is_fric])

        items.append({
            "text": plan.text + plan.punc,
            "era": era,
            "genre": f"{mode_key}_v15",
            "archetype": tag,
            "pz": "".join(t.pz for t in tones),
            "offset": round(offset, 4),
            "ph_seq": " ".join(ph_seq),
            "ph_dur": " ".join(f"{x:.4f}" for x in ph_dur),
            "f0_seq": " ".join(f"{float(x):.2f}" for x in hz),
            "f0_timestep": str(f0_step),
            "spk_name": speaker,
            "x_vital": [[round(x, 4) for x in v] for v in vital],
        })
        offset += tot + gap_after
    return items


def apply_chant_vitality_dsp(y: np.ndarray, sr: int, items: List[dict], female: bool = False) -> np.ndarray:
    """
    后处理管线：
      - 若为 V16 纯净棚录模式 (`x_pure_studio=True`，如兰亭调式 G1)：严格遵循《诗经》V9.6 零 EQ 染色、零逐字抽吸铁律，
        仅对压轴虚词大摇腔「也」末尾两波施加极柔和的自然气息渐收 (-1.5dB -> -9.5dB)；
      - 其它调式保留既有 V15 处理，互不干扰。
    """
    n = len(y)
    if items and all(it.get("x_pure_studio") for it in items):
        env_db = np.zeros(n, np.float32)
        has_fade = False
        for it in items:
            tf = it.get("x_tail_fade")
            if tf and len(tf) == 3:
                has_fade = True
                ta, tb, tc = tf
                ia = max(0, min(n, int(ta * sr)))
                ib = max(ia, min(n, int(tb * sr)))
                ic = max(ib, min(n, int(tc * sr)))
                if ib > ia:
                    env_db[ia:ib] = np.linspace(0.0, -1.5, ib - ia, dtype=np.float32)
                if ic > ib:
                    env_db[ib:ic] = np.linspace(-1.5, -9.5, ic - ib, dtype=np.float32)
                    env_db[ic:] = -9.5
        if not has_fade:
            return y.astype(np.float32)
        return (y * (10.0 ** (env_db / 20.0))).astype(np.float32)

    env_db = np.zeros(n, np.float32)
    has_vital = False

    for it in items:
        for seg in it.get("x_vital", []):
            has_vital = True
            t0, tv, t1, gh, gb, gt = seg[:6]
            is_fric = int(seg[6]) if len(seg) > 6 else 0
            i0 = max(0, int(t0 * sr))
            iv = max(i0 + 1, min(n, int(tv * sr)))
            i1 = max(iv + 1, min(n, int(t1 * sr)))
            if i1 <= i0 + 10:
                continue
            L_v = i1 - iv
            p_atk = iv + max(1, int(L_v * 0.26))
            p_rel = iv + max(2, int(L_v * 0.72))
            if is_fric:
                env_db[i0:iv] = 0.0
                env_db[iv:p_atk] = gh
            else:
                env_db[i0:iv] = gh * 0.65
                env_db[iv:p_atk] = gh
            if p_rel > p_atk:
                env_db[p_atk:p_rel] = np.linspace(gh, gb, p_rel - p_atk, dtype=np.float32)
            if i1 > p_rel:
                env_db[p_rel:i1] = np.linspace(gb, gt, i1 - p_rel, dtype=np.float32)
        if not has_vital:
            for t0, t1, g0, g1 in it.get("x_gain", []):
                i0, i1 = int(t0 * sr), min(n, int(t1 * sr))
                if i1 > i0:
                    env_db[i0:i1] = np.linspace(g0, g1, i1 - i0, dtype=np.float32)

    win = max(1, int((0.016 if has_vital else 0.045) * sr))
    k = np.hanning(win).astype(np.float32)
    k /= np.sum(k)
    env_db = np.convolve(env_db, k, mode="same")
    y_out = (y * (10.0 ** (env_db / 20.0))).astype(np.float64)

    # 线性相位温和人声中气均衡（消除流行录音气声单薄感，恢复书斋朗诵实声穿透力）
    n_fft = 1 << int(np.ceil(np.log2(max(n, 4096))))
    Y = np.fft.rfft(y_out, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    eq_db = np.zeros_like(freqs)
    f_ch = 300.0 if female else 235.0
    eq_db += (2.4 if female else 1.5) * np.exp(-0.5 * ((np.log2(np.maximum(freqs, 20.0) / f_ch)) / 0.55) ** 2)
    f_pr = 1850.0 if female else 1600.0
    eq_db += (3.2 if female else 3.0) * np.exp(-0.5 * ((np.log2(np.maximum(freqs, 20.0) / f_pr)) / 0.65) ** 2)
    shelf = 1.0 / (1.0 + (5600.0 / np.maximum(freqs, 1.0)) ** 3.5)
    eq_db -= (4.5 if female else 3.5) * shelf
    eq_db[freqs < 65.0] -= 18.0 * (1.0 - freqs[freqs < 65.0] / 65.0)
    Y *= 10.0 ** (eq_db / 20.0)
    return np.fft.irfft(Y, n=n_fft)[:n].astype(np.float32)


def apply_gain_plan(y: np.ndarray, sr: int, items: List[dict], female: bool = False) -> np.ndarray:
    """兼容旧入口，直接委派至 V15 挺拔中气后处理管线。"""
    return apply_chant_vitality_dsp(y, sr, items, female=female)
