#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汉火古音唱吟 · V6 格律辨体 × 真人唐调四式吟诵引擎 (tang_melody_prior.py)

核心格律吟诵范式（先辨平仄格式 → 再逐句选对应真人唐调调式 → 逐字依平上去入弹性微调）：
  1. 自动辨体 (`analyze_poem_meter`)：
     按中古四声自动识别五绝/七绝/五律/七律（仄起式/平起式、首句入韵/不入韵），严格遵循
     「一三五不论，二四六分明」：
       - 五言四大标准句式：
           W1: (仄)仄平平仄  → 匹配真人唐调 t17[0]（绿蚁新醅酒）
           W2: (平)平仄仄平  → 匹配真人唐调 t17[1]（红泥小火炉）
           W3: (平)平平仄仄  → 匹配真人唐调 t17[2]（晚来天欲雪）
           W4: (仄)仄仄平平  → 匹配真人唐调 t17[3]（能饮一杯无）
       - 七言四大标准句式（七言即五言加反向二字头！）：
           Q1: 平平 + W1 (平平仄仄平平仄)
           Q2: 仄仄 + W2 (仄仄平平仄仄平)
           Q3: 仄仄 + W3 (仄仄平平平仄仄)
           Q4: 平平 + W4 (平平仄仄仄平平)
  2. 真人四式原型直传 + 平上去入弹性变形（零公式机械感 + 入声短促不断裂）：
     - 直接以真人名家 `t17` 的真实人类 F0 轨迹与词内连腔骨架为底，彻底消除纯公式台阶机械感；
     - 若新诗某字与原型同位字声调不同（如原型为长上声，新诗为入声如「白日/月/泊/一」），
       自动弹性微调该字时长（入声保持 0.25~0.30s 利落短促且严禁在词内硬切 SP 导致断裂！）
       并对真人 F0 分段做平滑时间规整（Piecewise Time-Warp），完整保留真人活人气息与行腔！
"""
from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

SING_DIR = Path(__file__).resolve().parent
DATA_DIR = SING_DIR / "data_tang_processed"

if str(SING_DIR) not in sys.path:
    sys.path.insert(0, str(SING_DIR))

from hanfire_sing_phonemes import CODA_SET, VOWEL_SET, text_to_sing_syllables

_OPENCPOP_NORM = {
    "iou": "iu",
    "uei": "ui",
    "h_b": "b", "h_d": "d", "h_g": "g",
    "h_dz": "z", "h_dzh": "zh", "h_dj": "j",
    "h_z": "z", "h_zh": "zh", "h_zj": "x",
    "h_ng": "n", "h_ny": "n", "h_v": "w",
    "h_hm": "m", "h_hn": "n", "h_hl": "l", "h_hng": "h", "h_gh": "h",
    "coda_m": "m",
    "coda_p": "",
    "coda_t": "",
    "coda_k": "",
}

# OpenCpop 零声母齐齿/撮口/合口必须显式拆为 y/w + 韵核（严禁裸发 ia/ve/i/uei 导致读成“阿/厄”！）
_ZERO_ONSET_YW_SPLIT = {
    "i":    ("y", "i"),    "in":   ("y", "in"),   "ing":  ("y", "ing"),
    "ia":   ("y", "a"),    "ie":   ("y", "e"),    "iao":  ("y", "ao"),
    "iu":   ("y", "ou"),   "ian":  ("y", "an"),   "iang": ("y", "ang"),  "iong": ("y", "ong"),
    "v":    ("y", "v"),    "ve":   ("y", "ve"),   "van":  ("y", "van"),  "vn":   ("y", "vn"),
    "u":    ("w", "u"),    "ua":   ("w", "a"),    "uo":   ("w", "o"),
    "uai":  ("w", "ai"),   "ui":   ("w", "ei"),   "uan":  ("w", "an"),
    "un":   ("w", "en"),   "uang": ("w", "ang"),
}

# =============================================================================
# 1. 近体诗完整格律格式表（五绝 / 七绝 / 五律 / 七律 × 仄起式 / 平起式 × 首句入韵 / 不入韵）
# =============================================================================
# 四大五言基本句式 & 四大七言基本句式（P=平, Z=仄）
LINE_PATTERNS: Dict[str, str] = {
    "W1": "ZZPPZ",     # 仄仄平平仄
    "W2": "PPZZP",     # 平平仄仄平
    "W3": "PPPZZ",     # 平平平仄仄
    "W4": "ZZZPP",     # 仄仄仄平平
    "Q1": "PPZZPPZ",   # 平平仄仄平平仄 (= PP + W1)
    "Q2": "ZZPPZZP",   # 仄仄平平仄仄平 (= ZZ + W2)
    "Q3": "ZZPPPZZ",   # 仄仄平平平仄仄 (= ZZ + W3)
    "Q4": "PPZZZPP",   # 平平仄仄仄平平 (= PP + W4)
}

REGULATED_SCHEMAS: Dict[Tuple[int, int], List[Tuple[str, List[str]]]] = {
    # (句数, 每句字数) -> [(格式名, 逐句句型列表)]
    (4, 5): [
        ("五绝·仄起首句不入韵", ["W1", "W2", "W3", "W4"]),
        ("五绝·仄起首句入韵",   ["W4", "W2", "W3", "W4"]),
        ("五绝·平起首句不入韵", ["W3", "W4", "W1", "W2"]),
        ("五绝·平起首句入韵",   ["W2", "W4", "W1", "W2"]),
    ],
    (4, 7): [
        ("七绝·仄起首句入韵",   ["Q2", "Q4", "Q1", "Q2"]),
        ("七绝·仄起首句不入韵", ["Q3", "Q4", "Q1", "Q2"]),
        ("七绝·平起首句入韵",   ["Q4", "Q2", "Q3", "Q4"]),
        ("七绝·平起首句不入韵", ["Q1", "Q2", "Q3", "Q4"]),
    ],
    (8, 5): [
        ("五律·仄起首句不入韵", ["W1", "W2", "W3", "W4", "W1", "W2", "W3", "W4"]),
        ("五律·仄起首句入韵",   ["W4", "W2", "W3", "W4", "W1", "W2", "W3", "W4"]),
        ("五律·平起首句不入韵", ["W3", "W4", "W1", "W2", "W3", "W4", "W1", "W2"]),
        ("五律·平起首句入韵",   ["W2", "W4", "W1", "W2", "W3", "W4", "W1", "W2"]),
    ],
    (8, 7): [
        ("七律·仄起首句入韵",   ["Q2", "Q4", "Q1", "Q2", "Q3", "Q4", "Q1", "Q2"]),
        ("七律·仄起首句不入韵", ["Q3", "Q4", "Q1", "Q2", "Q3", "Q4", "Q1", "Q2"]),
        ("七律·平起首句入韵",   ["Q4", "Q2", "Q3", "Q4", "Q1", "Q2", "Q3", "Q4"]),
        ("七律·平起首句不入韵", ["Q1", "Q2", "Q3", "Q4", "Q1", "Q2", "Q3", "Q4"]),
    ],
}

# t17《问刘十九》恰好天然包含全部四大五言句式（W1/W2/W3/W4）的真人精确声学原型！
# 每句记录：(t17_clause_idx, [(start_s, end_s, tone_class), ...])
T17_W_PROTOTYPES: Dict[str, Tuple[int, List[Tuple[float, float, str]]]] = {
    # W1: 绿(入) 蚁(上) 新(平) 醅(平) 酒(上) = ZZPPZ
    "W1": (0, [(0.08, 0.28, "ru"), (0.30, 1.08, "shang"), (1.25, 1.72, "ping"), (1.76, 2.64, "ping"), (2.82, 3.28, "shang")]),
    # W2: 红(平) 泥(平) 小(上) 火(上) 炉(平) = PPZZP
    "W2": (1, [(0.08, 0.40, "ping"), (0.42, 1.05, "ping"), (1.36, 1.90, "shang"), (2.02, 2.74, "shang"), (2.80, 3.88, "ping")]),
    # W3: 晚(上) 来(平) 天(平) 欲(入) 雪(入) = (Z)PPZZ (典型平平平仄仄调式)
    "W3": (2, [(0.08, 0.33, "shang"), (0.36, 1.24, "ping"), (1.42, 2.04, "ping"), (2.08, 2.34, "ru"), (2.64, 2.90, "ru")]),
    # W4: 能(平) 饮(上) 一(入) 杯(平) 无(平) = (P)ZZPP (典型仄仄仄平平调式)
    "W4": (3, [(0.18, 0.80, "ping"), (0.84, 1.56, "shang"), (1.68, 1.92, "ru"), (1.98, 2.58, "ping"), (2.66, 3.68, "ping")]),
}

# 七言二字头原型（直接取自同一位真人名家 t17 的首二字声学实体，保证七言头尾音色、调高与气韵浑然一体！）
T17_HEAD2_PROTOTYPES: Dict[str, Tuple[int, List[Tuple[float, float, str]]]] = {
    "PP": (1, [(0.08, 0.40, "ping"), (0.42, 1.05, "ping")]),     # 红泥 (平平二字头)
    "ZZ": (0, [(0.08, 0.28, "ru"),   (0.30, 1.08, "shang")]),    # 绿蚁 (仄仄二字头)
}

# 非近体诗（诗经/辞赋/古文）所用平声落音骨架与真人母带库
TANG_CLAUSE_PING_ANCHORS: Dict[str, List[Tuple[float, float]]] = {
    "shijing_4": [(+0.5, -1.0), (-0.5, -3.0), (+1.5, 0.0),  (-0.5, -3.5)],
    "wujue_5":   [(0.0,  -1.0), (-0.5, -3.0), (+1.0, -1.5), (-0.5, -3.5)],
    "qijue_7":   [(+0.5, -1.0), (-0.5, -3.0), (+1.5, 0.0),  (-0.5, -3.5)],
    "guwen":     [(0.0,  -1.0), (+0.5, -1.5), (+1.2, 0.0),  (-0.5, -3.0)],
    "fu":        [(-0.5, -2.0), (0.0,  -2.5), (+1.0, -0.5), (-0.5, -3.5)],
}

GENRE_MASTER_TRACKS: Dict[str, List[str]] = {
    "wujue_5":   ["t17"],
    "shijing_4": ["t15", "t01", "t13"],
    "qijue_7":   ["t03", "t02"],
    "fu":        ["t18"],
    "guwen":     ["t16", "t09", "t06"],
}

_MANIFEST_CACHE: Optional[dict] = None
_T17_F0_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}


def _get_manifest() -> dict:
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is None:
        _MANIFEST_CACHE = json.loads((DATA_DIR / "tang_manifest.json").read_text(encoding="utf-8"))
    return _MANIFEST_CACHE


def _get_t17_clause_f0(cl_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    if cl_idx not in _T17_F0_CACHE:
        m = _get_manifest()
        tr = next(x for x in m["tracks"] if x["track_id"] == "t17")
        cl = tr["clauses"][cl_idx % len(tr["clauses"])]
        d_npz = np.load(str(DATA_DIR / cl["f0_npz"]))
        rel = d_npz["rel_semitones"].astype(np.float32)
        uv = d_npz["uv"] > 0
        sr_ref = int(d_npz["sr"])
        hop_ref = int(d_npz["hop"])
        idx_v = np.where(uv)[0]
        t_all = idx_v * hop_ref / float(sr_ref)
        st_all = np.clip(rel[idx_v], -6.5, 8.5)
        _T17_F0_CACHE[cl_idx] = (t_all, st_all)
    return _T17_F0_CACHE[cl_idx]


@dataclass
class MeterAnalysis:
    is_regulated: bool
    form_name: str               # 如 "五绝·平起首句不入韵" / "七律·仄起首句入韵" / "非近体(诗经/古文)"
    line_patterns: List[str]     # 每句匹配的调式代号 ["W3", "W4", "W1", "W2"] 或 ["Q4", "Q2", "Q3", "Q4"]
    actual_pz: List[str]         # 每句实测中古平仄串 ["PPZPP", ...]
    match_ratio: float           # 关键位（二四六与韵脚）匹配度 0..1


def _score_line_pattern(actual_pz: str, pat_code: str) -> float:
    """按「一三五不论、二四六分明、末字韵脚最重」计算某句与标准句型的匹配分。"""
    pat = LINE_PATTERNS[pat_code]
    n = len(pat)
    if len(actual_pz) != n:
        return 0.0
    # 关键位权重：五言第2、4、5字（索引1,3,4）；七言第2、4、6、7字（索引1,3,5,6）
    weights = [0.6, 2.2, 0.8, 2.2, 3.0] if n == 5 else [0.5, 2.0, 0.6, 2.2, 0.8, 2.2, 3.0]
    got = sum(w for a, b, w in zip(actual_pz, pat, weights) if a == b)
    return got / sum(weights)


def analyze_poem_meter(text: str) -> MeterAnalysis:
    """
    先分析诗文中古平仄格式，再精准判定诗体（五绝/七绝/五律/七律、仄起/平起、首句入韵/不入韵）
    与每一句对应的唐调标准句式（W1~W4 / Q1~Q4）。
    """
    raw_clauses = [c.strip() for c in re.split(r"[，。！？；、\n]+", text) if re.search(r"[\u4e00-\u9fff]", c)]
    if not raw_clauses:
        return MeterAnalysis(False, "空文本", [], [], 0.0)

    # 始终以中古四声（mc）判别平仄：入声（白/月/日/绿/欲/雪/一/泊/国/隔/觉/落）必归仄声！
    actual_pz: List[str] = []
    lens: List[int] = []
    for cl in raw_clauses:
        syls = [s for s in text_to_sing_syllables(cl, era="mc") if s.char != " "]
        pz = "".join("P" if s.tone_class == "ping" else "Z" for s in syls)
        actual_pz.append(pz)
        lens.append(len(pz))

    n_lines = len(raw_clauses)
    uniform_len = lens[0] if len(set(lens)) == 1 else -1

    # 1) 命中标准五绝/七绝/五律/七律格式表
    if (n_lines, uniform_len) in REGULATED_SCHEMAS:
        best_name, best_pats, best_score = "", [], -1.0
        for s_name, s_pats in REGULATED_SCHEMAS[(n_lines, uniform_len)]:
            sc = float(np.mean([_score_line_pattern(pz, p) for pz, p in zip(actual_pz, s_pats)]))
            if sc > best_score:
                best_name, best_pats, best_score = s_name, list(s_pats), sc

        # 若整首属于拗体/仄韵古绝（如《春晓》二四句押仄韵，与标准格律匹配度 < 0.72），
        # 逐句分别挑最贴合该句实测平仄的 W1..W4 / Q1..Q4，保证绝不张冠李戴！
        if best_score < 0.72:
            cands = ["W1", "W2", "W3", "W4"] if uniform_len == 5 else ["Q1", "Q2", "Q3", "Q4"]
            line_best = [max(cands, key=lambda c: _score_line_pattern(pz, c)) for pz in actual_pz]
            tag = "五绝·仄韵/古绝体" if uniform_len == 5 else "七绝·拗体/古绝"
            return MeterAnalysis(True, f"{tag}（逐句配式）", line_best, actual_pz, round(best_score, 3))
        return MeterAnalysis(True, best_name, best_pats, actual_pz, round(best_score, 3))

    # 2) 其他句数但仍为齐整五言或七言排律/古诗（如 6 句/10 句/12 句）：逐句匹配 W1..W4 / Q1..Q4
    if uniform_len in (5, 7):
        cands = ["W1", "W2", "W3", "W4"] if uniform_len == 5 else ["Q1", "Q2", "Q3", "Q4"]
        line_best = [max(cands, key=lambda c: _score_line_pattern(pz, c)) for pz in actual_pz]
        sc = float(np.mean([_score_line_pattern(pz, p) for pz, p in zip(actual_pz, line_best)]))
        return MeterAnalysis(True, f"{'五' if uniform_len == 5 else '七'}言诗（逐句配式）", line_best, actual_pz, round(sc, 3))

    return MeterAnalysis(False, "非近体（诗经/辞赋/古文）", [], actual_pz, 0.0)


def _build_char_phones(sy) -> Tuple[List[Tuple[str, str]], bool, bool]:
    raw_phs = list(sy.phones)
    has_checked = bool(sy.tone_class == "ru" or (len(raw_phs) >= 2 and raw_phs[-1] in ("coda_p", "coda_t", "coda_k")))
    has_m = bool(len(raw_phs) >= 2 and raw_phs[-1] == "coda_m")
    clean_pairs = []
    for rp in raw_phs:
        mp = _OPENCPOP_NORM.get(rp, rp)
        if mp:
            clean_pairs.append((rp, mp))
    if not clean_pairs:
        clean_pairs = [("a", "a")]
    if clean_pairs[0][0] in VOWEL_SET and clean_pairs[0][1] in _ZERO_ONSET_YW_SPLIT:
        g_on, g_nuc = _ZERO_ONSET_YW_SPLIT[clean_pairs[0][1]]
        clean_pairs = [(g_on, g_on), (clean_pairs[0][0], g_nuc)] + clean_pairs[1:]
    return clean_pairs, has_checked, has_m


def _adapt_slot_duration(tpl_dur: float, tpl_tone: str, act_tone: str, is_last: bool, is_foot_end: bool) -> float:
    """
    在真人原型时长基础上，按新诗该字的真实四声（平/上/去/入）做自然弹性微调：
      - 同声调类直接继承 100% 真人时长（入声下限保护 >= 0.24s 绝不突然断裂！）；
      - 原型非入声、新诗为入声（如 白/日/月/泊/国/一）：平滑收短至 0.26~0.30s（利落短促且饱满）；
      - 原型为入声、新诗为平声（如 一三五不论 首字平声）：平滑舒展至 0.42~0.58s；
      - 末字为平韵时保持充分饱满拖腔（>= 0.90s），末字为入声时绝不拖长（<= 0.30s）。
    """
    if act_tone == "ru":
        if tpl_tone == "ru":
            return round(min(0.31, max(0.24, tpl_dur)), 4)
        return round(0.29 if is_last else (0.28 if is_foot_end else 0.26), 4)

    if act_tone == "ping":
        if is_last:
            return round(max(0.92, tpl_dur), 4)
        if tpl_tone == "ru":
            return round(0.56 if is_foot_end else 0.44, 4)
        if tpl_tone == "ping":
            return round(tpl_dur, 4)
        # 原型是上/去声，新诗是平声（一三五不论）
        return round(max(0.42, min(0.76, tpl_dur * 0.95)), 4)

    if act_tone == "shang":
        if tpl_tone == "shang":
            return round(tpl_dur, 4)
        if tpl_tone == "ru":
            return round(0.44 if is_last else (0.50 if is_foot_end else 0.34), 4)
        return round(max(0.34, min(0.66, tpl_dur * 0.85)), 4)

    # 去声 (qu)
    if tpl_tone == "ru":
        return round(0.42 if is_last else (0.46 if is_foot_end else 0.33), 4)
    if is_last:
        return round(min(0.50, max(0.38, tpl_dur * 0.88)), 4)
    return round(max(0.32, min(0.58, tpl_dur * 0.82)), 4)


def _extract_char_f0_slice(cl_idx: int, ts: float, te: float, n_pts: int, pitch_shift_st: float = 0.0) -> np.ndarray:
    """从真人 t17 对应句的真实 F0 中截取 [ts, te] 区间并平滑重采样至 n_pts 帧。"""
    t_all, st_all = _get_t17_clause_f0(cl_idx)
    t_q = np.linspace(ts, te, max(4, n_pts), dtype=np.float32)
    vals = np.interp(t_q, t_all, st_all).astype(np.float32) + pitch_shift_st
    return vals


def compile_regulated_tang_ds(
    text: str,
    era: str = "mc",
    key_hz: float = 261.6,
    speaker: str = "opencpop_base",
    tempo_scale: float = 1.0,
) -> Tuple[List[dict], MeterAnalysis]:
    """
    V6 近体格律诗（五绝/七绝/五律/七律）真人四式调式引擎：
      1. 先用 `analyze_poem_meter` 判定诗体、仄起/平起、首句入韵与否，得出每句的 W1~W4 / Q1~Q4 句型；
      2. 逐句调取对应真人唐调四式声学原型（七言由二字头 + 五言身自然拼接），
         逐字按新诗平上去入做弹性时长微调与真人 F0 分段重采样，100% 保留真人活人韵腔！
    """
    meter = analyze_poem_meter(text)
    raw_clauses = [c.strip() for c in re.split(r"[，。！？；、\n]+", text) if re.search(r"[\u4e00-\u9fff]", c)]
    if not raw_clauses or not meter.is_regulated:
        return [], meter

    f0_step = 0.005
    ds_items: List[dict] = []
    running_offset = 0.0
    n_lines = len(raw_clauses)

    for ci, (clause, pat_code) in enumerate(zip(raw_clauses, meter.line_patterns)):
        sylls = [s for s in text_to_sing_syllables(clause, era=era) if s.char != " "]
        # 用中古四声作真实格律时长依据（哪怕 era="zh" 唱现代音，入声字白/日/绿/欲/雪/一仍精准识别为入声！）
        mc_sylls = [s for s in text_to_sing_syllables(clause, era="mc") if s.char != " "]
        n_ch = len(sylls)
        if n_ch not in (5, 7):
            continue

        # 组装本句对应的真人原型槽位列表: [(t17_cl_idx, ts, te, tpl_tone, gap_before_s)]
        slots: List[Tuple[int, float, float, str, float]] = []
        if n_ch == 5:
            cl_idx, spans = T17_W_PROTOTYPES[pat_code]
            prev_e = 0.0
            for k, (ts, te, tt) in enumerate(spans):
                gap = max(0.0, ts - prev_e) if k > 0 else 0.0
                # 严禁词内小间隔变成硬断 SP：仅保留真正的诗律大停顿 (>= 0.09s)
                if k == 1:
                    gap = 0.0
                elif k == 4 and not (mc_sylls[3].tone_class == "ru" and mc_sylls[4].tone_class == "ru"):
                    gap = min(gap, 0.05)
                slots.append((cl_idx, ts, te, tt, gap))
                prev_e = te
        else:
            # 七言 = 反向二字头 (PP 或 ZZ) + 125ms 诗律气口 + 对应五言身 (W1..W4)
            head_key = "PP" if pat_code in ("Q1", "Q4") else "ZZ"
            body_key = "W" + pat_code[1]
            h_cl, h_spans = T17_HEAD2_PROTOTYPES[head_key]
            b_cl, b_spans = T17_W_PROTOTYPES[body_key]
            # 二字头 (k=0, 1)
            slots.append((h_cl, h_spans[0][0], h_spans[0][1], h_spans[0][2], 0.0))
            slots.append((h_cl, h_spans[1][0], h_spans[1][1], h_spans[1][2], 0.0))
            # 五言身 (k=2..6)
            prev_e = b_spans[0][0]
            for j, (ts, te, tt) in enumerate(b_spans):
                if j == 0:
                    gap = 0.125  # 七言 2|5 第一大气口
                elif j == 1:
                    gap = 0.0
                else:
                    gap = max(0.0, ts - prev_e)
                    if j == 4 and not (mc_sylls[5].tone_class == "ru" and mc_sylls[6].tone_class == "ru"):
                        gap = min(gap, 0.05)
                slots.append((b_cl, ts, te, tt, gap))
                prev_e = te

        # 律诗（8句）微调起承转合宏观张力：颈联微扬 +0.5st，末句沉稳落韵 -0.4st
        line_shift = 0.0
        if n_lines == 8:
            if ci in (4, 5):
                line_shift = +0.5
            elif ci == 7:
                line_shift = -0.4

        ph_seq: List[str] = ["SP"]
        ph_dur: List[float] = [0.12]
        f0_chunks: List[Tuple[np.ndarray, bool]] = []  # (st_chunk, is_sp)

        for k, (sy, mcsy, (src_cl, ts, te, tpl_tone, gap_before)) in enumerate(zip(sylls, mc_sylls, slots)):
            # 气口停顿（仅 >= 0.065s 才插入真正自然换气 SP，绝不在词内生硬截断！）
            if gap_before >= 0.065:
                gd = round(min(0.24, gap_before * tempo_scale), 4)
                ph_seq.append("SP")
                ph_dur.append(gd)
                n_g = max(2, int(round(gd / f0_step)))
                last_val = float(f0_chunks[-1][0][-1]) if f0_chunks else 0.0
                f0_chunks.append((np.full(n_g, last_val, dtype=np.float32), True))

            is_last = (k == n_ch - 1)
            is_foot_end = (n_ch == 5 and k == 1) or (n_ch == 7 and k in (1, 3))
            tpl_d = max(0.18, te - ts) * (0.90 if n_ch == 7 else 1.0) * tempo_scale
            cdur = _adapt_slot_duration(tpl_d, tpl_tone, mcsy.tone_class, is_last=is_last, is_foot_end=is_foot_end)

            clean_pairs, _, has_m = _build_char_phones(sy)
            max_on = 0.048 if mcsy.tone_class == "ru" else 0.065
            onset_d = min(max_on, cdur * 0.24) if (clean_pairs[0][0] not in VOWEL_SET and clean_pairs[0][0] not in CODA_SET) else 0.0
            m_d = min(0.052, cdur * 0.16) if has_m else 0.0
            nuc_d = max(0.14, cdur - onset_d - m_d)

            # 抽取真人 t17 对应字的真实 F0 片段；若该位实际与模板平仄相反（一三五不论），做极轻柔音高贴合
            actual_char_dur = 0.0
            char_ph_durs: List[Tuple[str, float]] = []
            for rp, mp in clean_pairs:
                d = m_d if rp == "coda_m" else (nuc_d if rp in VOWEL_SET else (onset_d if onset_d > 0 else 0.052))
                d = round(d, 4)
                char_ph_durs.append((mp, d))
                actual_char_dur += d

            n_c = max(6, int(round(actual_char_dur / f0_step)))
            # 若一三五不论导致首字/第三字平仄与原型相反，微调 ±1.5st 而不破坏真人滑音曲线
            tone_nudge = 0.0
            if mcsy.tone_class == "ping" and tpl_tone != "ping":
                tone_nudge = -1.4
            elif mcsy.tone_class in ("shang", "qu") and tpl_tone == "ping":
                tone_nudge = +1.4
            st_slice = _extract_char_f0_slice(src_cl, ts, te, n_c, pitch_shift_st=line_shift + tone_nudge)
            # 若实际为去声且原型为高平/上声，末尾自然微落 1.0st 体现去声落音
            if mcsy.tone_class == "qu" and tpl_tone == "shang":
                st_slice = st_slice - np.linspace(0.0, 1.4, len(st_slice), dtype=np.float32)

            for mp, d in char_ph_durs:
                ph_seq.append(mp)
                ph_dur.append(d)
            f0_chunks.append((st_slice, False))

        tail_sp = 0.34 if mc_sylls[-1].tone_class == "ru" else 0.32
        ph_seq.append("SP")
        ph_dur.append(tail_sp)

        # 精确拼接全句真人 F0，并在连续音步内做轻度平滑避免拼接缝，保留全部真人活人微颤与滑腔！
        total_dur = sum(ph_dur)
        n_tot = max(12, int(round(total_dur / f0_step)))
        f0_st = np.zeros(n_tot, dtype=np.float32)
        first_v = float(f0_chunks[0][0][0]) if f0_chunks else 0.0
        cur_i = int(round(ph_dur[0] / f0_step))
        f0_st[:cur_i] = first_v
        last_v = first_v
        for arr, is_sp in f0_chunks:
            nx = min(n_tot, cur_i + len(arr))
            if nx > cur_i:
                f0_st[cur_i:nx] = arr[: nx - cur_i]
                last_v = float(f0_st[nx - 1])
            cur_i = nx
        if cur_i < n_tot:
            f0_st[cur_i:] = last_v

        # 极轻度平滑（sigma=2.0 ≈ 10ms），既完全保留真人 t17 的微腔活人感，又无缝衔接二字头与五言身！
        f0_st = gaussian_filter1d(f0_st, sigma=2.0)
        f0_st = np.clip(f0_st, -6.5, 8.5)
        f0_hz = key_hz * (2.0 ** (f0_st / 12.0))

        ds_items.append({
            "text": clause,
            "era": era,
            "genre": f"regulated_{pat_code}",
            "meter_form": meter.form_name,
            "line_pattern": pat_code,
            "offset": round(running_offset, 4),
            "ph_seq": " ".join(ph_seq),
            "ph_dur": " ".join(f"{x:.4f}" for x in ph_dur),
            "f0_seq": " ".join(f"{float(x):.2f}" for x in f0_hz),
            "f0_timestep": str(f0_step),
            "spk_name": speaker,
        })
        running_offset += total_dur + 0.32
    return ds_items, meter


# =============================================================================
# 2A. 《诗经》四言三大实录调式 · 16字逐字精确声学原型引擎 (V8 精雕版)
# =============================================================================
# 经对全库 5 首《诗经》录音逐帧声学拆解，归结出 3 大极具辨识度的经典四言调式：
#   A. shijing_jianjia (雅正苍凉咏叹调，先扬后抑·低回感怀) ← 逐字对齐自 t15《蒹葭》（与 t13《采薇》同源正调！）
#   B. shijing_guanju  (周南清丽起兴调，先抑后扬·婉转明丽) ← 逐字对齐自 t01《关雎》
#   C. shijing_wuyi    (秦风豪放铿锵调，明快顿挫·高亢雄浑) ← 逐字对齐自 t20《无衣》
# 每式含完整 4 句（起/承/转/合）× 每句 4 字的精确真人 (t_start, t_end, tpl_tone) 坐标，
# 保证每一个转腔、滑音、高扬、沉落 100% 精准踩在对应字眼上，并可泛化至任意四言诗！
SHIJING_MODES: Dict[str, dict] = {
    "shijing_jianjia": {
        "name": "诗经调式 A · 雅正苍凉咏叹调（先扬后抑）",
        "desc": "母带 t15《蒹葭》（与 t13《采薇》同源正调）：第1句高起舒展、第2句扬转沉落、第3/4句低五度深沉低回咏叹",
        "track_id": "t15",
        "body_wav": "bodies/t15_shijing_4_诗经·秦风·蒹葭.wav",
        "ref_clip_s": 11.4,
        "key_hz": 293.7,  # D4，最低 -6.5st 稳落女声清润共鸣区 G3(201Hz)
        "min_st": -6.5,
        "max_st": 7.0,
        "lines": [
            # 1. 蒹(平) 葭(平) 苍(平) 苍(平)
            [(0.13, 0.58, "ping"), (0.60, 1.08, "ping"), (1.12, 2.08, "ping"), (2.12, 2.94, "ping")],
            # 2. 白(入) 露(去) 为(平) 霜(平)
            [(3.05, 3.38, "ru"),   (3.40, 3.80, "qu"),   (3.82, 4.62, "ping"), (4.68, 5.32, "ping")],
            # 3. 所(上) 谓(去) 伊(平) 人(平)
            [(5.76, 6.18, "shang"),(6.20, 6.54, "qu"),   (6.62, 7.46, "ping"), (7.50, 7.98, "ping")],
            # 4. 在(上) 水(上) 一(入) 方(平) -> 第3字「一」接「方」不断气（10.10->10.22为平滑滑音）
            [(8.30, 8.74, "shang"),(8.76, 9.16, "shang"),(9.30, 10.10, "ru"),  (10.16, 11.05, "ping")],
        ],
    },
    "shijing_guanju": {
        "name": "诗经调式 B · 周南清丽起兴调（平低仄高 · 完整五声八度）",
        "desc": "母带 t01《关雎》（唐文治唐调）：完整还原 3₁-5₁-6₁-1-2-3 五声八度（13半音零削波），去除静音鬼峰污染与词内硬切 SP，精准对齐 2|2 句读与句间呼吸气口，让「在河之洲 → 窈窕淑女 → 君子好逑」行腔丝滑连贯！",
        "track_id": "t01",
        "body_wav": "bodies/t01_shijing_4_诗经·周南·关雎.wav",
        "ref_clip_s": 11.0,
        "key_hz": 256.0,  # ~C4：3₁(-6.3st)=178Hz(F3温润胸声) ~ 3(+6.8st)=379Hz(F#4/G4清亮松弛不紧绷)，完整保留 13.1 半音五声八度！
        "min_st": -7.0,
        "max_st": 7.5,
        "lines": [
            # 1. 关(阴平) 关(阴平) 雎(阴平) 鸠(阴平) —— 全平句：6₁→1 起扬，3₁(-6.3st)沉底长托 → 5₁(-3.1st)尤韵归平
            [(0.24, 0.62, "ping"), (0.62, 1.00, "ping"), (1.04, 1.78, "ping"), (1.80, 2.55, "ping")],
            # 2. 在(全浊上→去) 河(阳平) 之(阴平) 洲(阴平) —— 仄平平平：低起上扬，3₁(-6.0st)深沉长托 → 5₁(-3.1st)尤韵收束
            [(2.66, 3.00, "shang"),(3.00, 3.40, "ping"), (3.46, 4.30, "ping"), (4.32, 4.78, "ping")],
            # 3. 窈(阴上) 窕(全浊上) 淑(常三東入) 女(次浊上) —— 零断气连腔 + 「淑女」承接「窈窕」高音区，女字留足 0.58s 饱满收托！
            [(5.26, 5.58, "shang"),(5.58, 5.96, "shang"),(6.12, 6.82, "ru"),   (6.84, 7.42, "shang")],
            # 4. 君(阴平) 子(阴上) 好(阴上/去) 逑(全浊阳平) —— 2→3挑峰，「好」完整保留 8.68..9.55 下行大三度滑腔，「逑」9.70..10.85 饱满长韵落 5₁！
            [(7.82, 8.24, "ping"), (8.24, 8.66, "shang"),(8.68, 9.55, "shang"),(9.70, 10.85, "ping")],
        ],
    },
    "shijing_wuyi": {
        "name": "诗经调式 C · 秦风豪放铿锵调（2|2 均衡顿挫 · 末字急收）",
        "desc": "母带 t20《无衣》：精调前半步与后半步黄金比例（~0.84s : ~1.08s），岂曰饱满有力、无衣不拖沓、袍/衣/师/矛干脆急收！",
        "track_id": "t20",
        "body_wav": "bodies/t20_shijing_4_诗经·秦风·无衣.wav",
        "ref_clip_s": 10.7,
        "key_hz": 293.7,  # D4
        "min_st": -6.5,
        "max_st": 6.2,
        "lines": [
            # 1. 岂(上, 0.45s) 曰(入, 0.38s) | 无(平, 0.72s) 衣(平, 0.36s急收) -> 前后比例 0.83 : 1.08！
            [(0.12, 0.54, "shang"),(0.62, 0.98, "ru"),   (1.12, 1.84, "ping"), (1.98, 2.34, "ping")],
            # 2. 与(上, 0.44s) 子(上, 0.40s) | 同(平, 0.72s) 袍(平, 0.33s如入急收!) -> 0.84 : 1.05！
            [(2.90, 3.32, "shang"),(3.36, 3.76, "shang"),(3.86, 4.58, "ping"), (4.76, 5.09, "ping")],
            # 3. 王(平, 0.45s) 于(平, 0.41s) | 兴(平, 0.72s) 师(平, 0.37s干脆收!) -> 0.86 : 1.09！
            [(5.60, 6.05, "ping"), (6.08, 6.48, "ping"), (6.54, 7.26, "ping"), (7.44, 7.81, "ping")],
            # 4. 修(平, 0.45s) 我(上, 0.40s) | 戈(平, 0.74s) 矛(平, 0.35s急收!) -> 0.85 : 1.09！
            [(8.36, 8.81, "ping"), (8.86, 9.25, "shang"),(9.36, 10.10, "ping"),(10.24, 10.59, "ping")],
        ],
    },
}

_BODY_F0_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}


def _get_body_f0(track_id: str, body_wav_rel: str, max_s: float = 14.0) -> Tuple[np.ndarray, np.ndarray]:
    """对母带前 max_s 秒抽取高精度连续 F0 相对半音轨迹，经中值去毛刺（消除 BGM 偶发八度跳变）后缓存。"""
    if track_id not in _BODY_F0_CACHE:
        import librosa
        import soundfile as sf
        from scipy.ndimage import median_filter
        y, sr = sf.read(str(DATA_DIR / body_wav_rel), dtype="float32")
        y = y[: int(max_s * sr)]
        hop = 256
        f0, vf, _ = librosa.pyin(y, fmin=85.0, fmax=520.0, sr=sr, frame_length=2048, hop_length=hop)
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0]
        rms_db = 20.0 * np.log10(rms + 1e-8)
        n_f = min(len(f0), len(rms_db))
        f0, vf, rms_db = f0[:n_f], vf[:n_f], rms_db[:n_f]
        # 严格过滤静音底噪帧（< 峰值 -36 dB），彻底根除静音区 484Hz 鬼峰拉偏句首起音（如「窈」起音虚高下滑）！
        energy_ok = rms_db >= (float(np.percentile(rms_db, 98)) - 36.0)
        valid = (~np.isnan(f0)) & vf & energy_ok
        med = float(np.nanmedian(f0[valid]))
        idx = np.where(valid)[0]
        t_v = idx * hop / float(sr)
        st_v = np.clip(12.0 * np.log2(f0[idx] / med), -8.0, 8.5).astype(np.float32)
        if len(st_v) >= 9:
            st_v = median_filter(st_v, size=7).astype(np.float32)
        elif len(st_v) >= 7:
            st_v = median_filter(st_v, size=5).astype(np.float32)
        _BODY_F0_CACHE[track_id] = (t_v, st_v)
    return _BODY_F0_CACHE[track_id]


def compile_shijing_mode_ds(
    text: str,
    mode_key: str = "shijing_jianjia",
    era: str = "zh",
    key_hz: Optional[float] = None,
    speaker: str = "opencpop_base",
    tempo_scale: float = 1.0,
    use_v9: bool = True,
) -> List[dict]:
    """
    《诗经》三大调式编译器：默认路由至 V9 四层解耦「五声晶格 × 中古/近古四声」生成式引擎
    (`shijing_v9_engine.compile_shijing_v9_ds`)，传入 `use_v9=False` 可回退 V8 母带切片对照。
    """
    if use_v9:
        from shijing_v9_engine import compile_shijing_v9_ds
        return compile_shijing_v9_ds(
            text=text, mode_key=mode_key, era=era,
            key_hz=key_hz, speaker=speaker, tempo_scale=tempo_scale,
        )
    cfg = SHIJING_MODES.get(mode_key, SHIJING_MODES["shijing_jianjia"])
    use_key_hz = float(key_hz if key_hz is not None else cfg["key_hz"])
    min_st = float(cfg.get("min_st", -7.0))
    max_st = float(cfg.get("max_st", 7.5))
    t_body, st_body = _get_body_f0(cfg["track_id"], cfg["body_wav"], max_s=cfg["ref_clip_s"] + 2.0)

    raw_clauses = [c.strip() for c in re.split(r"[，。！？；、\n]+", text) if re.search(r"[\u4e00-\u9fff]", c)]
    if not raw_clauses:
        return []

    ds_items: List[dict] = []
    f0_step = 0.005
    running_offset = 0.0

    for ci, clause in enumerate(raw_clauses):
        sylls = [s for s in text_to_sing_syllables(clause, era=era) if s.char != " "]
        # 近古音(em)遵循《中原音韵》「入派三声」（如「淑」全浊入作阳平），中古/现代音严格依中古广韵四声判入声！
        tone_era = "em" if era.lower() == "em" else "mc"
        mc_sylls = [s for s in text_to_sing_syllables(clause, era=tone_era) if s.char != " "]
        n_ch = len(sylls)
        if n_ch == 0:
            continue

        tpl_line = cfg["lines"][ci % len(cfg["lines"])]
        slots = [tpl_line[min(3, int(round(k * 3.0 / max(1, n_ch - 1))))] for k in range(n_ch)] if n_ch != 4 else tpl_line

        # 第一遍：先算整句 4 字的弹性时长，并实施「四言节拍守恒代偿」防止整句变快赶拍！
        char_durs: List[float] = []
        tpl_durs: List[float] = []
        for k, (mcsy, (ts, te, tpl_tone)) in enumerate(zip(mc_sylls, slots)):
            td = max(0.24, te - ts) * tempo_scale
            # 《关雎》《无衣》第1/2字给足饱满立体的起拍（彻底消除「岂曰太短」「在河赶拍」）
            if mode_key == "shijing_guanju" and k in (0, 1):
                td = max(0.40 if k == 0 else 0.39, td)
            elif mode_key == "shijing_wuyi" and k in (0, 1):
                td = max(0.44 if k == 0 else 0.39, td)
            tpl_durs.append(td)

            if mcsy.tone_class == "ru":
                if mode_key == "shijing_wuyi" and k in (0, 1):
                    # 《无衣》前半步入声（如「岂曰」的「曰」）保留饱满顿挫 0.37s，确保前半步与后半步比例匀称！
                    cd = round(min(0.39, max(0.35, td * 0.95)), 4)
                elif mode_key == "shijing_guanju" and k == 2 and td >= 0.65:
                    # 《关雎》重拍第3字遇入声（如「淑」「室」）：
                    # 绝不在词内插硬断 SP 破坏「窈窕 | 淑女」2|2 节律！
                    # 中古音(mc)收紧至利落饱满的 0.50s 体现入声干脆短于平声（0.76s），现代音(zh)从容连腔 0.66s 忠实还原 t01 首章丝滑连贯！
                    cd = round((0.50 if era.lower() == "mc" else 0.66) * tempo_scale, 4)
                elif k == 2 and td >= 0.65:
                    # 其余调式重拍第3字遇入声（一/曰/作）：短于平声重拖，留足 0.44~0.50s 绝不赶拍或卡顿！
                    cd = round(min(0.50, max(0.42, td * 0.56)), 4)
                elif k == 3:
                    cd = round(min(0.36, max(0.28, td * 0.72)), 4)
                else:
                    cd = round(min(0.34, max(0.27, td * 0.85)), 4)
            elif mcsy.tone_class == "ping":
                if mode_key == "shijing_wuyi" and k == 3:
                    # 《无衣》第4字（衣/袍/师/矛）保持秦风特色短促急收！绝不拖慢！
                    cd = round(min(0.38, max(0.31, td)), 4)
                elif mode_key == "shijing_wuyi" and k == 2:
                    # 《无衣》第3字（无/同/兴/戈）稳控在 0.70~0.74s，既饱满又不显得过长压过前半步！
                    cd = round(min(0.75, max(0.68, td)), 4)
                elif mode_key == "shijing_guanju" and k == 2 and tpl_tone == "ru":
                    # 近古音「淑」入作阳平或新诗第3字为平声：保留饱满舒展的 0.72~0.78s 主干长腔！
                    cd = round(min(0.80, max(0.68, td * 0.95)), 4)
                elif tpl_tone == "ru":
                    cd = round(max(0.46, td * 1.18), 4)
                else:
                    cd = round(td, 4)
            else:
                # 上/去声：完整尊重原曲节拍
                if mode_key == "shijing_wuyi" and k == 3:
                    cd = round(min(0.36, max(0.28, td)), 4)
                elif mode_key == "shijing_wuyi" and k in (0, 1):
                    cd = round(max(0.42 if k == 0 else 0.38, td), 4)
                else:
                    cd = round(max(0.34, td * 0.97), 4)
            char_durs.append(cd)

        # 节拍守恒代偿：若因本句含多个入声（如「曰归曰归」「在水一方」「薇亦作止」）导致总长明显短于原句，
        # 将省出的拍长按 65% 平滑代偿给本句非入声字（优先第4字韵脚与第2字偶拍平声），确保诗经四言从容不赶拍！
        saved = sum(tpl_durs) - sum(char_durs)
        if saved > 0.16 and n_ch == 4:
            comp_pool = [i for i in (3, 1, 2, 0) if mc_sylls[i].tone_class != "ru" and not (mode_key == "shijing_wuyi" and i == 3)]
            if comp_pool:
                give = saved * (0.45 if (mode_key == "shijing_guanju" and mc_sylls[2].tone_class == "ru") else 0.65)
                weights = [1.8 if i == 3 else (1.0 if i == 1 else 0.6) for i in comp_pool]
                wsum = sum(weights)
                for idx_c, w in zip(comp_pool, weights):
                    char_durs[idx_c] = round(min(1.18, char_durs[idx_c] + give * (w / wsum)), 4)

        ph_seq: List[str] = ["SP"]
        # 《关雎》贴合 t01 原版自然气口（防止 lead_sp + tail_sp + gap_after 三重叠加制造 0.93s 断档！）
        lead_sp = 0.06 if mode_key == "shijing_guanju" else 0.10
        ph_dur: List[float] = [lead_sp]
        f0_chunks: List[np.ndarray] = []

        prev_te = slots[0][0]
        for k, (sy, mcsy, (ts, te, tpl_tone), cdur) in enumerate(zip(sylls, mc_sylls, slots, char_durs)):
            # 仅在 2|2 节拍交界 (k == 2) 允许自然气口停顿（>= 0.10s）；
            # 《关雎》首章「窈窕淑女」原版 0ms 连腔直承（插 SP 会叠加 sh 清擦造成 150ms 断谷），故 shijing_guanju 句内严禁插 SP！
            raw_gap = max(0.0, ts - prev_te) if k > 0 else 0.0
            if k == 2 and raw_gap >= 0.10 and mode_key != "shijing_guanju":
                gd = round(min(0.16, raw_gap * tempo_scale), 4)
                ph_seq.append("SP")
                ph_dur.append(gd)
                n_g = max(2, int(round(gd / f0_step)))
                last_v = float(f0_chunks[-1][-1]) if f0_chunks else 0.0
                f0_chunks.append(np.full(n_g, last_v, dtype=np.float32))
            prev_te = te

            clean_pairs, _, has_m = _build_char_phones(sy)
            max_on = 0.048 if (mcsy.tone_class == "ru" or (mode_key == "shijing_wuyi" and k == 3)) else 0.068
            onset_d = min(max_on, cdur * 0.24) if (clean_pairs[0][0] not in VOWEL_SET and clean_pairs[0][0] not in CODA_SET) else 0.0
            m_d = min(0.052, cdur * 0.16) if has_m else 0.0
            nuc_d = max(0.15, cdur - onset_d - m_d)

            actual_d = 0.0
            c_pairs: List[Tuple[str, float]] = []
            for rp, mp in clean_pairs:
                d = round(m_d if rp == "coda_m" else (nuc_d if rp in VOWEL_SET else (onset_d if onset_d > 0 else 0.055)), 4)
                c_pairs.append((mp, d))
                actual_d += d

            n_c = max(6, int(round(actual_d / f0_step)))
            st_slice = np.interp(np.linspace(ts, te, n_c, dtype=np.float32), t_body, st_body).astype(np.float32)
            # 《关雎》第3句（全仄高扬句「窈窕淑女」）：
            # 1) 第2字「窕」峰值柔限在 +6.6st（避免短段中位数偏低导致虚高）；
            # 2) 第3/4字「淑女」整体抬升 +1.2st 紧承「窈窕」高音区（消除 4.5st 断崖坠落与闭元音 u/v 发暗，同时完整保留先抑微扬的唐调神韵！）
            if mode_key == "shijing_guanju" and (ci % 4 == 2):
                if k == 1:
                    st_slice = np.minimum(st_slice, 6.6)
                elif k in (2, 3):
                    st_slice = st_slice + 1.2
            st_slice = np.clip(st_slice, min_st, max_st)

            for mp, d in c_pairs:
                ph_seq.append(mp)
                ph_dur.append(d)
            f0_chunks.append(st_slice)

        is_odd = (ci % 2 == 0)
        # 《无衣》句尾干脆利落收锋；《关雎》精准贴合 t01 原版句间呼吸（句1→2 约0.20s，句2→3 约0.42s，句3→4 约0.34s），彻底消除 0.93s 断档！
        if mode_key == "shijing_wuyi":
            tail_sp = 0.24 if is_odd else 0.34
        elif mode_key == "shijing_guanju":
            tail_sp = 0.14 if ci == 0 else (0.24 if ci == 1 else (0.20 if ci == 2 else 0.30))
        else:
            tail_sp = 0.22 if is_odd else 0.38
        ph_seq.append("SP")
        ph_dur.append(tail_sp)

        total_dur = sum(ph_dur)
        n_tot = max(12, int(round(total_dur / f0_step)))
        f0_st = np.zeros(n_tot, dtype=np.float32)
        first_v = float(f0_chunks[0][0]) if f0_chunks else 0.0
        cur_i = int(round(ph_dur[0] / f0_step))
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

        # 《关雎》用 sigma=1.6 保证「窈窕→淑女」零断气直承顺滑清晰；《无衣》用 sigma=1.8 保留铿锵干脆！
        sm_sigma = 1.6 if mode_key == "shijing_guanju" else (1.8 if mode_key == "shijing_wuyi" else 2.2)
        f0_st = np.clip(gaussian_filter1d(f0_st, sigma=sm_sigma), min_st, max_st)
        f0_hz = use_key_hz * (2.0 ** (f0_st / 12.0))

        ds_items.append({
            "text": clause,
            "era": era,
            "genre": mode_key,
            "offset": round(running_offset, 4),
            "ph_seq": " ".join(ph_seq),
            "ph_dur": " ".join(f"{x:.4f}" for x in ph_dur),
            "f0_seq": " ".join(f"{float(x):.2f}" for x in f0_hz),
            "f0_timestep": str(f0_step),
            "spk_name": speaker,
        })
        if mode_key == "shijing_guanju":
            # 与 tail_sp + 下一句 lead_sp(0.06) 合计恰好：句1→2=0.20s, 句2→3=0.42s, 句3→4=0.34s！
            gap_after = 0.00 if ci == 0 else (0.12 if ci == 1 else (0.08 if ci == 2 else 0.30))
        else:
            gap_after = 0.14 if (is_odd and mode_key == "shijing_jianjia") else (0.22 if is_odd else 0.42)
        running_offset += total_dur + gap_after
    return ds_items


# =============================================================================
# 2B. 非近体古诗文 · 四大真人实录调式引擎 (NONREG_MODES — 100% 真人母带曲调直传)
# =============================================================================
_GUWEN_PARTICLES = set("之乎者也矣焉哉兮而于以则乃然夫盖其与")

# 实测盘点全库 17 首非近体素材，归类为四大极具辨识度的真实人类调式：
NONREG_MODES: Dict[str, dict] = {
    "shijing_lyrical": {
        "name": "调式一 · 诗经雅正抒情调",
        "desc": "母带 t15《蒹葭》：大弧度咏叹（高起回旋、偶句大五度深沉落低），4+4成联、叠字舒展",
        "track_id": "t15",
        # (cl_idx, t_start_frac, t_end_frac) -> 精准对应奇句高扬咏叹 / 偶句深沉落韵！
        "odd_slices":  [(0, 0.00, 0.56), (2, 0.00, 0.48), (4, 0.00, 0.58), (7, 0.00, 0.55)],
        "even_slices": [(0, 0.56, 1.00), (2, 0.48, 1.00), (4, 0.58, 1.00), (7, 0.55, 1.00)],
        "speed_scale": 1.00,
        "pitch_scale": 0.92,
    },
    "shijing_heroic": {
        "name": "调式二 · 诗经秦风豪放调",
        "desc": "母带 t20《无衣》：节奏明快（~2.4s/句）、字字铿锵、高亢顿挫，适合豪放四言/军歌",
        "track_id": "t20",
        "odd_slices":  [(0, 0.0, 1.0), (2, 0.0, 1.0), (5, 0.0, 1.0), (6, 0.0, 1.0)],
        "even_slices": [(1, 0.0, 1.0), (3, 0.0, 1.0), (8, 0.0, 1.0), (9, 0.0, 1.0)],
        "speed_scale": 0.80,
        "pitch_scale": 0.90,
    },
    "guwen_elegant": {
        "name": "调式三 · 书斋清雅文言调",
        "desc": "母带 t09《兰亭集序》：江南书生温润平叙、句尾轻扬悬收留白，适合《兰亭》《论语》《爱莲说》",
        "track_id": "t09",
        "short_odd":   [(1, 0.0, 1.0), (4, 0.0, 1.0), (8, 0.0, 1.0)],
        "short_even":  [(5, 0.0, 1.0), (0, 0.5, 1.0), (3, 0.5, 1.0)],
        "long_odd":    [(0, 0.0, 1.0), (3, 0.0, 1.0), (6, 0.0, 1.0)],
        "long_even":   [(7, 0.0, 1.0), (9, 0.0, 1.0), (2, 0.0, 1.0)],
        "speed_scale": 0.92,
        "pitch_scale": 0.88,
    },
    "guwen_tang": {
        "name": "调式四 · 唐调沉雄古文赋调",
        "desc": "母带 t16《陋室铭》+ t18《前赤壁赋》：句前紧凑快诵、句尾大跨度沉落压韵，大开大合",
        "track_id": "t18",
        "short_track": "t16",
        "short_odd":   [(0, 0.0, 1.0), (2, 0.0, 0.48), (4, 0.0, 1.0)],
        "short_even":  [(1, 0.0, 1.0), (2, 0.50, 1.0), (3, 0.0, 1.0)],
        "long_odd":    [(0, 0.0, 1.0), (2, 0.0, 1.0), (4, 0.0, 1.0)],
        "long_even":   [(1, 0.0, 1.0), (3, 0.0, 1.0), (5, 0.0, 1.0), (7, 0.0, 1.0)],
        "speed_scale": 0.94,
        "pitch_scale": 0.92,
    },
}


def _extract_master_f0_slice(track_id: str, cl_idx: int, f0_s: float, f0_e: float, n_frames: int) -> np.ndarray:
    """从真人母带指定句的 [f0_s, f0_e] 比例区间 100% 完整抽取真实人类 F0 曲线（零公式扁平化！）。"""
    m = _get_manifest()
    tr = next(x for x in m["tracks"] if x["track_id"] == track_id)
    cl = tr["clauses"][cl_idx % len(tr["clauses"])]
    d_npz = np.load(str(DATA_DIR / cl["f0_npz"]))
    rel = d_npz["rel_semitones"].astype(np.float32)
    uv = d_npz["uv"] > 0
    idx_v = np.where(uv)[0]
    if len(idx_v) < 12:
        return np.zeros(n_frames, dtype=np.float32)
    st_v = np.clip(rel[idx_v], -8.5, 9.0)
    # 按句内比例截取（如 t15 联句的前半/后半）
    i0 = int(round(f0_s * (len(st_v) - 1)))
    i1 = max(i0 + 8, int(round(f0_e * len(st_v))))
    sub = st_v[i0:i1]
    # 将本段中值温和锚定在原录音相对调高内，保留完整的大起大落与句尾落韵！
    x_src = np.linspace(0.0, 1.0, len(sub), dtype=np.float32)
    x_dst = np.linspace(0.0, 1.0, max(8, n_frames), dtype=np.float32)
    out = np.interp(x_dst, x_src, sub).astype(np.float32)
    return out


def _mode_char_duration(
    ch: str,
    tone_class: str,
    k: int,
    n_ch: int,
    mode_key: str,
    base_unit: float,
    prev_ch: str = "",
    next_tone: str = "",
    clause_idx: int = 0,
) -> float:
    """按四大非近体实录调式的真实节奏比例分配字长（严守入声短促不断裂 + 句尾拖腔规律）。"""
    is_last = (k == n_ch - 1)
    is_penult = (k == n_ch - 2)
    is_odd = (clause_idx % 2 == 0)
    rubato = 1.0 + 0.04 * math.sin(1.9 * k + 2.3 * clause_idx + 0.3)

    if mode_key == "shijing_lyrical":
        # t15《蒹葭》抒情调：2|2 悠长舒展、叠字后字大拉长、偶句韵脚深沉拖腔！
        if tone_class == "ru":
            return round(min(0.32, max(0.25, (0.31 if is_last else 0.27) * base_unit * rubato)), 4)
        if ch == prev_ch:  # 叠字后字（苍「苍」、关「关」）
            return round(0.95 * base_unit * rubato, 4)
        if is_last:
            return round(((0.95 if is_odd else 1.25) if tone_class == "ping" else (0.68 if is_odd else 0.85)) * base_unit * rubato, 4)
        if n_ch == 4 and k == 1:
            return round((0.68 if tone_class == "ping" else 0.52) * base_unit * rubato, 4)
        return round((0.48 if tone_class == "ping" else 0.38) * base_unit * rubato, 4)

    if mode_key == "shijing_heroic":
        # t20《无衣》秦风豪放调：明快顿挫（~2.4s/句），末字铿锵有力不拖沓！
        if tone_class == "ru":
            return round(min(0.29, max(0.23, 0.26 * base_unit * rubato)), 4)
        if is_last:
            return round(((0.76 if is_odd else 0.92) if tone_class == "ping" else 0.62) * base_unit * rubato, 4)
        if n_ch == 4 and k == 1:
            return round((0.52 if tone_class == "ping" else 0.42) * base_unit * rubato, 4)
        return round((0.40 if tone_class == "ping" else 0.33) * base_unit * rubato, 4)

    if mode_key == "guwen_elegant":
        # t09《兰亭集序》书斋清雅调：从容均匀、文雅流转、句尾轻扬微延！
        is_long = (n_ch >= 6)
        if tone_class == "ru":
            return round(min(0.30, max(0.23, (0.29 if is_last else 0.25) * base_unit * rubato)), 4)
        if is_last:
            return round(((0.76 if is_odd else 0.88) if tone_class == "ping" else 0.62) * base_unit * rubato, 4)
        if is_long:
            return round((0.36 if ch in _GUWEN_PARTICLES else (0.34 if tone_class == "ping" else 0.28)) * base_unit * rubato, 4)
        return round((0.42 if k == 1 else (0.38 if tone_class == "ping" else 0.32)) * base_unit * rubato, 4)

    # guwen_tang: t16《陋室铭》+ t18《前赤壁赋》唐调沉雄顿挫调：句前快诵滚雪球、句末重锤大拉腔！
    is_long = (n_ch >= 6)
    if tone_class == "ru":
        return round(min(0.30, max(0.22, (0.30 if is_last else 0.24) * base_unit * rubato)), 4)
    if is_last:
        return round(((0.82 if is_odd else 1.10) if tone_class == "ping" else (0.64 if is_odd else 0.78)) * base_unit * rubato, 4)
    if is_long:
        if is_penult and next_tone in ("ru", "qu"):
            return round(0.42 * base_unit * rubato, 4)
        return round((0.33 if ch in _GUWEN_PARTICLES else (0.29 if tone_class == "ping" else 0.24)) * base_unit * rubato, 4)
    # 4~5字骈偶短句：前三字快诵铺垫，末字重锤拉长！
    return round((0.35 if k == 1 else (0.31 if tone_class == "ping" else 0.26)) * base_unit * rubato, 4)


def compile_nonreg_mode_ds(
    text: str,
    mode_key: str = "shijing_lyrical",
    era: str = "mc",
    key_hz: float = 261.6,
    speaker: str = "opencpop_base",
    tempo_scale: float = 1.0,
) -> List[dict]:
    """
    V7 非近体古诗文 · 四大真人实录调式编译器：
      直接 100% 抽取对应调式真人母带（t15 / t20 / t09 / t16+t18）的真实人类 F0 旋律大弧与微腔，
      只叠加极克制的四声方向微调（±0.7st）与入声防断裂时长约束，彻底摆脱公式扁平感！
    """
    cfg = NONREG_MODES.get(mode_key, NONREG_MODES["shijing_lyrical"])
    raw_clauses = [c.strip() for c in re.split(r"[，。！？；、\n]+", text) if re.search(r"[\u4e00-\u9fff]", c)]
    if not raw_clauses:
        return []

    base_unit = float(cfg["speed_scale"]) * tempo_scale
    p_scale = float(cfg["pitch_scale"])
    ds_items: List[dict] = []
    f0_step = 0.005
    running_offset = 0.0

    for ci, clause in enumerate(raw_clauses):
        sylls = [s for s in text_to_sing_syllables(clause, era=era) if s.char != " "]
        mc_sylls = [s for s in text_to_sing_syllables(clause, era="mc") if s.char != " "]
        if not sylls:
            continue
        n_ch = len(sylls)
        is_odd = (ci % 2 == 0)

        # 按调式与句长挑选最匹配的真人母带原句切片！
        if mode_key in ("shijing_lyrical", "shijing_heroic"):
            tid = cfg["track_id"]
            pool = cfg["odd_slices"] if is_odd else cfg["even_slices"]
            cl_idx, fs, fe = pool[(ci // 2) % len(pool)]
        else:
            is_short = (n_ch <= 5)
            tid = cfg.get("short_track", cfg["track_id"]) if is_short else cfg["track_id"]
            key_p = ("short_odd" if is_odd else "short_even") if is_short else ("long_odd" if is_odd else "long_even")
            pool = cfg[key_p]
            cl_idx, fs, fe = pool[(ci // 2) % len(pool)]

        ph_seq: List[str] = ["SP"]
        ph_dur: List[float] = [0.12]
        char_spans_sec: List[Tuple[float, float, str]] = []  # (t0, t1, tone_class)
        cur_t = 0.12

        for k, (sy, mcsy) in enumerate(zip(sylls, mc_sylls)):
            prev_ch = sylls[k - 1].char if k > 0 else ""
            next_t = mc_sylls[k + 1].tone_class if k + 1 < n_ch else ""
            c_dur = _mode_char_duration(
                sy.char, mcsy.tone_class, k, n_ch, mode_key, base_unit,
                prev_ch=prev_ch, next_tone=next_t, clause_idx=ci,
            )
            clean_pairs, _, has_m = _build_char_phones(sy)
            max_onset = 0.048 if mcsy.tone_class == "ru" else (0.056 if mode_key.startswith("guwen") else 0.065)
            onset_d = min(max_onset, c_dur * 0.24) if (clean_pairs[0][0] not in VOWEL_SET and clean_pairs[0][0] not in CODA_SET) else 0.0
            m_d = min(0.050, c_dur * 0.16) if has_m else 0.0
            nuc_d = max(0.14, c_dur - onset_d - m_d)

            c_start = cur_t
            for rp, mp in clean_pairs:
                d = round(m_d if rp == "coda_m" else (nuc_d if rp in VOWEL_SET else (onset_d if onset_d > 0 else 0.052)), 4)
                ph_seq.append(mp)
                ph_dur.append(d)
                cur_t += d
            char_spans_sec.append((c_start, cur_t, mcsy.tone_class))

            # 自然意群气口（严禁词内硬切 SP！）
            pause_d = 0.0
            if k < n_ch - 1:
                if mode_key == "shijing_lyrical" and n_ch == 4 and k == 1:
                    pause_d = 0.130
                elif mode_key == "shijing_heroic" and n_ch == 4 and k == 1:
                    pause_d = 0.105
                elif mode_key.startswith("guwen") and n_ch >= 8:
                    next_ch = sylls[k + 1].char
                    if (next_ch in ("于", "而", "以", "则", "游", "泛", "之") and 2 <= k <= n_ch - 4) or (
                        k == n_ch // 2 - 1 and not any(c.char in ("于", "而", "以", "则") for c in sylls[2:-2])
                    ):
                        pause_d = 0.095
            if pause_d > 0.0:
                ph_seq.append("SP")
                ph_dur.append(pause_d)
                cur_t += pause_d

        body_end_t = cur_t
        tail_sp = 0.22 if is_odd else 0.36
        ph_seq.append("SP")
        ph_dur.append(tail_sp)
        total_dur = sum(ph_dur)

        # 100% 抽取该调式真人母带的真实人类 F0 旋律弧线！
        n_tot = max(12, int(round(total_dur / f0_step)))
        i_start = int(round(0.12 / f0_step))
        i_end = min(n_tot, int(round(body_end_t / f0_step)))
        body_n = max(10, i_end - i_start)
        human_f0 = _extract_master_f0_slice(tid, cl_idx, fs, fe, body_n) * p_scale

        # 防止极端低/高段超出 OpenCpop 甜区，同时完整保留大起大落与句尾扬/落！
        med_h = float(np.median(human_f0))
        if abs(med_h) > 2.2:
            human_f0 = human_f0 - (med_h - np.clip(med_h, -2.0, 2.0))

        f0_st = np.zeros(n_tot, dtype=np.float32)
        f0_st[:i_start] = float(human_f0[0])
        f0_st[i_start:i_end] = human_f0
        f0_st[i_end:] = float(human_f0[-1])

        # 仅叠加极温柔的字调微韵（±0.7st，绝不破坏真人母带原汁原味的曲调！）
        for cs, ce, tc in char_spans_sec:
            a = max(i_start, int(round(cs / f0_step)))
            b = min(i_end, int(round(ce / f0_step)))
            if b - a < 4:
                continue
            w = np.sin(math.pi * np.linspace(0.0, 1.0, b - a, dtype=np.float32))
            if tc == "shang":
                f0_st[a:b] += 0.75 * w
            elif tc == "qu":
                f0_st[a:b] += np.linspace(0.6, -0.7, b - a, dtype=np.float32)
            elif tc == "ru":
                f0_st[a:b] -= 0.45 * np.linspace(0.0, 1.0, b - a, dtype=np.float32)

        f0_st = np.clip(gaussian_filter1d(f0_st, sigma=2.2), -6.8, 7.2)
        f0_hz = key_hz * (2.0 ** (f0_st / 12.0))

        ds_items.append({
            "text": clause,
            "era": era,
            "genre": mode_key,
            "offset": round(running_offset, 4),
            "ph_seq": " ".join(ph_seq),
            "ph_dur": " ".join(f"{x:.4f}" for x in ph_dur),
            "f0_seq": " ".join(f"{float(x):.2f}" for x in f0_hz),
            "f0_timestep": str(f0_step),
            "spk_name": speaker,
        })
        running_offset += total_dur + (0.16 if is_odd else 0.34)
    return ds_items


def compile_ancient_tang_ds(
    text: str,
    era: str = "mc",
    genre: str = "auto",
    key_hz: float = 261.6,
    speaker: str = "opencpop_base",
    tempo_scale: float = 1.0,
) -> List[dict]:
    """
    统一古诗文格律唐调吟诵入口：
      - 五言/七言绝句、律诗 → V6 格律辨体 × 真人四式调式引擎（W1~W4 / Q1~Q4）；
      - 非近体古诗文 → V7 四大真人实录调式直传引擎（shijing_lyrical / shijing_heroic / guwen_elegant / guwen_tang）！
    """
    if genre in SHIJING_MODES:
        return compile_shijing_mode_ds(text=text, mode_key=genre, era=era, key_hz=key_hz, speaker=speaker, tempo_scale=tempo_scale)
    if genre in NONREG_MODES:
        return compile_nonreg_mode_ds(text=text, mode_key=genre, era=era, key_hz=key_hz, speaker=speaker, tempo_scale=tempo_scale)

    meter = analyze_poem_meter(text)
    if meter.is_regulated and genre in ("auto", "wujue_5", "qijue_7", "wulv_5", "qilv_7"):
        ds_items, _ = compile_regulated_tang_ds(text=text, era=era, key_hz=key_hz, speaker=speaker, tempo_scale=tempo_scale)
        if ds_items:
            return ds_items

    raw_clauses = [c.strip() for c in re.split(r"[，。！？；、\n]+", text) if re.search(r"[\u4e00-\u9fff]", c)]
    if not raw_clauses:
        return []
    if genre == "shijing_4":
        mode_key = "shijing_lyrical"
    elif genre in ("fu", "guwen"):
        mode_key = "guwen_tang"
    else:
        lens = [len(re.findall(r"[\u4e00-\u9fff]", c)) for c in raw_clauses]
        mode_key = "shijing_lyrical" if (len(set(lens)) == 1 and lens[0] == 4) else "guwen_tang"
    return compile_nonreg_mode_ds(text=text, mode_key=mode_key, era=era, key_hz=key_hz, speaker=speaker, tempo_scale=tempo_scale)


def compile_from_tang_master_exact(
    text: str,
    track_id: str,
    clause_indices: List[int],
    char_durs_per_clause: Optional[List[List[Tuple[float, float]]]] = None,
    era: str = "mc",
    key_hz: float = 261.6,
    speaker: str = "opencpop_base",
) -> List[dict]:
    """保留精确逐字扒谱接口供基准对照。"""
    m = _get_manifest()
    tr = next(x for x in m["tracks"] if x["track_id"] == track_id)
    raw_clauses = [c.strip() for c in re.split(r"[，。！？；、\n]+", text) if re.search(r"[\u4e00-\u9fff]", c)]
    ds_items: List[dict] = []
    f0_step = 0.005
    running_offset = 0.0

    for ci, (clause, ref_idx) in enumerate(zip(raw_clauses, clause_indices)):
        cl = tr["clauses"][ref_idx % len(tr["clauses"])]
        d_npz = np.load(str(DATA_DIR / cl["f0_npz"]))
        rel = d_npz["rel_semitones"].astype(np.float32)
        uv = d_npz["uv"] > 0
        sr_ref, hop_ref = int(d_npz["sr"]), int(d_npz["hop"])
        idx_v = np.where(uv)[0]
        if len(idx_v) < 10:
            continue
        t_all = idx_v * hop_ref / float(sr_ref)
        st_all = np.clip(rel[idx_v], -7.0, 9.0)

        syl_chars = [s for s in text_to_sing_syllables(clause, era=era) if s.char != " "]
        spans = char_durs_per_clause[ci]
        ph_seq: List[str] = []
        ph_dur: List[float] = []
        cursor = 0.0
        for sy, (cs, ce) in zip(syl_chars, spans):
            cs = max(cursor, cs)
            gap = cs - cursor
            if gap >= 0.04:
                ph_seq.append("SP")
                ph_dur.append(round(gap, 4))
            elif gap > 0 and ph_dur:
                ph_dur[-1] = round(ph_dur[-1] + gap, 4)
            cdur = max(0.14, ce - cs)
            clean_pairs, _, has_m = _build_char_phones(sy)
            max_on = 0.042 if sy.tone_class == "ru" else 0.062
            onset_d = min(max_on, cdur * 0.24) if (clean_pairs[0][0] not in VOWEL_SET and clean_pairs[0][0] not in CODA_SET) else 0.0
            m_d = min(0.050, cdur * 0.16) if has_m else 0.0
            nuc_d = max(0.09, cdur - onset_d - m_d)
            for rp, mp in clean_pairs:
                d = m_d if rp == "coda_m" else (nuc_d if rp in VOWEL_SET else (onset_d if onset_d > 0 else 0.05))
                ph_seq.append(mp)
                ph_dur.append(round(d, 4))
            cursor = cs + sum(m_d if rp == "coda_m" else (nuc_d if rp in VOWEL_SET else (onset_d if onset_d > 0 else 0.05)) for rp, _ in clean_pairs)

        ph_seq.append("SP")
        ph_dur.append(0.32)
        total_dur = sum(ph_dur)
        n_f = max(10, int(round(total_dur / f0_step)))
        st_dst = gaussian_filter1d(np.interp(np.arange(n_f, dtype=np.float32) * f0_step, t_all, st_all).astype(np.float32), sigma=1.5)
        f0_hz = key_hz * (2.0 ** (st_dst / 12.0))
        ds_items.append({
            "text": clause, "era": era, "genre": f"master_{track_id}",
            "offset": round(running_offset, 4),
            "ph_seq": " ".join(ph_seq), "ph_dur": " ".join(f"{x:.4f}" for x in ph_dur),
            "f0_seq": " ".join(f"{float(x):.2f}" for x in f0_hz),
            "f0_timestep": str(f0_step), "spk_name": speaker,
        })
        running_offset += total_dur + 0.35
    return ds_items


compile_from_tang_master = compile_from_tang_master_exact


if __name__ == "__main__":
    tests = [
        ("问刘十九", "绿蚁新醅酒，红泥小火炉。晚来天欲雪，能饮一杯无。"),
        ("登鹳雀楼", "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。"),
        ("宿建德江", "移舟泊烟渚，日暮客愁新。野旷天低树，江清月近人。"),
        ("泊秦淮",   "烟笼寒水月笼沙，夜泊秦淮近酒家。商女不知亡国恨，隔江犹唱后庭花。"),
        ("早发白帝城", "朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。"),
        ("山居秋暝", "空山新雨后，天气晚来秋。明月松间照，清泉石上流。竹喧归浣女，莲动下渔舟。随意春芳歇，王孙自可留。"),
    ]
    print("✅ V6 近体格律自动辨体 + 句式匹配校验：")
    for title, txt in tests:
        mt = analyze_poem_meter(txt)
        print(f"  《{title}》 -> {mt.form_name} (匹配度={mt.match_ratio:.2f}) | 句式={mt.line_patterns} | 实测平仄={mt.actual_pz}")
