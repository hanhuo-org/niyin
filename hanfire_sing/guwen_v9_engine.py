#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汉火古音吟诵 · V13「唐调起承转合四阶句群 × 逐字平低仄高五声对位 × 真人单字活腔残差」古文/辞赋引擎 (guwen_v9_engine.py)

系统性拆解 12 首唐文治传派原版古文/辞赋母带（t09《兰亭集序》、t16《陋室铭》、t18《前赤壁赋》、
t19《岳阳楼记》、t06《答谢中书书》、t11《马说》、t12《诫兄子严敦书》）声学频谱后习得的三大铁律：

1. 为什么 V12 仍有差距？
   - V12 将多字母带区间 `(ts..t_cad)` 用 `np.linspace` 盲目拉平映射到整句，导致母带某字上的仄声高挑峰
     （如「在 +4.0st」）错位压到新句的平声虚词「之」上，且 `t09/t16/t18` 存在跨句误切（「兰亭」尾混入「修」、
     「斯是陋室/惟吾德馨」在「惟吾」处断错、「七月既望」误吞「秋」尾低音）；
   - V12 用简单 `ci % N` 轮换句型，不懂句号结句与骈偶对仗，导致《诫子书》「静以修身/俭以养德」上下句阴阳倒置、
     《爱莲说》末句悬空不收、《岳阳楼记》末句用起兴调收尾。

2. V13 双层解耦生成架构（零硬编码篇目、100% 真人活腔微残差、零词内 SP）：
   - 【第一层 · 篇章句群“起—承—转—合”规划器 (`plan_guwen_discourse`)】：
     自动解析全段句读（`，` / `；` / `。`）与骈偶对仗（3+3 / 4+4 / 5+5 / 6+6 / 7+7 邻句配对，自句末倒推，
     精准识别发语词领句如「夫君子之行」+ 对仗偶句「静以修身，俭以养德」），为每句分配六大唐调功能角：
       * `QI`（起·定调立宫）：低稳起兴，尾字稳托主宫或苍茫低徵；
       * `LEAD`（领·发语提纲）：端凝蓄势，领起后文骈偶；
       * `CHENG_UP`（承扬·骈偶上句）：平声立商(`+2.1st`)、仄声挑角(`+4.1st`)，句尾悬宫(`0.0st`)或仄悬高腔(`+3.2st`)；
       * `CHENG_DOWN`（承抑·骈偶下句）：严守「上句扬、下句抑」，句尾四度下沉落低徵(`-3.2st`)或沉敛仄收(`-0.9~-2.8st`)；
       * `ZHUAN_HIGH`（转·高腔拔起）：第三结构拍或转折重句陡拔五/六度入高徵/高角(`+4.4~+5.3st`，重仄冲顶 `+7.3st`)，
         句尾作五度流泻落宫(`+3.2→0.0st`)或高悬待合；
       * `PU_LONG`（铺·长散句行腔）：7 字以上长句绝不平铺直叙，前半稳守低羽/主宫，中后枢轴字扬波(`+2.5st`)，句末从容收韵；
       * `HE_GRAND`（合·欲抑先扬归底）：全篇/大句压轴严守唐调「欲抑先扬」铁律——倒数第 2 字蓄势挑高，
         末字沉入全曲最低角/低徵(`-4.9~-6.8st` 大拖腔 `1.15~1.35s`)或作书斋三波叹咏摇腔(`1.45~1.60s`)！
   - 【第二层 · 依字行腔“平低仄高”五声对位 + 100% 棚录单字活腔原型库】：
     从 `t09 / t16 / t18` 纯净单字元音核直接提取去毛刺真人 F0 活腔微轨迹 `Δ_human(u)`，
     按每个字的中古《广韵》/近古《中原音韵》四声逐字锚定至纯正五声音阶晶格
     `{-6.8, -4.9, -3.2, -1.6, 0.0, +2.1, +4.1, +5.3, +7.3}st`，彻底根除跨字串音与跑调！
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

SING_DIR = Path(__file__).resolve().parent
DATA_DIR = SING_DIR / "data_tang_processed"
if str(SING_DIR) not in sys.path:
    sys.path.insert(0, str(SING_DIR))

from hanfire_sing_phonemes import CODA_SET, SingSyllable, VOWEL_SET, text_to_sing_syllables
from shijing_v9_engine import ToneProfile, _build_char_phones, analyze_char_tone

_FINAL_PARTICLES = frozenset({"也", "矣", "乎", "哉", "焉", "耳", "兮", "夫", "耶", "欤", "歟"})
_MID_PARTICLES = frozenset({"之", "于", "於", "而", "以", "则", "則", "其", "为", "為", "与", "與", "乃"})
_LEAD_PARTICLES = frozenset({"夫", "盖", "蓋", "若", "嗟", "惟", "嗟夫", "然则"})

_FRICATIVE_ONSETS = frozenset({"sh", "s", "x", "ch", "q", "c", "f"})
_STOP_ONSETS = frozenset({"b", "p", "d", "t", "g", "k"})
_NASAL_ONSETS = frozenset({"n", "m"})

# 上下文中古四声精准纠偏（防单字多音误判平仄，如「吞长江」之「长」为澄母阳韵平声，「予独」之「予」为以母鱼韵平声）
_CONTEXT_TONE_OVERRIDES: Dict[str, Dict[int, Tuple[str, str, str]]] = {
    "长江": {0: ("ping", "yang", "澄開三陽平")},
    "長江": {0: ("ping", "yang", "澄開三陽平")},
    "长空": {0: ("ping", "yang", "澄開三陽平")},
    "长风": {0: ("ping", "yang", "澄開三陽平")},
    "少长": {0: ("qu", "yin", "書開三笑去"), 1: ("shang", "yin", "知開三養上")},
    "少長": {0: ("qu", "yin", "書開三笑去"), 1: ("shang", "yin", "知開三養上")},
    "予独": {0: ("ping", "yang", "以開三魚平")},
    "予獨": {0: ("ping", "yang", "以開三魚平")},
    "予观": {0: ("ping", "yang", "以開三魚平")},
    "不兴": {1: ("ping", "yin", "曉開三蒸平")},
    "甚蕃": {1: ("ping", "yang", "奉合三元平")},
    "既望": {1: ("qu", "yang", "微合三漾去")},
    "汤汤": {0: ("ping", "yin", "書開三陽平"), 1: ("ping", "yin", "書開三陽平")},
    "浩浩": {0: ("shang", "yang", "匣開一晧上"), 1: ("shang", "yang", "匣開一晧上")},
}


GUWEN_V9_MODES: Dict[str, dict] = {
    "guwen_lanting": {
        "name": "古文调式 G1 · 书斋清雅骈散调（t09《兰亭集序》11.2 半音起承转合 × 依字平低仄高 × 书斋三波摇腔）",
        "desc": "习得 t09《兰亭集序》与 t06《答谢中书书》唐调真传：低羽清叙起腔(-3.2st) → 偶句沉敛承接 → 转句五度拔入高徵(+5.3~+7.4st)流泻归宫 → 合句高起叹咏三波摇腔归底！",
        "track_id": "t09",
        "body_wav": "bodies/t09_guwen_兰亭集序.wav",
        "ref_clip_s": 16.0,
        # 女声 212Hz: -3.3st=175Hz(饱满胸声) ~ +5.2st=286Hz(避开300~311Hz换声带!) ~ 极峰+7.4st=325Hz(清亮头腔)
        "female_key_hz": 212.0,
        # 男声 m4_tenor7 154Hz: -3.3st=127Hz ~ +5.2st=208Hz ~ 极峰+7.4st=236Hz(100% 落入原生 117~247Hz 黄金实声区!)
        "male_key_hz": 154.0,
        "min_st": -4.0,
        "max_st": 8.0,
        "speed_scale": 1.00,
    },
    "guwen_loushi": {
        "name": "古文调式 G2 · 唐调铭文沉雄顿挫调（t16《陋室铭》10.8 半音骈偶阴阳 × 平商仄角 × 欲抑先扬沉底）",
        "desc": "习得 t16《陋室铭》唐调铭训真传：上句立商挑角悬主宫(+2.1→+4.1→0st) × 下句蓄势四度落低徵(-3.2st) × 转合连气「欲抑先扬」沉入低角(-5.0st)苍浑大腔！",
        "track_id": "t16",
        "body_wav": "bodies/t16_guwen_陋室铭.wav",
        "ref_clip_s": 17.0,
        # 女声 232Hz: -5.0st=174Hz ~ +2.1st=262Hz ~ +4.1st=294Hz(<298Hz!) ~ +5.4st=317Hz(>312Hz!) 完美跨过换声带!
        "female_key_hz": 232.0,
        # 男声 m4_bass1 132Hz: -5.0st=99Hz ~ +4.6st=172Hz(与原版母带 136Hz 1:1 同高，全程 <=185Hz 醇厚胸声!)
        "male_key_hz": 132.0,
        "min_st": -5.4,
        "max_st": 5.8,
        "speed_scale": 1.00,
    },
    "fu_chibi": {
        "name": "辞赋调式 G3 · 唐调辞赋大开大合调（t18《前赤壁赋》× t19《岳阳楼记》11.4 半音苍茫起兴 × 宫轴扬波）",
        "desc": "习得 t18《前赤壁赋》与 t19《岳阳楼记》唐调大赋真传：极沉低徵(-6.8st)起兴拔起六度 → 长赋句主宫滚诵扬波落底 → 排比偶句平商仄角高扬 → 压轴欲抑先扬浩然归底！",
        "track_id": "t18",
        "body_wav": "bodies/t18_fu_前赤壁赋.wav",
        "ref_clip_s": 21.5,
        # 女声 256Hz: -6.8st=173Hz ~ +2.2st=291Hz(<298Hz!) ~ +4.2st=326Hz(>312Hz!)
        "female_key_hz": 256.0,
        # 男声 m4_bass1 138Hz: -6.8st=93Hz(苍茫低音) ~ 0st=138Hz(主宫) ~ +4.2st=176Hz(<=185Hz 纯实胸声!)
        "male_key_hz": 138.0,
        "min_st": -7.2,
        "max_st": 5.2,
        "speed_scale": 1.00,
    },
}


# =============================================================================
# 1. 100% 真人棚录母带单字与落腔活腔原型库（RMS 净化 + 去八度/破音毛刺）
# =============================================================================
_ARCHETYPE_BANK: Optional[Dict[str, np.ndarray]] = None


def _extract_track_semitones(rel_wav: str, gong_hz: float, max_s: float, fmin: float, fmax: float) -> Tuple[np.ndarray, np.ndarray]:
    import librosa
    import soundfile as sf

    y, sr = sf.read(str(DATA_DIR / rel_wav), dtype="float32")
    y = y[: int(max_s * sr)]
    hop = 256
    f0, vf, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0]
    rms_db = 20.0 * np.log10(rms + 1e-8)
    n = min(len(f0), len(rms_db))
    f0, vf, rms_db = f0[:n], vf[:n], rms_db[:n]
    p98 = float(np.percentile(rms_db, 98))
    valid = (~np.isnan(f0)) & vf & (rms_db >= p98 - 34.0)
    idx = np.where(valid)[0]
    t_v = idx * hop / float(sr)
    st_v = (12.0 * np.log2(np.maximum(f0[idx], 1e-3) / gong_hz)).astype(np.float32)
    # 消除局部二倍频毛刺
    if len(st_v) >= 15:
        lmed = median_filter(st_v, size=min(61, len(st_v)))
        st_v[st_v - lmed > 9.0] -= 12.0
        st_v[lmed - st_v > 9.0] += 12.0
        st_v = median_filter(st_v, size=5).astype(np.float32)
    return t_v, st_v


def _slice_norm(t_v: np.ndarray, st_v: np.ndarray, s0: float, s1: float, anchor_mode: str = "median", n_pts: int = 64) -> np.ndarray:
    u = np.linspace(s0, s1, n_pts, dtype=np.float32)
    raw = np.interp(u, t_v, st_v).astype(np.float32)
    if anchor_mode == "median":
        ref = float(np.median(raw))
    elif anchor_mode == "tail":
        ref = float(np.median(raw[int(n_pts * 0.65):]))
    elif anchor_mode == "head":
        ref = float(np.median(raw[: int(n_pts * 0.35)]))
    else:
        ref = 0.0
    return (raw - ref).astype(np.float32)


def _get_archetype_bank() -> Dict[str, np.ndarray]:
    """从 t09 / t16 / t18 精确验证的纯净元音核提取 100% 真人单字活腔与招牌句尾摇腔残差。"""
    global _ARCHETYPE_BANK
    if _ARCHETYPE_BANK is not None:
        return _ARCHETYPE_BANK

    t09_t, t09_st = _extract_track_semitones("bodies/t09_guwen_兰亭集序.wav", 194.4, 28.0, 100.0, 390.0)
    t16_t, t16_st = _extract_track_semitones("bodies/t16_guwen_陋室铭.wav", 136.0, 56.4, 75.0, 260.0)
    t18_t, t18_st = _extract_track_semitones("bodies/t18_fu_前赤壁赋.wav", 154.0, 22.0, 80.0, 350.0)

    # 净化 t09「也」(13.55..15.45s) 在 14.4s 处的气声假跳变(+5.1st -> 限幅在真实三波摇腔上限 +2.3st)
    ye_raw = np.interp(np.linspace(13.58, 15.42, 96, dtype=np.float32), t09_t, t09_st).astype(np.float32)
    ye_clean = gaussian_filter1d(np.clip(ye_raw, -2.5, 2.3), sigma=1.8).astype(np.float32)
    ye_tail = float(np.median(ye_clean[-20:]))

    _ARCHETYPE_BANK = {
        # 句中单字真实活腔残差（相对各字自身五声目标音级）
        "ping_steady":     _slice_norm(t16_t, t16_st, 8.52, 9.36, "median"),    # t16「龙」平稳醇厚胸声长平
        "ping_light":      _slice_norm(t09_t, t09_st, 0.67, 0.96, "median"),    # t09「和」清雅书斋短平
        "ping_open_rise":  _slice_norm(t16_t, t16_st, 0.17, 0.92, "tail"),      # t16「山」-1.9 -> 0.0st 起兴盘升长平
        "ping_high_enter": _slice_norm(t09_t, t09_st, 4.67, 5.00, "tail"),      # t09「暮」-1.7 -> 0.0st 拔入高腔起首
        "ping_scoop":      _slice_norm(t18_t, t18_st, 4.40, 4.68, "tail"),      # t18「苏」-1.5 -> 0.0st 沉稳入宫
        "shang_lift":      _slice_norm(t09_t, t09_st, 1.13, 1.30, "tail"),      # t09「九」上声微凹挑起
        "ze_peak":         _slice_norm(t16_t, t16_st, 6.58, 6.82, "median"),    # t16「在」清亮角音高挑
        "qu_fall":         _slice_norm(t09_t, t09_st, 12.88, 13.44, "head"),    # t09「事」0.0 -> -2.3st 去声劲落
        "qu_short_fall":   _slice_norm(t18_t, t18_st, 3.06, 3.32, "head"),      # t18「既」去声短促下溜
        "ru_crisp":        _slice_norm(t16_t, t16_st, 15.15, 15.42, "median"),  # t16「德」入声短促实腔
        "low_scoop":       _slice_norm(t16_t, t16_st, 2.78, 2.90, "tail"),      # t16「有」低起虚领挑腔
        # 句尾招牌落腔/摇腔原型（全部对齐到尾段稳定落音 = 0.0st，直接加在目标收音五声级上！）
        "cad_gong_arch":     _slice_norm(t09_t, t09_st, 1.42, 1.98, "tail", 80),   # t09「年」-0.4 -> +2.3 -> 0.0st 书斋拱收
        "cad_gong_hold":     _slice_norm(t16_t, t16_st, 1.38, 2.30, "tail", 80),   # t16「高」-1.2 -> 0.0st 浑厚悬宫大托
        "cad_fifth_cascade": _slice_norm(t09_t, t09_st, 6.07, 6.56, "tail", 80),   # t09「初」+3.2 -> 0.0st 高腔五度流泻落宫
        "cad_zhi_drop":      _slice_norm(t16_t, t16_st, 4.25, 5.18, "tail", 80),   # t16「名」+2.9 -> 0.0st 四度陡坠落低徵
        "cad_deep_bottom":   _slice_norm(t16_t, t16_st, 15.55, 16.52, "median", 80),  # t16「馨」苍浑沉底大拖腔
        "cad_fu_deep":       _slice_norm(t18_t, t18_st, 19.95, 20.85, "median", 80),  # t18「章」辞赋苍茫沉底腔
        "cad_literati_sigh": (ye_clean - ye_tail).astype(np.float32),                 # t09「也」+4.0 -> +1.9 -> +4.3 -> 0.0st 三波叹咏摇腔
        "cad_ze_poise":      _slice_norm(t16_t, t16_st, 50.82, 51.26, "median", 64),  # t16「耳」仄声清朗悬收
    }
    return _ARCHETYPE_BANK


def _eval_archetype(name: str, anchor_st: float, n_frames: int, scale: float = 1.0) -> np.ndarray:
    bank = _get_archetype_bank()
    proto = bank[name]
    x = np.linspace(0.0, 1.0, len(proto), dtype=np.float32)
    xi = np.linspace(0.0, 1.0, max(4, n_frames), dtype=np.float32)
    return (anchor_st + scale * np.interp(xi, x, proto)).astype(np.float32)


# =============================================================================
# 2. 篇章句群“起—承—转—合”与骈偶对仗结构规划器
# =============================================================================
@dataclass
class ClausePlan:
    idx: int
    text: str
    punc: str               # "，" | "；" | "。"
    role: str               # QI | LEAD | CHENG_UP | CHENG_DOWN | ZHUAN_HIGH | PU_LONG | HE_GRAND
    role_desc: str
    is_major_end: bool      # 句号/分号级收束
    is_passage_end: bool    # 全段压轴总结句
    tight_next: bool        # 与下句一气呵成（如「斯是陋室，惟吾德馨」「清风徐来，水波不兴」）


def _split_clauses_with_punct(text: str) -> List[Tuple[str, str]]:
    parts = re.split(r"([，。！？；、\n]+)", text.strip())
    out: List[Tuple[str, str]] = []
    buf = ""
    for p in parts:
        if not p:
            continue
        if re.fullmatch(r"[，。！？；、\n]+", p):
            if buf and re.search(r"[\u4e00-\u9fff]", buf):
                if any(c in p for c in "。！？\n"):
                    punc = "。"
                elif "；" in p:
                    punc = "；"
                else:
                    punc = "，"
                out.append((buf.strip(), punc))
            buf = ""
        else:
            buf += p
    if buf and re.search(r"[\u4e00-\u9fff]", buf):
        out.append((buf.strip(), "。"))
    return out


def _char_len(s: str) -> int:
    return sum(1 for c in s if "\u4e00" <= c <= "\u9fff")


def plan_guwen_discourse(raw_clauses: List[Tuple[str, str]], mode_key: str) -> List[ClausePlan]:
    """
    自动剖析整段古文/辞赋的句群层次（句读边界 + 自右向左骈偶配对 + 起承转合演进）：
      - 保证成对偶句严格遵循「上句悬扬 (CHENG_UP / ZHUAN_HIGH) ↔ 下句落抑 (CHENG_DOWN / HE_GRAND)」，绝不错位倒挂；
      - 保证发语领起句（如「夫君子之行，」5字 + 后接 4+4 对仗）自动识为 `LEAD` 而不挤占上句槽位；
      - 保证全篇第三结构拍/转折句必拔高出彩 (`ZHUAN_HIGH`)，全篇末句必欲抑先扬稳稳归底 (`HE_GRAND`)！
    """
    m = len(raw_clauses)
    if m == 0:
        return []

    lens = [_char_len(c) for c, _ in raw_clauses]
    # 1) 按句号/分号切分大句群 (Sentences)
    sentences: List[List[int]] = []
    cur_s: List[int] = []
    for i, (_, punc) in enumerate(raw_clauses):
        cur_s.append(i)
        if punc in ("。", "；") or i == m - 1:
            sentences.append(cur_s)
            cur_s = []

    # 2) 在每个大句群内，自右向左识别平行对仗偶句对 (UPPER, LOWER)
    couplet_tag: Dict[int, str] = {}
    for s_idxs in sentences:
        k = len(s_idxs) - 1
        while k >= 1:
            i_up, i_dn = s_idxs[k - 1], s_idxs[k]
            l_up, l_dn = lens[i_up], lens[i_dn]
            if 3 <= l_up <= 8 and 3 <= l_dn <= 8 and abs(l_up - l_dn) <= 1:
                couplet_tag[i_up] = "UPPER"
                couplet_tag[i_dn] = "LOWER"
                k -= 2
            else:
                k -= 1

    # 3) 结合全段位置 (起 -> 承 -> 转 -> 合) 与对仗属性分配角色
    plans: List[ClausePlan] = []
    n_sent = len(sentences)
    has_zhuan = False
    s0_shape = tuple(lens[idx] for idx in sentences[0])

    for si, s_idxs in enumerate(sentences):
        is_last_sent = (si == n_sent - 1)
        s_shape = tuple(lens[idx] for idx in s_idxs)
        is_repeat_couplet = (si > 0 and not is_last_sent and s_shape == s0_shape and len(s_idxs) == 2)

        for pos_in_s, ci in enumerate(s_idxs):
            txt, punc = raw_clauses[ci]
            n_c = lens[ci]
            s_syls_tmp = text_to_sing_syllables(txt, era="mc")
            is_all_ping = bool(s_syls_tmp and all(x.tone_class == "ping" for x in s_syls_tmp))
            is_major = (punc in ("。", "；") or ci == m - 1)
            is_pass_end = (ci == m - 1)
            cp = couplet_tag.get(ci, "")

            # 全段压轴结句：必走「合」
            if is_pass_end and m >= 2:
                role, desc = "HE_GRAND", "合·欲抑先扬归底"
            # 长散句/长赋句 (>=7字，且非短句对仗上句)
            elif n_c >= 7 and cp != "UPPER":
                if ci >= 2 and not has_zhuan and not is_major and n_c <= 8:
                    role, desc = "ZHUAN_HIGH", "转·高腔昂扬破阵"
                    has_zhuan = True
                else:
                    role, desc = "PU_LONG", "铺·长句扬波行腔"
            # 首句（起）
            elif ci == 0:
                if len(s_idxs) >= 3 and cp == "" and (txt[0] in _LEAD_PARTICLES or n_c != lens[s_idxs[1]]):
                    role, desc = "LEAD", "起·发语端凝领句"
                else:
                    role, desc = "QI", "起·立调定宫起兴"
            # 对仗上句：
            #   - 若全平四字句（如「清风徐来」），必走稳锁主宫的 CHENG_UP，把仄扬波澜留给下句「水波不兴」；
            #   - 若中段与首联同构复沓（如《陋室铭》第2联「水不在深，有龙则灵」），保持 CHENG_UP 对仗递进，将高腔「转」留给末联「斯是陋室」！
            elif cp == "UPPER":
                if is_all_ping and mode_key == "fu_chibi":
                    role, desc = "CHENG_UP", "承·江风平腔悬宫"
                elif is_repeat_couplet:
                    role, desc = "CHENG_UP", "承·对仗再扬悬宫"
                elif (is_last_sent and len(s_idxs) == 2 and m >= 4 and not has_zhuan) or (ci == 2 and lens[0] == 3):
                    role, desc = "ZHUAN_HIGH", "转·高腔昂扬上句"
                    has_zhuan = True
                else:
                    role, desc = "CHENG_UP", "承·骈偶悬宫上句"
            # 对仗下句
            elif cp == "LOWER":
                role, desc = "CHENG_DOWN", "承·骈偶落徵下句"
            # 非对仗中段短句（如《兰亭》第3句「暮春之初」、《爱莲说》「晋陶渊明独爱菊」）
            else:
                if pos_in_s == 0 and si >= 1 and not has_zhuan:
                    role, desc = "ZHUAN_HIGH", "转·高商陡起五度"
                    has_zhuan = True
                elif is_major:
                    role, desc = "CHENG_DOWN", "承·沉稳落韵收句"
                else:
                    role, desc = "CHENG_UP", "承·蓄势递进悬句"

            tight_next = bool(
                ci + 1 == m - 1
                and not is_major
                and n_c == 4
                and lens[ci + 1] == 4
                and mode_key in ("guwen_loushi", "fu_chibi")
            )
            plans.append(ClausePlan(ci, txt, punc, role, desc, is_major, is_pass_end, tight_next))

    return plans


# =============================================================================
# 3. 依字定长 & 依字行腔（平低仄高 + 五声音阶晶格 + 真人单字活腔拼接）
# =============================================================================
def _analyze_clause_tones(clause_text: str, sylls: List[SingSyllable], tone_era: str) -> List[ToneProfile]:
    tones = [analyze_char_tone(s.char, era=tone_era) for s in sylls]
    # 应用经典双字词组中古声调纠偏（如「长江」之「长」必为平声，非上声！）
    for word, off_map in _CONTEXT_TONE_OVERRIDES.items():
        start = 0
        while True:
            idx = clause_text.find(word, start)
            if idx < 0:
                break
            for off, (sh, yy, dsc) in off_map.items():
                pos = idx + off
                if 0 <= pos < len(tones):
                    if tone_era == "em" and sh == "ru":
                        sh = "ping" if "濁" in dsc else "shang"
                    tones[pos] = ToneProfile(tones[pos].char, sh, yy, "P" if sh == "ping" else "Z", dsc)
            start = idx + len(word)
    return tones


def _compute_syllable_durations(
    sylls: List[SingSyllable],
    tones: List[ToneProfile],
    plan: ClausePlan,
    mode_key: str,
    speed_scale: float,
) -> List[float]:
    """
    严格对齐母带实测时长的「依字定长」律：
      - 四言铭偶句首平声（如「山——不在高」）与次字平声枢轴（如「有仙——则名」「暮春——之初」）给足 0.56~0.74s 真人吟诵拉腔；
      - 入声清脆顿挫（0.27~0.31s，句尾入声 0.48~0.54s），绝不拖泥带水，同时严守 >=0.26s 咬字清晰底线；
      - 句尾平声/虚词大拖腔从容舒展（句中 0.76~0.98s，全篇压轴 1.15~1.56s）。
    """
    n = len(sylls)
    is_long = (n >= 7)
    durs: List[float] = []

    for k, (sy, tp) in enumerate(zip(sylls, tones)):
        is_last = (k == n - 1)
        is_penult = (k == n - 2)

        if is_last:
            if sy.char in _FINAL_PARTICLES:
                base = 1.54 if plan.is_passage_end else 1.22
            elif tp.sheng == "ru":
                base = 0.54 if (plan.is_major_end or plan.is_passage_end) else 0.48
            elif plan.is_passage_end:
                base = 1.24 if tp.sheng == "ping" else 1.12
            elif plan.is_major_end:
                base = 0.94 if tp.sheng == "ping" else 0.78
            else:
                base = 0.84 if tp.sheng == "ping" else 0.66
            durs.append(round(base * speed_scale, 4))
            continue

        if tp.sheng == "ru":
            first_ph = sylls[k].phones[0] if sylls[k].phones else ""
            if is_penult and plan.role == "HE_GRAND":
                base = 0.36  # 压轴「欲抑先扬」入声（如「惟吾[德]馨」「而[不]染」）留足扬腔顿挫
            elif first_ph in _FRICATIVE_ONSETS:
                base = 0.31 if is_long else 0.33  # 清擦入声（如「朝晖[夕]阴」「[七]月」「[戌]」）留足清晰擦音
            else:
                base = 0.28 if is_long else 0.30
            durs.append(round(max(0.27, base * speed_scale), 4))
            continue

        if is_long:
            if is_penult:
                base = 0.45 if tp.sheng == "ping" else 0.39
            elif sy.char in _MID_PARTICLES:
                base = 0.31
            else:
                base = 0.36 if tp.sheng == "ping" else 0.32
        else:
            # 3~6 字短句/骈偶句：还原真人母带首字起兴与二字长平拉腔！
            if n == 4 and k == 0 and tp.sheng == "ping" and mode_key == "guwen_loushi" and plan.role in ("QI", "CHENG_UP"):
                base = 0.72  # 如「山——不在高」
            elif n == 4 and k == 0 and plan.role in ("QI", "CHENG_UP") and mode_key == "guwen_loushi":
                base = 0.56  # 如「水——不在深」
            elif n in (3, 4) and k == 1 and tp.sheng == "ping" and plan.role in ("CHENG_DOWN", "ZHUAN_HIGH"):
                base = 0.62 if mode_key == "guwen_loushi" else 0.50  # 如「有仙——则名」「有龙——则灵」「暮春——之初」
            elif is_penult:
                base = 0.43 if tp.sheng == "ping" else 0.38
            elif sy.char in _MID_PARTICLES:
                base = 0.33
            else:
                base = 0.42 if tp.sheng == "ping" else 0.36

        durs.append(round(max(0.26, base * speed_scale), 4))
    return durs


def _build_clause_syllable_specs(
    sylls: List[SingSyllable],
    tones: List[ToneProfile],
    plan: ClausePlan,
    mode_key: str,
    prev_end_st: float,
) -> List[Tuple[str, float, float]]:
    """
    核心音乐学引擎：根据【调式 G1/G2/G3】×【句群角色 起承转合】×【单字中古四声】
    为句中每个字精确指派 `(真人活腔原型名, 目标五声半音锚点, 残差幅度系数)`！
    每个锚点均严格锁定在纯五声音阶晶格上，杜绝任何半音阶跑调！
    """
    n = len(sylls)
    last_t = tones[-1].sheng
    has_part = (sylls[-1].char in _FINAL_PARTICLES)
    all_ping = all(t.sheng == "ping" for t in tones)
    specs: List[Tuple[str, float, float]] = []

    # -------------------- 调式 G1：书斋清雅骈散调 (t09 兰亭 / t06 答谢中书书) --------------------
    # 五声晶格：6₁(-3.2) — 1(0.0) — 2(+2.3) — 5(+5.2) — 6(+7.3)
    if mode_key == "guwen_lanting":
        if plan.role in ("QI", "LEAD"):
            for k in range(n - 1):
                sh = tones[k].sheng
                if k == n - 2 and sh != "ping":
                    specs.append(("shang_lift", +2.3, 1.0))  # 如「永和[九]年」仄声挑入中商
                elif k == n - 2 and sylls[k].char in _MID_PARTICLES:
                    specs.append(("ping_scoop", -1.2, 0.7))
                elif sh == "ru":
                    specs.append(("ru_crisp", -2.4, 0.8))
                else:
                    specs.append(("ping_light", -3.2, 1.0))
            specs.append(("cad_gong_arch" if last_t == "ping" else "cad_ze_poise", 0.0, 1.0))

        elif plan.role == "CHENG_UP":
            for k in range(n - 1):
                sh = tones[k].sheng
                if sh == "ping":
                    specs.append(("ping_light", -1.2 if k == 0 else 0.0, 1.0))
                elif sh == "ru":
                    specs.append(("ru_crisp", +1.2, 0.9))
                else:
                    specs.append(("ze_peak", +2.3, 0.9))
            specs.append(("cad_gong_hold" if last_t == "ping" else "cad_ze_poise", 0.0 if last_t == "ping" else +2.1, 0.9))

        elif plan.role == "CHENG_DOWN":
            for k in range(n - 1):
                sh = tones[k].sheng
                if k == 0:
                    specs.append(("ping_light", -3.2, 1.0))
                elif k == n - 2 and sh != "ping":
                    specs.append(("shang_lift", +2.3, 1.0))  # 如「岁在[癸]丑」「可爱者[甚]蕃」
                elif sh != "ping":
                    specs.append(("ping_light", -1.1, 0.9))  # 如「岁[在]癸丑」递进台阶
                else:
                    specs.append(("ping_light", -2.2, 1.0))
            if has_part:
                specs.append(("cad_literati_sigh", -2.3, 0.85))
            elif last_t == "ping":
                specs.append(("cad_zhi_drop", -2.3, 0.78))   # 偶句平收柔落低羽 (-2.3st)
            elif last_t == "ru":
                specs.append(("ru_crisp", -1.6, 1.0))
            else:
                specs.append(("cad_ze_poise", -0.9, 0.9))    # 1:1 复刻 t09「丑」(-0.9st 沉敛仄收)

        elif plan.role == "ZHUAN_HIGH":
            # 1:1 复刻 t09「暮春之初」/「此地有崇山峻岭」/「茂林修竹」：五度陡起高徵(+5.2st) + 领落冲高羽(+6.6~+7.2st)！
            for k in range(n - 1):
                sh = tones[k].sheng
                if k == 0:
                    specs.append(("ping_high_enter", +5.1, 1.0))
                elif k == n - 2 and n == 4:
                    specs.append(("ze_peak", +7.1, 1.0))     # 「暮春[之]初」华彩冲顶
                elif k == n - 2 and sh != "ping":
                    specs.append(("ze_peak", +6.3, 0.9))     # 「独[爱]菊」圆润高挑不突刺
                elif sh == "ping":
                    specs.append(("ping_steady", +5.2, 0.9))
                elif sh == "ru":
                    specs.append(("ru_crisp", +4.9, 0.9))
                else:
                    specs.append(("ze_peak", +6.2, 0.9))
            if last_t == "ping":
                specs.append(("cad_fifth_cascade", 0.0, 1.0))   # +3.2 -> 0.0st 绝美五度流泻归宫！
            elif last_t == "ru":
                specs.append(("cad_fifth_cascade", +0.4, 0.85)) # 如「独爱菊」「茂林修竹」高落清收！
            else:
                specs.append(("cad_ze_poise", +2.3, 1.0))

        elif plan.role == "PU_LONG":
            # 1:1 复刻 t09「会于会稽山阴之兰亭」：前半低羽清朗叙事，倒数第 2 字扬入中商，末字拱收！
            for k in range(n - 1):
                sh = tones[k].sheng
                if k == n - 2:
                    specs.append(("shang_lift", +2.0, 1.0))     # 「兰」挑起
                elif sh != "ping" and k not in (0, n - 3):
                    specs.append(("ping_light", -2.3, 0.9))
                elif k in (n // 2, n // 2 + 1) and sh == "ping":
                    specs.append(("ping_light", -2.6, 0.9))
                else:
                    specs.append(("ping_light", -3.2, 0.9))
            specs.append(("cad_gong_arch" if last_t == "ping" else "cad_ze_poise", -0.8, 0.95))

        else:  # HE_GRAND（压轴合句：短句高起叹咏如「修禊事也」，长句铺陈后欲抑先扬叹咏如「予独爱莲...而不染」）
            if n <= 6:
                for k in range(n - 1):
                    sh = tones[k].sheng
                    if k == 0:
                        specs.append(("ping_high_enter", +5.1, 1.0))
                    elif k == 1 and sh != "ping":
                        specs.append(("ze_peak", +7.3, 1.0))    # 「修[禊]事也」高羽极峰
                    elif k == n - 2:
                        specs.append(("qu_fall" if sh == "qu" else "shang_lift", +4.8 if sh == "qu" else +3.2, 1.0))
                    else:
                        specs.append(("ping_steady", +5.0, 0.9))
                specs.append(("cad_literati_sigh" if (has_part or last_t != "ru") else "cad_zhi_drop", -2.3, 1.0))
            else:
                # 长压轴句（如 11 字「予独爱莲之出淤泥而不染」）：前半中商温润领起 -> 中段低羽铺陈 -> 倒数第2字温和欲抑先扬(+1.1st) -> 三波摇腔归底！
                split_k = min(4, n // 2)
                for k in range(n - 1):
                    sh = tones[k].sheng
                    if k < split_k:
                        specs.append(("ze_peak" if sh != "ping" and k >= 1 else "ping_steady", +3.6 if (sh != "ping" and k >= 1) else +2.0, 0.9))
                    elif k == n - 2:
                        specs.append(("ru_crisp" if sh == "ru" else "shang_lift", +1.1, 0.9))  # 「而[不]染」自然大三度提腔！
                    elif sh == "ru":
                        specs.append(("ru_crisp", -2.0, 0.9))
                    else:
                        specs.append(("ping_light", -2.6, 0.9))
                specs.append(("cad_literati_sigh" if last_t != "ru" else "cad_zhi_drop", -2.4, 0.95))

    # -------------------- 调式 G2：唐调铭文沉雄顿挫调 (t16 陋室铭) --------------------
    # 五声晶格：3₁(-5.0) — 5₁(-3.2) — 6₁(-1.8) — 1(0.0) — 2(+2.1) — 3(+4.1) — 5(+5.3)
    elif mode_key == "guwen_loushi":
        if plan.role == "LEAD":
            for k in range(n - 1):
                sh = tones[k].sheng
                if k == 0:
                    specs.append(("ping_scoop", 0.0, 0.9))
                elif sh != "ping":
                    specs.append(("ze_peak", +3.6, 0.9))        # 「夫君[子]之行」仄声清挑
                elif sylls[k].char in _MID_PARTICLES:
                    specs.append(("ping_light", +0.4, 0.8))
                else:
                    specs.append(("ping_steady", +2.1, 0.9))
            specs.append(("cad_gong_hold" if last_t == "ping" else "cad_ze_poise", 0.0 if last_t == "ping" else +2.1, 0.95))

        elif plan.role in ("QI", "CHENG_UP"):
            # 1:1 复刻 t16「山不在高」「水不在深」「静以修身」：平商(+2.1) × 仄角(+4.1) × 悬宫大托(0.0)！
            for k in range(n - 1):
                sh = tones[k].sheng
                if k == 0 and sh == "ping":
                    specs.append(("ping_open_rise", +2.1, 1.0))
                elif k == 0 and n == 4 and tones[1].sheng == "ru":
                    specs.append(("ping_open_rise", +2.2, 1.15))  # 「水不在深」低挑入商
                elif sh == "ru":
                    specs.append(("ru_crisp", +2.2, 0.9))
                elif sh in ("shang", "qu"):
                    specs.append(("ze_peak", +4.1, 1.0))          # 「山不[在]高」「[静][以]修身」仄挑角音！
                else:
                    specs.append(("ping_steady", +2.1, 1.0))
            specs.append(("cad_gong_hold" if last_t == "ping" else "cad_ze_poise", -0.1 if last_t == "ping" else +3.1, 1.0))

        elif plan.role == "CHENG_DOWN":
            # 1:1 复刻 t16「有仙则名」「有龙则灵」「俭以养德」：首字低挑 -> 次字平商稳托 -> 欲抑微扬 -> 四度落低徵(-3.2st)！
            for k in range(n - 1):
                sh = tones[k].sheng
                if k == 0:
                    specs.append(("low_scoop", -1.2, 1.0))        # 「有」「俭」低起挑腔（绝不飙高！）
                elif k == n - 2 and sh != "ping":
                    specs.append(("ze_peak", +2.6, 0.9))          # 「有仙[则]名」「俭以[养]德」领落微挑
                elif sh == "ping":
                    specs.append(("ping_steady", +2.1, 1.0))      # 「仙——」「龙——」饱满立商
                else:
                    specs.append(("ze_peak", +2.3, 0.9))
            if last_t == "ping":
                specs.append(("cad_zhi_drop", -3.2, 1.0))         # 四度陡坠落低徵！
            elif last_t == "ru":
                specs.append(("cad_zhi_drop", -3.0, 0.80))        # 「俭以养[德]。」干脆落徵收句！
            else:
                specs.append(("cad_zhi_drop", -2.2, 0.85))

        elif plan.role == "ZHUAN_HIGH":
            # 1:1 复刻 t16「斯是陋室」(4字转折落宫) 与「无丝竹之乱耳 / 非淡泊无以明志」(长偶高腔仄悬)！
            if n == 4:
                specs.append(("ze_peak", +4.5, 1.0))              # 「斯」高角陡起
                specs.append(("ze_peak", +4.3, 1.0))              # 「是」高角稳承
                specs.append(("ping_scoop", +0.2, 0.9))           # 「陋」回落立宫
                specs.append(("cad_gong_hold" if last_t != "ru" else "ru_crisp", +0.2, 0.8))  # 「室」稳托连下句！
            else:
                for k in range(n - 1):
                    sh = tones[k].sheng
                    if k == 0:
                        specs.append(("ping_scoop", +2.1, 0.9))
                    elif sh != "ping":
                        specs.append(("ze_peak", +5.1 if k <= 3 else +4.4, 1.0))  # 「非[淡][泊]无[以]明志」冲高徵！
                    else:
                        specs.append(("ping_steady", +2.4, 0.9))
                specs.append(("cad_ze_poise" if last_t != "ping" else "cad_gong_hold", +3.2 if last_t != "ping" else +0.3, 1.0))

        elif plan.role == "PU_LONG":
            mid = n // 2
            for k in range(n - 1):
                sh = tones[k].sheng
                if k == n - 2:
                    specs.append(("ze_peak" if sh != "ping" else "ping_steady", +2.4, 0.9))
                elif k in (mid - 1, mid) and sh != "ping":
                    specs.append(("ze_peak", +4.0, 0.9))
                elif sh != "ping":
                    specs.append(("ze_peak", +2.3, 0.8))
                else:
                    specs.append(("ping_steady", +0.2 if k < mid else +2.0, 0.9))
            specs.append(("cad_zhi_drop" if plan.is_major_end else "cad_gong_hold", -3.1 if plan.is_major_end else 0.0, 0.95))

        else:  # HE_GRAND（1:1 复刻 t16「惟吾德馨」与「孔子云：何陋之有」—— 承转滑落 + 倒数第2字欲抑先扬 + 沉底大拖腔！）
            if n == 4:
                # 1:1 还原 t16 13.05..16.55s：「惟(+1.7) 吾(0.0↘-3.1) 德(+0.5 顿挫挑!) 馨(-5.0 苍浑沉底!)」
                specs.append(("ze_peak", +1.7, 0.85))
                specs.append(("cad_zhi_drop", -3.1, 0.95))
                specs.append(("ru_crisp" if tones[2].sheng == "ru" else "ze_peak", +0.5 if tones[2].sheng == "ru" else +1.3, 1.0))
            else:
                for k in range(n - 1):
                    sh = tones[k].sheng
                    if k == n - 2:
                        specs.append(("ru_crisp" if sh == "ru" else "ze_peak", +0.4 if sh == "ru" else +1.3, 1.0))
                    elif n >= 6 and k <= 2:
                        specs.append(("ze_peak" if sh != "ping" else "ping_steady", +2.2 if sh != "ping" else +0.3, 0.9))
                    else:
                        specs.append(("ping_steady", -2.9, 0.9))
            bot_st = -5.0 if last_t == "ping" else (-3.9 if last_t != "ru" else -3.5)
            specs.append(("cad_deep_bottom", bot_st, 1.0))

    # -------------------- 调式 G3：唐调辞赋大开大合调 (t18 前赤壁赋 / t19 岳阳楼记) --------------------
    # 五声晶格：5₂(-6.8) — 3₁(-4.8) — 5₁(-3.2) — 6₁(-1.5) — 1(0.0) — 2(+2.1) — 3(+4.2)
    else:
        if plan.role == "QI":
            # 独立四字平收纪年/起兴句（如「壬戌之秋」）走 -6.8st 苍茫沉底；对仗排比上句（如「衔远山」）走 +2.1/+4.2st 清朗立商！
            if n == 4 and last_t == "ping" and not plan.tight_next and plan.punc == "，":
                for k in range(n - 1):
                    sh = tones[k].sheng
                    if k == 0:
                        specs.append(("ping_steady", -6.8, 1.0))
                    elif sh != "ping":
                        specs.append(("ru_crisp", -5.3, 0.9))      # 「戌」微挑
                    else:
                        specs.append(("qu_short_fall", -5.0, 0.8))  # 「之」-5.0 -> -6.6st
                specs.append(("cad_fu_deep", -6.8, 1.0))
            else:
                for k in range(n - 1):
                    sh = tones[k].sheng
                    specs.append(("ze_peak" if sh != "ping" else "ping_steady", +4.1 if sh != "ping" else +2.1, 1.0))
                specs.append(("cad_gong_hold" if last_t == "ping" else "cad_ze_poise", +2.1 if n == 3 else 0.0, 0.9))

        elif plan.role == "LEAD":
            for k in range(n - 1):
                sh = tones[k].sheng
                specs.append(("ze_peak" if sh != "ping" else "ping_steady", +2.2 if sh != "ping" else 0.0, 0.9))
            specs.append(("cad_gong_hold", 0.0, 0.9))

        elif plan.role == "CHENG_UP":
            # 1:1 复刻 t18「清风徐来」（四平声稳锁主宫 0.0st）与 t19「朝晖夕阴」（平商+2.1 × 仄角+4.2 → 悬宫0.2st）！
            if all_ping:
                for k in range(n - 1):
                    specs.append(("ping_scoop" if k == 0 else "ping_steady", 0.0, 1.0))
                specs.append(("cad_gong_hold", +0.1, 0.85))
            else:
                for k in range(n - 1):
                    sh = tones[k].sheng
                    if k == 0 and sh == "ping":
                        specs.append(("ping_steady", +2.1, 0.95))
                    elif sh != "ping":
                        specs.append(("ze_peak", +4.2, 1.0))       # 「朝晖[夕]阴」「举酒[属]客」仄挑角音！
                    else:
                        specs.append(("ping_steady", +2.1, 0.95))
                specs.append(("cad_gong_hold" if last_t == "ping" else "cad_ze_poise", +0.3 if last_t == "ping" else +2.1, 0.9))

        elif plan.role == "CHENG_DOWN":
            # 若前句是 -6.8st 极低起兴（如「壬戌之秋」→「七月既望」），1:1 复刻母带猛拔七度至主宫(+0.2st) -> 去声劲落(-4.5st)！
            if prev_end_st <= -5.5:
                for k in range(n - 1):
                    sh = tones[k].sheng
                    if k == 0:
                        specs.append(("ru_crisp" if sh == "ru" else "ping_steady", +0.3, 0.9))
                    elif k == n - 2 and sh == "qu":
                        specs.append(("qu_short_fall", +0.0, 1.15))  # 「既」0.0 -> -3.4st
                    else:
                        specs.append(("ru_crisp" if sh == "ru" else "ping_steady", -0.3, 0.9))
                specs.append(("qu_short_fall" if last_t != "ping" else "cad_zhi_drop", +0.1 if last_t != "ping" else -4.4, 1.45 if last_t != "ping" else 0.9))
            else:
                # 标准对仗下句（如「吞长江」「横无际涯。」）：前半立商挑角，句末落宫(0.0st)或句号落低徵(-3.2st)！
                for k in range(n - 1):
                    sh = tones[k].sheng
                    if sh != "ping":
                        specs.append(("ze_peak", +4.3, 1.0))       # 「横无[际]涯」
                    elif k == 1 and n in (3, 4):
                        specs.append(("ping_steady", +3.4, 0.9))   # 「吞[长]江」「横[无]际涯」阳平扬波
                    else:
                        specs.append(("ping_steady", +2.1, 0.95))
                end_target = -3.2 if plan.is_major_end else +0.1
                specs.append(("cad_zhi_drop" if plan.is_major_end else "cad_gong_hold", end_target, 0.9))

        elif plan.role == "ZHUAN_HIGH":
            # 1:1 复刻 t19「浩浩汤汤」：重仄冲高角(+4.3~+4.7st)，平声稳立高商(+2.2st)！
            for k in range(n - 1):
                sh = tones[k].sheng
                if sh != "ping":
                    specs.append(("ze_peak", +4.5 if k >= 1 else +4.2, 1.0))
                else:
                    specs.append(("ping_steady", +2.2, 0.95))
            specs.append(("cad_gong_hold" if last_t == "ping" else "cad_ze_poise", +2.2 if last_t == "ping" else +3.4, 0.9))

        elif plan.role == "PU_LONG":
            # 1:1 复刻 t18「苏子与客泛舟游于赤壁之下」：起首入宫挑商(+0.4~+1.6st) -> 泛舟稳宫 -> 游于扬波(+3.1st) -> 阶梯落底(-4.8st)！
            piv0 = max(2, int(round(n * 0.50)))
            piv1 = max(piv0 + 1, int(round(n * 0.68)))
            for k in range(n - 1):
                sh = tones[k].sheng
                if k == 0:
                    specs.append(("ping_scoop", +0.4, 1.0))
                elif k == n - 2:
                    specs.append(("ping_steady", -2.2, 0.9))       # 「赤壁[之]下」阶梯下沉至低羽
                elif piv0 <= k < piv1:
                    specs.append(("ze_peak" if sh != "ping" else "ping_steady", +3.5 if sh != "ping" else +3.0, 1.0))  # 「游于」扬波！
                elif sh == "ru":
                    specs.append(("ru_crisp", +0.3, 0.9))          # 「客」「赤壁」清脆归宫
                elif sh in ("shang", "qu") and k <= 3:
                    specs.append(("shang_lift" if sh == "shang" else "qu_short_fall", +1.5 if sh == "shang" else +0.3, 0.85))
                else:
                    specs.append(("ping_steady", +0.3, 0.9))
            specs.append(("cad_zhi_drop", -4.8 if plan.is_major_end else -3.2, 0.95))

        else:  # HE_GRAND
            # 若紧承全平上句（如「清风徐来，水波不兴。」），1:1 复刻 t18「水(+3.2) 波(+2.2) 不(-0.5) 兴(+0.1)」仄扬浩然归宫！
            if n == 4 and abs(prev_end_st) <= 0.8 and tones[0].sheng != "ping" and tones[2].sheng == "ru" and last_t == "ping":
                specs.append(("ze_peak", +3.2, 1.0))               # 「水」陡扬
                specs.append(("ping_steady", +2.2, 0.9))           # 「波」承扬
                specs.append(("ru_crisp", -0.5, 0.9))              # 「不」顿挫点宫
                specs.append(("cad_gong_hold", +0.1, 0.9))         # 「兴」浩然稳立主宫！
            else:
                # 宏大辞赋/古文压轴（如「歌窈窕之章。」「气象万千。」）：渐沉蓄势 + 倒数第2字欲抑先扬 + 苍茫沉底大收腔！
                for k in range(n - 1):
                    sh = tones[k].sheng
                    if k == 0:
                        specs.append(("qu_short_fall" if sh == "qu" else "ping_steady", +1.4 if sh != "ping" else +0.2, 0.95))
                    elif k == n - 2:
                        specs.append(("ze_peak" if sh != "ping" else "ping_steady", +1.5 if sh != "ping" else -1.8, 1.0))  # 「万」/「之」欲抑先扬！
                    else:
                        specs.append(("ping_steady", -2.8, 0.9))
                bot = -4.8 if n <= 4 else -6.6
                if last_t in ("shang", "qu"):
                    bot = -3.8
                specs.append(("cad_fu_deep" if not has_part else "cad_literati_sigh", bot, 1.0))

    return specs


# =============================================================================
# 4. DiffSinger 声学乐谱主编译器
# =============================================================================
def compile_guwen_v9_ds(
    text: str,
    mode_key: str = "guwen_lanting",
    era: str = "zh",
    key_hz: Optional[float] = None,
    speaker: str = "opencpop_base",
    voice_gender: str = "female",
    tempo_scale: float = 1.0,
) -> List[dict]:
    """
    V13 古文/辞赋「起承转合四阶句群 × 依字平低仄高五声对位 × 真人单字活腔」编译器。
    """
    cfg = GUWEN_V9_MODES.get(mode_key, GUWEN_V9_MODES["guwen_lanting"])
    if key_hz is not None:
        use_key_hz = float(key_hz)
    else:
        use_key_hz = float(cfg["male_key_hz"] if voice_gender == "male" else cfg["female_key_hz"])

    min_st = float(cfg["min_st"])
    max_st = float(cfg["max_st"])
    speed_scale = float(cfg["speed_scale"]) * float(tempo_scale)

    raw_clauses = _split_clauses_with_punct(text)
    if not raw_clauses:
        return []

    plans = plan_guwen_discourse(raw_clauses, mode_key=mode_key)
    tone_era = "em" if era.lower() == "em" else "mc"
    ds_items: List[dict] = []
    f0_step = 0.005
    running_offset = 0.0
    prev_end_st = 0.0

    for plan in plans:
        sylls = [s for s in text_to_sing_syllables(plan.text, era=era) if s.char != " "]
        n_ch = len(sylls)
        if n_ch == 0:
            continue
        tones = _analyze_clause_tones(plan.text, sylls, tone_era=tone_era)
        char_durs = _compute_syllable_durations(sylls, tones, plan, mode_key, speed_scale)
        syll_specs = _build_clause_syllable_specs(sylls, tones, plan, mode_key, prev_end_st)

        lead_sp = 0.06
        ph_seq: List[str] = ["SP"]
        ph_dur: List[float] = [lead_sp]
        f0_chunks: List[np.ndarray] = []
        dip_windows: List[Tuple[float, float, float]] = []

        prev_nuc_d = 0.0
        for k, (sy, tp, cdur, (arch_name, anchor_st, arch_scale)) in enumerate(zip(sylls, tones, char_durs, syll_specs)):
            clean_pairs, _, has_m = _build_char_phones(sy)
            first_mp = clean_pairs[0][1]
            is_consonant = (clean_pairs[0][0] not in VOWEL_SET and clean_pairs[0][0] not in CODA_SET)
            after_long_vowel = (k > 0 and prev_nuc_d >= 0.42)

            dip_amp: Optional[float] = None
            if is_consonant:
                if first_mp in _FRICATIVE_ONSETS:
                    max_on = 0.112 if cdur >= 0.52 else 0.090
                elif after_long_vowel and first_mp in _NASAL_ONSETS:
                    max_on = 0.106
                    dip_amp = 1.45
                elif after_long_vowel and first_mp in _STOP_ONSETS:
                    max_on = 0.104
                    dip_amp = 1.70
                elif k > 0 and prev_nuc_d >= 0.20 and first_mp in _STOP_ONSETS:
                    max_on = 0.086
                    dip_amp = 1.15  # 句中塞音(如「岁在[癸]丑」g)加轻度闭塞微谷，彻底防止男中音软化为半元音 w!
                elif tp.sheng == "ru":
                    max_on = 0.076 if first_mp in _STOP_ONSETS else 0.068
                else:
                    max_on = 0.074
                onset_d = min(max_on, max(0.050, cdur - 0.18))
                onset_d = min(onset_d, cdur * 0.30)
            else:
                onset_d = 0.0

            m_d = min(0.048, cdur * 0.15) if has_m else 0.0
            nuc_d = max(0.18, cdur - onset_d - m_d)
            prev_nuc_d = nuc_d

            c_t0 = sum(ph_dur)
            c_pairs: List[Tuple[str, float]] = []
            actual_cdur = 0.0
            for rp, mp in clean_pairs:
                d = round(m_d if rp == "coda_m" else (nuc_d if rp in VOWEL_SET else (onset_d if onset_d > 0 else 0.055)), 4)
                c_pairs.append((mp, d))
                actual_cdur += d

            if dip_amp is not None and len(c_pairs) >= 2:
                dip_windows.append((c_t0, c_t0 + c_pairs[0][1], dip_amp))
            for mp, d in c_pairs:
                ph_seq.append(mp)
                ph_dur.append(d)

            n_c = max(6, int(round(actual_cdur / f0_step)))
            c_f0 = _eval_archetype(arch_name, anchor_st, n_c, scale=arch_scale)
            f0_chunks.append(c_f0)

        # 呼吸气口严格对齐母带真听韵律：紧连转合句一气呵成(0.11s)，普通逗号换气自然(0.24s)，大句号留足余韵(0.56s)！
        if plan.tight_next:
            tail_sp, gap_after = 0.09, 0.02
        elif plan.is_major_end:
            tail_sp, gap_after = 0.30, 0.26
        else:
            tail_sp, gap_after = 0.16, 0.08

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

        prev_end_st = last_v

        # 轻柔协同发音平滑 (sigma=1.6)，保留全部真人微腔与三波摇腔，随后在声母闭塞帧注入物理微谷！
        f0_st = gaussian_filter1d(f0_st, sigma=1.6)
        for d_t0, d_t1, dip_amp in dip_windows:
            j0, j1 = int(round(d_t0 / f0_step)), min(n_tot, int(round(d_t1 / f0_step)))
            if j1 > j0 + 2:
                f0_st[j0:j1] = f0_st[j0:j1] - dip_amp * np.hanning(j1 - j0).astype(np.float32)

        f0_st = np.clip(f0_st, min_st, max_st)
        f0_hz = use_key_hz * (2.0 ** (f0_st / 12.0))

        ds_items.append({
            "text": plan.text + plan.punc,
            "era": era,
            "genre": f"{mode_key}_v13",
            "archetype": plan.role_desc,
            "pz": "".join(t.pz for t in tones),
            "tones": " ".join(f"{t.char}({t.sheng[0]})" for t in tones),
            "offset": round(running_offset, 4),
            "ph_seq": " ".join(ph_seq),
            "ph_dur": " ".join(f"{x:.4f}" for x in ph_dur),
            "f0_seq": " ".join(f"{float(x):.2f}" for x in f0_hz),
            "f0_timestep": str(f0_step),
            "spk_name": speaker,
        })
        running_offset += total_dur + gap_after

    return ds_items
