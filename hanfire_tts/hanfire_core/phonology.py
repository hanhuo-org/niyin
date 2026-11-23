#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单核 4 槽位音系解析器与 42 维口腔物理特征编码器 (phonology.py)

核心职责：
1. 将任意上古/中古/近古 IPA 音节严格归并为单音节核 4 槽位：
   [ONSET (声母) | MEDIAL (介音滑音, -syl) | NUCLEUS (唯一韵腹, +syl) | CODA (韵尾, -syl)]
   彻底消灭王力多合元音（如 keau, ʔiei, ɣiwəi）导致的“一字多拍”。
2. 将每个槽位编码为 42 维连续物理向量：
   - [0..28]  29维口腔发音特征 (部位9 + 方法7 + 清浊1 + 舌高3 + 舌前后3 + 圆唇1 + 送气1 + 唇化1 + 腭化1 + 咽化ˤ1 + 长音1)
   - [29..33] 5维音节槽位 One-Hot (PAD/PAUSE, ONSET, MEDIAL, NUCLEUS, CODA)
   - [34..37] 4维声调轮廓层 [T_start, T_mid, T_end, T_checked] (上古按韵尾自然派生，中古/近古直传五度调值)
   - [38..39] 2维「音·形·义」生理动态控制量 [log_alpha_dur (急促<0/舒缓>0), alpha_openness (冷闭<0/暖开>0)]
   - [40..41] 2维官方发音人性别基底 [is_male, is_female]
3. 输出每个槽位的固有声学时长权重与硬上限帧数（唯闭塞尾 -p/-t/-k/-ʔ 强制 <=4帧 ≈42ms，物理断绝双音节拖尾）。
"""
from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import regex as re

# ---------------------------------------------------------------------------
# 常量与特征维定义
# ---------------------------------------------------------------------------
ARTIC_DIM = 29
SLOT_DIM = 5       # 0:PAUSE/PAD, 1:ONSET, 2:MEDIAL, 3:NUCLEUS, 4:CODA
TONE_DIM = 4       # [t_start, t_mid, t_end, is_checked] in [-1, 1]
SEM_DIM = 2        # [log_alpha_dur, alpha_openness]
SPK_DIM = 2        # [is_male, is_female]
TOTAL_FEAT_DIM = ARTIC_DIM + SLOT_DIM + TONE_DIM + SEM_DIM + SPK_DIM  # = 42

SLOT_PAUSE = 0
SLOT_ONSET = 1
SLOT_MEDIAL = 2
SLOT_NUCLEUS = 3
SLOT_CODA = 4

PLACE = ["labial", "labiodental", "dental", "alveolar", "retroflex", "palatal", "velar", "uvular", "guttural"]
MANNER = ["stop", "affricate", "fricative", "nasal", "lateral", "rhotic", "glide"]
P_IDX = {n: i for i, n in enumerate(PLACE)}
M_IDX = {n: i for i, n in enumerate(MANNER)}

CONSONANTS: Dict[str, Tuple[int, int, int]] = {
    "p": (P_IDX["labial"], M_IDX["stop"], 0), "b": (P_IDX["labial"], M_IDX["stop"], 1),
    "t": (P_IDX["alveolar"], M_IDX["stop"], 0), "d": (P_IDX["alveolar"], M_IDX["stop"], 1),
    "ʈ": (P_IDX["retroflex"], M_IDX["stop"], 0), "ɖ": (P_IDX["retroflex"], M_IDX["stop"], 1),
    "ȶ": (P_IDX["palatal"], M_IDX["stop"], 0), "ȡ": (P_IDX["palatal"], M_IDX["stop"], 1),
    "c": (P_IDX["palatal"], M_IDX["stop"], 0), "ɟ": (P_IDX["palatal"], M_IDX["stop"], 1),
    "k": (P_IDX["velar"], M_IDX["stop"], 0), "ɡ": (P_IDX["velar"], M_IDX["stop"], 1),
    "g": (P_IDX["velar"], M_IDX["stop"], 1),
    "q": (P_IDX["uvular"], M_IDX["stop"], 0), "ɢ": (P_IDX["uvular"], M_IDX["stop"], 1),
    "ʔ": (P_IDX["guttural"], M_IDX["stop"], 0),
    "t͡s": (P_IDX["alveolar"], M_IDX["affricate"], 0), "d͡z": (P_IDX["alveolar"], M_IDX["affricate"], 1),
    "t͡ʃ": (P_IDX["palatal"], M_IDX["affricate"], 0), "d͡ʒ": (P_IDX["palatal"], M_IDX["affricate"], 1),
    "t͡ʂ": (P_IDX["retroflex"], M_IDX["affricate"], 0), "d͡ʐ": (P_IDX["retroflex"], M_IDX["affricate"], 1),
    "t͡ɕ": (P_IDX["palatal"], M_IDX["affricate"], 0), "d͡ʑ": (P_IDX["palatal"], M_IDX["affricate"], 1),
    "p͡f": (P_IDX["labiodental"], M_IDX["affricate"], 0),
    "f": (P_IDX["labiodental"], M_IDX["fricative"], 0), "v": (P_IDX["labiodental"], M_IDX["fricative"], 1),
    "ɸ": (P_IDX["labial"], M_IDX["fricative"], 0), "β": (P_IDX["labial"], M_IDX["fricative"], 1),
    "θ": (P_IDX["dental"], M_IDX["fricative"], 0), "ð": (P_IDX["dental"], M_IDX["fricative"], 1),
    "s": (P_IDX["alveolar"], M_IDX["fricative"], 0), "z": (P_IDX["alveolar"], M_IDX["fricative"], 1),
    "ʃ": (P_IDX["palatal"], M_IDX["fricative"], 0), "ʒ": (P_IDX["palatal"], M_IDX["fricative"], 1),
    "ʂ": (P_IDX["retroflex"], M_IDX["fricative"], 0), "ʐ": (P_IDX["retroflex"], M_IDX["fricative"], 1),
    "ɕ": (P_IDX["palatal"], M_IDX["fricative"], 0), "ʑ": (P_IDX["palatal"], M_IDX["fricative"], 1),
    "ç": (P_IDX["palatal"], M_IDX["fricative"], 0), "ʝ": (P_IDX["palatal"], M_IDX["fricative"], 1),
    "x": (P_IDX["velar"], M_IDX["fricative"], 0), "ɣ": (P_IDX["velar"], M_IDX["fricative"], 1),
    "χ": (P_IDX["uvular"], M_IDX["fricative"], 0), "ʁ": (P_IDX["uvular"], M_IDX["fricative"], 1),
    "ħ": (P_IDX["guttural"], M_IDX["fricative"], 0), "ʕ": (P_IDX["guttural"], M_IDX["fricative"], 1),
    "h": (P_IDX["guttural"], M_IDX["fricative"], 0), "ɦ": (P_IDX["guttural"], M_IDX["fricative"], 1),
    "ɬ": (P_IDX["alveolar"], M_IDX["fricative"], 0), "ɮ": (P_IDX["alveolar"], M_IDX["fricative"], 1),
    "m": (P_IDX["labial"], M_IDX["nasal"], 1), "ɱ": (P_IDX["labiodental"], M_IDX["nasal"], 1),
    "n": (P_IDX["alveolar"], M_IDX["nasal"], 1), "ɳ": (P_IDX["retroflex"], M_IDX["nasal"], 1),
    "ɲ": (P_IDX["palatal"], M_IDX["nasal"], 1), "ȵ": (P_IDX["palatal"], M_IDX["nasal"], 1),
    "ŋ": (P_IDX["velar"], M_IDX["nasal"], 1), "ɴ": (P_IDX["uvular"], M_IDX["nasal"], 1),
    "l": (P_IDX["alveolar"], M_IDX["lateral"], 1), "ɭ": (P_IDX["retroflex"], M_IDX["lateral"], 1),
    "ʎ": (P_IDX["palatal"], M_IDX["lateral"], 1), "ʟ": (P_IDX["velar"], M_IDX["lateral"], 1),
    "r": (P_IDX["alveolar"], M_IDX["rhotic"], 1), "ɾ": (P_IDX["alveolar"], M_IDX["rhotic"], 1),
    "ɽ": (P_IDX["retroflex"], M_IDX["rhotic"], 1), "ʀ": (P_IDX["uvular"], M_IDX["rhotic"], 1),
    "ɹ": (P_IDX["alveolar"], M_IDX["glide"], 1), "ɻ": (P_IDX["retroflex"], M_IDX["glide"], 1),
    "j": (P_IDX["palatal"], M_IDX["glide"], 1), "w": (P_IDX["labial"], M_IDX["glide"], 1),
    "ɥ": (P_IDX["palatal"], M_IDX["glide"], 1), "ɰ": (P_IDX["velar"], M_IDX["glide"], 1),
    "ʋ": (P_IDX["labiodental"], M_IDX["glide"], 1), "ʍ": (P_IDX["labial"], M_IDX["glide"], 0),
}

# 元音: (height: 2=close, 1=mid, 0=open; back: 0=front, 1=central, 2=back; round: 0/1; sonority)
VOWELS: Dict[str, Tuple[int, int, int, float]] = {
    "i": (2, 0, 0, 4.0), "y": (2, 0, 1, 4.0), "ɪ": (2, 0, 0, 5.0),
    "ɨ": (2, 1, 0, 4.5), "ʉ": (2, 1, 1, 4.5), "ɯ": (2, 2, 0, 4.5),
    "u": (2, 2, 1, 4.0), "ʊ": (2, 2, 1, 5.0),
    "ɿ": (2, 1, 0, 4.2), "ʅ": (2, 1, 0, 4.2),
    "e": (1, 0, 0, 7.0), "ø": (1, 0, 1, 7.0), "ɛ": (1, 0, 0, 8.0), "œ": (1, 0, 1, 8.0),
    "ə": (1, 1, 0, 6.5), "ɜ": (1, 1, 0, 7.5),
    "o": (1, 2, 1, 7.0), "ɤ": (1, 2, 0, 7.0), "ɔ": (1, 2, 1, 8.0), "ʌ": (1, 2, 0, 7.5),
    "æ": (0, 0, 0, 9.5), "ɶ": (0, 0, 1, 9.5),
    "a": (0, 1, 0, 10.0), "ɐ": (0, 1, 0, 9.0),
    "ɑ": (0, 2, 0, 10.0), "ɒ": (0, 2, 1, 9.5),
}

MODIFIERS_MAP = {
    "ʰ": "asp", "ʱ": "asp",
    "ʷ": "lab",
    "ʲ": "pal",
    "ˤ": "pha",
    "ː": "long", "ˑ": "long",
    "̥": "devoice", "̊": "devoice",
    "̈": "round",
}

CHECKED_CODAS = {"p", "t", "k", "ʔ", "b", "d", "ɡ", "g", "wk", "kʷ"}
BILABIAL_CLOSED = {"p", "b", "m"}

_MOD_REGEX = r"[̀-ͯʰʱʲʷˤːˑ̥̠̪̺̻̊͜͡ˠˡˢˣ̤̩̆]*"
_BASE_REGEX = (
    r"[a-zɑæɐəɛɜɪɨɔœøɶɒʌʊʉɯɤɥβðɣχʁħʕʔɕʑȶȡȵpbtdʈɖcɟkɡqɢmɱnɳɲŋɴ"
    r"ʙrʀɾɽɸfvθszʃʒʂʐçʝxɬɮʋɹɻjɰlɭʎʟʍ]"
)
_ATOM_PAT = re.compile(rf"(?:{_BASE_REGEX}[͜͡])?{_BASE_REGEX}{_MOD_REGEX}")


def normalize_ipa_string(raw: str) -> str:
    """统一多学派拟音写法（如王力 tsh/ph/kh -> tsʰ/pʰ/kʰ，郑张双写元音 -> ː）。"""
    s = unicodedata.normalize("NFD", raw.strip())
    for ch in ("*", "[", "]", "(", ")", "<", ">"):
        s = s.replace(ch, "")
    s = s.replace("?", "ʔ").replace("ʼ", "ʔ").replace("ʻ", "ʔ").replace("'", "ʰ")
    # 剥离 C. 次要音节未定记号
    if s.startswith("C.") or s.startswith("Cə."):
        s = s.split(".", 1)[1]
    # 王力常见不带 ʰ 的送气写法归一: ph th kh tsh tɕh tʃh tʂh -> ʰ (仅在辅音后跟元音/介音时)
    s = re.sub(r"(ts|tɕ|tʃ|tʂ|p|t|k|q)h(?=[aeiouɑæɐəɛɜɪɨɔœøɒʌʊʉɯɤjwɥr])", r"\1ʰ", s)
    # 塞擦音加连音符绑为单一原子
    for aff in ("tʂʰ", "dʐ", "tʂ", "tɕʰ", "dʑ", "tɕ", "tʃʰ", "dʒ", "tʃ", "tsʰ", "dz", "ts"):
        s = s.replace(aff, aff[0] + "͡" + aff[1:])
    # 郑张双写元音 aa -> aː
    s = re.sub(r"([aɑeəiɨouɯɔ])\1", r"\1ː", s)
    return s


def tokenize_ipa_atoms(ipa: str) -> List[str]:
    s = normalize_ipa_string(ipa)
    parts = re.split(r"[.\-]", s)
    atoms: List[str] = []
    for p in parts:
        if p:
            atoms.extend(_ATOM_PAT.findall(p))
    return atoms


def _strip_mods(atom: str) -> Tuple[str, set]:
    mods = set()
    base = atom
    # 剥离组合符与后缀修饰符，保留连音符如 t͡s
    chars = []
    i = 0
    while i < len(base):
        ch = base[i]
        if ch in MODIFIERS_MAP:
            mods.add(MODIFIERS_MAP[ch])
        elif unicodedata.category(ch) == "Mn" and ch not in ("͡", "͜"):
            pass
        else:
            chars.append(ch)
        i += 1
    clean_base = "".join(chars)
    return clean_base, mods


def atom_to_artic29(atom: str) -> np.ndarray:
    """单个 IPA 原子 -> 29 维发音特征向量。"""
    v = np.zeros(ARTIC_DIM, dtype=np.float32)
    if not atom:
        return v
    base, mods = _strip_mods(atom)
    if base in CONSONANTS:
        pl, mn, vo = CONSONANTS[base]
        v[pl] = 1.0
        v[9 + mn] = 1.0
        v[16] = 1.0 if (vo and "devoice" not in mods) else 0.0
    elif base in VOWELS:
        h, b, r, _son = VOWELS[base]
        v[17 + h] = 1.0            # 17:open, 18:mid, 19:close
        v[20 + b] = 1.0            # 20:front, 21:central, 22:back
        v[23] = 1.0 if (r or "round" in mods) else 0.0
        v[16] = 0.0 if "devoice" in mods else 1.0  # 元音默认浊音
    else:
        # 未知单字符回退按首字符试探
        if base[:1] in CONSONANTS:
            pl, mn, vo = CONSONANTS[base[:1]]
            v[pl], v[9 + mn], v[16] = 1.0, 1.0, float(vo and "devoice" not in mods)
        elif base[:1] in VOWELS:
            h, b, r, _ = VOWELS[base[:1]]
            v[17 + h], v[20 + b], v[23], v[16] = 1.0, 1.0, float(r), 1.0

    if "asp" in mods:
        v[24] = 1.0
    if "lab" in mods:
        v[25] = 1.0
    if "pal" in mods:
        v[26] = 1.0
    if "pha" in mods:
        v[27] = 1.0
    if "long" in mods:
        v[28] = 1.0
    return v


@dataclass
class MononuclearSyllable:
    raw_ipa: str
    onset: str          # 如 "k", "m̥", "ŋ̊ˤ", "t͡sʰ", ""
    medial: str         # 滑音槽: "j", "w", "ɥ", "r", "" (严格非音节核)
    nucleus: str        # 唯一成音节主元音: "a", "ə", "e", "o", "i", "u" ...
    coda: str           # 韵尾槽: "p", "t", "k", "ʔ", "m", "n", "ŋ", "j", "w", "h", ""
    is_checked: bool    # 是否入声/喉塞唯闭短促尾
    mouth_openness: float  # 物理开口度 [-1.0 闭敛 .. +1.0 张大]，供「音形义」裁判使用

    def to_dict(self) -> dict:
        return asdict(self)


def _collapse_pre_vowels_to_glide(pre_vowels: List[str], existing_glides: List[str]) -> str:
    """将主元音前的前置元音序列折叠为单一过渡滑音 (j / w / ɥ / r)。"""
    has_front = any(v in {"i", "ɪ", "ɨ", "e", "ɛ", "y", "ø"} for v in pre_vowels) or ("j" in existing_glides) or ("ɥ" in existing_glides)
    has_round = any(v in {"u", "ʊ", "ʉ", "o", "ɔ", "y", "ø"} for v in pre_vowels) or ("w" in existing_glides) or ("ɥ" in existing_glides)
    has_r = any(g in {"r", "ɾ", "ɹ", "ɻ"} for g in existing_glides)
    if has_r and not (has_front or has_round):
        return "ɾ"  # 克制弹舌：上古介音 r 一律用轻闪音/近音而非重颤音
    if has_front and has_round:
        return "ɥ"
    if has_front:
        return "j"
    if has_round:
        return "w"
    return existing_glides[0] if existing_glides else ""


def parse_mononuclear_4slot(ipa: str) -> MononuclearSyllable:
    """
    核心算法：将任意复杂古音 IPA 严格归并为单核 4 槽位，100% 保证全音节只有 1 个元音核。
    示例：
      - 'keau' (交)  -> onset='k',  medial='j', nucleus='a', coda='w'
      - 'ʔiei' (伊)  -> onset='ʔ',  medial='j', nucleus='e', coda='j'
      - 'ɣiwəi'(位)  -> onset='ɣ',  medial='ɥ', nucleus='ə', coda='j'
      - 'kiəp' (急)  -> onset='k',  medial='j', nucleus='ə', coda='p' (is_checked=True)
      - 'liəm' (凜)  -> onset='l',  medial='j', nucleus='ə', coda='m'
    """
    atoms = tokenize_ipa_atoms(ipa)
    if not atoms:
        return MononuclearSyllable(ipa, "", "", "ə", "", False, 0.0)

    bases = [_strip_mods(a)[0] for a in atoms]
    v_indices = [i for i, b in enumerate(bases) if b in VOWELS]

    # 极端无元音成音节鼻音（如 m̩, ŋ̩）补央元音核
    if not v_indices:
        return MononuclearSyllable(ipa, atoms[0], "", "ə", "".join(atoms[1:]), False, -0.4)

    # 选响度最高的主元音；响度并列时（如 iu, ui, ie->e>i 已区分，iu 均为4.0）取靠后者（汉语上升复元音核在后）
    best_idx = v_indices[0]
    best_son = VOWELS[bases[best_idx]][3]
    for idx in v_indices[1:]:
        son = VOWELS[bases[idx]][3]
        if son >= best_son:
            best_idx = idx
            best_son = son

    nucleus_atom = atoms[best_idx]
    nuc_base = bases[best_idx]

    # 解析 best_idx 之前的部分 -> Onset + Medial
    before_atoms = atoms[:best_idx]
    before_bases = bases[:best_idx]
    onset_parts: List[str] = []
    pre_glides: List[str] = []
    pre_vowels: List[str] = []

    for a, b in zip(before_atoms, before_bases):
        if b in VOWELS:
            pre_vowels.append(b)
        elif b in {"j", "w", "ɥ", "r", "ɾ", "ɹ", "ɻ"} and onset_parts:
            pre_glides.append(b)
        else:
            onset_parts.append(a)

    # 若复合声母超过1个辅音（如 s + k），融合特征或保留冠音主辅音
    onset_str = "".join(onset_parts)
    medial_str = _collapse_pre_vowels_to_glide(pre_vowels, pre_glides)

    # 解析 best_idx 之后的部分 -> Coda (将尾随元音折叠为滑音 j/w，或与辅音尾合并)
    after_atoms = atoms[best_idx + 1:]
    after_bases = bases[best_idx + 1:]
    offglide = ""
    coda_cons: List[str] = []
    for a, b in zip(after_atoms, after_bases):
        if b in VOWELS:
            if b in {"i", "ɪ", "e", "y"} and not offglide:
                offglide = "j"
            elif b in {"u", "ʊ", "o", "ɔ"} and not offglide:
                offglide = "w"
        else:
            coda_cons.append(b if b in CHECKED_CODAS or b in {"m", "n", "ŋ", "h", "s", "j", "w"} else a)

    if coda_cons:
        # 例如 eauk -> nuc=a, offglide=w, coda_cons=[k] -> 圆唇化塞尾 kʷ（单槽位收尾，绝不拆两拍）
        raw_coda = "".join(coda_cons)
        if offglide == "w" and raw_coda == "k":
            coda_str = "kʷ"
        else:
            coda_str = raw_coda
    else:
        coda_str = offglide

    is_checked = coda_str in CHECKED_CODAS or coda_str.endswith(("p", "t", "k", "ʔ"))

    # 计算物理开口度 [-1.0 闭口锁温 .. +1.0 大开口舒展]
    # height: 0=open(+0.85), 1=mid(+0.1), 2=close(-0.65)
    h_val = VOWELS[nuc_base][0]
    open_score = {0: 0.85, 1: 0.10, 2: -0.65}.get(h_val, 0.0)
    if coda_str in {"m", "p"}:
        open_score -= 0.35   # 双唇紧闭收尾（如 凜 liəm, 急 kiəp）显著降低开口度
    elif coda_str in {"ʔ", "t", "k"}:
        open_score -= 0.15
    elif coda_str in {"", "ŋ", "n"} and h_val == 0:
        open_score += 0.15   # 暖 nuan / 阳 jiaŋ / 舒 ɕia
    if onset_parts and _strip_mods(onset_parts[0])[0] in BILABIAL_CLOSED and h_val == 2:
        open_score -= 0.15   # 冰 piəŋ
    open_score = float(np.clip(open_score, -1.0, 1.0))

    return MononuclearSyllable(
        raw_ipa=ipa,
        onset=onset_str,
        medial=medial_str,
        nucleus=nucleus_atom,
        coda=coda_str,
        is_checked=is_checked,
        mouth_openness=round(open_score, 3),
    )


def derive_oc_tone4(coda: str, is_checked: bool, explicit_tone: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
    """
    4维声调/句调层 [T_start, T_mid, T_end, T_checked] 归一化至 [-1, +1]。
    - 上古音无声调：按韵尾自然生理音高倾向下倾/微扬（平滑克制，绝不唱成现代普通话四声）；
    - 中古音/近古音：直接传入切韵/中原音韵五度标调值，零重构复用同一模型！
    """
    if explicit_tone is not None:
        return np.array([explicit_tone[0], explicit_tone[1], explicit_tone[2], 1.0 if is_checked else -1.0], dtype=np.float32)
    if coda == "ʔ":
        # 喉塞尾生理伴随声门张力微升
        return np.array([0.0, 0.08, 0.22, 1.0], dtype=np.float32)
    if coda in {"h", "s"}:
        # 去声源 -s/-h 气息松弛微降
        return np.array([0.12, 0.0, -0.22, -1.0], dtype=np.float32)
    if is_checked:
        # 唯闭入声 -p/-t/-k 短促平截
        return np.array([0.10, 0.08, 0.0, 1.0], dtype=np.float32)
    # 阴声/阳声舒声字：从容平调
    return np.array([0.05, 0.02, -0.04, -1.0], dtype=np.float32)


def _merge_cluster_artic(cluster_str: str) -> np.ndarray:
    """若声母为复合音（如 st, kr），加权融合为单个 29 维向量（主辅音 0.7 + 冠音 0.3）。"""
    atoms = _ATOM_PAT.findall(cluster_str)
    if not atoms:
        return np.zeros(ARTIC_DIM, dtype=np.float32)
    if len(atoms) == 1:
        return atom_to_artic29(atoms[0])
    v_main = atom_to_artic29(atoms[-1])
    v_pre = atom_to_artic29(atoms[0])
    return np.clip(0.75 * v_main + 0.35 * v_pre, 0.0, 1.0)


@dataclass
class EncodedSequence:
    feats: np.ndarray            # shape: (S, 42) float32
    dur_weights: np.ndarray      # shape: (S,) float32 相对时长分配权重
    max_frames: np.ndarray       # shape: (S,) int64 硬上限帧数 (93.75Hz = 24000/256; 4帧≈42.6ms)
    char_span_slots: List[Tuple[int, int, str]]  # 每个汉字对应的槽位起止索引 (s_start, s_end, char)


def encode_syllables_to_sequence(
    items: List[Dict],
    speaker_id: int = 0,
) -> EncodedSequence:
    """
    将一串已解析汉字/标点条目编码为 ZipVoice-Ancient 输入特征序列。
    items 每项支持:
      - {"type": "pause", "pause_kind": "comma" | "period"}
      - {"type": "char", "char": "急", "ipa": "kiəp", "alpha_dur": 0.75, "alpha_open": -0.35, "tone": None}
    speaker_id: 0 = 男声一号 (Male), 1 = 女声一号 (Female)
    """
    spk_vec = np.array([1.0, 0.0] if speaker_id == 0 else [0.0, 1.0], dtype=np.float32)
    rows: List[np.ndarray] = []
    weights: List[float] = []
    caps: List[int] = []
    spans: List[Tuple[int, int, str]] = []

    def _make_row(artic29: np.ndarray, slot_id: int, tone4: np.ndarray, sem2: np.ndarray) -> np.ndarray:
        slot_oh = np.zeros(SLOT_DIM, dtype=np.float32)
        slot_oh[slot_id] = 1.0
        return np.concatenate([artic29, slot_oh, tone4, sem2, spk_vec], axis=0)

    # 句首极短呼吸准备帧
    zero_artic = np.zeros(ARTIC_DIM, dtype=np.float32)
    zero_tone = np.zeros(TONE_DIM, dtype=np.float32)
    zero_sem = np.zeros(SEM_DIM, dtype=np.float32)
    rows.append(_make_row(zero_artic, SLOT_PAUSE, zero_tone, zero_sem))
    weights.append(0.35)
    caps.append(6)

    for it in items:
        if it.get("type") == "pause":
            p_kind = it.get("pause_kind", "comma")
            rows.append(_make_row(zero_artic, SLOT_PAUSE, zero_tone, zero_sem))
            weights.append(1.6 if p_kind == "comma" else 2.6)
            caps.append(22 if p_kind == "comma" else 38)
            continue

        ch = it.get("char", "")
        syl: MononuclearSyllable = it.get("parsed") or parse_mononuclear_4slot(it["ipa"])
        alpha_dur = float(it.get("alpha_dur", 0.76 if syl.is_checked else 1.0))
        alpha_open = float(it.get("alpha_open", syl.mouth_openness))
        sem2 = np.array([math.log(max(alpha_dur, 0.4)), alpha_open], dtype=np.float32)
        tone4 = derive_oc_tone4(syl.coda, syl.is_checked, it.get("tone"))

        s_start = len(rows)

        # 1. ONSET (若有)
        if syl.onset:
            v_on = _merge_cluster_artic(syl.onset)
            rows.append(_make_row(v_on, SLOT_ONSET, tone4, sem2))
            is_asp_or_vl = bool(v_on[24] > 0.5 or (v_on[9 + M_IDX["nasal"]] > 0.5 and v_on[16] < 0.5))
            weights.append(0.75 if is_asp_or_vl else 0.58)
            caps.append(10)  # <= 106ms

        # 2. MEDIAL GLIDE (若有 —— 硬上限 3 帧 ≈ 32ms，绝不允许变成第二元音核！)
        if syl.medial:
            v_med = atom_to_artic29(syl.medial)
            rows.append(_make_row(v_med, SLOT_MEDIAL, tone4, sem2))
            weights.append(0.30)
            caps.append(3)

        # 3. NUCLEUS VOWEL (全字唯一成音节核！吸收主要能量与时长调控)
        v_nuc = atom_to_artic29(syl.nucleus)
        rows.append(_make_row(v_nuc, SLOT_NUCLEUS, tone4, sem2))
        nuc_w = (1.35 if syl.is_checked else 1.95) * alpha_dur
        weights.append(nuc_w)
        caps.append(18 if syl.is_checked else 42)

        # 4. CODA (若有 —— 入声唯闭塞尾硬上限 4 帧 ≈ 42ms，物理斩断双音节拖尾！)
        if syl.coda:
            v_coda = _merge_cluster_artic(syl.coda)
            if syl.is_checked:
                v_coda[16] = 0.0  # 唯闭入声塞尾强制无声带振动
            rows.append(_make_row(v_coda, SLOT_CODA, tone4, sem2))
            if syl.is_checked:
                weights.append(0.28)
                caps.append(4)
            else:
                weights.append(0.85 * alpha_dur)
                caps.append(18)

        s_end = len(rows)
        spans.append((s_start, s_end, ch))

    # 句尾自然收音帧
    rows.append(_make_row(zero_artic, SLOT_PAUSE, zero_tone, zero_sem))
    weights.append(0.55)
    caps.append(10)

    return EncodedSequence(
        feats=np.vstack(rows).astype(np.float32),
        dur_weights=np.array(weights, dtype=np.float32),
        max_frames=np.array(caps, dtype=np.int64),
        char_span_slots=spans,
    )
