#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汉火古音唱吟 · 去声调三代古音唱吟音素编译器 (hanfire_sing_phonemes.py)

核心设计原则：
  1. 彻底解耦「音色音素」与「曲调音高 F0」：剥离朗读四声，将每个古音音节拆为
     [onset(可选短辅音)] + [nucleus(可无限拖腔的主元音核)] + [coda(可选短韵尾: -m/-p/-t/-k)]；
  2. 与开源中文歌唱库 (M4Singer / OpenCpop) 共享 100% 相同的现代汉语核心元音流形
     (a/o/e/i/u/v/ai/ei/ao/ou/an/en/ang/eng/ong/ia/ie/iao/iou/ian/in/iang/ing/iong/ua/uo/uai/uei/uan/un/uang/ve/van/vn/er/ir/i0)，
     使上古/中古/近古古音在训练中直接继承 30 小时棚录清唱的大八度拖腔、滑音与颤音共鸣；
  3. 完整保留历史特有音位：全浊塞音/塞擦/擦音 (OC_b/d/g/dz/dʑ/z/ʑ/ʒ)、清鼻流音 (hm/hn/hl/hŋ)、
     疑母 ŋ / 日母 ɲ / 微母 ʋ，以及闭口 -m 与入声塞尾 -p/-t/-k；
  4. 同步保留每个字的「历史声调类 (ping/shang/qu/ru)」供唐调多体裁旋律预测器依字行腔使用。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HANFIRE_TTS = PROJECT_ROOT / "hanfire_tts"
if str(HANFIRE_TTS) not in sys.path:
    sys.path.insert(0, str(HANFIRE_TTS))

# =============================================================================
# 1. 统一唱吟音素表（兼容 M4Singer/OpenCpop 现代汉语 + 三代古音扩展音素）
# =============================================================================
SPECIAL_TOKENS = ["<pad>", "SP", "AP"]  # SP=停顿静音, AP=换气呼吸

# 现代汉语标准声母（与 M4Singer / OpenCpop 100% 对齐）
MANDARIN_ONSETS = [
    "b", "p", "m", "f", "d", "t", "n", "l",
    "g", "k", "h", "j", "q", "x",
    "zh", "ch", "sh", "r", "z", "c", "s",
    "y", "w",
]

# 历史古音扩展声母（全浊、清鼻流音、疑母、日母、微母等）
HISTORICAL_ONSETS = [
    "h_b", "h_d", "h_g", "h_dz", "h_dzh", "h_dj",
    "h_z", "h_zh", "h_zj", "h_ng", "h_ny", "h_v",
    "h_hm", "h_hn", "h_hl", "h_hng", "h_gh",
]

# 可无限拉伸的共享主元音核（M4Singer / OpenCpop 与三代古音共享的核心歌唱元音！）
SHARED_NUCLEI = [
    "a", "o", "e", "i", "u", "v", "er", "i0", "ir",
    "ai", "ei", "ao", "ou",
    "ia", "ie", "iao", "iou",
    "ua", "uo", "uai", "uei",
    "ve",
    "an", "en", "in", "un", "vn",
    "ang", "eng", "ing", "ong",
    "ian", "iang", "iong",
    "uan", "uang", "van",
]

# 古音专用短收尾韵尾（只放在音符末尾 50~90ms，严禁无限拉长防卡带！）
HISTORICAL_CODAS = ["coda_m", "coda_p", "coda_t", "coda_k"]

SING_PHONEMES: List[str] = (
    SPECIAL_TOKENS + MANDARIN_ONSETS + HISTORICAL_ONSETS + SHARED_NUCLEI + HISTORICAL_CODAS
)
PHONEME_TO_ID: Dict[str, int] = {p: i for i, p in enumerate(SING_PHONEMES)}
ID_TO_PHONEME: Dict[int, str] = {i: p for i, p in enumerate(SING_PHONEMES)}
VOWEL_SET = set(SHARED_NUCLEI)
CODA_SET = set(HISTORICAL_CODAS)

# 将 M4Singer / pypinyin 常见变体归一化到 SHARED_NUCLEI
_PINYIN_FIN_NORM = {
    "iu": "iou",
    "ui": "uei",
    "ue": "ve",
    "üe": "ve",
    "ü": "v",
    "ün": "vn",
    "üan": "van",
    "van": "van",
    "iong": "iong",
    "ung": "ong",
}


def mandarin_pinyin_to_sing_phones(py_no_tone: str) -> List[str]:
    """将 M4Singer/OpenCpop 的无调汉语拼音音节转为统一唱吟音素列表 [onset?, nucleus]。"""
    s = py_no_tone.strip().lower().replace("u:", "v").replace("ü", "v")
    if not s or s in ("sp", "ap", "sil", "pau"):
        return ["AP" if s == "ap" else "SP"]

    # 识别声母
    onset = ""
    for cand in ("zh", "ch", "sh"):
        if s.startswith(cand):
            onset, s = cand, s[len(cand):]
            break
    if not onset and s[0] in "bpmfdtnlgkhjqxrzcsyw":
        onset, s = s[0], s[1:]

    # 整体认读音节归一化
    if onset in ("z", "c", "s") and s == "i":
        return [onset, "i0"]
    if onset in ("zh", "ch", "sh", "r") and s == "i":
        return [onset, "ir"]
    if onset in ("j", "q", "x", "y") and s in ("u", "ue", "uan", "un"):
        s = {"u": "v", "ue": "ve", "uan": "van", "un": "vn"}[s]
    if onset == "y":
        # 严格对齐 OpenCpop dictionary.txt（ya->y a, yao->y ao, you->y ou, yang->y ang, yong->y ong, ye->y ie/e, yan->y an）
        # 严禁拼出词典外 OOD 组合 `y iao / y iu`！
        if s in ("i", "in", "ing", "v", "ve", "van", "vn", "a", "ao", "ou", "an", "ang", "ong", "o"):
            return ["y", s]
        if s == "e":
            return ["y", "e"]
    if onset == "w":
        # 严格对齐 OpenCpop dictionary.txt（wa->w a, wo->w o, wai->w ai, wei->w ei, wan->w an, wen->w en, wang->w ang, weng->w eng）
        # 严禁拼出词典外 OOD 组合 `w ui / w uo / w uang`！
        if s in ("u", "a", "o", "ai", "ei", "an", "en", "ang", "eng"):
            return ["w", s]

    fin = _PINYIN_FIN_NORM.get(s, s)
    if fin not in PHONEME_TO_ID:
        # 兜底映射
        if fin.endswith("ng"):
            fin = "eng"
        elif fin.endswith("n"):
            fin = "en"
        else:
            fin = "a"
    return [onset, fin] if onset else [fin]


# 将汉火 V12 词元名转为去声调唱吟音素序列
_ZV_INI_TO_SING = {
    "b0": "b", "p0": "p", "m0": "m", "f0": "f",
    "d0": "d", "t0": "t", "n0": "n", "l0": "l",
    "g0": "g", "k0": "k", "h0": "h",
    "j0": "j", "q0": "q", "x0": "x",
    "zh0": "zh", "ch0": "ch", "sh0": "sh", "r0": "r",
    "z0": "z", "c0": "c", "s0": "s",
    "y0": "y", "w0": "w",
    "OC_b": "h_b", "OC_d": "h_d", "OC_g": "h_g",
    "OC_dz": "h_dz", "OC_dʒ": "h_dzh", "OC_dʑ": "h_dj",
    "OC_z": "h_z", "OC_ʒ": "h_zh", "OC_ʑ": "h_zj",
    "OC_ŋ": "h_ng", "OC_ɲ": "h_ny", "EM_ʋ": "h_v",
    "OC_hm": "h_hm", "OC_hn": "h_hn", "OC_hl": "h_hl", "OC_hŋ": "h_hng",
    "Q_並": "h_b", "Q_定": "h_d", "Q_羣": "h_g", "Q_從": "h_dz",
    "Q_崇": "h_dzh", "Q_船": "h_dj", "Q_邪": "h_z", "Q_常": "h_zh",
    "Q_俟": "h_z", "Q_澄": "h_dzh", "Q_匣": "h_gh", "Q_疑": "h_ng",
    "Q_日": "h_ny", "Q_幫": "b", "Q_滂": "p", "Q_明": "m",
    "Q_端": "d", "Q_透": "t", "Q_泥": "n", "Q_來": "l",
    "Q_知": "zh", "Q_徹": "ch", "Q_精": "z", "Q_清": "c", "Q_心": "s",
    "Q_章": "j", "Q_昌": "q", "Q_書": "x", "Q_莊": "zh", "Q_初": "ch", "Q_生": "sh",
    "Q_見": "g", "Q_溪": "k", "Q_曉": "h", "Q_以": "y", "Q_云": "y", "Q_影": "",
}


def ipa_or_token_to_sing_phones(ipa: str, tone_class: str = "ping") -> List[str]:
    """
    将三代古音 IPA 串（如 `kjem`, `sjwɛt`, `ŋjəi`, `kwaŋ`）精准解析为
    [onset?, nucleus(可无限拖腔), coda?(可选 -m/-p/-t/-k)]。
    """
    s = re.sub(r"[0-5˥˦˧˨˩ʔʰʱ\s/\[\]]+", "", ipa or "")
    # 识别送气标记以保留清送气声母
    raw = (ipa or "").strip().strip("/[]")
    # 简单解析声母
    onset_rules = [
        ("tɕʰ", "q"), ("t͡ɕʰ", "q"), ("tɕ", "j"), ("t͡ɕ", "j"), ("dʑ", "h_dj"), ("d͡ʑ", "h_dj"),
        ("tʂʰ", "ch"), ("t͡ʂʰ", "ch"), ("tʂ", "zh"), ("t͡ʂ", "zh"), ("dʐ", "h_dzh"), ("d͡ʐ", "h_dzh"),
        ("tʃʰ", "ch"), ("t͡ʃʰ", "ch"), ("tʃ", "zh"), ("t͡ʃ", "zh"), ("dʒ", "h_dzh"), ("d͡ʒ", "h_dzh"),
        ("tsʰ", "c"), ("t͡sʰ", "c"), ("ts", "z"), ("t͡s", "z"), ("dz", "h_dz"), ("d͡z", "h_dz"),
        ("ʈʰ", "ch"), ("ʈ", "zh"), ("ɖ", "h_dzh"),
        ("pʰ", "p"), ("tʰ", "t"), ("kʰ", "k"), ("kʷʰ", "k"), ("kʷ", "g"),
        ("m̥", "h_hm"), ("n̥", "h_hn"), ("l̥", "h_hl"), ("ŋ̊", "h_hng"),
        ("p", "b"), ("b", "h_b"), ("m", "m"), ("f", "f"), ("ʋ", "h_v"), ("v", "h_v"),
        ("t", "d"), ("d", "h_d"), ("ɳ", "n"), ("n", "n"), ("l", "l"), ("r", "l"), ("ɻ", "r"),
        ("k", "g"), ("g", "h_g"), ("ɡ", "h_g"), ("ŋ", "h_ng"), ("x", "h"), ("h", "h"), ("ɣ", "h_gh"), ("ɦ", "h_gh"),
        ("ɕ", "x"), ("ʑ", "h_zj"), ("ɲ", "h_ny"), ("ʂ", "sh"), ("ʃ", "sh"), ("ʒ", "h_zh"), ("ʐ", "r"),
        ("s", "s"), ("z", "h_z"), ("ʔ", ""),
    ]
    onset = ""
    rest = raw
    for pat, sym in onset_rules:
        if rest.startswith(pat):
            onset = sym
            rest = rest[len(pat):]
            break

    # 剥离尾调数字/声调符号
    rest = re.sub(r"[0-5˥˦˧˨˩ʔʰʱh]+$", "", rest)
    # 识别历史韵尾 -p / -t / -k / -m
    coda = ""
    if rest.endswith("p"):
        coda, rest = "coda_p", rest[:-1]
    elif rest.endswith("t"):
        coda, rest = "coda_t", rest[:-1]
    elif rest.endswith("k"):
        coda, rest = "coda_k", rest[:-1]
    elif rest.endswith("m"):
        coda, rest = "coda_m", rest[:-1]

    # 剩余主元音 + 介音 + -n/-ŋ 归一化至 SHARED_NUCLEI（严格遵守 AGENTS.md §3 防 ɨ/ʉ/ɛ/ew 退化！）
    r = rest
    # 0) 优先处理 魚韵三等 ɨə (MC 雎 tsʰɨə / 女 ɳɨə / 与 jɨə / 所 ʂɨə) 与近古撮口 ju (EM 雎 tsju / 女 nju / 与 ju / 于 ju)
    if r == "ɨə":
        if onset in ("zh", "ch", "sh", "r", "h_dzh", "h_zh"):
            r = "uo"   # 庄组魚韵三等（所 ʂɨə → shuo）
        else:
            r = "v"    # 精/泥/以组魚韵三等（雎 tsʰɨə → cv, 女 ɳɨə → nv, 与 jɨə → yv）
    elif r in ("jɨə", "ju"):
        r = "v"        # 近古鱼模韵撮口细音（雎 tsju / 女 nju / 与 ju / 于 ju / 欲 ju → v，绝不退化为 iou“牛/油”！）
    elif r == "ɨ" and onset in ("zh", "ch", "sh", "r", "h_dzh", "h_zh"):
        r = "ir"       # 近古/中古卷舌舌尖元音（之 tʂɨ / 师 ʂɨ → zhir/shir，绝不退化为 a“渣/沙”！）
    elif r == "ɨ" and onset in ("z", "c", "s", "h_dz", "h_z"):
        r = "i0"       # 精组舌尖前元音（子 tsɨ → zi0）
    elif coda == "coda_k" and r in ("ɨu", "iu"):
        r = "u"        # 屋韵三等入声（淑 dʑɨuk / 服 bɨuk → u+coda_k，绝不退化为 iou“就”！）
    elif r in ("ɨw", "ɨu") and not coda:
        # 尤韵三等 -ɨu/-ɨw（九 kɨu / 丑 ʈʰɨu / 修 sɨu / 矛 mɨw）：非舌面/泥来端声母后统一归 ou（防 g+iu / s+iu OOD 塌缩成 gu/chu）
        if onset in ("j", "q", "x", "n", "l", "d", "y", "h_ny", "h_zj", "h_dj", ""):
            r = "iou"
        else:
            r = "ou"

    # 先处理 ɨ（切韵三等央介音：唇音/庄组后不腭化为 i，仅在舌尖前/精知组保留细音感）
    if "ɨ" in r:
        if onset in ("b", "p", "m", "f", "h_b", "zh", "ch", "sh", "r", "h_dzh", "h_zh"):
            r = r.replace("ɨ", "")
        else:
            r = r.replace("ɨ", "i")
    # 处理 -ew / -ɛw -> -iao（切韵蕭韵四等 窈 ʔew / 窕 dew 与宵韵三等 小 siew 均保留齐齿细音 -iao；豪韵一等为 -aw -> -ao！）
    r = r.replace("iew", "iao").replace("iɛw", "iao").replace("jew", "iao").replace("jɛw", "iao")
    r = r.replace("ew", "iao").replace("ɛw", "iao").replace("aw", "ao").replace("ɑw", "ao")
    # 处理 ʉ（合口高元音，如 为 ʉe / 谓 ʉj / 绿 lʉok / 君 kʉn）
    r = r.replace("ʉe", "uei").replace("ʉj", "uei").replace("ʉo", "uo").replace("ʉ", "u")
    # 唇/牙喉前元音 e/ɛ 配闭口 -m 时用 ie（防退化为普通话后央元音 [ɤ] “哥姆”）
    if coda == "coda_m" and r in ("e", "ɛ"):
        r = "ie"
    # 零声母 w- 开头保留 u/w 介音（防“谓 wəs”丢掉 w 变成裸 e）
    if not onset and r.startswith("w"):
        onset = "w"
        r = r[1:]

    r = (
        r.replace("ɥi", "uei").replace("ɥe", "ue").replace("ɥa", "ua").replace("ɥn", "un").replace("ɥ", "v")
        .replace("j", "i").replace("w", "u")
        .replace("ɑ", "a").replace("ɐ", "a").replace("æ", "a").replace("ɒ", "o")
        .replace("ɛ", "e").replace("ɜ", "e").replace("ə", "e").replace("ɤ", "e").replace("œ", "e").replace("ʌ", "e")
        .replace("ɔ", "o").replace("ɯ", "u").replace("y", "v")
        .replace("ŋ", "ng").replace("n", "n")
    )
    r = re.sub(r"i+", "i", r)
    r = re.sub(r"u+", "u", r)
    # 常见双元音/鼻音组合规整
    nuc_map = {
        "ir": "ir", "i0": "i0",
        "ia": "ia", "ie": "ie", "io": "uo", "iu": "iou", "iau": "iao", "iao": "iao", "iou": "iou", "ieu": "iao",
        "iue": "ve", "iua": "van", "iuan": "van", "iuen": "vn", "iun": "vn", "iuo": "ve",
        "ua": "ua", "uo": "uo", "ue": "uei", "ui": "uei", "uai": "uai", "uei": "uei", "uoi": "uei",
        "ve": "ve", "va": "van", "van": "van", "ven": "vn", "vin": "vn", "vn": "vn", "vei": "uei", "vi": "uei", "v": "v",
        "ai": "ai", "ei": "ei", "oi": "ei", "au": "ao", "ao": "ao", "ou": "ou", "eu": "ou",
        "an": "an", "en": "en", "in": "in", "un": "un", "on": "un",
        "ang": "ang", "eng": "eng", "ing": "ing", "ong": "ong", "ung": "ong",
        "ian": "ian", "ien": "ian", "iang": "iang", "ieng": "ing", "iong": "iong", "iung": "iong",
        "uan": "uan", "uen": "un", "uang": "uang", "ueng": "ong", "uong": "ong",
        "a": "a", "o": "o", "e": "e", "i": "i", "u": "u",
    }
    nuc = nuc_map.get(r, "")
    if not nuc:
        if r.endswith("ng"):
            nuc = "iang" if r.startswith("i") else ("uang" if r.startswith("u") else ("ong" if "o" in r else "ang" if "a" in r else "eng"))
        elif r.endswith("n"):
            nuc = "ian" if r.startswith("i") else ("uan" if r.startswith("u") else ("an" if "a" in r else "en"))
        elif "a" in r:
            nuc = "ia" if r.startswith("i") else ("ua" if r.startswith("u") else "a")
        elif "o" in r:
            nuc = "uo" if r.startswith("u") else "o"
        elif "e" in r:
            nuc = "ie" if r.startswith("i") else "e"
        elif "v" in r:
            nuc = "v"
        elif "i" in r:
            nuc = "i"
        elif "u" in r:
            nuc = "u"
        else:
            nuc = "a"

    # 防 OOD 声韵拼合：
    # 1) OpenCpop/M4Singer 中仅 j/q/x/n/l/y 可拼撮口 v/ve/van/vn，牙喉/唇/舌尖声母须用合口 u/uei/uo/uan/un
    if nuc in ("v", "ve", "van", "vn") and onset and onset not in ("j", "q", "x", "n", "l", "y", "h_ny", "h_zj", "h_dj"):
        nuc = {"v": "u", "ve": "uo", "van": "uan", "vn": "un"}[nuc]
    # 2) 章组/书母 j/q/x 遇合口 u/uei/ua/uai/uan/un/uang/uo 时（如中古「水 ɕyi」），转为正齿 zh/ch/sh 防 x+ui OOD
    if onset in ("j", "q", "x", "h_dj", "h_zj") and nuc in ("u", "uei", "ua", "uai", "uan", "un", "uang", "uo"):
        onset = {"j": "zh", "q": "ch", "x": "sh", "h_dj": "h_dzh", "h_zj": "h_zh"}[onset]
    # 3) 精组 z/c/s 遇齐齿韵 ia/ie/iao/iou/ian/in/iang/ing/iong 时（如中古「仙 sien / 斯 sie」），防 s+ian / s+ie OOD
    if onset in ("z", "c", "s", "h_dz", "h_z") and nuc in ("ia", "ie", "iao", "iou", "ian", "in", "iang", "ing", "iong"):
        if nuc == "ie":
            nuc = "i0"
        else:
            onset = {"z": "j", "c": "q", "s": "x", "h_dz": "h_dj", "h_z": "h_zj"}[onset]
    # 4) 疑母/泥母后无单韵母 o（如中古「吾 ŋo -> n+o」OOD），统一规整为合口 uo (nuo)
    if onset in ("n", "h_ng", "l", "d", "t", "g", "k", "h", "z", "c", "s", "zh", "ch", "sh", "r") and nuc == "o":
        nuc = "uo"
    # 5) 唇音 b/p/m/f 在 OpenCpop 词典中拼 o 而非 uo（如中古「无 mʉo -> m+o」，防 m+uo OOD）
    if onset in ("b", "p", "m", "f", "h_b") and nuc == "uo":
        nuc = "o"
    # 6) 知庄组卷舌 zh/ch/sh/r 不拼齐齿韵 ia/ie/iao/iou/ian/in/iang/ing/iong（如中古「朝 ʈiew -> zh+ao」，防 zh+iao OOD）
    if onset in ("zh", "ch", "sh", "r", "h_dzh", "h_zh") and nuc in ("ia", "ie", "iao", "iou", "ian", "in", "iang", "ing", "iong"):
        nuc = {
            "ia": "a", "ie": "e", "iao": "ao", "iou": "ou",
            "ian": "an", "in": "en", "iang": "ang", "ing": "eng", "iong": "ong",
        }[nuc]

    out = []
    if onset:
        out.append(onset)
    out.append(nuc)
    if coda:
        out.append(coda)
    return out


from dataclasses import dataclass

@dataclass
class SingSyllable:
    char: str
    ipa: str
    phones: List[str]
    tone_class: str  # ping | shang | qu | ru


_LEXICON_CACHE: Dict[str, Dict[str, Tuple[str, str]]] | None = None
_S2T = None
_T2S = None


def _load_3era_lexicon() -> Dict[str, Dict[str, Tuple[str, str]]]:
    global _LEXICON_CACHE, _S2T, _T2S
    if _LEXICON_CACHE is not None:
        return _LEXICON_CACHE
    import json
    try:
        from opencc import OpenCC
        _S2T = OpenCC("s2t")
        _T2S = OpenCC("t2s")
    except Exception:
        _S2T = _T2S = None
    lex_path = PROJECT_ROOT / "hanfire_tts" / "v5_bundle" / "v5_three_era_lexicon.json"
    raw = json.loads(lex_path.read_text(encoding="utf-8"))
    _LEXICON_CACHE = raw.get("eras", {})
    return _LEXICON_CACHE


def _norm_tone_class(raw_tone: str, era: str = "mc") -> str:
    t = str(raw_tone).lower()
    # 近古音《中原音韵》已「入派三声」（入作平/入作上/入作去），不再作为独立入声短促类
    if era == "em" and "入作" in t:
        if "上" in t:
            return "shang"
        if "去" in t:
            return "qu"
        return "ping"
    if any(k in t for k in ("checked", "入", "ru")):
        return "ru"
    if any(k in t for k in ("rising", "上", "shang")):
        return "shang"
    if any(k in t for k in ("depart", "去", "qu")):
        return "qu"
    return "ping"


# 古诗文高频多音字现代音(zh)权威主读（防 pypinyin 默认冷僻读音，如 俟 qí→sì、蕃 bō→fán）
_ZH_POLYPHONE_OVERRIDE: Dict[str, Tuple[str, str]] = {
    "俟": ("si", "qu"),
    "蕃": ("fan", "ping"),
}

# 古诗文经典上下文多音词覆写（按位置精准纠偏：会稽 kuài jī、属客 zhǔ kè、冯虚 píng xū、扁舟 piān zhōu 等）
_ZH_CONTEXT_POLYPHONES: Dict[str, List[Tuple[str, str]]] = {
    "会稽": [("kuai", "qu"), ("ji", "ping")],
    "會稽": [("kuai", "qu"), ("ji", "ping")],
    "属客": [("zhu", "shang"), ("ke", "ru")],
    "屬客": [("zhu", "shang"), ("ke", "ru")],
    "冯虚": [("ping", "ping"), ("xu", "ping")],
    "馮虛": [("ping", "ping"), ("xu", "ping")],
    "扁舟": [("pian", "ping"), ("zhou", "ping")],
    "少长": [("shao", "qu"), ("zhang", "shang")],
    "少長": [("shao", "qu"), ("zhang", "shang")],
    "汤汤": [("shang", "ping"), ("shang", "ping")],
    "湯湯": [("shang", "ping"), ("shang", "ping")],
    "淡泊": [("dan", "qu"), ("bo", "ru")],
    "澹泊": [("dan", "qu"), ("bo", "ru")],
    "长江": [("chang", "ping"), ("jiang", "ping")],
    "長江": [("chang", "ping"), ("jiang", "ping")],
    "朝晖": [("zhao", "ping"), ("hui", "ping")],
    "朝暉": [("zhao", "ping"), ("hui", "ping")],
    "予独": [("yu", "ping"), ("du", "ru")],
    "予獨": [("yu", "ping"), ("du", "ru")],
    "予观": [("yu", "ping"), ("guan", "ping")],
    "予觀": [("yu", "ping"), ("guan", "ping")],
    "夫君子": [("fu", "ping"), ("jun", "ping"), ("zi", "shang")],
    "不兴": [("bu", "ru"), ("xing", "ping")],
    "不興": [("bu", "ru"), ("xing", "ping")],
}

# 中古音(mc)上下文破读纠偏（如「会稽」之「会」读古外切見母去声 guai，非匣母黄外切）
_MC_CONTEXT_POLYPHONES: Dict[str, Dict[int, Tuple[str, List[str], str]]] = {
    "会稽": {0: ("kuɑiH", ["g", "uai"], "qu")},
    "會稽": {0: ("kuɑiH", ["g", "uai"], "qu")},
    "长江": {0: ("ɖjɑŋ", ["h_dzh", "ang"], "ping")},
    "長江": {0: ("ɖjɑŋ", ["h_dzh", "ang"], "ping")},
}


def text_to_sing_syllables(text: str, era: str = "mc") -> List[SingSyllable]:
    """将任意汉字诗文按指定朝代 (oc/mc/em/zh) 编译为去声调唱吟音节序列（100% 一字一音零吞字）。"""
    from pypinyin import Style, pinyin

    era = era.lower()
    lex = _load_3era_lexicon()
    era_tbl = lex.get(era, {}) if era in ("oc", "mc", "em") else {}
    out: List[SingSyllable] = []

    # 整句分词级普通话拼音（保证「长江/朝晖/淡泊/重修」等常规多音词按语境正确定音，单字符 1:1 对齐）
    ctx_pinyin_list = pinyin(
        text,
        style=Style.TONE3,
        neutral_tone_with_five=True,
        errors=lambda seg: [[c] for c in seg],
    )

    # 预先标注现代音经典多音词位置
    ctx_map: Dict[int, Tuple[str, str]] = {}
    if era == "zh":
        for word, readings in _ZH_CONTEXT_POLYPHONES.items():
            start = 0
            while True:
                idx = text.find(word, start)
                if idx < 0:
                    break
                for off, rd in enumerate(readings):
                    ctx_map[idx + off] = rd
                start = idx + len(word)

    mc_ctx_map: Dict[int, Tuple[str, List[str], str]] = {}
    if era == "mc":
        for word, off_map in _MC_CONTEXT_POLYPHONES.items():
            start = 0
            while True:
                idx = text.find(word, start)
                if idx < 0:
                    break
                for off, spec in off_map.items():
                    mc_ctx_map[idx + off] = spec
                start = idx + len(word)

    for pos, ch in enumerate(text):
        if not re.match(r"[\u4e00-\u9fff]", ch):
            continue
        if pos in mc_ctx_map:
            ipa_s, phs_override, t_cls = mc_ctx_map[pos]
            out.append(SingSyllable(char=ch, ipa=ipa_s, phones=list(phs_override), tone_class=t_cls))
            continue

        entry = None
        if era_tbl:
            entry = era_tbl.get(ch)
            if entry is None and _S2T is not None:
                entry = era_tbl.get(_S2T.convert(ch))
            if entry is None and _T2S is not None:
                entry = era_tbl.get(_T2S.convert(ch))

        if entry is not None:
            ipa_str, raw_tone = entry[0], entry[1]
            t_cls = _norm_tone_class(raw_tone, era=era)
            phs = ipa_or_token_to_sing_phones(ipa_str, tone_class=t_cls)
            out.append(SingSyllable(char=ch, ipa=ipa_str, phones=phs, tone_class=t_cls))
        elif pos in ctx_map:
            py_plain, t_cls = ctx_map[pos]
            phs = mandarin_pinyin_to_sing_phones(py_plain)
            out.append(SingSyllable(char=ch, ipa=py_plain, phones=phs, tone_class=t_cls))
        elif ch in _ZH_POLYPHONE_OVERRIDE:
            py_plain, t_cls = _ZH_POLYPHONE_OVERRIDE[ch]
            phs = mandarin_pinyin_to_sing_phones(py_plain)
            out.append(SingSyllable(char=ch, ipa=py_plain, phones=phs, tone_class=t_cls))
        else:
            # 保底回退整句语境普通话拼音（永不静默吞字！）
            py_num = ctx_pinyin_list[pos][0] if pos < len(ctx_pinyin_list) and ctx_pinyin_list[pos] else ch
            tone_d = py_num[-1] if py_num and py_num[-1].isdigit() else "1"
            t_cls = {"1": "ping", "2": "ping", "3": "shang", "4": "qu", "5": "ping"}.get(tone_d, "ping")
            py_plain = re.sub(r"\d+$", "", py_num)
            phs = mandarin_pinyin_to_sing_phones(py_plain)
            out.append(SingSyllable(char=ch, ipa=py_plain, phones=phs, tone_class=t_cls))
    return out


if __name__ == "__main__":
    print(f"Unified Hanfire-Sing Phoneme Vocab: {len(SING_PHONEMES)} tokens")
    for sample in ["kjem", "kwaŋ", "sjwɛt", "ŋjəi", "tsʰaŋ", "kʰap", "bək"]:
        print(f"  IPA /{sample:6s}/ -> {ipa_or_token_to_sing_phones(sample)}")
    for py in ["jian", "jia", "cang", "shuang", "shi", "zi", "lv", "qu"]:
        print(f"  PY  '{py:6s}' -> {mandarin_pinyin_to_sing_phones(py)}")
    for era in ("oc", "mc", "em"):
        syls = text_to_sing_syllables("蒹葭苍苍白露为霜", era=era)
        print(f"  [{era}] " + " | ".join(f"{s.char}({s.tone_class}):{'+'.join(s.phones)}" for s in syls))
