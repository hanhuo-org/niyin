#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汉火上古音 4槽单核 -> DiaMoE-TTS 原生汉语方言专家 IPA 编译器 (diamoe_compiler.py)

核心设计原理：
1. 彻底根治“直接读音标的感觉”：
   - 弃用仅在美式英语 (en-us) 上训练、且含 139 个未训练随机向量的 ZipVoice 英文 IPA 词表；
   - 采用 100% 在中国十一大方言（粤、吴、闽、江淮、秦陇、中原、冀鲁、幽燕等）真人母语上训练的
     DiaMoE-TTS 原生汉语声韵词表 (`vocab.txt`)：
     * 全浊声母 `[b], [d], [ɡ], [d͜z], [d͜ʑ], [ʑ], [ɣ], [ɦ]` 由吴语/老湘语母语专家原生发音；
     * 入声唯闭塞尾 `[p], [t], [k]` 与短促核 `[ˈV̆ᴴ]/[ˈV̆ᴸ]` 由粤语/闽南语/江淮官话专家原生发音；
     * 闭口鼻尾 `[m]`、疑母 `[ŋ]`、日母硬腭鼻音 `[ȵ]` 由粤/吴/闽专家原生发音；
     * 音节间以汉字词界 `|` 切分，按“一字一拍”控制流匹配时长，复元音尾统一采用非音节滑音 `[ɪ̯]/[ʊ̯]`，
       物理级锁死“一字一音、绝无俄式弹舌、绝无老外拼读音标腔”！
2. 融合古四声（平、上 `*-ʔ`、去 `*-h/-s`、入 `*-p/-t/-k`）与声母清浊阴阳：
   - 阴平（清声母舒声平调）：`ᴴᴴ`（清越平展）
   - 阳平（浊声母舒声平调）：`ᴹᴴ`（沉吟低升）
   - 上声（古 `-ʔ` 源 / 上声字）：清上 `ᴹᴸ` / 浊上 `ᴸᴹ`（古雅微抑扬）
   - 去声（古 `-h/-s` 源 / 去声字）：`ᴴᴸ`（曼声长叹降调）
   - 入声（`-p/-t/-k` 促声）：清入短促 `̆ᴴ` / 浊入短促 `̆ᴸ`（短促急收，不拖元音尾巴）
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

DIAMOE_ROOT = Path("/mnt/s790/project/hanzi/diamoe-rust")
DEFAULT_VOCAB_PATH = DIAMOE_ROOT / "models" / "vocab.txt"
DEFAULT_LEXICON_PATH = Path("/mnt/s790/project/tts/hanfire_tts/gdrive_bundle/lexicon/hanfire_oc_v1.json")
SHELF_TSV_PATH = Path("/mnt/s790/project/tts/voxtts/collect/data/pron_shelf_v3.tsv")

VOICED_ONSETS = {
    "b", "d", "g", "ɡ", "dz", "d͡z", "d͜z", "dʑ", "d͡ʑ", "d͜ʑ", "dʒ", "d͡ʒ", "ɖʐ",
    "z", "ʑ", "ʒ", "ʐ", "ɣ", "ɦ", "v",
    "m", "n", "ŋ", "ɲ", "ȵ", "l", "r", "ɻ", "j", "w", "ɥ",
}

SONORANT_ONSETS = {"m", "n", "ŋ", "ɲ", "ȵ", "l", "j", "w", "ɥ"}

# 4槽声母 -> DiaMoE vocab.txt 标准辅音 token 序列（不含外层方括号）
ONSET_MAP: Dict[str, List[str]] = {
    "": ["∅"],
    "∅": ["∅"],
    "ʔ": ["∅"],
    "p": ["p"],
    "pʰ": ["pʰ"],
    "b": ["b"],
    "m": ["m"],
    "m̥": ["h", "m"],
    "f": ["f"],
    "v": ["v"],
    "t": ["t"],
    "tʰ": ["tʰ"],
    "d": ["d"],
    "n": ["n"],
    "n̥": ["h", "n"],
    "l": ["l"],
    "l̥": ["ɬ̩↓"],
    "r": ["l"],       # 严守弹舌克制铁律：严禁俄式大弹舌 [r]
    "ɻ": ["l"],
    "ts": ["t͜s"],
    "t͡s": ["t͜s"],
    "tsʰ": ["t͜sʰ"],
    "t͡sʰ": ["t͜sʰ"],
    "dz": ["d͜z"],
    "d͡z": ["d͜z"],
    "s": ["s"],
    "z": ["z"],
    "tɕ": ["t͜ɕ"],
    "t͡ɕ": ["t͜ɕ"],
    "tɕʰ": ["t͜ɕʰ"],
    "t͡ɕʰ": ["t͜ɕʰ"],
    "dʑ": ["d͜ʑ"],
    "d͡ʑ": ["d͜ʑ"],
    "ɕ": ["ɕ"],
    "ʑ": ["ʑ"],
    "ɲ": ["ȵ"],
    "ɲ̊": ["h", "ȵ"],
    "tʃ": ["t͡ʃ"],
    "t͡ʃ": ["t͡ʃ"],
    "ʈʂ": ["t͡ʂ"],
    "tʃʰ": ["t͡ʃʰ"],
    "t͡ʃʰ": ["t͡ʃʰ"],
    "ʈʂʰ": ["t͡ʂʰ"],
    "dʒ": ["d͡ʒ"],
    "d͡ʒ": ["d͡ʒ"],
    "ɖʐ": ["d͡ʒ"],
    "ʃ": ["ʃ"],
    "ʂ": ["ʂ"],
    "ʒ": ["ʒ"],
    "ʐ": ["ʒ"],
    "k": ["k"],
    "kʰ": ["kʰ"],
    "g": ["ɡ"],
    "ɡ": ["ɡ"],
    "ŋ": ["ŋ"],
    "ŋ̊": ["ŋ̊"],
    "x": ["x"],
    "h": ["x"],
    "χ": ["χ"],
    "ɣ": ["ɣ"],
    "ɦ": ["ɦ"],
    "q": ["k"],       # 汉火体系软腭化原则
    "qʰ": ["kʰ"],
    "ɢ": ["ɡ"],
    "j": ["j"],
    "w": ["w"],
    "ɥ": ["ɥ"],
}

MEDIAL_MAP: Dict[str, List[str]] = {
    "": [],
    "j": ["j"],
    "w": ["w"],
    "ɥ": ["ɥ"],
    "r": [],          # 严禁介音弹舌，二等洪音由开口元音承载
    "ɹ": [],
}

CODA_MAP: Dict[str, List[str]] = {
    "": [],
    "m": ["m"],
    "n": ["n"],
    "ŋ": ["ŋ"],
    "p": ["p"],
    "t": ["t"],
    "k": ["k"],
    "ʔ": [],          # 喉塞上声已由声调/促核承载，不追加塞擦杂音
    "h": [],          # 去声已由 ᴴᴸ 降调承载
    "s": [],
    "j": ["ɪ̯"],       # 非音节性滑音，锁死“一字一音”不拆拍
    "i": ["ɪ̯"],
    "w": ["ʊ̯"],       # 非音节性滑音，锁死“一字一音”不拆拍
    "u": ["ʊ̯"],
}

PUNCT_MAP: Dict[str, str] = {
    "，": "，",
    ",": "，",
    "、": "，",
    "；": "，",
    ";": "，",
    "：": "，",
    ":": "，",
    "。": "。",
    ".": "。",
    "？": "？",
    "?": "？",
    "！": "！",
    "!": "！",
}


def load_diamoe_vocab(vocab_path: Path = DEFAULT_VOCAB_PATH) -> Set[str]:
    return {ln.strip() for ln in vocab_path.read_text(encoding="utf-8").splitlines() if ln.strip()}


def load_historical_tone_hints(shelf_tsv: Path = SHELF_TSV_PATH) -> Dict[str, str]:
    """
    从 `pron_shelf_v3.tsv` 提取上古声调渊源标记：
      - 'checked': *-p / *-t / *-k
      - 'rising' : *-ʔ（上声源）
      - 'departing': *-h / *-s（去声源）
      - 'level'  : 平声舒声
    同时注册简体字映射，确保简繁输入一致命中。
    """
    tones: Dict[str, str] = {}
    if not shelf_tsv.exists():
        return tones

    for idx, ln in enumerate(shelf_tsv.read_text(encoding="utf-8").splitlines()):
        if idx == 0 or not ln.strip():
            continue
        parts = ln.split("\t")
        if len(parts) < 4:
            continue
        ch, sel, _src, xt = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
        ref = sel or xt
        if not ch or not ref:
            continue
        ref_clean = ref.rstrip(")")
        if ref_clean.endswith(("p", "t", "k")):
            t = "checked"
        elif ref_clean.endswith("ʔ"):
            t = "rising"
        elif ref_clean.endswith(("h", "s")):
            t = "departing"
        else:
            t = "level"
        tones[ch] = t

    try:
        from opencc import OpenCC
        t2s = OpenCC("t2s")
        for ch, t in list(tones.items()):
            s = t2s.convert(ch)
            if s != ch and s not in tones:
                tones[s] = t
    except Exception:
        pass
    return tones


class HanfireDiaMoECompiler:
    """将汉火上古音词表条目编译为 100% 合法的 DiaMoE 括号化/裸 IPA 序列。"""

    def __init__(
        self,
        lexicon_path: Path = DEFAULT_LEXICON_PATH,
        vocab_path: Path = DEFAULT_VOCAB_PATH,
        shelf_tsv: Path = SHELF_TSV_PATH,
    ) -> None:
        raw = json.loads(lexicon_path.read_text(encoding="utf-8"))
        self.chars: Dict[str, Dict] = raw["chars"]
        self.anthology: List[Dict] = raw.get("shijing_anthology", [])
        self.vocab: Set[str] = load_diamoe_vocab(vocab_path)
        self.tone_hints: Dict[str, str] = load_historical_tone_hints(shelf_tsv)

    def _pick_nucleus_token(
        self,
        nucleus: str,
        coda: str,
        is_checked: bool,
        voiced: bool,
        tone_cat: str,
    ) -> str:
        """
        根据主元音、韵尾与古四声调类，精确挑选 DiaMoE 粤/吴/官话高频训练过的带调元音核 token（不含方括号）。
        """
        v = nucleus or "a"
        # 归并罕见元音符号到 DiaMoE 主力母语元音核
        v_map = {
            "ɑ": "ɑ",
            "a": "a",
            "æ": "æ",
            "ɐ": "ɐ",
            "ə": "ə",
            "ɯ": "ə",
            "ɨ": "ɨ",
            "ɤ": "ɤ",
            "e": "e",
            "ɛ": "ɛ",
            "i": "i",
            "ɪ": "ɪ",
            "o": "o",
            "ɔ": "ɔ",
            "u": "u",
            "ʊ": "ʊ",
            "y": "y",
        }
        v = v_map.get(v, "a")

        # 阳声 -ŋ 韵腹采用后开元音 ɑ（如 苍/霜/方），共鸣更浑厚洪亮
        if v == "a" and coda == "ŋ":
            v = "ɑ"

        if is_checked or coda in ("p", "t", "k"):
            # 入声短促核：优先匹配粤语/吴语真实高频入声短元音，使唯闭塞尾干净利落！
            checked_vowel_map = {
                "ə": "ɐ",   # 粤语 黑 [h ˈɐ̆ᴴ k] / 职部短促核
                "a": "a",   # 白 [b j ˈăᴸ k]
                "ɑ": "ɑ",
                "e": "ɛ",   # 质/月部 一 [∅ j ˈɛ̆ᴴ t]
                "i": "ɪ",   # 粤语 色 [s ˈɪ̆ᴴ k]
                "o": "ɔ",
                "u": "u",   # 觉部 淑 [ʑ j ˈŭᴸ k]
            }
            vc = checked_vowel_map.get(v, v)
            cand_primary = f"ˈ{vc}̆ᴸ" if voiced else f"ˈ{vc}̆ᴴ"
            cand_fallbacks = [
                cand_primary,
                f"ˈ{vc}̆ᴴ",
                f"ˈ{vc}̆ᴹ",
                f"ˈ{v}̆ᴸ" if voiced else f"ˈ{v}̆ᴴ",
                f"ˈ{v}̆ᴴ",
                "ˈɐ̆ᴸ" if voiced else "ˈɐ̆ᴴ",
            ]
            for c in cand_fallbacks:
                if f"[{c}]" in self.vocab:
                    return c

        # 正常上古读音调值：平声用自然清高平 ᴴᴴ / 浊中低平 ᴹᴹ~ᴸᴸ（彻底消除普通话 35 阳平上挑现代腔！）
        if tone_cat == "departing" or coda in ("h", "s"):
            t_order = ["ᴴᴸ", "ᴹᴸ", "ᴴᴴ"] if not voiced else ["ᴹᴸ", "ᴸᴸ", "ᴴᴸ"]
        elif tone_cat == "rising" or coda == "ʔ":
            t_order = ["ᴸᴹ", "ᴹᴴ", "ᴹᴹ"] if voiced else ["ᴹᴴ", "ᴸᴹ", "ᴹᴸ"]
        else:
            t_order = ["ᴹᴹ", "ᴸᴸ", "ᴹᴴ"] if voiced else ["ᴴᴴ", "ᴹᴹ", "ᴴᴹ"]

        for t in t_order:
            cand = f"ˈ{v}{t}"
            if f"[{cand}]" in self.vocab:
                return cand

        # 兜底
        for t in ("ᴴᴴ", "ᴹᴴ", "ᴹᴸ", "ᴴᴸ"):
            if f"[ˈa{t}]" in self.vocab:
                return f"ˈa{t}"
        raise ValueError(f"无法为元音 {nucleus} 匹配 DiaMoE 词表核")

    def compile_entry(
        self,
        ch: str,
        entry: Optional[Dict] = None,
        override_ipa: Optional[str] = None,
        is_rhyme_word: bool = False,
    ) -> Tuple[List[str], List[str]]:
        """
        返回 (bare_tokens, bracketed_tokens)。
        每个 token 均严格通过 `self.vocab` 校验。
        """
        if override_ipa:
            from .phonology import parse_mononuclear_4slot
            syl = parse_mononuclear_4slot(override_ipa)
            onset, medial, nucleus, coda, is_checked = (
                syl.onset, syl.medial, syl.nucleus, syl.coda, syl.is_checked
            )
        else:
            if entry is None:
                entry = self.chars.get(ch)
            if entry is None:
                raise KeyError(f"汉字未收录于汉火词表: {ch!r}")
            onset = entry["onset"]
            medial = entry["medial"]
            nucleus = entry["nucleus"]
            coda = entry["coda"]
            is_checked = bool(entry["is_checked"])

        # 声母映射
        onset_toks = list(ONSET_MAP.get(onset, ["∅"]))
        # 若零声母后直接跟介音 j / w / ɥ，在汉语方言词表中可直接以 [j]/[w]/[ɥ] 起首（避免多余喉塞停顿），
        # 但保留 [∅] 对纯元音起首（如 安 [∅] [ˈaᴴᴴ] [n]）
        if onset_toks == ["∅"] and medial in ("j", "w", "ɥ"):
            onset_toks = []

        # 介音映射（防与同部位声母或主元音重叠成双音节，如 j+j -> j, j+i -> i, w+u -> u）
        med_toks = list(MEDIAL_MAP.get(medial, []))
        if med_toks and onset_toks == med_toks:
            med_toks = []
        elif med_toks == ["j"] and onset_toks in (["ȵ"], ["t͜ɕ"], ["t͜ɕʰ"], ["d͜ʑ"], ["ɕ"], ["ʑ"]):
            # 龈腭音/硬腭鼻音本身已含硬腭色彩，再叠加显式 [j] 在部分方言专家中易偏紧，保留轻滑或按核决定
            if nucleus in ("i", "ɪ", "e", "ɛ"):
                med_toks = []
        if med_toks == ["j"] and nucleus in ("i", "ɪ"):
            med_toks = []
            if not onset_toks:
                onset_toks = ["j"]
        elif med_toks == ["w"] and nucleus in ("u", "ʊ", "o"):
            if nucleus in ("u", "ʊ"):
                med_toks = []
                if not onset_toks:
                    onset_toks = ["w"]
        elif med_toks == ["ɥ"] and nucleus == "y":
            med_toks = []
            if not onset_toks:
                onset_toks = ["ɥ"]

        if not onset_toks and not med_toks:
            onset_toks = ["∅"]

        voiced = onset in VOICED_ONSETS
        tone_cat = self.tone_hints.get(ch, "level")
        if is_checked or coda in ("p", "t", "k"):
            tone_cat = "checked"
        elif coda == "ʔ":
            tone_cat = "rising"
        elif coda in ("h", "s"):
            tone_cat = "departing"

        # 句末舒声韵脚字：若原本是短促上声，在《诗经》叶韵吟诵时采用舒展缓升/平展调（依字行腔、韵脚曼声）
        if is_rhyme_word and tone_cat != "checked" and tone_cat == "rising":
            tone_cat = "level"

        nuc_tok = self._pick_nucleus_token(nucleus, coda, is_checked, voiced, tone_cat)
        coda_toks = list(CODA_MAP.get(coda, []))

        # 防 i + ɪ̯ 或 u + ʊ̯ 同音滑音冗余（如 伊 ʔiei -> [j] [ˈeᴴᴴ] [ɪ̯] 保留，但 i+j -> [ˈiᴴᴴ]）
        if coda_toks == ["ɪ̯"] and nuc_tok.startswith(("ˈi", "ˈɪ")):
            coda_toks = []
        if coda_toks == ["ʊ̯"] and nuc_tok.startswith(("ˈu", "ˈʊ")):
            coda_toks = []

        bare = onset_toks + med_toks + [nuc_tok] + coda_toks
        bracketed = [f"[{t}]" for t in bare]
        for bt in bracketed:
            if bt not in self.vocab:
                raise ValueError(f"字符 {ch!r} 编译出未登录 token {bt!r} (bare={bare})")
        return bare, bracketed

    def compile_text(
        self,
        text: str,
        context_overrides: Optional[Dict[str, str]] = None,
        rhyme_chars: Optional[Set[str]] = None,
    ) -> Tuple[str, str, int]:
        """
        将整句/整章古文编译为 (bare_ipa_str, bracketed_ipa_str, n_chars)。
        字间统一以 `|` 分隔，标点保持裸 token。
        """
        context_overrides = context_overrides or {}
        rhyme_chars = rhyme_chars or set()

        bare_seq: List[str] = []
        br_seq: List[str] = []
        n_chars = 0

        chars_list = list(text.strip())
        for i, ch in enumerate(chars_list):
            if ch.isspace():
                continue
            if ch in PUNCT_MAP:
                p = PUNCT_MAP[ch]
                bare_seq.append(p)
                br_seq.append(p)
                continue
            if ch not in self.chars and ch not in context_overrides:
                continue

            # 判断是否位于句末韵脚位置（后接句号/问号/叹号或文末，或在韵脚集合中）
            next_ch = chars_list[i + 1] if i + 1 < len(chars_list) else "。"
            is_clause_end = next_ch in PUNCT_MAP and PUNCT_MAP[next_ch] in ("。", "？", "！")
            is_rhyme = (ch in rhyme_chars) or is_clause_end

            b_toks, br_toks = self.compile_entry(
                ch,
                override_ipa=context_overrides.get(ch),
                is_rhyme_word=is_rhyme,
            )
            bare_seq.extend(b_toks)
            bare_seq.append("|")
            br_seq.extend(br_toks)
            br_seq.append("|")
            n_chars += 1

        return " ".join(bare_seq), " ".join(br_seq), n_chars

    def verify_full_lexicon(self) -> Dict[str, int]:
        """对全表 13,074 个汉字逐一编译校验，确保 0 OOV！"""
        ok = 0
        errors: List[str] = []
        for ch, entry in self.chars.items():
            try:
                self.compile_entry(ch, entry=entry)
                ok += 1
            except Exception as exc:
                errors.append(f"{ch}: {exc}")
        if errors:
            raise RuntimeError(f"词表存在 {len(errors)} 个编译异常: {errors[:10]}")
        return {"total_verified": ok, "oov_errors": 0}


if __name__ == "__main__":
    comp = HanfireDiaMoECompiler()
    stats = comp.verify_full_lexicon()
    print("[全表校验通过]", stats)
    for poem_id in ("guanju_1", "jianjia_1"):
        st = next(x for x in comp.anthology if x["id"] == poem_id)
        rhyme_set = {c for grp in st["rhymes"] for c in grp}
        bare, br, n = comp.compile_text(
            st["text"],
            context_overrides=st.get("context_rhyme_overrides", {}),
            rhyme_chars=rhyme_set,
        )
        print(f"\n=== {st['title']} ({n}字) ===")
        print("BARE:", bare)
        print("BRAC:", br)
