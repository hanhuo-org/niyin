#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZipVoice 华夏声韵流形手术引擎 + 一字一拍吟诵时长控制器 (zipvoice_zh_manifold.py)

根因修复说明：
1. 为什么 V2 听起来“很多地方是直接读音标的感觉”？
   - ZipVoice 的 `tokens.txt` 分为两个完全割裂的子空间：
     * IDs 14..158 是美式英语 (`en-us`) IPA 子空间，且其中 139 个符号（含 `q, ɢ, ɯ, ʰ, ˤ, g, ʦ, ̊`）
       根本从未训练过（L2 范数仅 ~1.3 的随机高斯噪声，正常训练词元范数为 ~8.2）！
     * 更致命的是，ZipVoice 内部用 `prepare_avg_tokens_durations` 给每个 token 均分相等的帧数！
       当一个汉字被拆成 6 个英文 IPA 字符 + 空格时，每个音标字母被硬生生拖长到 120ms 逐个拼读！
   - 反观 IDs 159..355（`b0..zh0` 声母 + `a1..uo4` 带调韵母）是在 5 万小时原生中文语音（Emilia）上训练的
     “声母 + 韵母”双 token 汉字流形，零空格、字正腔圆、声韵自然协同发音！

2. 本模块的三大核心机制：
   - 【流形内声韵向量代数手术】：利用 `embed.weight` 中未使用的空闲槽位（IDs 46..118），在中文声母/韵母流形内
     通过向量代数精确合成上古全浊声母（`b, d, g, dz, z, dʑ, ʑ, dʒ, ʒ, ɣ`）、鼻音首（`ɲ, ŋ`）、清化响音
     （`hm, hn, hŋ, hl`）、上古特色合口/之幽韵母（`iə, iwən, iwai, iwei, iu`）以及无爆破唯闭尾（`-m, -p, -t, -k`），
     L2 范数严格对齐中文预训练词元（~8.2），彻底消灭未训练随机噪声与美式英语拼读腔！
   - 【一字一拍同步时长分配器】：废除按 token 个数均分时长的 `prepare_avg_tokens_durations`，改为严格按“汉字”
     分配帧数：
     * 普通舒声字 = 1.00 拍（声母 ~28% + 韵母 ~72%，若带 `-m` 尾则内部切出 ~22% 闭唇鼻尾，总时长仍严格等于 1 字！）
     * 入声促声字 = 0.70 拍（仄短急收：声母 + 短促降调核 + ~40ms 唯闭塞尾 `-p/-t/-k`，绝不弹音拆拍！）
     * 诗经韵脚字 = 1.48 拍（平长曼声：辅音声母保持原速不拖泥带水，仅延长韵母核，自然呈现古雅吟诵余韵！）
   - 【古四声平仄行腔】：融合 `pron_shelf_v3.tsv` 的古上声 (`*-ʔ`) / 古去声 (`*-h/-s`) / 古入声 (`*-p/-t/-k`)
     与声母清浊阴阳（清平阴平 Tone 1 高平曼声、浊平阳平 Tone 2 沉吟低升、上声 Tone 3 微抑扬、去声 Tone 4 叹降）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from pypinyin import Style, lazy_pinyin
from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials

from zipvoice.utils.common import get_tokens_index, make_pad_mask
from .diamoe_compiler import DEFAULT_LEXICON_PATH, SHELF_TSV_PATH, VOICED_ONSETS, load_historical_tone_hints

# 预留的纯净死槽位池（100% 位于 ID < 159 的非英语生僻 IPA / 数字区及 356..359 ê 区，L2 范数均 ~1.3，绝不触碰任何普通话拼音词元 159..355！）
DEAD_SLOT_POOL: List[int] = [
    1, 2, 6, 7, 9, 16, 29, 37, 40, 42, 43, 45,
    46, 47, 48, 49, 52, 53, 55, 56, 57, 58,
    63, 64, 65, 67, 68, 69, 70, 71, 72, 73,
    75, 77, 78, 79, 80, 81, 82, 83, 84, 85,
    86, 87, 89, 90, 91, 93, 94, 95, 97, 98,
    99, 101, 103, 104, 105, 106, 107, 110, 111, 112,
    113, 114, 115, 116, 117, 118, 123, 124, 125, 127,
    129, 130, 132, 133, 134, 135, 136, 137, 138, 139,
    142, 143, 145, 146, 147, 148, 149, 150, 151, 152,
    153, 154, 155, 156, 157, 158, 356, 357, 358, 359,
]


def load_zv_token2id(tok_path: Path) -> Dict[str, int]:
    t2i: Dict[str, int] = {}
    for ln in tok_path.read_text(encoding="utf-8").splitlines():
        parts = ln.rstrip("\r\n").split("\t")
        if len(parts) == 2:
            t2i[parts[0]] = int(parts[1])
    return t2i


def _scale_to_norm(vec: torch.Tensor, target_norm: float) -> torch.Tensor:
    n = float(vec.norm().item())
    if n < 1e-8:
        return vec
    return vec * (target_norm / n)


def inject_hanfire_zh_manifold_embeddings(model: torch.nn.Module, base_t2i: Dict[str, int]) -> Dict[str, int]:
    """
    在 `model.embed.weight` 的空闲死槽位中注入华夏声韵流形合成向量，返回扩充后的 `t2i` 字典。
    """
    t2i = dict(base_t2i)
    emb = model.embed.weight.data  # [360, 192]

    zh_inits = ["b0", "p0", "m0", "f0", "d0", "t0", "n0", "l0", "g0", "k0", "h0",
                "j0", "q0", "x0", "z0", "c0", "s0", "zh0", "ch0", "sh0"]
    zh_finals = ["a1", "a2", "a3", "a4", "ai1", "an1", "ang1", "ao1", "e1", "ei1",
                 "en1", "eng1", "i1", "ia1", "ian1", "iang1", "iao1", "ie1", "in1",
                 "iu1", "o1", "ong1", "ou1", "u1", "ua1", "uan1", "uang1", "ui1", "un1", "uo1"]

    init_norm = float(torch.stack([emb[t2i[k]].norm() for k in zh_inits]).mean().item())
    final_norm = float(torch.stack([emb[t2i[k]].norm() for k in zh_finals]).mean().item())

    slot_iter = iter(DEAD_SLOT_POOL)

    def register(name: str, vec: torch.Tensor, is_final: bool = False) -> None:
        slot = next(slot_iter)
        emb[slot] = _scale_to_norm(vec, final_norm if is_final else init_norm)
        t2i[name] = slot

    # 1. 全浊塞音 / 塞擦音 / 擦音 / 鼻音 / 清化响音声母（强化真浊声带振动权重，防止被清声母吞没！）
    register("OC_b",  0.46 * emb[t2i["b0"]]  + 0.62 * emb[t2i["b"]] + 0.14 * emb[t2i["m0"]])
    register("OC_d",  0.46 * emb[t2i["d0"]]  + 0.62 * emb[t2i["d"]] + 0.14 * emb[t2i["n0"]])
    register("OC_g",  0.45 * emb[t2i["g0"]]  + 0.65 * emb[t2i["ɡ"]])
    register("OC_dz", 0.48 * emb[t2i["z0"]]  + 0.60 * emb[t2i["z"]])
    register("OC_z",  0.40 * emb[t2i["s0"]]  + 0.68 * emb[t2i["z"]])
    register("OC_dʑ", 0.48 * emb[t2i["j0"]]  + 0.58 * emb[t2i["ʒ"]])
    register("OC_ʑ",  0.45 * emb[t2i["x0"]]  + 0.62 * emb[t2i["ʒ"]])
    register("OC_dʒ", 0.48 * emb[t2i["zh0"]] + 0.58 * emb[t2i["ʒ"]])
    register("OC_ʒ",  0.45 * emb[t2i["sh0"]] + 0.60 * emb[t2i["r0"]])
    register("OC_ɣ",  0.48 * emb[t2i["h0"]]  + 0.54 * emb[t2i["ɡ"]] + 0.15 * emb[t2i["w0"]])
    register("OC_ɲ",  0.66 * emb[t2i["n0"]]  + 0.42 * emb[t2i["y0"]])
    register("OC_ŋ",  0.38 * emb[t2i["n0"]]  + 0.25 * emb[t2i["g0"]] + 0.52 * emb[t2i["ŋ"]])
    register("OC_hm", 0.48 * emb[t2i["h0"]]  + 0.62 * emb[t2i["m0"]])
    register("OC_hn", 0.48 * emb[t2i["h0"]]  + 0.62 * emb[t2i["n0"]])
    register("OC_hŋ", 0.48 * emb[t2i["h0"]]  + 0.38 * emb[t2i["n0"]] + 0.40 * emb[t2i["ŋ"]])
    register("OC_hl", 0.45 * emb[t2i["h0"]]  + 0.55 * emb[t2i["l0"]] + 0.28 * emb[t2i["ɬ"]])
    register("OC_ʔ",  0.75 * emb[t2i["ʔ"]]   + 0.25 * emb[t2i["e5"]])

    # 2. 上古特色单核韵母（四声齐全 1..4，彻底修复 ie 污染导致的“之->皆、水->血、伊->也、不->波”！）
    for tone in ("1", "2", "3", "4"):
        # 之部细音 -iə（如 之 tɕiə、子 tsiə、思 siə、侧 tʃiək）：严禁混入 ie（否则变成“皆”），用 i 介音 + 央元音 e[ə]！
        register(f"OC_iə{tone}",   0.36 * emb[t2i[f"i{tone}"]]   + 0.72 * emb[t2i[f"e{tone}"]], is_final=True)
        # 之部唇合口 -wə（如 不 piwə）：央元音 e[ə] + 轻圆唇 u，绝不变成“波 uo”！
        register(f"OC_wə{tone}",   0.70 * emb[t2i[f"e{tone}"]]   + 0.36 * emb[t2i[f"u{tone}"]], is_final=True)
        # 微/脂部细音 -iei / -iəi（如 伊 ʔiei、寐 miəi）：必须带 -i 闭口滑音 ei+i，绝不变成“也/灭 ie”！
        register(f"OC_iei{tone}",  0.68 * emb[t2i[f"ei{tone}"]]  + 0.42 * emb[t2i[f"i{tone}"]], is_final=True)
        # 文部合口细音 -ɥən（如 君 kiwən）：圆唇前高介音 + 央鼻韵 [ɥən]
        register(f"OC_iwən{tone}", 0.58 * emb[t2i[f"un{tone}"]]  + 0.48 * emb[t2i[f"in{tone}"]], is_final=True)
        # 歌部合口细音 -ɥai（如 为 ɣiwai）
        register(f"OC_iwai{tone}", 0.72 * emb[t2i[f"uai{tone}"]] + 0.32 * emb[t2i[f"ia{tone}"]], is_final=True)
        # 微/脂部合口细音 -ɥəi/-ɥei（如 谓 ɣiwəi、水 ɕiwei）：严禁混入 ie/ue（否则 x0+ue 变成“血”！），用纯正 ui+ei！
        register(f"OC_iwei{tone}", 0.72 * emb[t2i[f"ui{tone}"]]  + 0.35 * emb[t2i[f"ei{tone}"]], is_final=True)
        # 幽部纯正细音 -iu [ju]（如 鸠 kiu、洲 tiu、逑 giu）
        register(f"OC_iu{tone}",   0.62 * emb[t2i[f"iu{tone}"]]  + 0.45 * emb[t2i[f"u{tone}"]], is_final=True)
        # 幽/宵四等半闭敛口叠韵 -eu（如 窈 ʔeu、窕 deu：半闭前元音收圆唇，严禁张大口读成 ao/iao！）
        register(f"OC_eu{tone}",   0.52 * emb[t2i[f"ei{tone}"]]  + 0.56 * emb[t2i[f"ou{tone}"]], is_final=True)
        # 侵/谈部闭口韵过渡核
        register(f"OC_əm{tone}",   0.74 * emb[t2i[f"en{tone}"]]  + 0.35 * emb[t2i["m"]], is_final=True)
        register(f"OC_am{tone}",   0.74 * emb[t2i[f"an{tone}"]]  + 0.35 * emb[t2i["m"]], is_final=True)
        register(f"OC_iam{tone}",  0.74 * emb[t2i[f"ian{tone}"]] + 0.35 * emb[t2i["m"]], is_final=True)
        register(f"OC_iəm{tone}",  0.74 * emb[t2i[f"in{tone}"]]  + 0.35 * emb[t2i["m"]], is_final=True)

    # 3. 自然收口韵尾（用于闭口 -m 及入声融合韵母：严禁混入普通话声母 b0/d0/g0，严禁硬切静音！）
    # 用柔和鼻音/喉塞过渡融合部位特征，构建单 token 入声韵母，让流匹配自然收住气息，绝无“突然掐断感”也绝无“咋可”弹音！
    register("OC_coda_m", 0.80 * emb[t2i["m"]] + 0.22 * emb[t2i["n2"]])
    register("OC_ak_chk",  0.78 * emb[t2i["a4"]]  + 0.18 * emb[t2i["k"]] + 0.14 * emb[t2i["ʔ"]], is_final=True)
    register("OC_iak_chk", 0.78 * emb[t2i["ia4"]] + 0.18 * emb[t2i["k"]] + 0.14 * emb[t2i["ʔ"]], is_final=True)
    register("OC_ək_chk",  0.78 * emb[t2i["e4"]]  + 0.18 * emb[t2i["k"]] + 0.14 * emb[t2i["ʔ"]], is_final=True)
    register("OC_iək_chk", 0.32 * emb[t2i["i4"]]  + 0.52 * emb[t2i["e4"]] + 0.18 * emb[t2i["k"]] + 0.14 * emb[t2i["ʔ"]], is_final=True)
    register("OC_uk_chk",  0.78 * emb[t2i["u4"]]  + 0.18 * emb[t2i["k"]] + 0.14 * emb[t2i["ʔ"]], is_final=True)
    register("OC_iuk_chk", 0.55 * emb[t2i["iu4"]] + 0.30 * emb[t2i["u4"]] + 0.18 * emb[t2i["k"]] + 0.14 * emb[t2i["ʔ"]], is_final=True)
    register("OC_iet_chk", 0.76 * emb[t2i["ie4"]] + 0.18 * emb[t2i["t"]] + 0.14 * emb[t2i["ʔ"]], is_final=True)
    register("OC_at_chk",  0.76 * emb[t2i["a4"]]  + 0.18 * emb[t2i["t"]] + 0.14 * emb[t2i["ʔ"]], is_final=True)
    register("OC_ət_chk",  0.76 * emb[t2i["e4"]]  + 0.18 * emb[t2i["t"]] + 0.14 * emb[t2i["ʔ"]], is_final=True)
    register("OC_ap_chk",  0.76 * emb[t2i["a4"]]  + 0.18 * emb[t2i["p"]] + 0.14 * emb[t2i["ʔ"]], is_final=True)
    register("OC_əp_chk",  0.76 * emb[t2i["e4"]]  + 0.18 * emb[t2i["p"]] + 0.14 * emb[t2i["ʔ"]], is_final=True)

    # 4. 近古音 (中原音韵) 与中古音 (切韵) 专属补充槽位（总占用 86 / 96，完全隔离于普通话区）
    register("EM_ʋ", 0.60 * emb[t2i["w0"]] + 0.45 * emb[t2i["v"]])
    for tone in ("1", "2", "3", "4"):
        # 中古咍/灰韵 ɒi（如 在 dzɒi、海 xɒi）
        register(f"MC_ɒi{tone}", 0.45 * emb[t2i[f"uo{tone}"]] + 0.62 * emb[t2i[f"ai{tone}"]], is_final=True)
        # 中古鱼/虞韵 ɨə / ʉo（如 处 tɕʰɨə、树 dʑʉo）
        register(f"MC_ɨə{tone}", 0.56 * emb[t2i[f"e{tone}"]]  + 0.38 * emb[t2i[f"u{tone}"]] + 0.22 * emb[t2i[f"i{tone}"]], is_final=True)

    return t2i


def tokenize_mandarin_prompt_exact(text: str, t2i: Dict[str, int]) -> Tuple[List[Tuple[int, int]], int]:
    """
    使用与 ZipVoice `EmiliaTokenizer` 100% 一致的 `pypinyin` 规则切分参考音频中文文本，
    返回 `(token_id_and_weight_list, n_hanzi)`，供精确帧长分配。
    每个 token 条目为 `(tok_id, rel_frames_weight)`，参考句单字标准权重 = 100。
    """
    norm_punct = {
        "，": ",", "。": ".", "！": "!", "？": "?", "；": ";", "：": ":", "、": ",",
    }
    for k, v in norm_punct.items():
        text = text.replace(k, v)

    py_list = lazy_pinyin(
        list(text),
        style=Style.TONE3,
        tone_sandhi=True,
        neutral_tone_with_five=True,
    )
    items: List[Tuple[int, int]] = []
    n_chars = 0
    for py in py_list:
        py = py.strip()
        if not py:
            continue
        if len(py) >= 2 and py[:-1].isalpha() and py[-1] in ("1", "2", "3", "4", "5"):
            init = to_initials(py, strict=False)
            fin = to_finals_tone3(py, strict=False, neutral_tone_with_five=True)
            # 若遇到轻声 5 落在未训练槽位（如 uan5, ian5），优雅回退到 1/4 声以避开死向量
            if fin not in t2i or fin.endswith("5"):
                alt_fin = fin[:-1] + "4"
                if fin not in t2i and alt_fin in t2i:
                    fin = alt_fin
            init_tok = (init + "0") if init else ""
            if init_tok and init_tok in t2i and fin in t2i:
                items.append((t2i[init_tok], 32))
                items.append((t2i[fin], 68))
                n_chars += 1
            elif fin in t2i:
                items.append((t2i[fin], 100))
                n_chars += 1
        elif py in t2i:
            items.append((t2i[py], 16))
    return items, max(n_chars, 1)


class HanfireZipVoiceCompiler:
    """汉火上古音 4槽单核 -> ZipVoice 华夏声韵流形编译器（附一字一拍吟诵权重）。"""

    ONSET_TO_ZV: Dict[str, str] = {
        "": "",
        "∅": "",
        "ʔ": "",
        "p": "b0",
        "pʰ": "p0",
        "b": "OC_b",
        "m": "m0",
        "m̥": "OC_hm",
        "f": "f0",
        "v": "EM_ʋ",
        "ʋ": "EM_ʋ",
        "t": "d0",
        "tʰ": "t0",
        "d": "OC_d",
        "n": "n0",
        "ʈ": "zh0",
        "ʈʰ": "ch0",
        "ɖ": "OC_dʒ",
        "ɳ": "n0",
        "n̥": "OC_hn",
        "l": "l0",
        "l̥": "OC_hl",
        "r": "l0",
        "ɻ": "r0",
        "ts": "z0",
        "t͡s": "z0",
        "tsʰ": "c0",
        "t͡sʰ": "c0",
        "dz": "OC_dz",
        "d͡z": "OC_dz",
        "s": "s0",
        "z": "OC_z",
        "tɕ": "j0",
        "t͡ɕ": "j0",
        "tɕʰ": "q0",
        "t͡ɕʰ": "q0",
        "dʑ": "OC_dʑ",
        "d͡ʑ": "OC_dʑ",
        "ɕ": "x0",
        "ʑ": "OC_ʑ",
        "ɲ": "OC_ɲ",
        "tʃ": "zh0",
        "t͡ʃ": "zh0",
        "ʈʂ": "zh0",
        "tʃʰ": "ch0",
        "t͡ʃʰ": "ch0",
        "ʈʂʰ": "ch0",
        "dʒ": "OC_dʒ",
        "d͡ʒ": "OC_dʒ",
        "ɖʐ": "OC_dʒ",
        "ʃ": "sh0",
        "ʃʰ": "ch0",
        "ʂ": "sh0",
        "ʒ": "OC_ʒ",
        "ʐ": "r0",
        # V8 补全：此前缺失导致 1,455 字声母被静默丢弃（近古照组 804 字、中古日母 186 字、庄组 235 字等）
        "tʂ": "zh0", "t͡ʂ": "zh0",
        "tʂʰ": "ch0", "t͡ʂʰ": "ch0",
        "dʐ": "OC_dʒ", "d͡ʐ": "OC_dʒ",
        "ɲʑ": "OC_ɲ", "ɲ͡ʑ": "OC_ɲ",
        # 上古圆唇软腭音与前置辅音/复声母：取主声母（前置成分暂不建模）
        "kʷ": "g0", "kʷʰ": "k0", "gʷ": "OC_g", "ɡʷ": "OC_g",
        "sŋ": "OC_ŋ", "st": "d0", "mt": "d0", "ss": "s0", "ml": "l0", "skʰ": "k0", "tk": "g0",
        "kl": "g0", "sg": "OC_g", "mk": "g0", "sn": "n0", "sgʷ": "OC_g", "mn": "n0",
        "k": "g0",
        "kʰ": "k0",
        "g": "OC_g",
        "ɡ": "OC_g",
        "ŋ": "OC_ŋ",
        "ŋ̊": "OC_hŋ",
        "x": "h0",
        "h": "h0",
        "χ": "h0",
        "ɣ": "OC_ɣ",
        "ɦ": "OC_ɣ",
        "q": "g0",
        "qʰ": "k0",
        "ɢ": "OC_g",
        "j": "y0",
        "w": "w0",
        "ɥ": "y0",
    }

    def __init__(
        self,
        t2i: Dict[str, int],
        lexicon_path: Path = DEFAULT_LEXICON_PATH,
        shelf_tsv: Path = SHELF_TSV_PATH,
    ) -> None:
        raw = json.loads(lexicon_path.read_text(encoding="utf-8"))
        self.chars: Dict[str, Dict] = raw["chars"]
        self.anthology: List[Dict] = raw.get("shijing_anthology", [])
        self.t2i = t2i
        self.tone_hints = load_historical_tone_hints(shelf_tsv)

    def _select_tone_digit(self, ch: str, onset: str, coda: str, is_checked: bool, is_rhyme_word: bool) -> str:
        if is_checked or coda in ("p", "t", "k"):
            return "4"
        hint = self.tone_hints.get(ch, "level")
        if coda in ("h", "s") or hint == "departing":
            return "4"
        if coda == "ʔ" or hint == "rising":
            return "3"
        # 正常上古平声读音：统一采用平稳舒展的平调 (1)，不再套现代普通话阳平 35 上扬调，消除“现代腔”！
        return "1"

    def _resolve_final_tokens(
        self,
        medial: str,
        nucleus: str,
        coda: str,
        tone: str,
        is_checked: bool,
    ) -> Tuple[str, Optional[str]]:
        """
        将 (medial, nucleus, coda, tone) 映射为 `(main_final_tok, optional_coda_tok)`。
        """
        m = medial if medial in ("j", "w", "ɥ") else ""
        v = nucleus or "a"
        c = coda

        # 1. 闭口鼻韵尾 -m：使用预融合闭口韵核 + 轻收双唇 m
        if c == "m":
            if m in ("j", "ɥ"):
                base = f"OC_iam{tone}" if v in ("a", "ɑ", "æ", "e", "ɛ") else f"OC_iəm{tone}"
            elif m == "w":
                base = f"uan{tone}"
            else:
                base = f"OC_am{tone}" if v in ("a", "ɑ", "æ") else f"OC_əm{tone}"
            return base, "OC_coda_m"

        # 2. 入声单 token 自然融合韵母（不拆分独立塞音尾 token，不加人工切音门：既无“咋可”弹音，又保留自然胸腔余韵不突兀掐断！）
        if is_checked or c in ("p", "t", "k"):
            if c == "p":
                return ("OC_ap_chk" if v in ("a", "ɑ", "æ") else "OC_əp_chk"), None
            if c == "t":
                if m in ("j", "ɥ") or v in ("e", "ɛ", "i", "ɪ"):
                    return "OC_iet_chk", None
                return ("OC_at_chk" if v in ("a", "ɑ", "æ") else "OC_ət_chk"), None
            # c == "k" or default checked
            # V8-A：按韵腹前后/开口度分流，不再把所有合口 -k 都塞进 OC_uk_chk（此前 28 种韵母挤在一起）
            back_round = v in ("u", "ʊ", "ʉ", "o")
            low = v in ("a", "ɑ", "æ", "ɐ", "ɔ", "ɒ")
            if m == "j" or m == "ɥ":
                if low or v in ("e", "ɛ"):
                    return "OC_iak_chk", None
                if back_round:
                    return "OC_iuk_chk", None
                return "OC_iək_chk", None
            if back_round:
                return "OC_uk_chk", None
            if low:
                return "OC_ak_chk", None
            return "OC_ək_chk", None

        # 3. 阳声韵 -n / *-r(上古舌尖流音尾与元文部通押) / -ŋ 与阴声韵开尾 / -j / -w（单 token 浑然一体！）
        if c in ("n", "r"):
            if m == "ɥ":
                return (f"uan{tone}" if v in ("a", "ɑ", "æ", "ɐ", "e", "ɛ", "ɜ") else f"un{tone}"), None
            if m == "w":
                return (f"uan{tone}" if v in ("a", "ɑ", "æ", "ɐ", "e", "ɛ", "ɔ", "ɒ") else f"un{tone}"), None
            if m == "j":
                if v in ("u", "ʊ", "ʉ", "o", "ɔ"):
                    return f"un{tone}", None
                return (f"ian{tone}" if v in ("a", "ɑ", "æ", "ɐ", "e", "ɛ", "ɜ") else f"in{tone}"), None
            if v == "e":
                return f"ian{tone}", None
            if v in ("i", "ɪ"):
                return f"in{tone}", None
            if v in ("u", "ʊ", "ʉ", "y", "o"):
                return f"un{tone}", None
            return (f"an{tone}" if v in ("a", "ɑ", "æ", "ɐ", "ɛ", "ɔ", "ɒ") else f"en{tone}"), None

        if c == "ŋ":
            if m in ("w", "ɥ"):
                # V8-A：合口低/前元音 (wæŋ wɐŋ wɛŋ wɔŋ…) → uang；撮口前元音 (ɥæŋ ɥɛŋ) → iong；其余 → ong
                if m == "ɥ" and v in ("æ", "ɐ", "e", "ɛ", "i"):
                    return (f"iong{tone}" if f"iong{tone}" in self.t2i else "iong3"), None
                return (f"uang{tone}" if v in ("a", "ɑ", "æ", "ɐ", "e", "ɛ", "ɔ", "ɒ") else f"ong{tone}"), None
            if m == "j":
                if v in ("a", "ɑ"):
                    return f"iang{tone}", None
                if v in ("u", "o", "ɔ"):
                    return (f"iong{tone}" if f"iong{tone}" in self.t2i else "iong3"), None
                return f"ing{tone}", None
            if v in ("i", "ɪ"):
                return f"ing{tone}", None
            # V8-A：ɔŋ/ɒŋ（江韵、唐韵合口…）开口度大，近 ang 而非 ong[uŋ]（江 kɔŋ 不再读成“工”）
            if v in ("a", "ɑ", "æ", "ɔ", "ɒ"):
                return f"ang{tone}", None
            if v in ("o", "u", "ʊ"):
                return f"ong{tone}", None
            return f"eng{tone}", None

        if c in ("j", "i"):
            if m in ("w", "ɥ"):
                return (f"OC_iwai{tone}" if v in ("a", "ɑ", "æ") else f"OC_iwei{tone}"), None
            if m == "j":
                return (f"OC_iwai{tone}" if v in ("a", "ɑ", "æ") else f"OC_iei{tone}"), None
            if v in ("e", "ɛ"):
                return f"OC_iei{tone}", None
            if v == "ɒ":
                return f"MC_ɒi{tone}", None
            if v in ("u", "ʊ", "o", "ɔ"):
                return f"ui{tone}", None
            return (f"ai{tone}" if v in ("a", "ɑ", "æ", "ɐ") else f"ei{tone}"), None

        if c in ("w", "u"):
            if m in ("j", "ɥ"):
                if v in ("a", "ɑ", "æ", "ɐ", "e", "ɛ"):
                    return f"iao{tone}", None
                return f"OC_iu{tone}", None
            if v in ("e", "ɛ"):
                return f"OC_eu{tone}", None
            return (f"ao{tone}" if v in ("a", "ɑ", "æ", "ɐ", "ʌ", "ɒ") else f"ou{tone}"), None

        # 纯开尾 (c == "" or c in {"ʔ", "h", "s"})
        if m == "j":
            if v in ("a", "ɑ", "æ", "ɐ"):
                return f"ia{tone}", None
            if v in ("e", "ɛ"):
                return f"ie{tone}", None
            if v in ("u", "ʊ"):
                return f"OC_iu{tone}", None
            if v in ("o", "ɔ"):
                return f"iao{tone}", None
            if v in ("i", "ɪ"):
                return f"i{tone}", None
            return f"OC_iə{tone}", None

        if m == "ɥ":
            if v in ("a", "ɑ", "æ", "ɐ"):
                return f"ua{tone}", None
            if v in ("e", "ɛ"):
                return f"ue{tone}", None
            if v in ("i", "ɪ", "y"):
                return (f"v{tone}" if f"v{tone}" in self.t2i else "v2"), None
            return f"OC_wə{tone}", None

        if m == "w":
            if v in ("a", "ɑ", "æ", "ɐ"):
                return f"ua{tone}", None
            if v in ("o", "ɔ", "ə"):
                return f"uo{tone}", None
            if v in ("e", "ɛ"):
                return f"OC_iwei{tone}", None
            return f"u{tone}", None

        # 无介音纯单核
        if v in ("a", "ɑ", "æ", "ɐ"):
            return f"a{tone}", None
        if v in ("e", "ɛ"):
            return f"ie{tone}", None
        if v in ("i", "ɪ"):
            return f"i{tone}", None
        if v in ("o", "ɔ", "ɒ"):
            # V8-A：无介音 o 用单元音 o，不用自带 w 介音的 uo（孤 ko 不再像“郭”）
            return (f"o{tone}" if f"o{tone}" in self.t2i else f"uo{tone}"), None
        if v in ("u", "ʊ"):
            return f"u{tone}", None
        if v in ("ɨ", "ʉ"):
            return f"MC_ɨə{tone}", None
        if v == "y":
            return (f"v{tone}" if f"v{tone}" in self.t2i else "v2"), None
        return f"e{tone}", None

    def compile_char(
        self,
        ch: str,
        override_ipa: Optional[str] = None,
        is_rhyme_word: bool = False,
        chant_mode: bool = False,
    ) -> Tuple[List[Tuple[int, int]], str]:
        """
        将单个汉字编译为 `([(tok_id, rel_weight), ...], debug_str)`。
        正常读音模式 (`chant_mode=False`)：
          - 舒声单字总权重 = 100（不再人工拉长句末韵脚）；
          - 入声促声字总权重 = 80（自然短促收音，不硬切、不弹音）。
        """
        if override_ipa:
            from .phonology import parse_mononuclear_4slot
            s = parse_mononuclear_4slot(override_ipa)
            onset, medial, nucleus, coda, is_checked = s.onset, s.medial, s.nucleus, s.coda, s.is_checked
        else:
            entry = self.chars[ch]
            onset = entry["onset"]
            medial = entry["medial"]
            nucleus = entry["nucleus"]
            coda = entry["coda"]
            is_checked = bool(entry["is_checked"])

        init_tok = self.ONSET_TO_ZV.get(onset, "")
        if not init_tok:
            if medial in ("j", "ɥ"):
                init_tok = "y0"
            elif medial == "w":
                init_tok = "w0"

        tone = self._select_tone_digit(ch, onset, coda, is_checked, is_rhyme_word and chant_mode)
        fin_tok, coda_tok = self._resolve_final_tokens(medial, nucleus, coda, tone, is_checked)

        items: List[Tuple[int, int]] = []
        dbg_parts: List[str] = []

        if is_checked or coda in ("p", "t", "k"):
            total_w = (88 if is_rhyme_word else 78) if chant_mode else 80
            if init_tok:
                w_i = 24
                w_f = total_w - w_i
                items.append((self.t2i[init_tok], w_i))
                items.append((self.t2i[fin_tok], w_f))
                dbg_parts.extend([init_tok, fin_tok])
            else:
                items.append((self.t2i[fin_tok], total_w))
                dbg_parts.append(fin_tok)
        elif coda_tok is not None:
            # -m 闭口舒声字
            total_w = 148 if (is_rhyme_word and chant_mode) else 100
            if init_tok:
                w_i = 28
                w_c = 22
                w_f = total_w - w_i - w_c
                items.append((self.t2i[init_tok], w_i))
                items.append((self.t2i[fin_tok], w_f))
                items.append((self.t2i[coda_tok], w_c))
                dbg_parts.extend([init_tok, fin_tok, coda_tok])
            else:
                w_c = 22
                w_f = total_w - w_c
                items.append((self.t2i[fin_tok], w_f))
                items.append((self.t2i[coda_tok], w_c))
                dbg_parts.extend([fin_tok, coda_tok])
        else:
            total_w = 148 if (is_rhyme_word and chant_mode) else 100
            if init_tok:
                w_i = 30
                w_f = total_w - w_i
                items.append((self.t2i[init_tok], w_i))
                items.append((self.t2i[fin_tok], w_f))
                dbg_parts.extend([init_tok, fin_tok])
            else:
                items.append((self.t2i[fin_tok], total_w))
                dbg_parts.append(fin_tok)

        return items, "+".join(dbg_parts)

    def compile_clause(
        self,
        clause_text: str,
        context_overrides: Optional[Dict[str, str]] = None,
        rhyme_last: bool = False,
        end_punct: str = ".",
        chant_mode: bool = False,
    ) -> Tuple[List[Tuple[int, int]], str]:
        context_overrides = context_overrides or {}
        hanzi = [c for c in clause_text if c in self.chars or c in context_overrides]
        all_items: List[Tuple[int, int]] = []
        dbg: List[str] = []
        for idx, ch in enumerate(hanzi):
            is_rhyme = rhyme_last and (idx == len(hanzi) - 1)
            c_items, c_dbg = self.compile_char(
                ch,
                override_ipa=context_overrides.get(ch),
                is_rhyme_word=is_rhyme,
                chant_mode=chant_mode,
            )
            all_items.extend(c_items)
            dbg.append(f"{ch}[{c_dbg}]")
        if end_punct and end_punct in self.t2i:
            p_w = 38 if end_punct == "." else 26
            all_items.append((self.t2i[end_punct], p_w))
            dbg.append(end_punct)
        return all_items, " ".join(dbg)


def _weights_to_exact_frames(weights: List[int], target_total_frames: int) -> List[int]:
    """按相对权重精确分配整数帧数，保证总和严格等于 `target_total_frames` 且每个 token ≥ 1 帧。"""
    n = len(weights)
    if n == 0:
        return []
    target_total_frames = max(target_total_frames, n)
    w_arr = np.array(weights, dtype=np.float64)
    raw = w_arr / np.sum(w_arr) * target_total_frames
    durs = np.maximum(1, np.floor(raw).astype(np.int64))
    diff = int(target_total_frames - int(np.sum(durs)))
    if diff > 0:
        frac = raw - np.floor(raw)
        order = np.argsort(-frac)
        for i in range(diff):
            durs[order[i % n]] += 1
    elif diff < 0:
        order = np.argsort(-durs)
        idx = 0
        while diff < 0 and idx < n * 4:
            k = order[idx % n]
            if durs[k] > 1:
                durs[k] -= 1
                diff += 1
            idx += 1
    return [int(x) for x in durs]


@torch.no_grad()
def sample_with_character_durations(
    *,
    model: torch.nn.Module,
    prompt_items: List[Tuple[int, int]],
    target_items: List[Tuple[int, int]],
    prompt_features: torch.Tensor,
    prompt_features_lens: torch.Tensor,
    frames_per_100w: float = 27.0,
    num_step: int = 16,
    guidance_scale: float = 1.25,
    t_shift: float = 0.5,
    seed: int = 42,
) -> torch.Tensor:
    """
    使用一字一拍精确吟诵帧长表执行 ZipVoice 流匹配采样，返回生成段的 mel 特征 `(1, T_gen, 100)`。
    """
    device = next(model.parameters()).device
    pmt_ids = [x[0] for x in prompt_items]
    pmt_ws = [x[1] for x in prompt_items]
    tgt_ids = [x[0] for x in target_items]
    tgt_ws = [x[1] for x in target_items]

    pmt_frames = int(prompt_features_lens[0].item())
    tgt_frames = max(int(round(sum(tgt_ws) / 100.0 * frames_per_100w)), len(tgt_ids) + 4)
    total_frames = pmt_frames + tgt_frames

    pmt_durs = _weights_to_exact_frames(pmt_ws, pmt_frames)
    tgt_durs = _weights_to_exact_frames(tgt_ws, tgt_frames)
    cat_ids = [pmt_ids + tgt_ids]
    cat_durs = [pmt_durs + tgt_durs]

    cat_embed, _ = model.forward_text_embed(cat_ids)
    features_lens = torch.tensor([total_frames], dtype=torch.int64, device=device)
    padding_mask = make_pad_mask(features_lens, max_len=total_frames)
    tokens_index = get_tokens_index(cat_durs, total_frames).to(device)
    text_condition = torch.gather(
        cat_embed,
        dim=1,
        index=tokens_index.unsqueeze(-1).expand(1, total_frames, cat_embed.size(-1)),
    )

    speech_condition = torch.nn.functional.pad(
        prompt_features, (0, 0, 0, total_frames - prompt_features.size(1))
    )
    speech_condition_mask = make_pad_mask(prompt_features_lens, total_frames)
    speech_condition = torch.where(
        speech_condition_mask.unsqueeze(-1),
        torch.zeros_like(speech_condition),
        speech_condition,
    )

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    x0 = torch.randn(1, total_frames, prompt_features.size(-1), generator=gen, device=device)

    x1 = model.solver.sample(
        x=x0,
        text_condition=text_condition,
        speech_condition=speech_condition,
        padding_mask=padding_mask,
        num_step=num_step,
        guidance_scale=guidance_scale,
        t_shift=t_shift,
    )
    return x1[:, pmt_frames:, :]


def hanfire_deterministic_randn(shape: Tuple[int, ...], seed: int = 42) -> np.ndarray:
    """
    跨平台逐位一致的标准高斯噪声生成器（向量化 Mulberry32 + Box-Muller）：
    在 Python CPU、Python CUDA、浏览器 WebGPU 与 WASM 间产生完全相同的 float32 序列（max_abs_diff == 0.0），
    彻底杜绝 CUDA Philox vs CPU MT19937 vs 浏览器 PRNG 不同步导致的跨端发音漂移。
    """
    n = int(np.prod(shape))
    n_pairs = (n + 1) // 2
    with np.errstate(over="ignore"):
        k = np.arange(1, n_pairs * 2 + 1, dtype=np.uint64)
        st = ((np.uint64(seed & 0xFFFFFFFF) + k * np.uint64(0x6D2B79F5)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
        t = (st ^ (st >> np.uint32(15))) * (np.uint32(1) | st)
        t = (t + ((t ^ (t >> np.uint32(7))) * (np.uint32(61) | t))) ^ t
        u = ((t ^ (t >> np.uint32(14))).astype(np.float64)) / 4294967296.0
    u1 = np.maximum(u[0::2], 1e-12)
    u2 = u[1::2]
    r = np.sqrt(-2.0 * np.log(u1))
    th = 2.0 * np.pi * u2
    out = np.empty(n_pairs * 2, dtype=np.float32)
    out[0::2] = (r * np.cos(th)).astype(np.float32)
    out[1::2] = (r * np.sin(th)).astype(np.float32)
    return out[:n].reshape(shape)


def hanfire_initial_noise(
    total_frames: int,
    feat_dim: int = 100,
    seed: int = 42,
    tau: float = 0.667,
    mode: str = "spherical",
    clip_sigma: float = 2.5,
) -> np.ndarray:
    """
    流匹配 ODE 稳定化初值构造器（返回 shape=(1, total_frames, feat_dim) float32）：
      - mode='spherical' (默认): 截尾 ±2.5σ + 每帧强制零均值 + 精确球壳缩放 (std=tau=0.667)，
        数学上根除帧级负均值能量空洞诱发的句中随机断句（G2 断裂归零）与 4~5σ 尖峰爆音，
        同时保留自然音高韵律表现力（与浏览器 hanfire_dsp.mjs makeSphericalInitialNoise 逐位一致）；
      - mode='gauss': 常规缩放高斯 N(0, tau^2)；
      - mode='zero': 零初值全确定性先验。
    """
    if mode == "zero":
        return np.zeros((1, total_frames, feat_dim), dtype=np.float32)
    z = hanfire_deterministic_randn((total_frames, feat_dim), seed=seed)
    if mode == "spherical":
        z = np.clip(z, -clip_sigma, clip_sigma).astype(np.float64)
        z = z - z.mean(axis=1, keepdims=True)
        z = z / (z.std(axis=1, keepdims=True) + 1e-8) * float(tau)
        z = z.astype(np.float32)
    else:
        z = (z * float(tau)).astype(np.float32)
    return z[None, :, :]


def get_sway_time_steps(
    num_step: int = 16,
    sway_s: float = -0.8,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    F5-TTS Sway Sampling 时间步调度：t = u + s * (cos(pi/2 * u) - 1 + u)。
    在 t→0 低信噪比关键分岔区提供约 2x 更密步长，显著降低音位/断句随机漂移且零额外 NFE。
    """
    u = torch.linspace(0.0, 1.0, num_step + 1, dtype=torch.float32, device=device)
    return u + float(sway_s) * (torch.cos((np.pi * 0.5) * u) - 1.0 + u)


def _is_initial_token(tok: str) -> bool:
    """判断词元是否为声母（普通话 b0..zh0 / 切韵 Q_<母> / 上古近古 OC_*/EM_* 辅音声母）。
    用于防单槽零声母首字（如“安/庵/翁/哀”）被误判为声母并误加 char0_boost_w（会裂变出双字“安安”并吹胀次字声母）。
    """
    if not tok:
        return False
    if len(tok) >= 2 and tok.endswith("0"):
        return True
    if tok.startswith("Q_"):
        return ("_舒_" not in tok) and ("_入_" not in tok)
    if tok.startswith(("OC_", "EM_", "MC_")):
        return (not tok[-1].isdigit()) and (not tok.endswith("_chk")) and ("coda" not in tok)
    return False


@torch.no_grad()
def sample_clean_zero_carrier(
    *,
    model: torch.nn.Module,
    prompt_items: List[Tuple[int, int]],
    target_items: List[Tuple[int, int]],
    t2i: Dict[str, int],
    prompt_features: torch.Tensor,
    prompt_features_lens: torch.Tensor,
    frames_per_100w: float = 27.5,
    char0_boost_w: int = 44,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    t_shift: float = 0.5,
    seed: int = 42,
    lead_w: int = 26,
    noise_mode: str = "spherical",
    noise_tau: float = 0.667,
    sway_s: Optional[float] = -0.8,
) -> torch.Tensor:
    """
    零前导词、零电音、跨平台确定性球面初值采样器 (Zero-Carrier Clean Studio Sampler)：
    1. 彻底废除任何发音前导词（如“吟”），100% 根除句首残留 "en" 杂音；
    2. 仅用纯静音逗号停顿 (",", lead_w) 衔接原生未拉伸参考音，同时为多槽首字补偿 char0_boost_w 权重（单槽零声母首字如“安/庵/翁”严禁加权以防裂变为“安安”）；
    3. 默认 guidance_scale=1.0 配合温和输入电平，彻底消除流匹配过饱和破音；
    4. 默认启用跨平台 Mulberry32 帧级零均值球面初值 (Spherical, clip=±2.5σ, τ=0.667) + F5-TTS Sway(s=-0.8) 调度，
       从底层根除帧级负均值空洞断句与 5σ 尖峰爆音，且在 Python GPU/CPU 与浏览器端生成逐位一致的初值。
    注：实测模型在参考音后固定要先生成 ~360ms 静音，lead_w=26 严禁随意放大（拉长会破坏全局韵律诱发断句失韵）。
    """
    device = next(model.parameters()).device
    tgt = list(target_items)
    if char0_boost_w > 0 and len(tgt) >= 2:
        i2t = {v: k for k, v in t2i.items()}
        # 仅当首词元确为声母时执行首字补偿；单槽零声母首字（如“安/庵/翁”）首词元即为韵母，严禁叠加 char0_boost_w（否则首字读两遍且把次字声母吹胀！）
        if _is_initial_token(i2t.get(tgt[0][0], "")):
            b_i = char0_boost_w // 4
            b_f = char0_boost_w - b_i
            tgt[0] = (tgt[0][0], tgt[0][1] + b_i)
            # 若首字带显式介音滑音 (ini + y0/w0 + fin)，将韵母补偿加在真正的第 3 槽主元音上，绝不把 14w 介音吹胀为 47w 引发齿音爆破！
            fin_slot = 2 if (len(tgt) >= 3 and tgt[1][0] in (t2i.get("y0", -1), t2i.get("w0", -1))) else 1
            tgt[fin_slot] = (tgt[fin_slot][0], tgt[fin_slot][1] + b_f)

    lead_block = [(t2i[","], lead_w)]
    tail_block = [(t2i["."], 38)]

    pmt_ids = [x[0] for x in prompt_items]
    pmt_ws = [x[1] for x in prompt_items]
    gen_all = lead_block + tgt + tail_block
    gen_ids = [x[0] for x in gen_all]

    pmt_frames = int(prompt_features_lens[0].item())
    lead_frames = max(6, int(round(lead_w * frames_per_100w / 100.0)))
    sp_frames = max(int(round(sum(w for _, w in tgt) * frames_per_100w / 100.0)), len(tgt) + 4)
    tail_frames = max(8, int(round(38 * frames_per_100w / 100.0)))
    total_frames = pmt_frames + lead_frames + sp_frames + tail_frames

    cat_embed, _ = model.forward_text_embed([pmt_ids + gen_ids])
    cat_durs = [
        _weights_to_exact_frames(pmt_ws, pmt_frames)
        + [lead_frames]
        + _weights_to_exact_frames([w for _, w in tgt], sp_frames)
        + [tail_frames]
    ]

    features_lens = torch.tensor([total_frames], dtype=torch.int64, device=device)
    padding_mask = make_pad_mask(features_lens, max_len=total_frames)
    tokens_index = get_tokens_index(cat_durs, total_frames).to(device)
    text_condition = torch.gather(
        cat_embed,
        dim=1,
        index=tokens_index.unsqueeze(-1).expand(1, total_frames, cat_embed.size(-1)),
    )

    speech_condition = torch.nn.functional.pad(
        prompt_features, (0, 0, 0, total_frames - prompt_features.size(1))
    )
    speech_condition_mask = make_pad_mask(prompt_features_lens, total_frames)
    speech_condition = torch.where(
        speech_condition_mask.unsqueeze(-1),
        torch.zeros_like(speech_condition),
        speech_condition,
    )

    x0_np = hanfire_initial_noise(
        total_frames=total_frames,
        feat_dim=int(prompt_features.size(-1)),
        seed=seed,
        tau=noise_tau,
        mode=noise_mode,
    )
    x0 = torch.from_numpy(x0_np).to(device=device, dtype=prompt_features.dtype)

    if sway_s is not None:
        timesteps = get_sway_time_steps(num_step=num_step, sway_s=sway_s, device=device)
        x = x0
        for step in range(num_step):
            v = model.solver.model(
                t=timesteps[step],
                x=x,
                text_condition=text_condition,
                speech_condition=speech_condition,
                padding_mask=padding_mask,
                guidance_scale=guidance_scale,
            )
            x = x + v * (timesteps[step + 1] - timesteps[step])
        x1 = x
    else:
        x1 = model.solver.sample(
            x=x0,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
            num_step=num_step,
            guidance_scale=guidance_scale,
            t_shift=t_shift,
        )
    s0 = pmt_frames + max(0, lead_frames - 2)
    s1 = total_frames
    return x1[:, s0:s1, :]


@torch.no_grad()
def sample_metrical_with_carrier(
    *,
    model: torch.nn.Module,
    prompt_items: List[Tuple[int, int]],
    carrier_items: List[Tuple[int, int]],
    target_items: List[Tuple[int, int]],
    t2i: Dict[str, int],
    prompt_features: torch.Tensor,
    prompt_features_lens: torch.Tensor,
    frames_per_100w: float = 35.0,
    num_step: int = 16,
    guidance_scale: float = 1.25,
    t_shift: float = 0.5,
    seed: int = 42,
) -> torch.Tensor:
    """
    格律稳速前导载波采样器 (Sacrificial Lead-Carrier Sampler)：
    1. 在目标诗句前拼接一个“牺牲前导字 + 逗号停顿 (68w)”，彻底吸收 ZipVoice 流匹配在句首产生的 200~450ms 吸气静音延迟，
       使真正的第 1 个字（如“床/春/花/千”）处于句中第 2 拍位置，获得 100% 饱满的起始音长（消除“开头的音时间太短”）；
    2. 在句尾拼接句号停顿缓冲区 (45w)，避免末字贴死卷积边界；
    3. 自动在逗号停顿区间内寻找能量谷底切分点，既不残留前导字尾音，也不误裁首字清擦/送气声母。
    """
    device = next(model.parameters()).device
    lead_block = [(t2i["."], 45)] + list(carrier_items) + [(t2i[","], 92)]
    tail_block = [(t2i["."], 50)]

    pmt_ids = [x[0] for x in prompt_items]
    pmt_ws = [x[1] for x in prompt_items]
    lead_ids = [x[0] for x in lead_block]
    lead_ws = [x[1] for x in lead_block]
    tgt_ids = [x[0] for x in target_items]
    tgt_ws = [x[1] for x in target_items]
    tail_ids = [x[0] for x in tail_block]
    tail_ws = [x[1] for x in tail_block]

    pmt_frames = int(prompt_features_lens[0].item())
    lead_frames = max(int(round(sum(lead_ws) / 100.0 * frames_per_100w)), len(lead_ids) + 4)
    sp_frames = max(int(round(sum(tgt_ws) / 100.0 * frames_per_100w)), len(tgt_ids) + 4)
    tail_frames = max(int(round(sum(tail_ws) / 100.0 * frames_per_100w)), 8)
    total_frames = pmt_frames + lead_frames + sp_frames + tail_frames

    pmt_durs = _weights_to_exact_frames(pmt_ws, pmt_frames)
    lead_durs = _weights_to_exact_frames(lead_ws, lead_frames)
    sp_durs = _weights_to_exact_frames(tgt_ws, sp_frames)
    tail_durs = [tail_frames]

    # Prompt Boundary Guard：将参考音与生成序列在 text_encoder 侧分块编码，消除普通话提示音尾部串扰
    pmt_emb, _ = model.forward_text_embed([pmt_ids])
    gen_emb, _ = model.forward_text_embed([lead_ids + tgt_ids + tail_ids])
    cat_embed = torch.cat([pmt_emb, gen_emb], dim=1)
    cat_durs = [pmt_durs + lead_durs + sp_durs + tail_durs]

    features_lens = torch.tensor([total_frames], dtype=torch.int64, device=device)
    padding_mask = make_pad_mask(features_lens, max_len=total_frames)
    tokens_index = get_tokens_index(cat_durs, total_frames).to(device)
    text_condition = torch.gather(
        cat_embed,
        dim=1,
        index=tokens_index.unsqueeze(-1).expand(1, total_frames, cat_embed.size(-1)),
    )

    speech_condition = torch.nn.functional.pad(
        prompt_features, (0, 0, 0, total_frames - prompt_features.size(1))
    )
    speech_condition_mask = make_pad_mask(prompt_features_lens, total_frames)
    speech_condition = torch.where(
        speech_condition_mask.unsqueeze(-1),
        torch.zeros_like(speech_condition),
        speech_condition,
    )

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    x0 = torch.randn(1, total_frames, prompt_features.size(-1), generator=gen, device=device)

    x1 = model.solver.sample(
        x=x0,
        text_condition=text_condition,
        speech_condition=speech_condition,
        padding_mask=padding_mask,
        num_step=num_step,
        guidance_scale=guidance_scale,
        t_shift=t_shift,
    )

    # 在逗号停顿的后半段寻找最安静的 mel 谷底作为切入点（彻底避开前导字尾音，同时保留首字完整声母起音）
    comma_len = lead_durs[-1]
    search_l = pmt_frames + lead_frames - max(4, int(comma_len * 0.50))
    search_r = pmt_frames + lead_frames - 2
    if search_r > search_l:
        frame_energy = x1[0, search_l:search_r, :].mean(dim=-1)
        valley_idx = int(torch.argmin(frame_energy).item())
        s0 = search_l + valley_idx
    else:
        s0 = pmt_frames + lead_frames - 3
    s1 = min(total_frames, pmt_frames + lead_frames + sp_frames + 8)
    return x1[:, s0:s1, :]


class ThreeEraHanfireCompiler(HanfireZipVoiceCompiler):
    # V8-A：各时代朗读语速（每 100 权重 ≈ 每字 帧数，24kHz/hop256），取自训练语料中位数：
    #   上古 shanggu 0.245s/字、中古 千字文 0.371s/字、近古 桃花庵 0.293s/字
    ERA_FRAMES_PER_100W: Dict[str, float] = {"oc": 25.0, "mc": 33.0, "em": 28.0, "em_zyyy": 28.0}
    # 中古阴阳调实验开关（普通话 token 无 33/31 低调，暂用半上 3 近似阳平，默认关闭待听辨）
    MC_YANG_REGISTER = False
    YANG_ONSETS = frozenset({
        "b", "d", "g", "ɡ", "dz", "d͡z", "dʑ", "d͡ʑ", "dʒ", "d͡ʒ", "ɖ", "dʐ", "d͡ʐ", "z", "ʑ", "ʒ", "ɣ", "ɦ", "ʐ",
        "m", "n", "ɲ", "ɲʑ", "ŋ", "l", "r", "ɳ", "j", "w", "v", "ʋ",
    })
    # 王力上古音真部(-n) vs 耕部(-ŋ)分野：旧词表把大批真部字标成了 -iŋ；以下 19 字确属耕部保留 -ŋ，其余 -iŋ 一律对齐王力真部 -n
    OC_GENG_ING_KEEP = frozenset({
        "伶", "佞", "命", "囹", "瓴", "翎", "聆", "苓", "鈴", "齡",
        "铃", "龄", "呤", "坽", "旍", "玲", "舲", "蘦", "魿",
    })
    """
    三代贯通编译器 (`era = "oc" | "mc" | "em"`)：
    单文件读取打包好的 `v5_three_era_lexicon.json`，无需外部数据库即可在本地或 Colab 云端编译：
      - `oc`：上古音（汉火体系，平声稳平正常读音）
      - `mc`：中古音（切韵/广韵体系，平上去入 × 清浊 = 四声八调 + 平长仄短格律）
      - `em`：近古音（中原音韵体系，尖团区分 + 见组细音不颚化 + 闭口 -m + 微母 ʋ + 入派三声）
    """

    VARIANT_MAP: Dict[str, str] = {
        "滾": "袞", "滚": "袞", "裏": "裡", "爲": "為", "衆": "眾",
        "線": "綫", "傘": "繖", "遊": "游", "願": "愿", "髮": "髪",
    }

    def __init__(self, t2i: Dict[str, int], three_era_json: Path, v8_fixes: bool = True) -> None:
        self.t2i = t2i
        self._i2t = {v: k for k, v in t2i.items()}
        data = json.loads(three_era_json.read_text(encoding="utf-8"))
        self.eras: Dict[str, Dict[str, List[str]]] = data["eras"]  # era -> {char: [ipa, tone_tag]}
        self._variant_cache: Dict[Tuple[str, str], Optional[str]] = {}
        self.chars = {}
        self.tone_hints = {}
        # V8 发音修复（经声学实验 + 人耳听辨确认，见 2026-11-18 诊断）：
        #   1. 上古浊塞音槽位 OC_b/d/g/dz/dʒ/dʑ 当前嵌入会产生“前浊音”额外一拍（在河之洲 → 五拍），暂回退清声母 token；
        #   2. 零声母：i/ü 类补 y0、u 类补 w0；开口零声母单 token 权重封顶，防止单 token 拉伸裂成两拍（岸、庵）；
        #   3. 元音结尾字 + 开口零声母字（花庵）之间插入极短喉塞停顿，防止两个相同元音粘连成“安安”；
        #   4. 擦音声母 (h/x/s/sh/f…) 时长下限，防止被解码器吞掉（花 → “a”）。
        self.v8_fixes = v8_fixes

    VOICED_STOP_FALLBACK: Dict[str, str] = {
        "OC_b": "b0", "OC_d": "d0", "OC_g": "g0", "OC_dz": "z0", "OC_dʒ": "zh0", "OC_dʑ": "j0",
    }
    FRICATIVE_ONSETS = frozenset({"h0", "x0", "s0", "sh0", "f0", "OC_z", "OC_ʑ", "OC_ʒ", "OC_ɣ"})
    FRICATIVE_ONSET_MIN_W = 40
    ZERO_OPEN_FINAL_MAX_W = 62
    HIATUS_PAUSE_W = 10

    @staticmethod
    def _is_initial_name(name: str) -> bool:
        if name in (",", ".", "!", "?"):
            return False
        if name.endswith("0") and not name.startswith(("OC_", "MC_", "EM_")):
            return True
        if name.startswith(("OC_", "MC_", "EM_")):
            return not (name[-1].isdigit() or name.endswith("_chk") or "coda" in name)
        return False

    def _v8_fix_char(self, items: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], bool, bool]:
        """返回 (修复后 items, 是否开口零声母, 是否以开元音收尾)。"""
        names = [self._i2t[t] for t, _ in items]
        ws = [int(w) for _, w in items]
        zero_open = False
        if names and names[0] in self.VOICED_STOP_FALLBACK and self.VOICED_STOP_FALLBACK[names[0]] in self.t2i:
            names[0] = self.VOICED_STOP_FALLBACK[names[0]]
        if names and not self._is_initial_name(names[0]):
            base = names[0].removeprefix("OC_").removeprefix("MC_").removeprefix("EM_")
            # 普通话里 ong 从不单独出现（零声母读 weng = w0+…），故 ong 类同 u 类补 w0（江雪“翁”）
            glide = "y0" if base.startswith(("i", "v", "ue")) else ("w0" if base.startswith(("u", "ong")) else "")
            if glide:
                names.insert(0, glide)
                ws.insert(0, 24)
                ws[1] = max(20, ws[1] - 24)
            else:
                zero_open = True
                ws[0] = min(ws[0], self.ZERO_OPEN_FINAL_MAX_W)
        elif names and names[0] in self.FRICATIVE_ONSETS and len(names) == 2:
            # 入声短促韵 (_chk, 总重 80w) 保持原生 22w 声母配比，绝不扩增至 40w 挤占主元音引发高频齿音爆破毛刺
            min_f_w = 22 if names[1].endswith("_chk") else self.FRICATIVE_ONSET_MIN_W
            if ws[0] < min_f_w:
                delta = min_f_w - ws[0]
                ws[0] += delta
                ws[1] = max(20, ws[1] - delta)
            elif names[1].endswith("_chk") and ws[0] > 22:
                delta = ws[0] - 22
                ws[0] = 22
                ws[1] += delta
        last = names[-1] if names else ""
        ends_open = bool(last) and not (
            last.endswith("_chk") or "coda" in last or self._is_initial_name(last)
            or last.rstrip("012345").endswith(("n", "ng", "m"))
        )
        return [(self.t2i[n], w) for n, w in zip(names, ws)], zero_open, ends_open

    def _era_tone_and_weight(
        self, era: str, onset: str, coda: str, is_checked: bool, tone_tag: str
    ) -> Tuple[str, int]:
        if is_checked or coda in ("p", "t", "k") or tone_tag == "入":
            return "4", (78 if era == "mc" else 80)

        if era == "mc":
            # 兼顾《切韵》声调格局与唐诗自然顺耳美感：
            # 平声 = 舒展平稳 Tone 1 (全浊/次浊由声母本身承载浑厚低频共振，不强压突兀折调)；
            # 上声 = 清越柔和升调 Tone 2 (“上声短而昂”)；去声 = 从容缓降 Tone 4 (“去声哀而远”)。
            if tone_tag == "上":
                return "2", 100
            if tone_tag == "去":
                return "4", 98
            if self.MC_YANG_REGISTER and onset in self.YANG_ONSETS:
                return "3", 104  # 阳平（低平 33）近似
            return "1", 104

        if era in ("em", "em_zyyy"):
            # 明代官话：纯 "入" 走短促入声；若词表已标明 "入作陽/入作上/入作去"，按实际归派声调走舒声时长！
            if era == "em" and tone_tag == "入":
                return "4", 86
            if "陽" in tone_tag:
                return "2", 100
            if "上" in tone_tag:
                return "3", 100
            if "去" in tone_tag:
                return "4", 100
            return "1", 100

        # 默认 oc 上古正常读音：遵循王力体系与汉语清高浊低源流——平声清声母走高平 1 声，全浊/次浊声母走阳平 2 声（如“伊”清1 vs “人”次浊2）
        if coda in ("h", "s") or tone_tag == "departing":
            return "4", 100
        if coda == "ʔ" or tone_tag == "rising":
            return "3", 100
        if onset in self.YANG_ONSETS:
            return "2", 100
        return "1", 100

    def _resolve_era_char_key(self, era_key: str, ch: str) -> Optional[str]:
        era_map = self.eras[era_key]
        if ch in era_map:
            return ch
        ck = (era_key, ch)
        if ck in self._variant_cache:
            return self._variant_cache[ck]
        cand = self.VARIANT_MAP.get(ch)
        if cand and cand in era_map:
            self._variant_cache[ck] = cand
            return cand
        try:
            if not hasattr(self, "_opencc_pool"):
                from opencc import OpenCC
                self._opencc_pool = [OpenCC(m) for m in ("s2t", "t2s", "s2tw", "tw2s")]
            for cc in self._opencc_pool:
                cc_ch = cc.convert(ch)
                if cc_ch in era_map:
                    self._variant_cache[ck] = cc_ch
                    return cc_ch
                if cc_ch in self.VARIANT_MAP and self.VARIANT_MAP[cc_ch] in era_map:
                    self._variant_cache[ck] = self.VARIANT_MAP[cc_ch]
                    return self.VARIANT_MAP[cc_ch]
        except Exception:
            pass
        self._variant_cache[ck] = None
        return None

    def compile_era_char(
        self,
        era: str,
        ch: str,
        override_ipa: Optional[str] = None,
    ) -> Tuple[List[Tuple[int, int]], str]:
        from .phonology import parse_mononuclear_4slot
        era_key = "em" if era == "em_zyyy" else era
        era_map = self.eras[era_key]
        resolved_ch = self._resolve_era_char_key(era_key, ch)
        explicit_tone_d: Optional[str] = None
        if override_ipa is not None:
            s_ov = override_ipa.strip()
            if len(s_ov) >= 2 and s_ov[-1] in ("1", "2", "3", "4"):
                explicit_tone_d = s_ov[-1]
                raw_ipa = s_ov[:-1]
                tone_tag = {"1": "平", "2": "陽平", "3": "上", "4": "去"}[explicit_tone_d]
            elif s_ov.endswith("h") and era in ("mc", "em", "em_zyyy"):
                raw_ipa = s_ov[:-1]
                tone_tag = "去"
            elif s_ov.endswith("ʔ") and era in ("mc", "em", "em_zyyy"):
                raw_ipa = s_ov[:-1]
                tone_tag = "上"
            else:
                raw_ipa = s_ov
                tone_tag = era_map.get(resolved_ch or ch, ["", "level"])[1]
        elif resolved_ch is not None:
            raw_ipa, tone_tag = era_map[resolved_ch]
        else:
            fb_it, _ = tokenize_mandarin_prompt_exact(ch, self.t2i)
            self._last_char_flags = (False, False)
            return fb_it, "+".join(self._i2t[t] for t, _ in fb_it)

        # 上古音 (oc) 后缀喉音 *-ʔ (上声源) 与 *-h/*-s (去声源) 先提取调类再剥离，
        # 既防止 *-ʔ 被误判成 -k 入声塞尾（如 水 sujʔ、在 dzəʔ、子 tsəʔ），又完整保留前接滑音韵尾 -j/-w（如 我 ŋajʔ、谓 wəjh）！
        if era == "oc" and len(raw_ipa) >= 2:
            if raw_ipa.endswith("ʔ"):
                raw_ipa = raw_ipa[:-1]
                tone_tag = "rising"
            elif raw_ipa.endswith(("h", "s")) and not raw_ipa.endswith(("ts", "tʂ", "tʃ")):
                raw_ipa = raw_ipa[:-1]
                tone_tag = "departing"

        s = parse_mononuclear_4slot(raw_ipa)
        onset, medial, nucleus, coda, is_checked = s.onset, s.medial, s.nucleus, s.coda, s.is_checked

        # 王力上古音 (oc)：真部归 -n 尾（排除确属耕部的 19 个 -iŋ 字，确保 人/真/身/新/天/年/田/千 等真部字收 -in）
        if (
            era == "oc"
            and nucleus == "i"
            and coda == "ŋ"
            and ch not in self.OC_GENG_ING_KEEP
            and (resolved_ch or ch) not in self.OC_GENG_ING_KEEP
        ):
            coda = "n"

        # 近古明代官话 (em)：1) 闭口韵 -m 在明中叶已并入 -n（如唐伯虎《桃花庵歌》庵/仙/錢通押），避免拆成双词元 OC_am+OC_coda_m 卡顿；
        # 2) 零声母 "iw"（如 又/有/友）补齐细音头 j，读为 y0+iu 而非孤立 ou（“怄”）。
        if era in ("em", "em_zyyy"):
            if coda == "m":
                coda = "n"
            if not onset and raw_ipa == "iw":
                medial, nucleus, coda = "j", "o", "w"

        # 中古章组（照三 tɕ/tɕʰ/dʑ/ɕ/ʑ）在合口或鱼虞央后元音前保持舌叶音 [tʃ/tʃʰ/dʒ/ʃ/ʒ]，绝不退化为现代尖音 j0/q0（如 春、处、出、书、树）
        init_tok = self.ONSET_TO_ZV.get(onset, "")
        if era == "mc" and onset in ("tɕ", "t͡ɕ", "tɕʰ", "t͡ɕʰ", "dʑ", "d͡ʑ", "ɕ", "ʑ"):
            if medial in ("w", "ɥ") or ("ɨ" in raw_ipa) or ("ʉ" in raw_ipa) or nucleus in ("ə", "u", "o", "ɔ", "ɑ"):
                mc_postalv = {
                    "tɕ": "zh0", "t͡ɕ": "zh0",
                    "tɕʰ": "ch0", "t͡ɕʰ": "ch0",
                    "dʑ": "OC_dʒ", "d͡ʑ": "OC_dʒ",
                    "ɕ": "sh0", "ʑ": "OC_ʒ",
                }
                init_tok = mc_postalv[onset]
        if not init_tok:
            if medial in ("j", "ɥ"):
                init_tok = "y0"
            elif medial == "w":
                init_tok = "w0"

        tone_d, total_w = self._era_tone_and_weight(era, onset, coda, is_checked, tone_tag)
        if explicit_tone_d is not None and not (is_checked or coda in ("p", "t", "k")):
            tone_d = explicit_tone_d
        # 近古明代官话特殊韵母归并：tsiw(酒)->z0+iu；tʂɛj(摘)->zh0+ai；ʂju(樹)->sh0+u；保留原生细音词元 (ian/ia/iu)，不插双声母 y0 防止爆破齿音毛刺！
        em_keep_palatal = False
        if era in ("em", "em_zyyy"):
            if raw_ipa.endswith("iw") and init_tok:
                fin_tok, coda_tok = f"iu{tone_d}", None
                em_keep_palatal = True
            elif init_tok in ("z0", "c0", "s0", "zh0", "ch0", "sh0", "r0") and nucleus == "ɨ" and not coda:
                fin_tok, coda_tok = f"i{tone_d}", None
            elif init_tok in ("zh0", "ch0", "sh0", "r0") and ("ɛj" in raw_ipa or "ej" in raw_ipa or (nucleus in ("ɛ", "e") and coda == "j")):
                fin_tok, coda_tok = f"ai{tone_d}", None
            elif init_tok in ("zh0", "ch0", "sh0", "r0") and nucleus == "u" and medial in ("j", "ɥ"):
                fin_tok, coda_tok = f"u{tone_d}", None
            else:
                fin_tok, coda_tok = self._resolve_final_tokens(medial, nucleus, coda, tone_d, is_checked and tone_tag == "入")
                if init_tok in ("z0", "c0", "s0", "g0", "k0", "h0") and fin_tok[:-1] in ("ia", "ian", "iang", "iao", "ie", "in", "ing", "iu", "iong", "ue", "ve"):
                    em_keep_palatal = True
        elif era == "mc" and raw_ipa.endswith("ɨə") and not is_checked:
            fin_tok, coda_tok = f"uo{tone_d}", None
        else:
            fin_tok, coda_tok = self._resolve_final_tokens(medial, nucleus, coda, tone_d, is_checked or tone_tag == "入")

        # 卷舌声母无 in 韵母（針/深/真 -> zh0/sh0+en）；零声母 y0 接前低元音对齐原生 ian/iao（烟/邀）
        if init_tok in ("zh0", "ch0", "sh0", "r0") and fin_tok[:-1] == "in":
            fin_tok = f"en{fin_tok[-1]}"
        elif init_tok == "y0" and fin_tok[:-1] == "an":
            fin_tok = f"ian{fin_tok[-1]}"
        elif init_tok == "y0" and fin_tok[:-1] == "OC_eu":
            fin_tok = f"iao{fin_tok[-1]}"

        # 防止 OOD 非法声韵组合（仅针对默认词表路径；显式 override_ipa 或近古尖音/见晓细音直接忠实保留！）
        if override_ipa is None and not em_keep_palatal and init_tok in ("g0", "k0", "h0", "zh0", "ch0", "sh0", "r0", "z0", "c0", "s0"):
            if not fin_tok.endswith("_chk") and fin_tok[-1:] in ("1", "2", "3", "4", "5"):
                base_fin, t_ch = fin_tok[:-1], fin_tok[-1]
                if base_fin == "iao":
                    fin_tok = f"OC_eu{t_ch}"
                elif base_fin == "ia":
                    fin_tok = f"a{t_ch}"
                elif base_fin == "ie":
                    fin_tok = f"MC_ɨə{t_ch}"
                elif base_fin == "ian":
                    fin_tok = f"an{t_ch}"
                elif base_fin == "iang":
                    fin_tok = f"ang{t_ch}"
                elif base_fin == "ue":
                    fin_tok = f"OC_iwei{t_ch}"
                elif base_fin in ("v", "OC_iu"):
                    fin_tok = f"u{t_ch}"

        items: List[Tuple[int, int]] = []
        dbg: List[str] = []
        if is_checked or coda in ("p", "t", "k") or tone_tag == "入":
            if init_tok:
                items.append((self.t2i[init_tok], 24))
                items.append((self.t2i[fin_tok], total_w - 24))
                dbg.extend([init_tok, fin_tok])
            else:
                items.append((self.t2i[fin_tok], total_w))
                dbg.append(fin_tok)
        elif coda_tok is not None:
            if init_tok:
                items.append((self.t2i[init_tok], 28))
                items.append((self.t2i[fin_tok], total_w - 50))
                items.append((self.t2i[coda_tok], 22))
                dbg.extend([init_tok, fin_tok, coda_tok])
            else:
                items.append((self.t2i[fin_tok], total_w - 22))
                items.append((self.t2i[coda_tok], 22))
                dbg.extend([fin_tok, coda_tok])
        else:
            if init_tok:
                items.append((self.t2i[init_tok], 30))
                items.append((self.t2i[fin_tok], total_w - 30))
                dbg.extend([init_tok, fin_tok])
            else:
                items.append((self.t2i[fin_tok], total_w))
                dbg.append(fin_tok)
        if self.v8_fixes:
            items, zero_open, ends_open = self._v8_fix_char(items)
            dbg = [self._i2t[t] for t, _ in items]
        else:
            zero_open, ends_open = False, False
        self._last_char_flags = (zero_open, ends_open)
        return items, "+".join(dbg)

    def _load_era_overrides(self, era_key: str) -> Dict[str, str]:
        cfg_p = Path(__file__).resolve().parent.parent / "configs" / f"overrides_{era_key}.json"
        out: Dict[str, str] = {}
        if cfg_p.exists():
            try:
                raw = json.loads(cfg_p.read_text(encoding="utf-8"))
                for k, v in raw.items():
                    if not k.startswith("_") and isinstance(v, str) and len(k) == 1:
                        out[k] = v
            except Exception:
                pass
        return out

    def _load_era_bigram_overrides(self, era_key: str) -> Dict[str, List[str]]:
        cfg_p = Path(__file__).resolve().parent.parent / "configs" / f"overrides_{era_key}.json"
        out: Dict[str, List[str]] = {}
        if cfg_p.exists():
            try:
                raw = json.loads(cfg_p.read_text(encoding="utf-8"))
                for k, v in raw.items():
                    if not k.startswith("_") and isinstance(v, list) and len(k) >= 2 and len(v) == len(k):
                        out[k] = [str(x) for x in v]
            except Exception:
                pass
        return out

    def _compile_era_spec(self, era: str, ch: str, spec: str) -> Tuple[List[Tuple[int, int]], str]:
        spec = spec.strip()
        if spec.startswith("同") and len(spec) >= 2:
            return self.compile_era_char(era, spec[1])
        if ":" in spec or "+" in spec:
            items: List[Tuple[int, int]] = []
            for p in spec.split("+"):
                p = p.strip()
                if not p:
                    continue
                tok_s, w_s = (p.split(":", 1) + ["50"])[:2]
                if tok_s in self.t2i:
                    items.append((self.t2i[tok_s], int(w_s)))
            if items:
                self._last_char_flags = (False, False)
                return items, spec
        return self.compile_era_char(era, ch, override_ipa=spec)

    def compile_era_clause(
        self,
        era: str,
        clause_text: str,
        context_overrides: Optional[Dict[str, str]] = None,
        end_punct: str = ".",
        rhyme_hold_w: int = 12,
    ) -> Tuple[List[Tuple[int, int]], str]:
        import re as _re
        era_key = "em" if era == "em_zyyy" else era
        merged_ov = self._load_era_overrides(era_key)
        bigram_ov = self._load_era_bigram_overrides(era_key)
        if context_overrides:
            for k, v in context_overrides.items():
                if isinstance(v, list) and len(k) >= 2 and len(v) == len(k):
                    bigram_ov[k] = [str(x) for x in v]
                elif isinstance(v, str):
                    merged_ov[k] = v
        era_map = self.eras[era_key]
        all_items: List[Tuple[int, int]] = []
        char_groups: List[List[Tuple[int, int]]] = []
        dbg: List[str] = []
        prev_ends_open = False
        pos = 0
        tag_re = _re.compile(r"\[([^\[\]=]+)=([^\[\]]+)\]")
        bigram_keys = sorted(bigram_ov.keys(), key=len, reverse=True)
        while pos < len(clause_text):
            m = tag_re.match(clause_text, pos)
            if m:
                ch_part, spec = m.group(1).strip(), m.group(2).strip()
                for ch in ch_part:
                    c_it, c_dbg = self._compile_era_spec(era, ch, spec)
                    zero_open, ends_open = self._last_char_flags
                    grp: List[Tuple[int, int]] = []
                    if self.v8_fixes and era == "oc" and prev_ends_open and zero_open:
                        grp.append((self.t2i[","], self.HIATUS_PAUSE_W))
                        dbg.append("ʔ")
                    grp.extend(c_it)
                    char_groups.append(grp)
                    dbg.append(f"{ch}[{c_dbg}]")
                    prev_ends_open = ends_open
                pos = m.end()
                continue
            matched_bg = None
            for bg in bigram_keys:
                if clause_text.startswith(bg, pos):
                    matched_bg = bg
                    break
            if matched_bg is not None:
                for ch, sp in zip(matched_bg, bigram_ov[matched_bg]):
                    sp_eff = (context_overrides or {}).get(ch) if isinstance((context_overrides or {}).get(ch), str) else sp
                    c_it, c_dbg = self._compile_era_spec(era, ch, sp_eff) if sp_eff else self.compile_era_char(era, ch)
                    zero_open, ends_open = self._last_char_flags
                    grp = []
                    if self.v8_fixes and era == "oc" and prev_ends_open and zero_open:
                        grp.append((self.t2i[","], self.HIATUS_PAUSE_W))
                        dbg.append("ʔ")
                    grp.extend(c_it)
                    char_groups.append(grp)
                    dbg.append(f"{ch}[{c_dbg}]")
                    prev_ends_open = ends_open
                pos += len(matched_bg)
                continue
            ch = clause_text[pos]
            pos += 1
            if ch not in era_map and ch not in merged_ov and not ("\u4e00" <= ch <= "\u9fff"):
                continue
            ov_spec = merged_ov.get(ch)
            if ov_spec:
                c_it, c_dbg = self._compile_era_spec(era, ch, ov_spec)
            else:
                c_it, c_dbg = self.compile_era_char(era, ch)
            zero_open, ends_open = self._last_char_flags
            grp = []
            if self.v8_fixes and era == "oc" and prev_ends_open and zero_open:
                grp.append((self.t2i[","], self.HIATUS_PAUSE_W))
                dbg.append("ʔ")
            grp.extend(c_it)
            char_groups.append(grp)
            dbg.append(f"{ch}[{c_dbg}]")
            prev_ends_open = ends_open

        # 防气泡破音上下文变调：
        #   - 近古官话 (em)：标准连上变调 (3+3 -> 2+3)；
        #   - 上古音 (oc)：仅对“促声 _chk 后紧邻非句末上声”（如“陟彼高岡”之“彼”）及“三连上声”（如“昔我往矣”之“我往”）
        #     变调为清越升调 2，既 100% 根除 67Hz 低频气泡断裂破音，又不惊动双上声诗句（如《关雎》“窈窕/君子好逑”）韵律！
        if era in ("oc", "em", "em_zyyy") and len(char_groups) >= 2:
            def _grp_nucleus(g: List[Tuple[int, int]]) -> Tuple[int, str]:
                for idx_g in range(len(g) - 1, -1, -1):
                    nm = self._i2t[g[idx_g][0]]
                    if nm and (nm[-1] in ("1", "2", "3", "4", "5") or nm.endswith("_chk")):
                        return idx_g, nm
                return -1, ""
            for gi in range(len(char_groups) - 1):
                i_cur, n_cur = _grp_nucleus(char_groups[gi])
                if i_cur < 0 or not n_cur.endswith("3"):
                    continue
                _, n_nxt = _grp_nucleus(char_groups[gi + 1])
                _, n_nxt2 = _grp_nucleus(char_groups[gi + 2]) if (gi + 2) < len(char_groups) else (-1, "")
                _, n_prv = _grp_nucleus(char_groups[gi - 1]) if gi > 0 else (-1, "")
                need_sandhi = (
                    n_nxt.endswith("3")
                    if era in ("em", "em_zyyy")
                    else (n_prv.endswith("_chk") or (n_nxt.endswith("3") and (n_nxt2.endswith("3") or n_prv.endswith("2"))))
                )
                if need_sandhi:
                    sandhi_nm = n_cur[:-1] + "2"
                    if sandhi_nm in self.t2i:
                        char_groups[gi][i_cur] = (self.t2i[sandhi_nm], char_groups[gi][i_cur][1])

        # 上古音 (oc) 连续高前元音防粘连：前字开元音 i 族收尾（如“伊”y0+i1）且后字为鼻音声母 + i 族韵母（如“人”OC_ɲ/n0+in2）时，
        # 将后字鼻音声母权重由 30w 加厚至 38w，避免夹在两高前元音之间被吞没成 30ms 模糊过渡！
        if era == "oc" and len(char_groups) >= 2:
            nasal_inis = {"n0", "m0", "OC_ɲ", "OC_ŋ"}
            for gi in range(1, len(char_groups)):
                prev_fin = self._i2t[char_groups[gi - 1][-1][0]]
                cur_grp = char_groups[gi]
                if len(cur_grp) >= 2:
                    c_ini = self._i2t[cur_grp[0][0]]
                    c_fin = self._i2t[cur_grp[1][0]]
                    prev_is_open_i = (
                        prev_fin.startswith(("i", "OC_i"))
                        and not prev_fin.rstrip("012345").endswith(("n", "ng", "m"))
                        and not prev_fin.endswith("_chk")
                    )
                    cur_is_i = c_fin.startswith(("i", "OC_i"))
                    if prev_is_open_i and c_ini in nasal_inis and cur_is_i and cur_grp[0][1] < 38:
                        delta = 38 - cur_grp[0][1]
                        cur_grp[0] = (cur_grp[0][0], 38)
                        cur_grp[1] = (cur_grp[1][0], max(36, cur_grp[1][1] - delta))

        for grp in char_groups:
            all_items.extend(grp)
        if rhyme_hold_w > 0 and all_items and not self._last_char_flags[0]:
            lt, lw = all_items[-1]
            all_items[-1] = (lt, lw + rhyme_hold_w)
        if end_punct and end_punct in self.t2i:
            p_w = 38 if end_punct == "." else 26
            all_items.append((self.t2i[end_punct], p_w))
            dbg.append(end_punct)
        return all_items, " ".join(dbg)


class FactorizedEraEmbedding(torch.nn.Module):
    """三代统一可扩展音韵因子化嵌入（源自 Swarm arch-phonology / arch-backbone 设计）：
    严格冻结原生普通话/底模区 `old[:n_old]`，在词表末尾追加可学习声母与 `[medial + nucleus + coda + tone]` 组合韵母，
    彻底摆脱固定 86 槽 `DEAD_SLOT_POOL` 容量上限，供下一代 OC/EM/MC 联合重训使用（零破坏现有 V12 1449 词元检查点）。
    """

    def __init__(
        self,
        base: torch.nn.Embedding,
        n_old: int,
        onset_init: torch.Tensor,
        medial_init: torch.Tensor,
        nucleus_init: torch.Tensor,
        coda_init: torch.Tensor,
        tone_init: torch.Tensor,
        fin_med_idx: torch.Tensor,
        fin_nuc_idx: torch.Tensor,
        fin_coda_idx: torch.Tensor,
        fin_tone_idx: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("old", base.weight.data[:n_old].clone())
        self.onset = torch.nn.Parameter(onset_init.clone())
        self.medial = torch.nn.Parameter(medial_init.clone())
        self.nucleus = torch.nn.Parameter(nucleus_init.clone())
        self.coda = torch.nn.Parameter(coda_init.clone())
        self.tone = torch.nn.Parameter(tone_init.clone())
        self.register_buffer("fin_med_idx", fin_med_idx)
        self.register_buffer("fin_nuc_idx", fin_nuc_idx)
        self.register_buffer("fin_coda_idx", fin_coda_idx)
        self.register_buffer("fin_tone_idx", fin_tone_idx)
        self.embedding_dim = base.embedding_dim
        self.padding_idx = base.padding_idx

    @property
    def weight(self) -> torch.Tensor:
        fin = (
            self.medial[self.fin_med_idx]
            + self.nucleus[self.fin_nuc_idx]
            + self.coda[self.fin_coda_idx]
            + self.tone[self.fin_tone_idx]
        )
        return torch.cat([self.old, self.onset, fin], dim=0)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(idx, self.weight, self.padding_idx)

    def materialize(self) -> torch.nn.Embedding:
        w = self.weight.detach()
        e = torch.nn.Embedding(w.size(0), w.size(1), padding_idx=self.padding_idx)
        e.weight.data.copy_(w)
        return e
