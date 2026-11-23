#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中古音「音韵地位」编译器 + 组合式嵌入 (mc_qieyun.py)  —— V9-B

替代旧做法（拟音 IPA → 普通话韵母近似，209 种韵母挤进 52 个 token），改为：
  - 读音来源：qieyun-js 广韵数据（CC0），每读音带完整音韵地位，调类直接取自数据（修复旧表 21.5% 调类错误）
  - 声母 token：38 个「母」各一个 (Q_幫, Q_並, Q_影 …)，全浊、知组、娘母、影母 ʔ、俟母等全部独立
  - 韵母 token：韵类 (韻×呼×等×類，共 133 类) × 调类 (平上去入 × 阴阳 = 8 类)
      嵌入 = seg[韵类] + tone[调类]  （组合式：少量数据也能泛化到未见过的“韵类×调类”组合）
  - 嵌入初始化：用旧编译器把该韵类的晚唐/通俗拟音映射到最近的普通话/古音槽位作起点
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

UNIFIED_TSV = Path(__file__).resolve().parent.parent / "tools" / "qieyun" / "mc_qieyun_unified.tsv"

# 38 母 → 初始化用的普通话/古音近似声母 token（只作嵌入起点，训练中可学习独立声学）
MU_INIT: Dict[str, List[str]] = {
    "幫": ["b0"], "滂": ["p0"], "並": ["b0", "OC_b"], "明": ["m0"],
    "端": ["d0"], "透": ["t0"], "定": ["d0", "OC_d"], "泥": ["n0"], "來": ["l0"],
    "知": ["zh0", "d0"], "徹": ["ch0", "t0"], "澄": ["zh0", "OC_dʒ"], "孃": ["n0"],
    "精": ["z0"], "清": ["c0"], "從": ["z0", "OC_dz"], "心": ["s0"], "邪": ["s0", "OC_z"],
    "莊": ["zh0"], "初": ["ch0"], "崇": ["zh0", "OC_dʒ"], "生": ["sh0"], "俟": ["sh0", "r0"],
    "章": ["zh0", "j0"], "昌": ["ch0", "q0"], "常": ["sh0", "r0"], "書": ["sh0", "x0"], "船": ["zh0", "r0"], "日": ["OC_ɲ", "r0"],
    "見": ["g0"], "溪": ["k0"], "羣": ["g0", "OC_g"], "疑": ["OC_ŋ", "y0"],
    "影": ["ʔ", "y0"], "曉": ["h0"], "匣": ["h0", "OC_ɣ"], "云": ["y0", "h0"], "以": ["y0"],
}
TONE_CATS = ["平陰", "平陽", "上陰", "上陽", "去陰", "去陽", "入陰", "入陽"]
# 调类向量初始化：普通话声调偏移的组合（阳调取低调成分 T3）
TONE_INIT: Dict[str, Dict[str, float]] = {
    "平陰": {"1": 1.0}, "平陽": {"1": 0.45, "3": 0.55},
    "上陰": {"2": 1.0}, "上陽": {"2": 0.6, "3": 0.4},
    "去陰": {"4": 1.0}, "去陽": {"4": 0.6, "3": 0.4},
    "入陰": {"4": 1.0}, "入陽": {"4": 0.6, "3": 0.4},
}
FRICATIVE_MU = frozenset({"心", "邪", "生", "俟", "書", "常", "曉", "匣"})
# 高频虚词/常用字的主读音（推导普通话与今音对不上时的人工指定）
PRIMARY_OVERRIDE: Dict[str, str] = {"不": "幫三文入"}


@dataclass
class Reading:
    char: str
    order: int
    desc: str
    mu: str
    hu: str
    deng: str
    lei: str
    yun: str
    sheng: str
    qingzhuo: str
    tupa: str
    unt: str
    mid_tang: str
    putonghua: str = ""

    @property
    def yunlei(self) -> str:  # 韵类 = 韻 × 呼 × 等 × 類 × (舒/入)：舒声(-m/-n/-ŋ/开)与入声(-p/-t/-k)必须分槽
        mode = "入" if self.sheng == "入" else "舒"
        return f"{self.yun}{self.hu}{self.deng}{self.lei}_{mode}"

    @property
    def tone_cat(self) -> str:
        yin = self.qingzhuo in ("全清", "次清")
        return f"{self.sheng}{'陰' if yin else '陽'}"


def load_unified(tsv: Path = UNIFIED_TSV) -> Dict[str, List[Reading]]:
    out: Dict[str, List[Reading]] = defaultdict(list)
    with open(tsv, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            out[r["字"]].append(Reading(r["字"], int(r["序"]), r["描述"], r["母"], r["呼"], r["等"], r["類"],
                                        r["韻"], r["聲"], r["清濁"], r["tupa"], r["unt"], r["mid_tang"], r.get("putonghua", "")))
    return dict(out)


def _mandarin_tones(ch: str) -> List[int]:
    try:
        from pypinyin import pinyin, Style
        res = pinyin(ch, style=Style.TONE3, heteronym=True, neutral_tone_with_five=True)[0]
        return [int(p[-1]) for p in res if p and p[-1].isdigit()]
    except Exception:
        return []


def _mandarin_pinyins(ch: str) -> List[str]:
    try:
        from pypinyin import pinyin, Style
        return [p for p in pinyin(ch, style=Style.TONE, heteronym=True)[0] if p]
    except Exception:
        return []


def _strip_tone(p: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", p) if not unicodedata.combining(c)).replace("ü", "v")


def choose_reading(ch: str, rs: List[Reading]) -> Reading:
    """多音字取主读音：
    1) qieyun-examples 的「推导普通话」与该字实际普通话读音（pypinyin 首读）完全一致者优先（含声调）；
    2) 其次音节相同（不计声调）；3) 再按普通话声调对应关系；4) 偏好韵书首读。"""
    if len(rs) == 1:
        return rs[0]
    pys = _mandarin_pinyins(ch)
    main = pys[0] if pys else ""
    mt = set(_mandarin_tones(ch))
    def score(r: Reading) -> float:
        s = -0.1 * r.order
        derived = [x for x in r.putonghua.split("/") if x]
        if main and main in derived:
            s += 6
        elif main and _strip_tone(main) in {_strip_tone(x) for x in derived}:
            s += 4
        elif any(p in derived for p in pys[1:]):
            s += 1.5
        if r.sheng == "平" and mt & {1, 2}:
            s += 1
        elif r.sheng == "上" and (3 in mt or (r.qingzhuo == "全濁" and 4 in mt)):
            s += 1
        elif r.sheng == "去" and 4 in mt:
            s += 1
        return s
    return max(rs, key=score)


class QieyunMCCompiler:
    """中古音韵地位编译器：字 → [(token_id, weight)]，token 名称：Q_母 / Q_韵类_调类。"""

    def __init__(self, t2i: Dict[str, int], readings: Optional[Dict[str, List[Reading]]] = None,
                 simp_map: Optional[Dict[str, str]] = None,
                 overrides_json: Optional[Path] = None) -> None:
        self.t2i = t2i
        self.readings = readings or load_unified()
        # 加载外部可编辑单字与双字词上下文覆写表 configs/overrides_mc.json
        self.default_overrides: Dict[str, str] = dict(PRIMARY_OVERRIDE)
        self.bigram_overrides: Dict[str, List[str]] = {}
        cfg_p = overrides_json or (Path(__file__).resolve().parent.parent / "configs" / "overrides_mc.json")
        if cfg_p.exists():
            try:
                import json as _json
                raw_ov = _json.loads(cfg_p.read_text(encoding="utf-8"))
                for k, v in raw_ov.items():
                    if k.startswith("_"):
                        continue
                    if isinstance(v, str) and len(k) == 1:
                        self.default_overrides[k] = v
                    elif isinstance(v, list) and len(k) >= 2 and len(v) == len(k):
                        self.bigram_overrides[k] = [str(x) for x in v]
            except Exception:
                pass
        # 简体/异体 → 广韵正字（双向 OpenCC + 古籍常见异体字表，确保一字不漏、一字一音）
        self.s2t: Dict[str, str] = {
            "滾": "袞", "滚": "袞", "裏": "裡", "爲": "為", "衆": "眾",
            "線": "綫", "傘": "繖", "遊": "游", "願": "愿", "髮": "髪",
            "間": "閒", "间": "閒", "劍": "劒", "剑": "劒", "窗": "窻",
            "淘": "陶", "瀟": "蕭", "潇": "蕭", "敍": "敘", "粧": "妝", "隣": "鄰",
        }
        try:
            from opencc import OpenCC
            t2s = OpenCC("t2s")
            s2t_cc = OpenCC("s2t")
            for ch in self.readings:
                s = t2s.convert(ch)
                if s != ch and s not in self.readings:
                    self.s2t.setdefault(s, ch)
                t = s2t_cc.convert(ch)
                if t != ch and t not in self.readings:
                    self.s2t.setdefault(t, ch)
        except Exception:
            pass
        if simp_map:
            self.s2t.update(simp_map)
        self._last_flags = (False, False)

    # ---- 词元清单 ----
    @staticmethod
    def initial_name(r: Reading) -> str:
        return f"Q_{r.mu}"

    @staticmethod
    def final_name(r: Reading) -> str:
        return f"Q_{r.yunlei}_{r.tone_cat}"

    def lookup(self, ch: str, override: Optional[str] = None) -> Optional[Reading]:
        key = ch if ch in self.readings else self.s2t.get(ch)
        if key is None or key not in self.readings:
            try:
                if not hasattr(self, "_opencc_pool"):
                    from opencc import OpenCC
                    self._opencc_pool = [OpenCC(m) for m in ("s2t", "t2s", "s2tw", "tw2s")]
                for cc in self._opencc_pool:
                    cand = cc.convert(ch)
                    if cand in self.readings:
                        key = cand
                        break
                    if cand in self.s2t and self.s2t[cand] in self.readings:
                        key = self.s2t[cand]
                        break
            except Exception:
                pass
        if key is None or key not in self.readings:
            return None
        rs = self.readings[key]
        override = override or self.default_overrides.get(ch) or self.default_overrides.get(key)
        if override:  # 允许用 “描述”（如 “昌合三魚去”）或调类字（平/上/去/入）指定读音
            for r in rs:
                if r.desc == override or r.sheng == override:
                    return r
        return choose_reading(key, rs)

    def all_token_names(self) -> Tuple[List[str], List[str]]:
        inis, fins = set(), set()
        for rs in self.readings.values():
            for r in rs:
                inis.add(self.initial_name(r))
                fins.add(self.final_name(r))
        return sorted(inis), sorted(fins)

    JING_MU = frozenset({"精", "清", "從", "心", "邪"})
    JIAN_MU = frozenset({"見", "溪", "羣"})
    BANG_MU = frozenset({"幫", "滂", "並", "明"})

    def _compile_ipa_override(self, ipa_spec: str, metrical: bool = True) -> Tuple[List[Tuple[int, int]], str]:
        """L1 直接 IPA / 拼音+声调覆写（如 [花=huɑ1]、[光=kwaŋ1]、[處=tɕʰy4]），经统一 5 槽解析器直达预训练流形，不触发 OOD 静默改写。"""
        from .phonology import parse_mononuclear_4slot
        from .zipvoice_zh_manifold import HanfireZipVoiceCompiler
        s_raw = ipa_spec.strip()
        tone_d = "1"
        if s_raw and s_raw[-1] in ("1", "2", "3", "4"):
            tone_d = s_raw[-1]
            s_raw = s_raw[:-1]
        elif s_raw.endswith("h"):
            tone_d = "4"
            s_raw = s_raw[:-1]
        elif s_raw.endswith("ʔ"):
            tone_d = "3"
            s_raw = s_raw[:-1]
        # 兼容常见拼音缩写如 huɑ / xua / kwaŋ
        norm = s_raw.replace("hu", "xu").replace("sh", "ʂ").replace("zh", "tʂ").replace("ch", "tʂʰ")
        slot = parse_mononuclear_4slot(norm)
        checked = slot.is_checked or slot.coda in ("p", "t", "k")
        if checked:
            tone_d = "4"
        helper = HanfireZipVoiceCompiler.__new__(HanfireZipVoiceCompiler)
        helper.t2i = self.t2i
        init_tok = HanfireZipVoiceCompiler.ONSET_TO_ZV.get(slot.onset, "")
        if not init_tok:
            if slot.medial in ("j", "ɥ"):
                init_tok = "y0"
            elif slot.medial == "w":
                init_tok = "w0"
        fin_tok, coda_tok = helper._resolve_final_tokens(slot.medial, slot.nucleus, slot.coda, tone_d, checked)
        total = (92 if metrical else 80) if checked else 100
        items: List[Tuple[int, int]] = []
        dbg: List[str] = []
        if init_tok and init_tok in self.t2i and fin_tok in self.t2i:
            w_i = 26 if checked else 30
            if coda_tok and coda_tok in self.t2i and not checked:
                items = [(self.t2i[init_tok], w_i), (self.t2i[fin_tok], total - w_i - 20), (self.t2i[coda_tok], 20)]
                dbg = [init_tok, fin_tok, coda_tok]
            else:
                items = [(self.t2i[init_tok], w_i), (self.t2i[fin_tok], total - w_i)]
                dbg = [init_tok, fin_tok]
        elif fin_tok in self.t2i:
            items = [(self.t2i[fin_tok], min(total, 76 if checked else 72))]
            dbg = [fin_tok]
        return items, "+".join(dbg)

    def compile_spec(self, ch: str, spec: str, metrical: bool = True) -> Tuple[List[Tuple[int, int]], str, Optional[Reading]]:
        """支持五种零重训单字覆写规格（源自 Swarm L1 架构设计）：
        1) '同某' -> 借用某字读音（如 [滅=同絕] 强制同韵同呼）
        2) 'tok1:w1+tok2:w2' -> 直接指定底层词元与时长权重（如 [花=Q_曉:26+ua1:74]）
        3) '韵类_调类' -> 仅替换韵母槽而保留原字切韵声母（如 [光=唐合一_平陰] 或 [光=Q_唐合一_舒_平陰]）
        4) 切韵描述或声调字（如 [處=昌開三魚去] 或 [行=去]）
        5) 直接 IPA/拼音+声调（如 [花=huɑ1] 或 [光=kwaŋ1]），走规范 5 槽直编
        """
        spec = spec.strip()
        if spec.startswith("同") and len(spec) >= 2:
            return self.compile_char(spec[1], metrical=metrical)
        if ":" in spec or "+" in spec:
            items: List[Tuple[int, int]] = []
            parts = [p.strip() for p in spec.split("+") if p.strip()]
            for p in parts:
                tok_s, w_s = (p.split(":", 1) + ["50"])[:2]
                if tok_s in self.t2i:
                    items.append((self.t2i[tok_s], int(w_s)))
            if items:
                return items, spec, None
        # 3) 韵类_调类直接覆写（如 唐合一_平陰 / Q_唐合一_舒_平陰）
        if "_" in spec:
            cand_fin = spec if spec.startswith("Q_") else f"Q_{spec}"
            if cand_fin not in self.t2i:
                parts_u = spec.removeprefix("Q_").rsplit("_", 1)
                if len(parts_u) == 2:
                    mode_tag = "入" if "入" in parts_u[1] else "舒"
                    alt_fin = f"Q_{parts_u[0]}_{mode_tag}_{parts_u[1]}"
                    if alt_fin in self.t2i:
                        cand_fin = alt_fin
            if cand_fin in self.t2i:
                r = self.lookup(ch)
                checked = "_入_" in cand_fin
                total = (92 if metrical else 80) if checked else 100
                if r is not None and self.initial_name(r) in self.t2i:
                    ini = self.initial_name(r)
                    w_i = 12 if r.mu == "影" else (26 if checked else 30)
                    return [(self.t2i[ini], w_i), (self.t2i[cand_fin], total - w_i)], f"{ini}+{cand_fin}", r
                return [(self.t2i[cand_fin], total)], cand_fin, r
        # 5) 纯拉丁/IPA 串（不含汉字，避免误把重纽切韵描述如 '羣開三B支去' 当成 IPA）直接走 5 槽音位编译（如 [花=huɑ1] / [光=kwaŋ1]）
        if (not any("\u4e00" <= c <= "\u9fff" for c in spec)) and (
            any(c.isascii() and c.isalpha() for c in spec) or any("\u0250" <= c <= "\u02af" for c in spec)
        ):
            items_ipa, dbg_ipa = self._compile_ipa_override(spec, metrical=metrical)
            if items_ipa:
                return items_ipa, f"ipa:{dbg_ipa}", None
        return self.compile_char(ch, override=spec, metrical=metrical)

    def compile_char(
        self, ch: str, override: Optional[str] = None, metrical: bool = True
    ) -> Tuple[List[Tuple[int, int]], str, Optional[Reading]]:
        eff_ov = override or self.default_overrides.get(ch) or self.default_overrides.get(self.s2t.get(ch, ""))
        _is_ipa_ov = bool(
            eff_ov
            and (not any("\u4e00" <= c <= "\u9fff" for c in eff_ov))
            and (any(c.isascii() and c.isalpha() for c in eff_ov) or any("\u0250" <= c <= "\u02af" for c in eff_ov))
        )
        if eff_ov and (
            eff_ov.startswith("同")
            or ":" in eff_ov
            or "+" in eff_ov
            or "_" in eff_ov
            or _is_ipa_ov
        ):
            return self.compile_spec(ch, eff_ov, metrical=metrical)
        r = self.lookup(ch, eff_ov)
        if r is None:
            if len(ch) == 1 and "\u4e00" <= ch <= "\u9fff":
                from .zipvoice_zh_manifold import tokenize_mandarin_prompt_exact
                fb_it, _ = tokenize_mandarin_prompt_exact(ch, self.t2i)
                if fb_it:
                    return fb_it, "mandarin_fallback", None
            return [], "", None
        ini, fin = self.initial_name(r), self.final_name(r)
        # 中古崇母止摄仄声擦化律（士/仕/事/柿/俟…）：盛中唐崇母止摄仄声已擦化为浊擦 [ʐ/ʑ]（平声如“床/崇/愁”仍保塞擦 Q_崇），
        # 改用无塞音闭锁停顿的 Q_常（sh0+r0），彻底消除“寒士/人事/萬事”中因 OC_dʒ 闭锁导致的 150ms 误断句；
        # 其中全濁上聲（士/仕/柹/戺，中唐濁上變去 ˨˨˦）同步归入去陽，消除低频 i3 气声微弱化。
        if r.mu == "崇" and r.sheng != "平" and r.yun in ("之", "支", "脂", "微") and "Q_常" in self.t2i:
            ini = "Q_常"
            if r.sheng == "上":
                qu_fin = f"Q_{r.yunlei}_去陽"
                if qu_fin in self.t2i:
                    fin = qu_fin
        checked = r.sheng == "入"
        # 格律吟诵模式 (metrical=True)：入声总重设为 92（而非 80），消除“不覺/花落”句中连入声字突然加速 25% 的不稳定感
        total = (92 if metrical else 80) if checked else 100
        # 1) 精组三四等尖音 (絕/雪/斜/細/須)、见组三等魚韵 (舉/去/渠/居 [kɨə/ky]) 或 唇音重纽三等A类入声 (滅/別/必 [mjɛt/pjɛt]):
        # 注：见组虞韵（俱/拘/區/驅/句 [kʉu/ku]）保留原生二词元 (Q_見/溪/羣 + Q_虞三_舒_*)，严禁误送 OOD 组合 g0+v1 裂变为双音“gui”！
        need_y0 = (
            (r.mu in self.JING_MU and ("三" in r.yunlei or "四" in r.yunlei))
            or (r.mu in self.JIAN_MU and r.yun == "魚")
            or (r.mu in self.BANG_MU and "三A_入" in r.yunlei)
        )
        if need_y0 and "y0" in self.t2i:
            # 精组舒声魚虞韵（須/徐/取/聚/序…）与见组舒声魚韵（居/舉/去/渠…）：y0 在普通话里恒为音节起首，
            # [声母+y0+后/央圆唇韵] 三词元会分裂为双音节（如“舉→ji+ü”、“須→思+与”）。
            # 统一改用 [声母 + 纯撮口音 v_tone] 两词元（平陰复用纯合成 v1 槽 Q_魚開三_舒_平陰，平陽 v2、上 v3、去 v4），
            # 既杜绝双音节分裂，又完整保留精组尖音与四声调型！
            if not checked and ((r.mu in self.JING_MU and r.yun in ("魚", "虞")) or (r.mu in self.JIAN_MU and r.yun == "魚")):
                v_tok = (
                    "Q_魚開三_舒_平陰" if r.tone_cat == "平陰"
                    else ("v2" if r.tone_cat == "平陽"
                    else ("v3" if r.sheng == "上" else "v4"))
                )
                if v_tok in self.t2i:
                    return [(self.t2i[ini], 26), (self.t2i[v_tok], total - 26)], f"{ini}+{v_tok}", r
            # 入声三槽位 (絕/滅/雪 等): y0 至少占 16w (约 4 帧，对齐 Zipformer 4x 下采样步长)，防止 12w (2.9 帧) 被下采样吞没退化为后元音
            w_i = 18 if checked else 20
            w_m = 16 if checked else 14
            w_f = total - w_i - w_m
            return [(self.t2i[ini], w_i), (self.t2i["y0"], w_m), (self.t2i[fin], w_f)], f"{ini}+y0+{fin}", r
        if r.mu == "影":
            # 影母 [ʔ-] 无固有腭化/唇化，介音严格由呼/等/韵决定，严禁统一用偏 y0 的 Q_影 污染洪音（否则“翁→雍”给前字“笠”挂出 y 尾巴、“安→烟”）：
            # 1) 一二等合口、模一、東/冬一入（烏/屋/沃/溫/彎/蛙）及三四等合口微支脂（威/委）：短唇化轻起音 w0:12
            # 2) 三四等细音（衣/伊/一/因/煙/於/雍/淵/冤）：短腭化轻起音 y0:12
            # 3) 東/冬一舒（翁，普通话无 wong，w0+ong 属 OOD）及一二等开口洪音（安/庵/哀/愛/阿/奧/歐/鴉/惡）：纯零声母单词元
            is_labial_ying = (
                (r.deng in ("一", "二") and (r.hu == "合" or r.yun == "模" or (checked and r.yun in ("東", "冬"))))
                or (r.deng in ("三", "四") and r.hu == "合" and r.yun in ("微", "支", "脂"))
            )
            if is_labial_ying and "w0" in self.t2i:
                return [(self.t2i["w0"], 12), (self.t2i[fin], total - 12)], f"w0+{fin}", r
            if r.deng in ("三", "四") and "y0" in self.t2i:
                return [(self.t2i["y0"], 12), (self.t2i[fin], total - 12)], f"y0+{fin}", r
            w_zero = 88 if (not checked and r.yun in ("東", "冬")) else (76 if checked else 72)
            return [(self.t2i[fin], w_zero)], fin, r
        if r.mu == "云":
            # 中古云母（匣母三等）至盛中唐已脱离喉擦 [h0] 并入轻半元音滑音 [w/j/ɥ]；
            # 沿用含 h0(cos=0.79) 的 Q_云:30 会在去声后（如“細雨”）制造 225ms 气声静音空洞！
            # 按呼/韵分派 14w 短滑音：合口微支祭廢陽庚舒声（王/為/衛/圍/榮/永）用 w0:14，其余（雨/羽/宇/于/雲/有/友/遠/圓/越/炎）用 y0:14。
            is_labial_yun = (
                not checked
                and r.hu == "合"
                and r.yun in ("微", "支", "祭", "廢", "陽", "庚")
            )
            glide_tok = "w0" if (is_labial_yun and "w0" in self.t2i) else "y0"
            if glide_tok in self.t2i:
                return [(self.t2i[glide_tok], 14), (self.t2i[fin], total - 14)], f"{glide_tok}+{fin}", r
        else:
            w_i = 26 if checked else 30
            # 仅在纯开口舒声字略微保底擦音；合口/三四等/入声字不抢占韵母时间（避免“花→哈”“雪”入声收尾被挤薄）
            if r.mu in FRICATIVE_MU and not checked and r.hu != "合" and r.deng not in ("三", "四"):
                w_i = 34
        items = [(self.t2i[ini], w_i), (self.t2i[fin], total - w_i)]
        return items, f"{ini}+{fin}", r

    def compile_clause(
        self,
        text: str,
        overrides: Optional[Dict[str, str]] = None,
        end_punct: str = "",
        metrical: bool = True,
        rhyme_hold_w: int = 12,
    ) -> Tuple[List[Tuple[int, int]], str]:
        """支持全局 overrides 字典、configs/overrides_mc.json 及句内位置级标注 '[處=昌開三魚去]處聞啼鳥'。"""
        import re as _re
        overrides = overrides or {}
        char_groups: List[List[Tuple[int, int]]] = []
        dbg = []
        pos = 0
        tag_re = _re.compile(r"\[([^\[\]=]+)=([^\[\]]+)\]")
        bigram_keys = sorted(self.bigram_overrides.keys(), key=len, reverse=True)
        while pos < len(text):
            m = tag_re.match(text, pos)
            if m:
                ch_part, spec = m.group(1).strip(), m.group(2).strip()
                for ch in ch_part:
                    it, d, r = self.compile_spec(ch, spec, metrical=metrical)
                    if it:
                        char_groups.append(list(it))
                        dbg.append(f"{ch}[{d}|{r.desc if r else 'custom'}]")
                pos = m.end()
                continue
            # 多字词上下文覆写（如 可汗 / 單于），优先级次于句内显式 [...] 标注、高于单字默认读音
            matched_bg = None
            for bg in bigram_keys:
                if text.startswith(bg, pos):
                    matched_bg = bg
                    break
            if matched_bg is not None:
                specs = self.bigram_overrides[matched_bg]
                for ch, sp in zip(matched_bg, specs):
                    sp_eff = overrides.get(ch) or sp
                    it, d, r = (
                        self.compile_spec(ch, sp_eff, metrical=metrical)
                        if sp_eff
                        else self.compile_char(ch, None, metrical=metrical)
                    )
                    if it:
                        char_groups.append(list(it))
                        dbg.append(f"{ch}[{d}|{r.desc if r else 'custom'}]")
                pos += len(matched_bg)
                continue
            ch = text[pos]
            pos += 1
            it, d, r = self.compile_char(ch, overrides.get(ch), metrical=metrical)
            if not it:
                continue
            char_groups.append(list(it))
            dbg.append(f"{ch}[{d}|{r.desc if r else 'custom'}]")
        # 句末舒声韵脚稳态延音（入声韵脚保持原长，不额外拉长也不截短，防止“望明月/寒江雪”末字被前字吞没！）
        if metrical and char_groups:
            last_tok_id, last_w = char_groups[-1][-1]
            last_is_checked = len(dbg) > 0 and ("_入_" in dbg[-1] or dbg[-1].endswith("入]"))
            if not last_is_checked and rhyme_hold_w > 0:
                char_groups[-1][-1] = (last_tok_id, last_w + rhyme_hold_w)
        items = [pair for grp in char_groups for pair in grp]
        if end_punct and end_punct in self.t2i:
            items.append((self.t2i[end_punct], 38 if end_punct == "." else 26))
        return items, " ".join(dbg)


class FactorizedMCEmbedding(nn.Module):
    """正交音韵因子化嵌入 (L3 Rhyme-Group Locking)：
    旧行可冻结；新声母独立可学；
    新韵母 = rhyme_core[韻_舒入] + medial[呼_等] + seg[韵类细微残差] + tone[调类]。
    同一韵部与舒入属性（如 仙_入 的“絕/滅/雪”、陽_舒 的“霜/鄉”）在参数层天然共享同一个 rhyme_core 向量！
    """

    def __init__(
        self,
        base: nn.Embedding,
        n_old: int,
        ini_init: torch.Tensor,
        seg_init: torch.Tensor,
        tone_init: torch.Tensor,
        fin_seg_idx: torch.Tensor,
        fin_tone_idx: torch.Tensor,
        rhyme_init: Optional[torch.Tensor] = None,
        medial_init: Optional[torch.Tensor] = None,
        fin_rhyme_idx: Optional[torch.Tensor] = None,
        fin_med_idx: Optional[torch.Tensor] = None,
        coda_init: Optional[torch.Tensor] = None,
        fin_coda_idx: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.register_buffer("old", base.weight.data[:n_old].clone())
        self.ini = nn.Parameter(ini_init.clone())
        self.seg = nn.Parameter(seg_init.clone())
        self.tone = nn.Parameter(tone_init.clone())
        self.register_buffer("fin_seg_idx", fin_seg_idx)
        self.register_buffer("fin_tone_idx", fin_tone_idx)
        self.has_rhyme_lock = rhyme_init is not None and medial_init is not None
        if self.has_rhyme_lock:
            self.rhyme_core = nn.Parameter(rhyme_init.clone())
            self.medial = nn.Parameter(medial_init.clone())
            self.register_buffer("fin_rhyme_idx", fin_rhyme_idx)
            self.register_buffer("fin_med_idx", fin_med_idx)
        self.has_coda_lock = coda_init is not None and fin_coda_idx is not None
        if self.has_coda_lock:
            self.coda = nn.Parameter(coda_init.clone())
            self.register_buffer("fin_coda_idx", fin_coda_idx)
        self.embedding_dim = base.embedding_dim
        self.padding_idx = base.padding_idx

    @property
    def weight(self) -> torch.Tensor:
        fin = self.seg[self.fin_seg_idx] + self.tone[self.fin_tone_idx]
        if self.has_rhyme_lock:
            fin = fin + self.rhyme_core[self.fin_rhyme_idx] + self.medial[self.fin_med_idx]
        if self.has_coda_lock:
            fin = fin + self.coda[self.fin_coda_idx]
        return torch.cat([self.old, self.ini, fin], dim=0)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return nn.functional.embedding(idx, self.weight, self.padding_idx)

    def materialize(self) -> nn.Embedding:
        w = self.weight.detach()
        e = nn.Embedding(w.size(0), w.size(1), padding_idx=self.padding_idx)
        e.weight.data.copy_(w)
        return e


def build_mc_embedding(model: nn.Module, t2i: Dict[str, int], old_compiler, comp: Optional[QieyunMCCompiler] = None):
    """扩充 ZipVoice 嵌入表，注册 Q_* 词元。返回 (新 t2i, FactorizedMCEmbedding, QieyunMCCompiler)。

    old_compiler: 旧 ThreeEraHanfireCompiler（用其 _resolve_final_tokens 把每个韵类的晚唐拟音映射到近似 token 作初始化）
    """
    from .phonology import parse_mononuclear_4slot

    emb = model.embed.weight.data
    n_old = emb.size(0)
    t2i = dict(t2i)
    comp = comp or QieyunMCCompiler(t2i)
    ini_names, fin_names = comp.all_token_names()

    def row(name: str) -> torch.Tensor:
        return emb[t2i[name]]

    # 1. 声母
    ini_vecs = []
    for n in ini_names:
        srcs = [s for s in MU_INIT[n[2:]] if s in t2i]
        v = torch.stack([row(s) for s in srcs]).mean(0) if srcs else emb[t2i["y0"]]
        ini_vecs.append(v + 0.01 * torch.randn_like(v))

    # 2. 声调偏移（普通话 4 声相对“去调均值”的偏移）
    tone_off = {}
    bases = [b for b in ("a", "an", "ang", "ai", "ao", "e", "en", "eng", "i", "in", "ing", "u", "un", "ong", "ou", "ia", "ian", "iang", "ua", "uan", "uo")
             if all(f"{b}{t}" in t2i for t in "1234")]
    for t in "1234":
        tone_off[t] = torch.stack([row(f"{b}{t}") - torch.stack([row(f"{b}{u}") for u in "1234"]).mean(0) for b in bases]).mean(0)
    tone_vecs = torch.stack([sum(w * tone_off[t] for t, w in TONE_INIT[c].items()) for c in TONE_CATS])

    # 3. 韵类段向量：取该韵类任一读音的晚唐拟音 → 旧映射 → 去调平均
    sample: Dict[str, Reading] = {}
    for rs in comp.readings.values():
        for r in rs:
            sample.setdefault(r.yunlei, r)
    yunleis = sorted({n[2:].rsplit("_", 1)[0] for n in fin_names})
    seg_vecs = []
    yl_coda_key: Dict[str, str] = {}
    for yl in yunleis:
        r = sample[yl]
        # 归一化 mid_tang 中的潘悟云/切韵特殊元音符号 (ɨ/ʉ/ɯ/ɜ/œ/ɒ)，防止 parse_mononuclear_4slot 退化为央元音 ə ("了/乐/葛")
        ipa = re.sub(r"[˥˦˧˨˩ʼ]", "", r.mid_tang)
        ipa = (
            ipa.replace("ɨuŋ", "juŋ").replace("ɨu", "ju").replace("ɨw", "ju")
            .replace("ɨɯ", "jo").replace("iɯ", "jo").replace("ɯ", "uo")
            .replace("ʉɜ", "yɛ").replace("ʉ", "y").replace("ɨi", "i").replace("ɨ", "i")
            .replace("ɜ", "ɛ").replace("œ", "ɑ").replace("uɒ", "ua").replace("ɒ", "ɑ").replace("ʌ", "ɑ")
        )
        s = parse_mononuclear_4slot(ipa)
        yl_coda_key[yl] = f"{s.coda or 'open'}_{'入' if r.sheng == '入' else '舒'}"
        fin_tok, coda_tok = old_compiler._resolve_final_tokens(s.medial, s.nucleus, s.coda, "1", r.sheng == "入")
        cands = []
        for t in "1234":
            nm = re.sub(r"\d$", t, fin_tok)
            if nm in t2i:
                cands.append(row(nm))
        v = torch.stack(cands).mean(0) if cands else row(fin_tok)
        if coda_tok and coda_tok in t2i:
            v = 0.85 * v + 0.15 * row(coda_tok)
        seg_vecs.append(v)
    seg_vecs = torch.stack(seg_vecs)

    # 4. 正交分解 seg_vecs = rhyme_core[韻_舒入] + medial[呼_等] + coda[韵尾_舒入] + seg_residual (L3 韵部+韵尾硬共享锁定)
    yl_rhyme_key = {yl: f"{sample[yl].yun}_{'入' if sample[yl].sheng == '入' else '舒'}" for yl in yunleis}
    yl_med_key = {yl: f"{sample[yl].hu or '開'}_{sample[yl].deng or '一'}" for yl in yunleis}
    rhyme_groups = sorted(set(yl_rhyme_key.values()))
    med_groups = sorted(set(yl_med_key.values()))
    coda_groups = sorted(set(yl_coda_key.values()))
    rg_idx = {g: i for i, g in enumerate(rhyme_groups)}
    mg_idx = {m: i for i, m in enumerate(med_groups)}
    cg_idx = {c: i for i, c in enumerate(coda_groups)}

    rhyme_init = torch.stack([
        seg_vecs[[i for i, yl in enumerate(yunleis) if yl_rhyme_key[yl] == g]].mean(0)
        for g in rhyme_groups
    ])
    rem1 = torch.stack([seg_vecs[i] - rhyme_init[rg_idx[yl_rhyme_key[yl]]] for i, yl in enumerate(yunleis)])
    medial_init = torch.stack([
        rem1[[i for i, yl in enumerate(yunleis) if yl_med_key[yl] == m]].mean(0)
        for m in med_groups
    ])
    rem2 = torch.stack([rem1[i] - medial_init[mg_idx[yl_med_key[yl]]] for i, yl in enumerate(yunleis)])
    coda_init = torch.stack([
        rem2[[i for i, yl in enumerate(yunleis) if yl_coda_key[yl] == c]].mean(0)
        for c in coda_groups
    ])
    seg_residual = torch.stack([
        rem2[i] - coda_init[cg_idx[yl_coda_key[yl]]] for i, yl in enumerate(yunleis)
    ])

    yl_idx = {y: i for i, y in enumerate(yunleis)}
    tc_idx = {c: i for i, c in enumerate(TONE_CATS)}
    fin_seg_idx = torch.tensor([yl_idx[n[2:].rsplit("_", 1)[0]] for n in fin_names])
    fin_tone_idx = torch.tensor([tc_idx[n.rsplit("_", 1)[1]] for n in fin_names])
    fin_rhyme_idx = torch.tensor([rg_idx[yl_rhyme_key[n[2:].rsplit("_", 1)[0]]] for n in fin_names])
    fin_med_idx = torch.tensor([mg_idx[yl_med_key[n[2:].rsplit("_", 1)[0]]] for n in fin_names])
    fin_coda_idx = torch.tensor([cg_idx[yl_coda_key[n[2:].rsplit("_", 1)[0]]] for n in fin_names])

    for i, n in enumerate(ini_names):
        t2i[n] = n_old + i
    for i, n in enumerate(fin_names):
        t2i[n] = n_old + len(ini_names) + i
    comp.t2i = t2i
    fe = FactorizedMCEmbedding(
        model.embed, n_old, torch.stack(ini_vecs), seg_residual, tone_vecs,
        fin_seg_idx, fin_tone_idx,
        rhyme_init=rhyme_init, medial_init=medial_init,
        fin_rhyme_idx=fin_rhyme_idx, fin_med_idx=fin_med_idx,
        coda_init=coda_init, fin_coda_idx=fin_coda_idx,
    )
    meta = {
        "n_old": n_old, "ini_names": ini_names, "fin_names": fin_names,
        "yunleis": yunleis, "tone_cats": TONE_CATS,
        "rhyme_groups": rhyme_groups, "med_groups": med_groups, "coda_groups": coda_groups,
    }
    return t2i, fe, comp, meta
