#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出浏览器端 V12 Studio 预编译四模式词表 (lexicon_3era.json[.gz])
全面对齐 Python 端 render_v10_studio.py 与 AGENTS.md 红线：
1. mc 模式直接采用 1449 词元 QieyunMCCompiler (38 声母 + 249 韵类 + 尖音/重纽 y0 桥接 + 12w 影母 + overrides_mc.json)；
2. oc / em / em_zyyy 模式采用 ThreeEraHanfireCompiler + overrides_{oc,em}.json + 双向简繁扩充 + 异体字映射；
3. 导出多音字位置覆写字典 polyphones (支持浏览器端直接写 [處=昌開三魚去]、[行=去]、[種=tʂuŋh]、[滅=同絕]、[仙=s0:30+ian1:70])；
4. 导出连上变调表 sandhi_3to2 与促声集合 chk_ids，使前端 JS 编译出的 [tokId, weight] 与 Python 100% 逐字一致！
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "vendor" / "ZipVoice")):
    if p not in sys.path:
        sys.path.insert(0, p)

from hanfire_core.mc_qieyun import QieyunMCCompiler  # noqa: E402
from hanfire_core.zipvoice_zh_manifold import ThreeEraHanfireCompiler  # noqa: E402


def build_web_lexicon(t2i: Dict[str, int], bundle_dir: Path, assets_dir: Path) -> int:
    mc_comp = QieyunMCCompiler(t2i)
    old_comp = ThreeEraHanfireCompiler(t2i=t2i, three_era_json=bundle_dir / "v5_three_era_lexicon.json", v8_fixes=True)
    i2t = {v: k for k, v in t2i.items()}

    # 收集全量汉字集合（含简繁双向扩展与异体字）
    all_chars = set(mc_comp.readings.keys())
    for src in ("oc", "mc", "em"):
        all_chars.update(old_comp.eras[src].keys())
    all_chars.update(ThreeEraHanfireCompiler.VARIANT_MAP.keys())
    try:
        from opencc import OpenCC
        s2t, t2s = OpenCC("s2t"), OpenCC("t2s")
        expanded = set()
        for ch in list(all_chars):
            if len(ch) == 1:
                c1, c2 = t2s.convert(ch), s2t.convert(ch)
                if len(c1) == 1:
                    expanded.add(c1)
                if len(c2) == 1:
                    expanded.add(c2)
        all_chars.update(expanded)
    except Exception:
        pass

    compiled: Dict[str, Dict[str, List]] = {"oc": {}, "mc": {}, "em": {}, "em_zyyy": {}}
    polyphones: Dict[str, List] = {}

    # 1. 编译 MC (QieyunMCCompiler, metrical=True) + 导出全部切韵多音字规格
    for ch in sorted(all_chars):
        if len(ch) != 1 or not ("\u4e00" <= ch <= "\u9fff"):
            continue
        it, dbg, r = mc_comp.compile_char(ch, metrical=True)
        if it:
            is_chk = bool(r is not None and r.sheng == "入")
            desc = r.desc if r is not None else "fallback"
            flat = [int(v) for pair in it for v in pair]
            # flags: bit0=zero_open(0), bit1=ends_open(0), bit2=is_checked(4)
            compiled["mc"][ch] = [flat, desc, dbg, 4 if is_chk else 0]

    for ch, rs in mc_comp.readings.items():
        if len(rs) > 1 or ch in mc_comp.default_overrides:
            for r in rs:
                it, dbg, r_obj = mc_comp.compile_char(ch, override=r.desc, metrical=True)
                if it and r_obj is not None:
                    flat = [int(v) for pair in it for v in pair]
                    fl = 4 if r_obj.sheng == "入" else 0
                    entry = [flat, r_obj.desc, dbg, fl]
                    polyphones[f"mc|{ch}|{r.desc}"] = entry
                    polyphones.setdefault(f"mc|{ch}|{r.sheng}", entry)
                    # 同时映射到简体字键，方便浏览器输入简体如 [处=昌開三魚去]
                    try:
                        sc = t2s.convert(ch)
                        if len(sc) == 1 and sc != ch:
                            polyphones[f"mc|{sc}|{r.desc}"] = entry
                            polyphones.setdefault(f"mc|{sc}|{r.sheng}", entry)
                    except Exception:
                        pass

    # 2. 编译 OC / EM / EM_ZYYY (含 configs/overrides_{oc,em}.json 持久覆写)
    for mode in ("oc", "em", "em_zyyy"):
        era_key = "em" if mode == "em_zyyy" else mode
        ov_map = old_comp._load_era_overrides(era_key)
        for ch in sorted(all_chars):
            if len(ch) != 1 or not ("\u4e00" <= ch <= "\u9fff"):
                continue
            ov_spec = ov_map.get(ch)
            if not ov_spec:
                rk = old_comp._resolve_era_char_key(era_key, ch)
                if rk and rk in ov_map:
                    ov_spec = ov_map[rk]
            if ov_spec:
                it, dbg = old_comp._compile_era_spec(mode, ch, ov_spec)
                raw_ipa = ov_spec
            else:
                rk = old_comp._resolve_era_char_key(era_key, ch)
                if rk is None:
                    continue
                raw_ipa = old_comp.eras[era_key][rk][0]
                it, dbg = old_comp.compile_era_char(mode, ch)
            zero_open, ends_open = old_comp._last_char_flags
            is_chk = any(i2t[tid].endswith("_chk") for tid, _ in it)
            flat = [int(v) for pair in it for v in pair]
            flags = int(zero_open) | (int(ends_open) << 1) | ((4 if is_chk else 0))
            compiled[mode][ch] = [flat, raw_ipa, dbg, flags]

    # 预置常用句内 IPA 覆写规格（如《桃花庵歌》[種=tʂuŋh]、《采薇》[昔=sjak] [矣=jiʔ]）
    preset_ipa_specs = [
        ("em", "種", "tʂuŋh"), ("em", "种", "tʂuŋh"),
        ("em_zyyy", "種", "tʂuŋh"), ("em_zyyy", "种", "tʂuŋh"),
        ("oc", "昔", "sjak"), ("oc", "矣", "jiʔ"), ("oc", "為", "同維"), ("oc", "为", "同維"),
    ]
    for mode, ch, spec in preset_ipa_specs:
        it, dbg = old_comp._compile_era_spec(mode, ch, spec)
        flat = [int(v) for pair in it for v in pair]
        polyphones[f"{mode}|{ch}|{spec}"] = [flat, spec, dbg, 0]

    # 2.5 导出多字词上下文覆写表 bigrams（如 可汗 / 單于）
    bigrams: Dict[str, Dict[str, List]] = {"mc": {}, "oc": {}, "em": {}, "em_zyyy": {}}
    for bg, specs in mc_comp.bigram_overrides.items():
        entries = []
        for ch, sp in zip(bg, specs):
            it, dbg, r_obj = mc_comp.compile_spec(ch, sp, metrical=True) if sp else mc_comp.compile_char(ch, metrical=True)
            flat = [int(v) for pair in it for v in pair]
            fl = 4 if (r_obj is not None and r_obj.sheng == "入") else 0
            entries.append([flat, r_obj.desc if r_obj is not None else sp, dbg, fl])
        bigrams["mc"][bg] = entries
    for mode in ("oc", "em", "em_zyyy"):
        era_key = "em" if mode == "em_zyyy" else mode
        for bg, specs in old_comp._load_era_bigram_overrides(era_key).items():
            entries = []
            for ch, sp in zip(bg, specs):
                it, dbg = old_comp._compile_era_spec(mode, ch, sp) if sp else old_comp.compile_era_char(mode, ch)
                zero_open, ends_open = old_comp._last_char_flags
                is_chk = any(i2t[tid].endswith("_chk") for tid, _ in it)
                flat = [int(v) for pair in it for v in pair]
                fl = int(zero_open) | (int(ends_open) << 1) | (4 if is_chk else 0)
                entries.append([flat, sp, dbg, fl])
            bigrams[mode][bg] = entries

    # 3. 导出连上变调映射表 (Tone 3 -> Tone 2) 与促声/声调核集合
    sandhi_3to2: Dict[str, int] = {}
    tone_of_id: Dict[str, str] = {}
    for tok, tid in t2i.items():
        if tok.endswith("_chk"):
            tone_of_id[str(tid)] = "chk"
        elif tok and tok[-1] in ("1", "2", "3", "4", "5") and not tok.startswith("Q_"):
            tone_of_id[str(tid)] = tok[-1]
            if tok[-1] == "3":
                t2_nm = tok[:-1] + "2"
                if t2_nm in t2i:
                    sandhi_3to2[str(tid)] = int(t2i[t2_nm])

    payload = {
        "version": "v12_studio",
        "t2i": {k: int(v) for k, v in t2i.items()},
        "punct": {",": [int(t2i[","]), 26], ".": [int(t2i["."]), 38], "?": [int(t2i["?"]), 38], "!": [int(t2i["!"]), 38]},
        "hiatus_pause": [int(t2i[","]), int(old_comp.HIATUS_PAUSE_W)],
        "lead_block": [int(t2i[","]), 26],
        "tail_block": [int(t2i["."]), 38],
        "y0_id": int(t2i["y0"]),
        "w0_id": int(t2i["w0"]),
        "char0_boost_w": 44,
        "rhyme_hold_w": 16,
        "era_rate": {"mc": 24.5, "oc": 25.0, "em": 24.0, "em_zyyy": 24.0},
        "sandhi_3to2": sandhi_3to2,
        "tone_of_id": tone_of_id,
        "polyphones": polyphones,
        "bigrams": bigrams,
        "modes": compiled,
    }
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    assets_dir.mkdir(parents=True, exist_ok=True)
    (assets_dir / "lexicon_3era.json").write_bytes(raw)
    with gzip.open(assets_dir / "lexicon_3era.json.gz", "wb", compresslevel=9) as gz:
        gz.write(raw)
    return len(raw)


if __name__ == "__main__":
    ck_p = PROJECT_ROOT / "v10_outputs" / "best_fp32_v10_studio.pt"
    ck = torch.load(str(ck_p), map_location="cpu", weights_only=False)
    n = build_web_lexicon(ck["t2i_custom"], PROJECT_ROOT / "v5_bundle", PROJECT_ROOT / "web_wasm" / "assets")
    gz_mb = (PROJECT_ROOT / "web_wasm" / "assets" / "lexicon_3era.json.gz").stat().st_size / 1e6
    print(f"✓ lexicon_3era.json v12_studio 导出完成 (raw={n/1e6:.2f} MB, gz={gz_mb:.2f} MB)")
