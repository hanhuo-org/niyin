#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
打包 V10 三代同堂训练数据包 (prepare_v10_bundle.py)
合并：
  - OC (上古): v5_train_dataset.pt 中的全部 OC 切片 (is_single=False)
  - MC (中古): xiaoqieyun_all 单字音节 (is_single=True)
              + qzw2026 千字文 4 字句 (is_single=False)
              + 周南 MC 多字句 (mc_zn_verified_manifest.json, is_single=False)
  - EM (近古): v5_train_dataset.pt 中的全部 EM 切片 (is_single=False)
              + em_short_manifest.json (明清官话诗文, is_single=False)
              + em_xyj_manifest.json (明官话西游记第一回, is_single=False)
严格排除测试集三首唐诗：江雪 / 静夜思 / 春晓。
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf
import torch
from scipy import signal

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HERE))

from colab_train_v5_qat import PureVocosFbank, SR, ThreeEraHanfireCompiler, ZipVoice  # noqa: E402
from hanfire_core.mc_qieyun import build_mc_embedding  # noqa: E402
from train_v9_mc import build_qzw_items, build_xiaoqieyun_items  # noqa: E402

HELD_OUT_SUBSTRS = (
    "千山鳥飛絕", "千山鸟飞绝", "萬徑人蹤滅", "万径人踪灭", "孤舟蓑笠翁", "獨釣寒江雪", "独钓寒江雪",
    "床前明月光", "疑是地上霜", "舉頭望明月", "举头望明月", "低頭思故鄉", "低头思故乡",
    "春眠不覺曉", "春眠不觉晓", "處處聞啼鳥", "处处闻啼鸟", "夜來風雨聲", "夜来风雨声", "花落知多少",
)


def is_held_out(text: str) -> bool:
    clean = re.sub(r"[^\u4e00-\u9fff]", "", text)
    return any(s in clean for s in HELD_OUT_SUBSTRS)


def build_manifest_items(
    manifest_path: Path,
    era: str,
    mc_comp,
    old_comp: ThreeEraHanfireCompiler,
    fbank: PureVocosFbank,
) -> List[dict]:
    if not manifest_path.exists():
        print(f"  [跳过] 未找到 {manifest_path}")
        return []
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    hp_sos = signal.butter(4, 75.0 / (SR * 0.5), btype="highpass", output="sos")
    wav_cache: Dict[str, tuple[np.ndarray, int]] = {}
    items: List[dict] = []
    skipped = 0

    for idx, ent in enumerate(entries):
        text = str(ent.get("text", "")).strip()
        hanzi = re.sub(r"[^\u4e00-\u9fff]", "", text)
        if len(hanzi) < 2 or is_held_out(hanzi):
            skipped += 1
            continue
        wav_p = str(ent["wav"])
        if wav_p not in wav_cache:
            w, sr = sf.read(wav_p, dtype="float32")
            if w.ndim > 1:
                w = w.mean(axis=1)
            if sr != SR:
                w = signal.resample_poly(w, SR, sr).astype(np.float32)
            w = signal.sosfiltfilt(hp_sos, w).astype(np.float32)
            wav_cache[wav_p] = (w, SR)
        w, sr = wav_cache[wav_p]

        s_s, e_s = float(ent["start"]), float(ent["end"])
        dur = e_s - s_s
        if dur < 0.35 or dur > 12.5:
            skipped += 1
            continue
        a = max(0, int((s_s - 0.03) * sr))
        b = min(len(w), int((e_s + 0.05) * sr))
        y = w[a:b]
        if len(y) < int(0.35 * sr):
            skipped += 1
            continue
        rms = float(np.sqrt(np.mean(y ** 2)) + 1e-6)
        if rms < 1e-4:
            skipped += 1
            continue
        y = np.clip(y * (0.1 / rms), -0.95, 0.95).astype(np.float32)

        if era == "mc":
            tok, _ = mc_comp.compile_clause(hanzi)
            if len(tok) // 2 != len(hanzi):
                skipped += 1
                continue
        else:
            tok, _ = old_comp.compile_era_clause(era, hanzi, end_punct="")
            if not tok:
                skipped += 1
                continue

        mel = fbank.extract(y, sr=SR).cpu()
        items.append({
            "mel": mel,
            "token_items": tok,
            "valid_mel_bins": 100,
            "era": era,
            "is_single": False,
            "src": f"{ent.get('tag', era)}_{idx}_{hanzi[:12]}",
        })
    print(f"  ✓ {manifest_path.name} ({era}): {len(items)} 条 (跳过 {skipped})")
    return items


def main() -> None:
    out_dir = PROJECT_ROOT / "v10_bundle"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_pt = PROJECT_ROOT / "v5_outputs" / "best_fp32.pt"
    ck = torch.load(str(base_pt), map_location="cpu", weights_only=False)
    model = ZipVoice(**dict(ck["config"]))
    model.load_state_dict(ck["model"], strict=True)

    old_comp = ThreeEraHanfireCompiler(
        ck["t2i_custom"],
        PROJECT_ROOT / "v5_bundle" / "v5_three_era_lexicon.json",
        v8_fixes=True,
    )
    t2i, _fe, mc_comp, meta = build_mc_embedding(model, ck["t2i_custom"], old_comp)
    fbank = PureVocosFbank()

    v5_ds = torch.load(str(PROJECT_ROOT / "v5_bundle" / "v5_train_dataset.pt"), map_location="cpu", weights_only=False)
    old_items = v5_ds if isinstance(v5_ds, list) else v5_ds["items"]

    # 1. OC + existing EM
    oc_em_old = [{**it, "is_single": False} for it in old_items if it["era"] in ("oc", "em") and not is_held_out(it.get("src", ""))]
    print(f"  ✓ v5 既有 OC/EM: {len(oc_em_old)} 条 ({dict(Counter(x['era'] for x in oc_em_old))})")

    # 2. MC: xiaoqieyun (single) + qzw2026 (multi) + zn_mc (multi)
    D_mc = PROJECT_ROOT / "data_mc_raw"
    lab_p = D_mc / "xiaoqieyun_labels_v3.json"
    xqy_items = build_xiaoqieyun_items(mc_comp, fbank, lab_p, D_mc / "xiaoqieyun_all.wav")
    xqy_items = [{**it, "is_single": True} for it in xqy_items]
    qzw_items = [{**it, "is_single": False} for it in build_qzw_items(mc_comp, old_items)]
    zn_items = build_manifest_items(D_mc / "mc_zn_verified_manifest.json", "mc", mc_comp, old_comp, fbank)

    # 3. New EM: short poems + 西游记第一回
    D_em = PROJECT_ROOT / "data_em_raw"
    em_short = build_manifest_items(D_em / "em_short_manifest.json", "em", mc_comp, old_comp, fbank)
    em_xyj = build_manifest_items(D_em / "em_xyj_manifest.json", "em", mc_comp, old_comp, fbank)

    all_items = oc_em_old + xqy_items + qzw_items + zn_items + em_short + em_xyj
    print(f"\n[汇总] 总样本={len(all_items)} | era分布={dict(Counter(x['era'] for x in all_items))} | single={sum(1 for x in all_items if x['is_single'])} | multi={sum(1 for x in all_items if not x['is_single'])}")

    ds_out = out_dir / "v10_train_dataset.pt"
    torch.save({"items": all_items, "t2i_custom": t2i, "mc_meta": meta}, str(ds_out))
    shutil.copy2(base_pt, out_dir / "base_v5_fp32.pt")
    shutil.copy2(PROJECT_ROOT / "v5_bundle" / "v5_three_era_lexicon.json", out_dir / "v5_three_era_lexicon.json")
    print(f"✅ 已写入 {ds_out} ({ds_out.stat().st_size / 1e6:.1f} MB) 与 base_v5_fp32.pt")


if __name__ == "__main__":
    main()
