#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V8 发音修复样带渲染 + 声学回归测试（仅 FP32，不做 QAT）
(render_showcase_fp32.py)

用法:
  HF_HUB_OFFLINE=1 python scripts/render_showcase_fp32.py [--out-dir v5_outputs/audio_v8_fp32] [--regress-only]

回归测试集来自人耳听辨确认过的问题（2026-11-18）：
  岸读两次 / 庵读两次(花庵→安安) / 在河之洲五拍 / 伊人粘连
指标：逐分句浊音核个数 K 与字数 N 之差（男女双声 × 3 个随机种子），对比 v8_fixes 关闭/开启。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy import signal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from colab_train_v5_qat import (  # noqa: E402
    FEAT_SCALE, PROJECT_ROOT, SR, PureVocosDecoder, PureVocosFbank, ThreeEraHanfireCompiler,
    ZipVoice, sample_with_character_durations, synth_era_poem, tokenize_mandarin_prompt_exact,
    trim_edge_silence,
)

SHOWCASE = [
    ("OC_上古_关雎", "oc", [("关关雎鸠", "在河之洲", {}), ("窈窕淑女", "君子好逑", {}),
                         ("求之不得", "寤寐思服", {"服": "bək"}), ("悠哉悠哉", "辗转反侧", {"服": "bək"})]),
    ("OC_上古_蒹葭", "oc", [("蒹葭苍苍", "白露为霜", {}), ("所谓伊人", "在水一方", {}),
                         ("溯洄从之", "道阻且长", {"长": "daŋ"}), ("溯游从之", "宛在水中央", {})]),
    ("MC_中古_春晓", "mc", [("春眠不觉晓", "处处闻啼鸟", {"处": "tɕʰɨəh"}), ("夜来风雨声", "花落知多少", {})]),
    ("MC_中古_春望", "mc", [("国破山河在", "城春草木深", {}), ("感时花溅泪", "恨别鸟惊心", {"溅": "tsenh"})]),
    ("MC_中古_江雪", "mc", [("千山鸟飞绝", "万径人踪灭", {}), ("孤舟蓑笠翁", "独钓寒江雪", {})]),
    ("MC_中古_静夜思", "mc", [("床前明月光", "疑是地上霜", {}), ("举头望明月", "低头思故乡", {})]),
    ("EM_近古明代雅音_桃花庵歌", "em", [("桃花坞里桃花庵", "桃花庵下桃花仙", {}), ("酒醒只在花前坐", "酒醉还来花下眠", {})]),
    ("EM_近古明代雅音_琵琶行", "em", [("浔阳江头夜送客", "枫叶荻花秋瑟瑟", {}), ("千呼万唤始出来", "犹抱琵琶半遮面", {})]),
    ("EM_近古明代雅音_念奴娇", "em", [("乱石穿空惊涛拍岸", "卷起千堆雪", {}), ("江山如画", "一时多少豪杰", {"画": "xwah"})]),
    ("EM_近古元曲北音_天净沙", "em_zyyy", [("枯藤老树昏鸦", "小桥流水人家", {}), ("古道西风瘦马", "断肠人在天涯", {})]),
]
REGRESS = [("em", "乱石穿空惊涛拍岸"), ("em", "桃花坞里桃花庵"), ("em", "桃花庵下桃花仙"),
           ("oc", "在河之洲"), ("oc", "所谓伊人"), ("oc", "蒹葭苍苍")]


def voiced_nuclei(wav: np.ndarray) -> int:
    y = signal.resample_poly(wav.astype(np.float64), 2, 3)
    y = y / (np.max(np.abs(y)) + 1e-8)
    hop, win = 160, 480
    yb = signal.sosfiltfilt(signal.butter(4, [300 / 8000, 3400 / 8000], btype="band", output="sos"), y)
    n = max(1, (len(y) - win) // hop)
    env = np.zeros(n)
    for i in range(n):
        seg = y[i * hop:i * hop + win] - y[i * hop:i * hop + win].mean()
        ac = np.correlate(seg, seg, "full")[win - 1:]
        voi = (np.max(ac[35:230]) / ac[0]) if ac[0] > 1e-6 else 0.0
        env[i] = np.sqrt(np.mean(yb[i * hop:i * hop + win] ** 2)) * np.clip((voi - 0.35) / 0.35, 0, 1)
    if len(env) > 12:
        env = signal.sosfiltfilt(signal.butter(3, 12 / 50, output="sos"), env)
    env = np.maximum(env, 0) / (env.max() + 1e-8)
    return len(signal.find_peaks(env, distance=9, prominence=0.12, height=0.12)[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "v5_outputs" / "audio_v8_fp32")
    ap.add_argument("--regress-only", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(str(PROJECT_ROOT / "v5_outputs/best_fp32.pt"), map_location="cpu", weights_only=False)
    t2i = ck["t2i_custom"]
    model = ZipVoice(**ck["config"])
    model.load_state_dict(ck["model"], strict=True)
    model = model.to(dev).eval()
    vocos, fbank = PureVocosDecoder(dev), PureVocosFbank()
    lex = PROJECT_ROOT / "v5_bundle/v5_three_era_lexicon.json"
    comp_new = ThreeEraHanfireCompiler(t2i=t2i, three_era_json=lex, v8_fixes=True)
    comp_old = ThreeEraHanfireCompiler(t2i=t2i, three_era_json=lex, v8_fixes=False)

    voices = {
        "male_128Hz": (sf.read(str(PROJECT_ROOT / "v5_bundle/ref_male_128Hz.wav"), dtype="float32")[0],
                       tokenize_mandarin_prompt_exact("高分经典悬疑电影。", t2i)[0], 25.0),
        "female_252Hz": (sf.read(str(PROJECT_ROOT / "v5_bundle/ref_female_252Hz.wav"), dtype="float32")[0],
                         tokenize_mandarin_prompt_exact("今夜的月光如此清亮。", t2i)[0], 24.5),
    }

    # ---- 1. 声学回归测试 ----
    print("[1] 声学回归测试 (逐分句浊音核 K vs 字数 N, 男女 × 3 种子)")
    report = {"regression": {}}
    for tag, comp in (("v7_old_compile", comp_old), ("v8_fixed_compile", comp_new)):
        tot, rows = 0, []
        for era, text in REGRESS:
            ks = []
            for vname, (pw, pit, fp) in voices.items():
                pf = fbank.extract(pw, sr=SR).unsqueeze(0).to(dev) * FEAT_SCALE
                pl = torch.tensor([pf.size(1)], device=dev)
                it, dbg = comp.compile_era_clause(era, text, end_punct="")
                for seed in (42, 7, 123):
                    fp_era = comp.ERA_FRAMES_PER_100W[era] * (fp / 25.0)
                    mel = sample_with_character_durations(
                        model=model, prompt_items=pit, target_items=[(t2i["."], 120)] + it,
                        prompt_features=pf, prompt_features_lens=pl, frames_per_100w=fp_era,
                        num_step=16, guidance_scale=1.25, t_shift=0.5, seed=seed)
                    w = trim_edge_silence(vocos.decode_mel(mel.permute(0, 2, 1) / FEAT_SCALE))
                    if seed == 42:
                        sf.write(str(args.out_dir / f"regress_{tag}_{text}_{vname}.wav"), w, SR)
                    ks.append(voiced_nuclei(w))
            err = int(sum(abs(k - len(text)) for k in ks))
            tot += err
            rows.append({"text": text, "era": era, "K": ks, "N": len(text), "sum_abs_err": err, "tokens": dbg})
            print(f"  {tag:17s} {text:10s} K={ks} N={len(text)} Σ|K-N|={err} | {dbg}")
        report["regression"][tag] = {"total_abs_err": tot, "rows": rows}
        print(f"  ==> {tag}: Σ|K-N| = {tot}")

    # ---- 2. 全量样带 ----
    if not args.regress_only:
        print("\n[2] 渲染 V8 FP32 三代样带 (逐分句 + 首停顿预算 + 首尾静音切除)")
        for title, era, couplets in SHOWCASE:
            for vname, (pw, pit, fp) in voices.items():
                fp_era = comp_new.ERA_FRAMES_PER_100W[era] * (fp / 25.0)
                w, _ = synth_era_poem(model, vocos, fbank, comp_new, era, pw, pit, couplets, fp_era)
                sf.write(str(args.out_dir / f"{title}_{vname}_FP32_v8.wav"), w, SR)
                if era == "mc":  # 阴阳调 A/B（阳平→半上近似）
                    comp_new.MC_YANG_REGISTER = True
                    w2, _ = synth_era_poem(model, vocos, fbank, comp_new, era, pw, pit, couplets, fp_era)
                    comp_new.MC_YANG_REGISTER = False
                    sf.write(str(args.out_dir / f"{title}_{vname}_FP32_v8_阳平低调实验.wav"), w2, SR)
                print(f"  ✓ {title}_{vname} ({len(w)/SR:.2f}s)")
    (args.out_dir / "v8_regression_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 完成: {args.out_dir}")


if __name__ == "__main__":
    main()
