#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V9 中古音（音韵地位词元 + LoRA）测试集渲染：江雪 / 静夜思 / 春晓（均未参与训练）。

改进点（消除音色漂移 + 语速/字间隔不均）：
  - 所有分句锁定同一个随机种子 (seed=42)，消除每句重抽带来的音色跳变；
  - 分句级 RMS 响度对齐 (-20 dBFS)，消除句间忽大忽小；
  - 默认语速放慢至 0.45s/字 (FP_MC=42.0)，另输出慢速逐字版 (FP_MC=58.0, 0.62s/字，接近原版视频 0.78s/字) 供对比。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from colab_train_v5_qat import (  # noqa: E402
    FEAT_SCALE, PROJECT_ROOT, SR, PureVocosDecoder, PureVocosFbank, ZipVoice,
    sample_with_character_durations, tokenize_mandarin_prompt_exact, trim_edge_silence,
)
from hanfire_core.mc_qieyun import QieyunMCCompiler  # noqa: E402

POEMS = {
    "江雪": [("千山鳥飛絕", "萬徑人蹤滅"), ("孤舟蓑笠翁", "獨釣寒江雪")],
    "静夜思": [("床前明月光", "疑是地上霜"), ("舉頭望明月", "低頭思故鄉")],
    "春晓": [("春眠不覺曉", "處處聞啼鳥"), ("夜來風雨聲", "花落知多少")],
}
OVERRIDES = {"處": "昌開三魚去"}


def normalize_rms(w: np.ndarray, target_rms: float = 0.10) -> np.ndarray:
    rms = float(np.sqrt(np.mean(w ** 2) + 1e-9))
    if rms > 1e-5:
        w = w * (target_rms / rms)
    return np.clip(w, -0.95, 0.95).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=PROJECT_ROOT / "v9_outputs/best_fp32_v9mc_lora.pt")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "v9_outputs/audio_mc_lora_test")
    ap.add_argument("--fp100", type=float, default=42.0, help="每100权重帧数：42≈0.45s/字，58≈0.62s/字")
    ap.add_argument("--char-gap-w", type=int, default=12, help="字间固定微停顿权重 (0=连续，12≈50ms 均匀字间隔)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)
    t2i = ck["t2i_custom"]
    model = ZipVoice(**ck["config"])
    model.load_state_dict(ck["model"], strict=True)
    model = model.to(dev).eval()
    vocos, fbank = PureVocosDecoder(dev), PureVocosFbank()
    comp = QieyunMCCompiler(t2i)
    voices = {
        "male_128Hz": (sf.read(str(PROJECT_ROOT / "v5_bundle/ref_male_128Hz.wav"), dtype="float32")[0],
                       tokenize_mandarin_prompt_exact("高分经典悬疑电影。", t2i)[0], 1.0),
        "female_252Hz": (sf.read(str(PROJECT_ROOT / "v5_bundle/ref_female_252Hz.wav"), dtype="float32")[0],
                         tokenize_mandarin_prompt_exact("今夜的月光如此清亮。", t2i)[0], 0.98),
    }
    comma_id = t2i[","]
    for title, couplets in POEMS.items():
        for vname, (pw, pit, sc) in voices.items():
            pf = fbank.extract(pw, sr=SR).unsqueeze(0).to(dev) * FEAT_SCALE
            pl = torch.tensor([pf.size(1)], device=dev)
            chunks = []
            for ci, (c1, c2) in enumerate(couplets):
                for j, cl in enumerate((c1, c2)):
                    # 逐字编译并在字间插入均匀微停顿（防止字间隔忽长忽短）
                    it = [(t2i["."], 120)]
                    for idx_c, ch in enumerate(cl):
                        c_it, _, _ = comp.compile_char(ch, OVERRIDES.get(ch))
                        it.extend(c_it)
                        if args.char_gap_w > 0 and idx_c + 1 < len(cl):
                            it.append((comma_id, args.char_gap_w))
                    mel = sample_with_character_durations(
                        model=model, prompt_items=pit, target_items=it,
                        prompt_features=pf, prompt_features_lens=pl, frames_per_100w=args.fp100 * sc,
                        num_step=16, guidance_scale=1.25, t_shift=0.5, seed=42)
                    w = normalize_rms(trim_edge_silence(vocos.decode_mel(mel.permute(0, 2, 1) / FEAT_SCALE)))
                    f = int(0.012 * SR)
                    if len(w) > f * 2:
                        w[:f] *= np.linspace(0, 1, f)
                        w[-f:] *= np.linspace(1, 0, f)
                    chunks += [w, np.zeros(int((0.18 if j == 0 else 0.35) * SR), dtype=np.float32)]
            out = args.out_dir / f"MC_v9lora_{title}_{vname}.wav"
            sf.write(str(out), np.concatenate(chunks), SR)
            print(f"✓ {out.name}")


if __name__ == "__main__":
    main()
