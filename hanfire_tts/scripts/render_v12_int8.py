#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用磁盘真 INT8 模型 (best_int8_v10_studio.pt) 渲染 9 首回归基准诗到 v10_outputs/audio_v12_qat_int8，
并执行 audit_regression 六大门禁（无需重新跑 QAT；编译器/词表改动后用它验证 INT8 产出）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy import signal

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "vendor" / "ZipVoice"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from zipvoice.models.zipvoice import ZipVoice  # noqa: E402

from audit_regression import check_wav  # noqa: E402
from colab_train_v5_qat import (  # noqa: E402
    FEAT_SCALE, PureVocosDecoder, PureVocosFbank, SR, ThreeEraHanfireCompiler,
    tokenize_mandarin_prompt_exact, trim_edge_silence,
)
from hanfire_core.mc_qieyun import QieyunMCCompiler  # noqa: E402
from hanfire_core.qat_int8 import load_int8_checkpoint  # noqa: E402
from hanfire_core.zipvoice_zh_manifold import sample_clean_zero_carrier  # noqa: E402
from render_v10_studio import safe_normalize  # noqa: E402

JYS = ["床前明月光", "疑是地上霜", "舉頭望明月", "低頭思故鄉"]
CX = ["春眠不覺曉", "[處=昌開三魚去]處聞啼鳥", "夜來風雨聲", "花落知多少"]
JX = ["千山鳥飛絕", "萬徑人蹤滅", "孤舟蓑笠翁", "獨釣寒江雪"]
TH = ["桃花塢裏桃花庵", "桃花庵下桃花仙", "桃花仙人[種=tʂuŋh]桃樹", "又摘桃花換酒錢"]
JJ = ["蒹葭蒼蒼", "白露為霜", "所謂伊人", "在水一方"]
TASKS = [
    ("MC_v12_纯净自然_静夜思_male.wav", "mc", "male", 24.5, JYS, 0.20, False),
    ("MC_v12_纯净自然_静夜思_female.wav", "mc", "female", 26.5, JYS, 0.20, False),
    ("MC_v12_纯净自然_春晓_male.wav", "mc", "male", 24.5, CX, None, False),
    ("MC_v12_纯净自然_春晓_female.wav", "mc", "female", 26.5, CX, 0.35, False),
    ("MC_v12_纯净自然_江雪_male.wav", "mc", "male", 24.5, JX, 0.30, True),
    ("MC_v12_纯净自然_江雪_female.wav", "mc", "female", 26.5, JX, 0.25, True),
    ("EM_v12_纯净自然_桃花庵歌_male.wav", "em", "male", 24.0, TH, 0.45, False),
    ("EM_v12_纯净自然_桃花庵歌_female.wav", "em", "female", 25.0, TH, 0.45, False),
    ("OC_v12_纯净自然_蒹葭_male.wav", "oc", "male", 25.0, JJ, None, False),
]


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, t2i = load_int8_checkpoint(PROJECT_ROOT / "v10_outputs" / "best_int8_v10_studio.pt", ZipVoice, device=dev)
    vocos, fbank = PureVocosDecoder(dev), PureVocosFbank()
    mc = QieyunMCCompiler(t2i)
    old = ThreeEraHanfireCompiler(t2i, PROJECT_ROOT / "v5_bundle" / "v5_three_era_lexicon.json", v8_fixes=True)
    voices = {}
    for vk, wavf, txt in (("male", "ref_male_128Hz.wav", "高分经典悬疑电影，"), ("female", "ref_female_252Hz.wav", "今夜的月光如此清亮，")):
        pw, _ = sf.read(str(PROJECT_ROOT / "v5_bundle" / wavf), dtype="float32")
        pit, _ = tokenize_mandarin_prompt_exact(txt, t2i)
        pf = fbank.extract((pw * 0.45).astype(np.float32), sr=SR).unsqueeze(0).to(dev) * FEAT_SCALE
        voices[vk] = (pit, pf, torch.tensor([pf.size(1)], device=dev))
    sos_bp = signal.butter(4, [75.0 / (SR / 2), 9500.0 / (SR / 2)], btype="band", output="sos")
    out_dir = PROJECT_ROOT / "v10_outputs" / "audio_v12_qat_int8"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_errs = 0
    for fname, era, vk, fp100, lines, min_r, chk_xue in TASKS:
        pit, pf, pl = voices[vk]
        chunks = []
        for j, cl in enumerate(lines):
            if era == "mc":
                it, _ = mc.compile_clause(cl, metrical=True, rhyme_hold_w=16)
            else:
                it, _ = old.compile_era_clause(era, cl, end_punct="", rhyme_hold_w=16)
            mel = sample_clean_zero_carrier(model=model, prompt_items=pit, target_items=it, t2i=t2i,
                                            prompt_features=pf, prompt_features_lens=pl, frames_per_100w=fp100,
                                            char0_boost_w=44, num_step=16, guidance_scale=1.0, t_shift=0.5, seed=42)
            w = vocos.decode_mel(mel.permute(0, 2, 1) / FEAT_SCALE)
            w = signal.sosfiltfilt(sos_bp, w).astype(np.float32)
            w = trim_edge_silence(w, rel_db=52.0, pre_ms=35, post_ms=65)
            w = safe_normalize(w, target_rms=0.055, max_peak=0.72)
            f = int(0.025 * SR)
            if len(w) > f * 2:
                w[:f] *= 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, f, dtype=np.float32)))
                w[-f:] *= 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, f, dtype=np.float32)))
            chunks += [w, np.zeros(int((0.26 if j % 2 == 0 else 0.42) * SR), dtype=np.float32)]
        out_p = out_dir / fname
        sf.write(str(out_p), np.concatenate(chunks), SR)
        total_errs += len(check_wav(out_p, 4, (0, 1, 3), min_r, chk_xue))
    print(f"\n=== INT8 回归门禁结果: {'全部通过 ✅' if total_errs == 0 else f'失败 {total_errs} 项 ❌'} ===")
    sys.exit(0 if total_errs == 0 else 1)


if __name__ == "__main__":
    main()
