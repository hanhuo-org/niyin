#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V12 Studio 推理与声学验收脚本 (eval_v10.py)
已对齐 AGENTS.md 全部工程红线与 Swarm 审查修复项：
  - 废除 ffmpeg atempo 变速与错误参考文本，改用原生 24kHz 未拉伸棚录干声 (ref_male_128Hz.wav * 0.45) + 精确参考文本；
  - 改用 sample_clean_zero_carrier（确定性 Spherical + Sway(-0.8)，guidance_scale=1.0，零前导杂音）；
  - 废除 np.clip 硬削波，改用 [75Hz, 9500Hz] 零相位带通 + 无损峰值保护归一化；
  - 自动输出分句时长与 audit_rhyme.py 押韵对比度。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
import torch
from scipy import signal

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "vendor" / "ZipVoice"), str(HERE)]

from audit_rhyme import audit_poem, split_clauses  # noqa: E402
from colab_train_v5_qat import (  # noqa: E402
    FEAT_SCALE, PureVocosDecoder, PureVocosFbank, SR, ThreeEraHanfireCompiler,
    ZipVoice, tokenize_mandarin_prompt_exact, trim_edge_silence,
)
from hanfire_core.mc_qieyun import QieyunMCCompiler  # noqa: E402
from hanfire_core.zipvoice_zh_manifold import sample_clean_zero_carrier  # noqa: E402
from render_v10_studio import safe_normalize  # noqa: E402


def synthesize_poem(
    model: ZipVoice,
    vocos: PureVocosDecoder,
    fbank: PureVocosFbank,
    t2i: dict,
    mc_comp: QieyunMCCompiler,
    old_comp: ThreeEraHanfireCompiler,
    prompt_wav: np.ndarray,
    prompt_text: str,
    era: str,
    clauses: List[str],
    out_wav: Path,
    dev: torch.device,
    fp100: float = 24.5,
) -> None:
    pit, _ = tokenize_mandarin_prompt_exact(prompt_text, t2i)
    pf = fbank.extract((prompt_wav * 0.45).astype(np.float32), sr=SR).unsqueeze(0).to(dev) * FEAT_SCALE
    pl = torch.tensor([pf.size(1)], device=dev, dtype=torch.long)
    sos_bp = signal.butter(4, [75.0 / (SR / 2), 9500.0 / (SR / 2)], btype="band", output="sos")

    pieces: List[np.ndarray] = []
    durs_s: List[float] = []
    f_fade = int(0.025 * SR)

    for ci, clause in enumerate(clauses):
        if era == "mc":
            c_tok, _ = mc_comp.compile_clause(clause, metrical=True, rhyme_hold_w=16)
        else:
            c_tok, _ = old_comp.compile_era_clause(era, clause, end_punct="", rhyme_hold_w=16)

        pred_mel = sample_clean_zero_carrier(
            model=model,
            prompt_items=pit,
            target_items=c_tok,
            t2i=t2i,
            prompt_features=pf,
            prompt_features_lens=pl,
            frames_per_100w=fp100,
            char0_boost_w=44,
            num_step=16,
            guidance_scale=1.0,
            t_shift=0.5,
            seed=42,
        )
        wav = vocos.decode_mel(pred_mel.permute(0, 2, 1) / FEAT_SCALE)
        wav = signal.sosfiltfilt(sos_bp, wav).astype(np.float32)
        wav = trim_edge_silence(wav, rel_db=52.0, pre_ms=35, post_ms=65)
        wav = safe_normalize(wav, target_rms=0.055, max_peak=0.72)
        if len(wav) > f_fade * 2:
            wav[:f_fade] *= 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, f_fade, dtype=np.float32)))
            wav[-f_fade:] *= 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, f_fade, dtype=np.float32)))
        durs_s.append(round(len(wav) / SR, 2))
        pieces.append(wav)
        if ci < len(clauses) - 1:
            pieces.append(np.zeros(int((0.26 if ci % 2 == 0 else 0.42) * SR), dtype=np.float32))

    full = np.concatenate(pieces)
    sf.write(str(out_wav), full, SR)
    sub = split_clauses(full, SR, min_sil_ms=160)
    print(f"  ✓ {out_wav.name}: 分句时长={durs_s}s, 切出分句={len(sub)}")


def main() -> None:
    ckpt = PROJECT_ROOT / "v10_outputs" / "best_fp32_v10_studio.pt"
    out_dir = PROJECT_ROOT / "v10_outputs" / "audio_v10"
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    model = ZipVoice(**dict(ck["config"])).to(dev)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    t2i = ck["t2i_custom"]

    lex_path = PROJECT_ROOT / "v5_bundle" / "v5_three_era_lexicon.json"
    old_comp = ThreeEraHanfireCompiler(t2i, lex_path, v8_fixes=True)
    mc_comp = QieyunMCCompiler(t2i)
    fbank = PureVocosFbank()
    vocos = PureVocosDecoder(dev)

    prompt_wav, _ = sf.read(str(PROJECT_ROOT / "v5_bundle" / "ref_male_128Hz.wav"), dtype="float32")
    prompt_text = "高分经典悬疑电影，"

    tests = [
        ("MC_v10_静夜思.wav", "mc", 24.5, ["床前明月光", "疑是地上霜", "舉頭望明月", "低頭思故鄉"], (0, 1, 3), ["光", "霜", "月", "鄉"]),
        ("MC_v10_江雪.wav", "mc", 24.5, ["千山鳥飛絕", "萬徑人蹤滅", "孤舟蓑笠翁", "獨釣寒江雪"], (0, 1, 3), ["絕", "滅", "翁", "雪"]),
        ("MC_v10_春晓.wav", "mc", 24.5, ["春眠不覺曉", "[處=昌開三魚去]處聞啼鳥", "夜來風雨聲", "花落知多少"], (0, 1, 3), ["曉", "鳥", "聲", "少"]),
        ("EM_v10_桃花庵歌.wav", "em", 24.0, ["桃花塢裏桃花庵", "桃花庵下桃花仙", "桃花仙人[種=tʂuŋh]桃樹", "又摘桃花換酒錢"], (0, 1, 3), ["庵", "仙", "樹", "錢"]),
        ("OC_v10_蒹葭.wav", "oc", 25.0, ["蒹葭蒼蒼", "白露為霜", "所謂伊人", "在水一方"], (0, 1, 3), ["蒼", "霜", "人", "方"]),
    ]

    for fname, era, fp100, clauses, r_idx, r_labs in tests:
        out_p = out_dir / fname
        synthesize_poem(model, vocos, fbank, t2i, mc_comp, old_comp, prompt_wav, prompt_text, era, clauses, out_p, dev, fp100=fp100)
        audit_poem(out_p, r_idx, r_labs)


if __name__ == "__main__":
    main()
