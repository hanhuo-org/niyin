#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L1/L2 外科手术级单字/单韵校准工具 (calibrate_pronunciation.py) —— 源自 Swarm 架构设计：
  - 100% 冻结 text_encoder、fm_decoder 以及非目标词元（零底噪劣化、零多字连读退化）；
  - 支持两种无损精修方式：
    1) [L1 凸组合流形插值, 0.5秒]: --blend "Q_麻合二_舒_平陰=ua1:0.75"
    2) [L3 同韵部硬共享锁定, 0.5秒]: --lock-rhyme "Q_仙三A_入_入陽:Q_仙合三_入_入陽,Q_仙合三_入_入陰"
    3) [L2 参考音频 30 步掩码精修, ~5秒]: --target-tok "Q_麻合二_舒_平陰" --clause "花" --wav ref.wav
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "vendor" / "ZipVoice"), str(HERE)]

from colab_train_v5_qat import (  # noqa: E402
    FEAT_SCALE, PureVocosFbank, SR, ThreeEraHanfireCompiler, ZipVoice,
    _weights_to_exact_frames, condition_time_mask, forward_velocity, make_pad_mask,
)
from hanfire_core.mc_qieyun import QieyunMCCompiler  # noqa: E402
from render_v10_studio import safe_normalize  # noqa: E402


def apply_blend(W: torch.Tensor, t2i: dict, spec: str) -> None:
    # 格式: target_tok=donor_tok:alpha
    lhs, rhs = spec.split("=", 1)
    donor, alpha_s = rhs.split(":", 1)
    alpha = float(alpha_s)
    t_id, d_id = t2i[lhs.strip()], t2i[donor.strip()]
    with torch.no_grad():
        orig_norm = W[d_id].norm()
        v = (1.0 - alpha) * W[t_id] + alpha * W[d_id]
        W[t_id] = v * (orig_norm / (v.norm() + 1e-8))
    print(f"  ✓ [L1 Blend] {lhs} <- (1-{alpha:.2f})*{lhs} + {alpha:.2f}*{donor}")


def apply_rhyme_lock(W: torch.Tensor, t2i: dict, spec: str) -> None:
    # 格式: anchor_tok:tok1,tok2,...
    anchor, targets_s = spec.split(":", 1)
    a_id = t2i[anchor.strip()]
    with torch.no_grad():
        for t_name in targets_s.split(","):
            t_name = t_name.strip()
            if t_name in t2i:
                W[t2i[t_name]] = W[a_id].clone()
                print(f"  ✓ [L3 RhymeLock] {t_name} == {anchor}")


def calibrate_from_wav(
    model: ZipVoice,
    t2i: dict,
    target_toks: List[str],
    clause: str,
    wav_path: Path,
    era: str = "mc",
    rhyme_pull: str = "",
    steps: int = 35,
    lr: float = 8e-3,
    l2_anchor: float = 0.15,
    rhyme_pull_w: float = 0.08,
) -> None:
    dev = next(model.parameters()).device
    for p in model.parameters():
        p.requires_grad = False

    if era == "mc":
        comp = QieyunMCCompiler(t2i)
        tok_items, _ = comp.compile_clause(clause, metrical=False, rhyme_hold_w=0)
    else:
        old_comp = ThreeEraHanfireCompiler(t2i, PROJECT_ROOT / "v5_bundle" / "v5_three_era_lexicon.json", v8_fixes=True)
        tok_items, _ = old_comp.compile_era_clause(era, clause, end_punct="", rhyme_hold_w=0)
    w_wav, sr = sf.read(str(wav_path), dtype="float32")
    if w_wav.ndim > 1:
        w_wav = w_wav.mean(axis=1)
    # 无损峰值保护归一化（杜绝 np.clip 硬削波）
    w_wav = safe_normalize(w_wav, target_rms=0.065, max_peak=0.78)
    mel = PureVocosFbank().extract(w_wav, sr=SR).unsqueeze(0).to(dev) * FEAT_SCALE
    nf = mel.size(1)
    fl = torch.tensor([nf], dtype=torch.long, device=dev)
    pad = make_pad_mask(fl, max_len=nf)

    tok_ids = [[t for t, _ in tok_items]]
    durs = [_weights_to_exact_frames([w for _, w in tok_items], nf)]

    # 仅允许 target_toks 对应的行接收梯度，weight_decay=0.0 绝对保护非目标词元
    target_ids = [t2i[k] for k in target_toks if k in t2i]
    grad_mask = torch.zeros_like(model.embed.weight.data)
    for tid in target_ids:
        grad_mask[tid] = 1.0
    w_init = model.embed.weight.detach().clone()
    pull_vec = w_init[t2i[rhyme_pull]].clone() if (rhyme_pull and rhyme_pull in t2i) else None
    model.embed.weight.requires_grad = True
    opt = torch.optim.AdamW([model.embed.weight], lr=lr, weight_decay=0.0)

    model.eval()
    for st in range(1, steps + 1):
        scm = condition_time_mask(features_lens=fl, mask_percent=(0.85, 1.0), max_len=nf)
        sc = torch.where(scm.unsqueeze(-1), 0.0, mel)
        noise = torch.randn_like(mel)
        t = torch.rand(1, 1, 1, device=dev)
        xt = mel * t + noise * (1.0 - t)
        ut = mel - noise
        vt = forward_velocity(model, tok_ids, durs, xt, t, sc, pad)
        m = (scm & ~pad).unsqueeze(-1).expand_as(vt)
        fm_loss = torch.mean((vt[m] - ut[m]) ** 2)
        reg = l2_anchor * torch.mean(((model.embed.weight - w_init)[target_ids]) ** 2)
        if pull_vec is not None:
            reg = reg + rhyme_pull_w * torch.mean((model.embed.weight[target_ids] - pull_vec.unsqueeze(0)) ** 2)
        loss = fm_loss + reg
        opt.zero_grad()
        loss.backward()
        model.embed.weight.grad.mul_(grad_mask)
        opt.step()
        if st in (1, steps // 2, steps):
            print(f"  [L2 Step {st:02d}/{steps}] fm_loss={fm_loss.item():.4f} reg={reg.item():.5f}")
    model.embed.weight.requires_grad = False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=PROJECT_ROOT / "v10_outputs" / "best_fp32_v10_studio.pt")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--era", choices=["mc", "oc", "em"], default="mc")
    ap.add_argument("--blend", action="append", default=[], help="如 Q_麻合二_舒_平陰=ua1:0.75")
    ap.add_argument("--lock-rhyme", action="append", default=[], help="如 Q_仙三A_入_入陽:Q_仙合三_入_入陽,Q_仙合三_入_入陰")
    ap.add_argument("--target-tok", action="append", default=[])
    ap.add_argument("--clause", type=str, default="")
    ap.add_argument("--wav", type=Path, default=None)
    ap.add_argument("--rhyme-pull", type=str, default="", help="可选 L3 同韵牵引锚点词元名（如 Q_陽開三_舒_平陰）")
    ap.add_argument("--steps", type=int, default=35)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)
    t2i = ck["t2i_custom"]
    model = ZipVoice(**ck["config"]).to(dev)
    model.load_state_dict(ck["model"], strict=True)

    for b_spec in args.blend:
        apply_blend(model.embed.weight.data, t2i, b_spec)
    for l_spec in args.lock_rhyme:
        apply_rhyme_lock(model.embed.weight.data, t2i, l_spec)
    if args.target_tok and args.clause and args.wav:
        calibrate_from_wav(
            model, t2i, args.target_tok, args.clause, args.wav,
            era=args.era, rhyme_pull=args.rhyme_pull, steps=args.steps,
        )

    out_p = args.out or args.ckpt
    ck["model"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(ck, str(out_p))
    print(f"✅ 已保存校准后权重: {out_p}")


if __name__ == "__main__":
    main()
