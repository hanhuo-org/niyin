#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V10 三代同堂统一模型 A100 两阶段 FP32 训练脚本 (train_v10_a100.py)
- 不做 QAT（先出高保真 FP32 定版模型）
- Stage A (全数据含单字切韵切片): 仅训练 Embedding（FactorizedMCEmbedding + fe.old），冻结 text_encoder & fm_decoder
- Stage B (仅多字长句 is_single=False，OC+MC+EM 混训): 解冻 text_encoder + fm_decoder LoRA (rank=16) + Embedding
- A100 自动启用 BF16（无 FP16 溢出风险），Turing/2080Ti 自动回退 FP16+GradScaler
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HERE))

from colab_train_v5_qat import (  # noqa: E402
    FEAT_SCALE, ThreeEraDataset, ThreeEraHanfireCompiler, ZipVoice, collate_three_era,
    condition_time_mask, forward_velocity, make_pad_mask,
)
from hanfire_core.mc_qieyun import build_mc_embedding  # noqa: E402
from train_v9_mc import inject_fm_lora, merge_fm_lora  # noqa: E402


def _make_loader(items: List[dict], batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        ThreeEraDataset(items),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_three_era,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", type=Path, default=PROJECT_ROOT / "v10_bundle")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "v10_outputs")
    ap.add_argument("--stage-a-epochs", type=int, default=15)
    ap.add_argument("--stage-b-epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--emb-lr", type=float, default=1.0e-3)
    ap.add_argument("--lora-lr", type=float, default=1.0e-4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--rhyme-loss-weight", type=float, default=0.02, help="L3 同韵部/韵尾因子锚定权重")
    ap.add_argument(
        "--allow-backbone-unfreeze",
        action="store_true",
        help="显式确认训练集为 100% 棚录干声时才允许执行 Stage B（解冻 text_encoder + fm_decoder LoRA）；默认遵守 AGENTS.md 红线仅训 Embedding",
    )
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = dev.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_scaler = dev.type == "cuda" and not use_bf16

    ds_path = args.bundle_dir / "v10_train_dataset.pt"
    base_ckpt_path = args.bundle_dir / "base_v5_fp32.pt"
    lex_path = args.bundle_dir / "v5_three_era_lexicon.json"
    if not lex_path.exists():
        lex_path = PROJECT_ROOT / "v5_bundle" / "v5_three_era_lexicon.json"
    print(f"[1] 加载 V10 数据包: {ds_path} | device={dev} dtype={amp_dtype}")
    ds = torch.load(str(ds_path), map_location="cpu", weights_only=False)
    items: List[dict] = ds["items"]
    t2i: Dict[str, int] = ds["t2i_custom"]
    meta: dict = ds["mc_meta"]

    ck = torch.load(str(base_ckpt_path), map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    model = ZipVoice(**cfg)
    model.load_state_dict(ck["model"], strict=True)

    # 重建 FactorizedMCEmbedding 并从 base_v5_fp32 初始化 MC 行
    old_comp = ThreeEraHanfireCompiler(ck["t2i_custom"], lex_path, v8_fixes=True)
    _t2i_chk, fe, _comp, _meta = build_mc_embedding(model, ck["t2i_custom"], old_comp)
    model.embed = fe
    model = model.to(dev)
    print(f"  vocab={len(t2i)} (old={meta['n_old']}, mc={len(t2i) - meta['n_old']})")

    # 划分 Stage A (全量) 与 Stage B (仅多字长句 is_single=False)
    random.seed(42)
    single_items = [it for it in items if it.get("is_single", False)]
    multi_items = [it for it in items if not it.get("is_single", False)]
    random.shuffle(multi_items)
    n_val = max(24, len(multi_items) // 20)
    val_multi = multi_items[:n_val]
    train_multi = multi_items[n_val:]
    train_all = train_multi + single_items
    random.shuffle(train_all)

    from collections import Counter
    print(f"  Stage A train={len(train_all)} (single={len(single_items)}, multi={len(train_multi)})")
    print(f"  Stage B train={len(train_multi)} | val_multi={len(val_multi)} | eras={dict(Counter(it['era'] for it in train_multi))}")

    nw = 0 if args.smoke else args.num_workers
    dl_a = _make_loader(train_all, args.batch_size, True, nw)
    dl_b = _make_loader(train_multi, args.batch_size, True, nw)
    dl_v = _make_loader(val_multi, args.batch_size, False, nw)

    # fe.old 为 register_buffer（严格冻结原生普通话区），仅收集 FactorizedMCEmbedding 全部可学习因子参数
    # （ini, seg, tone 及 L3 韵部/介音/韵尾锁定的 rhyme_core, medial, coda）
    emb_params = [p for p in fe.parameters()]
    emb_init = [p.detach().clone() for p in emb_params]
    norm_ref = fe.old.detach().norm(dim=-1).mean().item()

    def run_epoch(loader, opt, sched, scaler, trainable: List[nn.Parameter], train_mode: bool, max_steps: int = 0) -> float:
        losses = []
        for step_i, batch in enumerate(loader):
            if max_steps and step_i >= max_steps:
                break
            feats = batch["mels"].to(dev, non_blocking=True) * FEAT_SCALE
            fl = batch["mel_lens"].to(dev, non_blocking=True)
            b_mask = batch["bin_masks"].to(dev, non_blocking=True)
            nf = int(fl.max())
            pad = make_pad_mask(fl, max_len=nf)
            scm = condition_time_mask(features_lens=fl, mask_percent=(0.7, 1.0), max_len=nf)
            sc = torch.where(scm.unsqueeze(-1), 0.0, feats)
            noise = torch.randn_like(feats)
            t = torch.rand(feats.size(0), 1, 1, device=dev)
            xt = feats * t + noise * (1 - t)
            ut = feats - noise
            with torch.set_grad_enabled(train_mode):
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(dev.type == "cuda")):
                    vt = forward_velocity(model, batch["tokens"], batch["durs"], xt, t, sc, pad)
                    m = (scm & ~pad).unsqueeze(-1) & b_mask
                    loss = torch.mean((vt[m].float() - ut[m].float()) ** 2)
                if train_mode:
                    reg = sum(args.rhyme_loss_weight * torch.mean((p - p0) ** 2) for p, p0 in zip(emb_params, emb_init))
                    # L3 韵类残差收窄：鼓励同韵部/同韵尾信息由共享 rhyme_core/coda 承载，避免低频 seg 过拟合漂移
                    if getattr(fe, "has_rhyme_lock", False):
                        reg = reg + (args.rhyme_loss_weight * 0.5) * torch.mean(fe.seg ** 2)
                    total = loss + reg
                    opt.zero_grad(set_to_none=True)
                    if use_scaler:
                        scaler.scale(total).backward()
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                        scaler.step(opt)
                        scaler.update()
                    else:
                        total.backward()
                        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                        opt.step()
                    sched.step()
            losses.append(float(loss.item()))
        return float(np.mean(losses))

    # ==================== Stage A: Embedding Only ====================
    for p in model.parameters():
        p.requires_grad = False
    for p in emb_params:
        p.requires_grad = True
    model.eval()

    ep_a = 2 if args.smoke else args.stage_a_epochs
    max_steps = 8 if args.smoke else 0
    opt_a = torch.optim.AdamW([{"params": emb_params, "lr": args.emb_lr, "weight_decay": 0.0}])
    sched_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=max(1, ep_a * len(dl_a)), eta_min=args.emb_lr * 0.1)
    scaler_a = torch.amp.GradScaler("cuda", enabled=use_scaler)

    best_a, best_a_fe = 1e9, None
    print(f"\n[Stage A] 仅训练 Embedding ({ep_a} epochs, 含单字切片)...")
    for ep in range(1, ep_a + 1):
        t0 = time.time()
        torch.manual_seed(100 + ep)
        tr = run_epoch(dl_a, opt_a, sched_a, scaler_a, emb_params, True, max_steps=max_steps)
        torch.manual_seed(1234)
        va = run_epoch(dl_v, None, None, None, [], False, max_steps=max_steps)
        if va < best_a:
            best_a = va
            best_a_fe = {k: v.detach().cpu().clone() for k, v in fe.state_dict().items()}
        print(f"  [A] ep {ep:02d}/{ep_a} train={tr:.4f} val={va:.4f} best={best_a:.4f} | norm={fe.weight[meta['n_old']:].norm(dim=-1).mean():.2f}/{norm_ref:.2f} | {time.time()-t0:.1f}s")

    fe.load_state_dict(best_a_fe)

    # ==================== Stage B: Multi-Char LoRA + TextEncoder + Embedding ====================
    best_b = best_a
    if not args.allow_backbone_unfreeze:
        print("\n[Stage B 已按 AGENTS.md 工程红线跳过] 如确认全部训练语料为无分离棚录干声，请显式传入 --allow-backbone-unfreeze。")
    else:
        ep_b = 2 if args.smoke else args.stage_b_epochs
        te_params = list(model.text_encoder.parameters())
        for p in te_params:
            p.requires_grad = True
        lora_params = inject_fm_lora(model.fm_decoder, rank=args.lora_rank, alpha=args.lora_rank * 2.0)
        trainable_b = emb_params + te_params + lora_params
        opt_b = torch.optim.AdamW([
            {"params": emb_params, "lr": args.emb_lr * 0.4, "weight_decay": 0.0},
            {"params": te_params, "lr": args.lora_lr * 0.4, "weight_decay": 1e-4},
            {"params": lora_params, "lr": args.lora_lr, "weight_decay": 0.0},
        ])
        sched_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=max(1, ep_b * len(dl_b)), eta_min=args.lora_lr * 0.08)
        scaler_b = torch.amp.GradScaler("cuda", enabled=use_scaler)

        print(f"\n[Stage B] 多字长句微调 ({ep_b} epochs | te={sum(p.numel() for p in te_params)/1e6:.2f}M, fm_lora={sum(p.numel() for p in lora_params)/1e6:.2f}M)...")
        best_b, best_b_state = 1e9, None
        for ep in range(1, ep_b + 1):
            t0 = time.time()
            torch.manual_seed(500 + ep)
            tr = run_epoch(dl_b, opt_b, sched_b, scaler_b, trainable_b, True, max_steps=max_steps)
            torch.manual_seed(1234)
            va = run_epoch(dl_v, None, None, None, [], False, max_steps=max_steps)
            if va < best_b:
                best_b = va
                best_b_state = {
                    "fe": {k: v.detach().cpu().clone() for k, v in fe.state_dict().items()},
                    "te": {k: v.detach().cpu().clone() for k, v in model.text_encoder.state_dict().items()},
                    "fm": {k: v.detach().cpu().clone() for k, v in model.fm_decoder.state_dict().items()},
                }
            print(f"  [B] ep {ep:02d}/{ep_b} train={tr:.4f} val={va:.4f} best={best_b:.4f} | {time.time()-t0:.1f}s")

        fe.load_state_dict(best_b_state["fe"])
        model.text_encoder.load_state_dict(best_b_state["te"])
        model.fm_decoder.load_state_dict(best_b_state["fm"])
        n_merged = merge_fm_lora(model.fm_decoder)
        print(f"\n  ✓ 已无损合并 {n_merged} 个 LoRA 线性层回标准 nn.Linear")

    model.embed = fe.materialize().to(dev)
    cfg["vocab_size"] = model.embed.weight.size(0)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    out = args.out_dir / "best_fp32_v10.pt"
    torch.save({
        "model": sd,
        "config": cfg,
        "t2i_custom": t2i,
        "mc_meta": meta,
        "best_val_a": best_a,
        "best_val_b": best_b,
    }, str(out))
    print(f"✅ 保存 V10 FP32 模型: {out} (vocab={cfg['vocab_size']}, val_A={best_a:.4f}, val_B={best_b:.4f})")


if __name__ == "__main__":
    main()
