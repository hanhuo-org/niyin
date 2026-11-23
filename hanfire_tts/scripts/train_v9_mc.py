#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V9-B 中古音「音韵地位」词元训练（支持 --lora + --fp16 混合精度，仅 FP32 权重输出，不做 QAT）
(train_v9_mc.py)

数据：
  1) 《一口气听完中古切韵音全常用音节》逐音节片段（OCR+指针锚点对齐）
  2) 千字文 2026 重制版 250 句（按统一词表主读音重新编码）
  3) 多说话人正则化（--mix-old-eras）：混入 v5_train_dataset.pt 中的上古/近古多音色切片，
     防止解冻 text_encoder + fm_decoder LoRA 后音色坍缩到单一朗读者。
  测试集（不参与训练）：江雪 / 静夜思 / 春晓
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from colab_train_v5_qat import (  # noqa: E402
    FEAT_SCALE, PROJECT_ROOT, SR, PureVocosFbank, ThreeEraDataset, ThreeEraHanfireCompiler, ZipVoice,
    _weights_to_exact_frames, collate_three_era, condition_time_mask, forward_velocity, make_pad_mask,
)
from hanfire_core.mc_qieyun import build_mc_embedding  # noqa: E402

TONE_FROM_MARK = {"\u0300": "去", "\u0301": "上"}


class LoRALinear(nn.Module):
    """纯 PyTorch 轻量 LoRA 线性层，支持训练后无损合并回标准 nn.Linear。"""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.0) -> None:
        super().__init__()
        self.base = base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False
        self.scaling = alpha / float(rank)
        dev, dt = base.weight.device, base.weight.dtype
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=dev, dtype=dt))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = F.linear(F.linear(x, self.lora_a), self.lora_b)
        return out + delta * self.scaling

    @torch.no_grad()
    def merge_to_linear(self) -> nn.Linear:
        delta = (self.lora_b.float() @ self.lora_a.float()) * self.scaling
        self.base.weight.data.add_(delta.to(self.base.weight.dtype))
        return self.base


def inject_fm_lora(fm_decoder: nn.Module, rank: int = 16, alpha: float = 32.0) -> List[nn.Parameter]:
    """为 fm_decoder 的注意力/前馈线性层注入 LoRA，返回可训练 LoRA 参数列表。"""
    lora_params: List[nn.Parameter] = []
    targets: List[tuple[nn.Module, str, nn.Linear]] = []
    for mod_name, mod in fm_decoder.named_modules():
        if "time_emb" in mod_name:
            continue
        for attr, child in mod.named_children():
            if isinstance(child, nn.Linear) and child.in_features >= 64 and child.out_features >= 64:
                targets.append((mod, attr, child))
    for parent, attr, lin in targets:
        wrapped = LoRALinear(lin, rank=rank, alpha=alpha)
        setattr(parent, attr, wrapped)
        lora_params.extend([wrapped.lora_a, wrapped.lora_b])
    return lora_params


def merge_fm_lora(fm_decoder: nn.Module) -> int:
    """将所有 LoRALinear 无损合并回标准 nn.Linear，保持导出权重与原版 ZipVoice 结构 100% 一致。"""
    merged = 0
    for mod in fm_decoder.modules():
        for attr, child in list(mod.named_children()):
            if isinstance(child, LoRALinear):
                setattr(mod, attr, child.merge_to_linear())
                merged += 1
    return merged


def ipa_tone(ipa: str) -> str:
    import unicodedata
    nfd = unicodedata.normalize("NFD", ipa)
    for mk, tone in TONE_FROM_MARK.items():
        if mk in nfd:
            return tone
    base = re.sub(r"[^a-zʔ]", "", unicodedata.normalize("NFD", ipa))
    return "入" if base.endswith(("p", "t", "k")) else "平"


def header_yun(header: str | None, tone: str) -> str | None:
    if not header:
        return None
    parts = [p for p in re.split(r"\s+", header.replace("*", " * ").strip()) if p]
    flat = []
    for p in parts:
        flat.extend(list(p) if p != "*" else ["*"])
    pos = {"平": 0, "上": 1, "去": 2, "入": 3}[tone]
    return flat[pos] if pos < len(flat) and flat[pos] != "*" else None


def build_xiaoqieyun_items(comp, fbank, labels_path: Path, wav_path: Path, max_items: int = 0):
    labels = json.load(open(labels_path))
    w, sr = sf.read(str(wav_path), dtype="float32")
    items, miss = [], 0
    for lab in labels:
        tone = ipa_tone(lab["ipa"])
        yun = header_yun(lab.get("header"), tone)
        r = None
        for ch in lab["chars"]:
            key = ch if ch in comp.readings else comp.s2t.get(ch)
            if not key:
                continue
            rs = comp.readings[key]
            cand = [x for x in rs if x.sheng == tone] or rs
            if yun:
                cand = [x for x in cand if x.yun == yun] or cand
            r = cand[0]
            break
        if r is None:
            miss += 1
            continue
        a, b = int(max(0, lab["start"] - 0.03) * sr), int(min(len(w) / sr, lab["end"] + 0.05) * sr)
        y = w[a:b]
        if len(y) < int(0.15 * sr):
            continue
        y = y * (0.1 / (np.sqrt(np.mean(y ** 2)) + 1e-6))
        mel = fbank.extract(np.clip(y, -0.95, 0.95), sr=SR).cpu()
        ini, fin = comp.initial_name(r), comp.final_name(r)
        checked = r.sheng == "入"
        tok = [(comp.t2i[ini], 24 if checked else 30), (comp.t2i[fin], 56 if checked else 70)]
        items.append({"mel": mel, "token_items": tok, "valid_mel_bins": 100, "era": "mc", "src": f"xqy_{lab['seg']}_{r.desc}"})
        if max_items and len(items) >= max_items:
            break
    print(f"  ✓ 全常用音节: {len(items)} 条 (无法查到读音 {miss})")
    return items


def build_qzw_items(comp, old_items):
    out = []
    for it in old_items:
        if it["era"] != "mc" or "qzw2026" not in it["src"]:
            continue
        text = it["src"].split("_")[-1]
        tok, _ = comp.compile_clause(text)
        if len(tok) // 2 != len(re.findall(r"[\u4e00-\u9fff]", text)):
            continue
        out.append({**it, "token_items": tok})
    print(f"  ✓ 千字文: {len(out)} 句重新编码")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--lora", action="store_true", help="同时微调 text_encoder 与 fm_decoder LoRA")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-lr", type=float, default=1.5e-4)
    ap.add_argument("--fp16", action="store_true", help="开启 CUDA FP16 混合精度加速")
    ap.add_argument("--mix-old-eras", action="store_true", help="混入上古/近古多说话人样本防止音色坍缩")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "v9_outputs")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0); random.seed(0)

    ck = torch.load(str(PROJECT_ROOT / "v5_outputs/best_fp32.pt"), map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    model = ZipVoice(**cfg)
    model.load_state_dict(ck["model"], strict=True)
    old_comp = ThreeEraHanfireCompiler(ck["t2i_custom"], PROJECT_ROOT / "v5_bundle/v5_three_era_lexicon.json")
    t2i, fe, comp, meta = build_mc_embedding(model, ck["t2i_custom"], old_comp)
    model.embed = fe
    model = model.to(dev)

    print("[1] 构建中古训练集")
    fbank = PureVocosFbank()
    D = PROJECT_ROOT / "data_mc_raw"
    lab = D / "xiaoqieyun_labels_v3.json" if (D / "xiaoqieyun_labels_v3.json").exists() else D / "xiaoqieyun_labels_v2.json"
    items = build_xiaoqieyun_items(comp, fbank, lab, D / "xiaoqieyun_all.wav", max_items=64 if args.smoke else 0)
    old_items = torch.load(str(PROJECT_ROOT / "v5_bundle/v5_train_dataset.pt"), map_location="cpu", weights_only=False)
    items += build_qzw_items(comp, old_items)
    if args.mix_old_eras:
        reg_items = [it for it in old_items if it["era"] in ("oc", "em")]
        random.shuffle(reg_items)
        if args.smoke:
            reg_items = reg_items[:32]
        items += reg_items
        print(f"  ✓ 多说话人正则化样本 (OC+EM): {len(reg_items)} 条")

    random.shuffle(items)
    n_val = max(16, len(items) // 25)
    val, train = items[:n_val], items[n_val:]
    dl = DataLoader(ThreeEraDataset(train), batch_size=args.batch_size, shuffle=True, collate_fn=collate_three_era)
    dlv = DataLoader(ThreeEraDataset(val), batch_size=args.batch_size, shuffle=False, collate_fn=collate_three_era)
    print(f"  train={len(train)} val={len(val)} | lora={args.lora} fp16={args.fp16}")

    for p in model.parameters():
        p.requires_grad = False
    emb_params = [fe.ini, fe.seg, fe.tone]
    for p in emb_params:
        p.requires_grad = True
    emb_init = [p.detach().clone() for p in emb_params]
    norm_ref = fe.old.norm(dim=-1).mean().item()

    opt_groups: List[Dict] = [{"params": emb_params, "lr": args.lr, "weight_decay": 0.0}]
    all_trainable: List[nn.Parameter] = list(emb_params)
    if args.lora:
        te_params: List[nn.Parameter] = []
        for p in model.text_encoder.parameters():
            p.requires_grad = True
            te_params.append(p)
        lora_params = inject_fm_lora(model.fm_decoder, rank=args.lora_rank, alpha=args.lora_rank * 2.0)
        opt_groups.append({"params": te_params, "lr": args.lora_lr * 0.5, "weight_decay": 1e-4})
        opt_groups.append({"params": lora_params, "lr": args.lora_lr, "weight_decay": 0.0})
        all_trainable.extend(te_params + lora_params)
        print(f"  ✓ 启用 LoRA (rank={args.lora_rank}): text_encoder={sum(p.numel() for p in te_params)/1e6:.2f}M, fm_lora={sum(p.numel() for p in lora_params)/1e6:.2f}M")

    opt = torch.optim.AdamW(opt_groups)
    epochs = 2 if args.smoke else args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * len(dl)), eta_min=args.lr * 0.08)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.fp16 and dev.type == "cuda"))
    model.eval()

    def run_epoch(loader, train_mode: bool) -> float:
        losses = []
        for batch in loader:
            feats = batch["mels"].to(dev) * FEAT_SCALE
            fl = batch["mel_lens"].to(dev)
            b_mask = batch["bin_masks"].to(dev)
            nf = int(fl.max())
            pad = make_pad_mask(fl, max_len=nf)
            scm = condition_time_mask(features_lens=fl, mask_percent=(0.7, 1.0), max_len=nf)
            sc = torch.where(scm.unsqueeze(-1), 0.0, feats)
            noise = torch.randn_like(feats)
            t = torch.rand(feats.size(0), 1, 1, device=dev)
            xt = feats * t + noise * (1 - t)
            ut = feats - noise
            with torch.set_grad_enabled(train_mode):
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(args.fp16 and dev.type == "cuda")):
                    vt = forward_velocity(model, batch["tokens"], batch["durs"], xt, t, sc, pad)
                    m = (scm & ~pad).unsqueeze(-1) & b_mask
                    loss = torch.mean((vt[m].float() - ut[m].float()) ** 2)
                if train_mode:
                    reg = sum(0.02 * torch.mean((p - p0) ** 2) for p, p0 in zip(emb_params, emb_init))
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss + reg).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
            losses.append(float(loss.item()))
        return float(np.mean(losses))

    best, best_state = 1e9, None
    print("[2] 开始训练 V9 中古音模型...")
    for ep in range(1, epochs + 1):
        t0 = time.time()
        torch.manual_seed(ep)
        tr = run_epoch(dl, True)
        torch.manual_seed(1234)
        va = run_epoch(dlv, False)
        if va < best:
            best = va
            best_state = {
                "fe": {k: v.detach().cpu().clone() for k, v in fe.state_dict().items()},
                "te": {k: v.detach().cpu().clone() for k, v in model.text_encoder.state_dict().items()} if args.lora else None,
                "fm": {k: v.detach().cpu().clone() for k, v in model.fm_decoder.state_dict().items()} if args.lora else None,
            }
        print(f"  epoch {ep:02d}/{epochs} train={tr:.4f} val={va:.4f} best={best:.4f} | row_norm={fe.weight[meta['n_old']:].norm(dim=-1).mean():.2f} (ref {norm_ref:.2f}) | {time.time()-t0:.1f}s")

    fe.load_state_dict(best_state["fe"])
    if args.lora and best_state["te"] is not None:
        model.text_encoder.load_state_dict(best_state["te"])
        model.fm_decoder.load_state_dict(best_state["fm"])
        n_merged = merge_fm_lora(model.fm_decoder)
        print(f"  ✓ 已将 {n_merged} 个 LoRA 线性层无损合并回标准 nn.Linear")

    model.embed = fe.materialize().to(dev)
    cfg["vocab_size"] = model.embed.weight.size(0)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    out_name = "best_fp32_v9mc_lora.pt" if args.lora else "best_fp32_v9mc.pt"
    out = args.out_dir / out_name
    torch.save({"model": sd, "config": cfg, "t2i_custom": t2i, "mc_meta": meta, "best_val": best}, str(out))
    print(f"✅ 保存 {out} (vocab={cfg['vocab_size']}, best_val={best:.4f})")


if __name__ == "__main__":
    main()
