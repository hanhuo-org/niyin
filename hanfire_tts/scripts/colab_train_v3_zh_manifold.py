#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汉火上古音 TTS V3.2 —— Colab L4 / L20 24kHz 华夏声韵流形精准精调主程序
(colab_train_v3_zh_manifold.py)

设计目标（解决零样本线性插值残余的“个别拟音偏移 / 全浊与入声不够地道”问题，同时绝不重犯 V1 电音错误）：
1. 教师声源 100% 原生 24kHz 汉语方言专家 (`DiaMoE-TTS` + `HanfireDiaMoECompiler`)：
   - 直接在 GPU 上调用中国十一大方言母语训练的 DiaMoE-TTS，为汉火词表 (`hanfire_oc_v1.json`) 的核心高频音节与
     《诗经》全篇四言短句生成 24kHz 高清母语发音（吴语真全浊 `b/d/g/dz/ʑ/ɣ` + 粤/闽唯闭入声 `-p̚/-t̚/-k̚` 与 `-m`），
   - 并由四维声学裁判 (`judge_utterance_acoustics`) 在入库前自动拦截任何多峰/吞字样本，仅保留 100% 严格“一字一音”切片！
2. 严密冻结声码器与高频解码层（零电音铁律）：
   - 100% 冻结 `Vocos` 与 `ZipVoice.fm_decoder` 的首尾高分辨率声学层 (`encoders.0`, `encoders.4`, `out_proj`)；
   - 对 `model.embed.weight` 施加梯度掩码：仅更新 65 个新注入的汉火声韵槽位 (`DEAD_SLOT_POOL`)，100% 锁死原装普通话词元；
   - 仅联合微调 `model.text_encoder` 与 `fm_decoder.in_proj`，用 ~15 分钟让模型把每个汉火音位咬得字正腔圆，同时保留
     100% 真人男女嗓音质感！
3. Google Drive 异步非阻塞双写存盘 (`AsyncCheckpointManager`)：支持 Colab L4 与外部 L20 随时无缝续训。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
VENDOR_ZV = PROJECT_ROOT / "vendor" / "ZipVoice"
for p in (str(VENDOR_ZV), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from huggingface_hub import hf_hub_download
from zipvoice.models.zipvoice import ZipVoice
from zipvoice.utils.common import condition_time_mask, get_tokens_index, make_pad_mask

from hanfire_core.async_checkpoint import AsyncCheckpointManager
from hanfire_core.zipvoice_ancient import PureVocosFbank
from hanfire_core.zipvoice_zh_manifold import (
    DEAD_SLOT_POOL,
    HanfireZipVoiceCompiler,
    _weights_to_exact_frames,
    inject_hanfire_zh_manifold_embeddings,
    load_zv_token2id,
)

SR = 24000
FEAT_SCALE = 0.1


class AlignedPhraseDataset(Dataset):
    """缓存好的 24kHz 四言短句与一字一拍精确权重数据集。"""

    def __init__(self, cache_pt: Path) -> None:
        self.items: List[Dict] = torch.load(str(cache_pt), map_location="cpu", weights_only=False)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]


def collate_aligned(batch: List[Dict]) -> Dict:
    max_t = max(x["mel"].shape[0] for x in batch)
    feat_dim = batch[0]["mel"].shape[1]
    mels = torch.zeros(len(batch), max_t, feat_dim, dtype=torch.float32)
    mel_lens = torch.zeros(len(batch), dtype=torch.int64)
    tokens_list: List[List[int]] = []
    durs_list: List[List[int]] = []

    for i, item in enumerate(batch):
        t_len = item["mel"].shape[0]
        mels[i, :t_len] = item["mel"]
        mel_lens[i] = t_len
        tok_ids = [ pair[0] for pair in item["token_items"] ]
        tok_ws  = [ pair[1] for pair in item["token_items"] ]
        tokens_list.append(tok_ids)
        durs_list.append(_weights_to_exact_frames(tok_ws, t_len))

    return {
        "mels": mels,
        "mel_lens": mel_lens,
        "tokens": tokens_list,
        "durs": durs_list,
    }


def configure_surgical_trainable_params(model: ZipVoice) -> Tuple[List[torch.nn.Parameter], torch.Tensor]:
    """
    冻结高分辨率声学输出层以 100% 保护真人音质；
    仅放开 `embed` (带汉火槽位梯度掩码)、`text_encoder` 与 `fm_decoder.in_proj`。
    """
    for p in model.parameters():
        p.requires_grad = False

    # 1. 词表嵌入（仅允许更新 DEAD_SLOT_POOL 中的汉火新音位槽！）
    model.embed.weight.requires_grad = True
    grad_mask = torch.zeros_like(model.embed.weight)
    for slot in DEAD_SLOT_POOL:
        if slot < grad_mask.size(0):
            grad_mask[slot] = 1.0
    model.embed.weight.register_hook(lambda g: g * grad_mask.to(g.device))

    # 2. 文本编码器 + 流匹配条件入口投影（不动声学高频层！）
    other_trainable: List[torch.nn.Parameter] = []
    for p in model.text_encoder.parameters():
        p.requires_grad = True
        other_trainable.append(p)
    if hasattr(model.fm_decoder, "in_proj"):
        for p in model.fm_decoder.in_proj.parameters():
            p.requires_grad = True
            other_trainable.append(p)

    return other_trainable, grad_mask


def set_zipformer_batch_count(model: torch.nn.Module, step: int) -> None:
    val = 100000.0 + float(step)
    for m in model.modules():
        if hasattr(m, "batch_count"):
            m.batch_count = val


def forward_character_aligned_fm_loss(
    model: ZipVoice,
    tokens: List[List[int]],
    durs: List[List[int]],
    features: torch.Tensor,
    features_lens: torch.Tensor,
    condition_drop_ratio: float = 0.1,
) -> torch.Tensor:
    device = features.device
    num_frames = int(features_lens.max().item())
    embed, _ = model.forward_text_embed(tokens)
    padding_mask = make_pad_mask(features_lens, max_len=num_frames)
    tokens_index = get_tokens_index(durs, num_frames).to(device)
    text_condition = torch.gather(
        embed,
        dim=1,
        index=tokens_index.unsqueeze(-1).expand(embed.size(0), num_frames, embed.size(-1)),
    )

    speech_condition_mask = condition_time_mask(
        features_lens=features_lens,
        mask_percent=(0.7, 1.0),
        max_len=num_frames,
    )
    speech_condition = torch.where(speech_condition_mask.unsqueeze(-1), 0.0, features)

    if condition_drop_ratio > 0.0:
        drop_mask = (torch.rand(text_condition.size(0), 1, 1, device=device) > condition_drop_ratio).float()
        text_condition = text_condition * drop_mask

    noise = torch.randn_like(features)
    t = torch.rand(features.size(0), 1, 1, device=device)
    xt = features * t + noise * (1.0 - t)
    ut = features - noise

    vt = model.forward_fm_decoder(
        t=t,
        xt=xt,
        text_condition=text_condition,
        speech_condition=speech_condition,
        padding_mask=padding_mask,
    )
    loss_mask = speech_condition_mask & (~padding_mask)
    return torch.mean((vt[loss_mask] - ut[loss_mask]) ** 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-pt", type=Path, required=True, help="24kHz 对齐短句特征缓存路径")
    ap.add_argument("--gdrive-dir", type=Path, default=Path("/content/drive/MyDrive/hanfire_ancient_tts/checkpoints_v3_zh"))
    ap.add_argument("--local-ckpt-dir", type=Path, default=Path("/tmp/hanfire_ckpts_v3_zh"))
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--resume", type=str, default="auto")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    p_json = Path(hf_hub_download("k2-fsa/ZipVoice", filename="zipvoice/model.json"))
    p_tok = Path(hf_hub_download("k2-fsa/ZipVoice", filename="zipvoice/tokens.txt"))
    p_pt = Path(hf_hub_download("k2-fsa/ZipVoice", filename="zipvoice/model.pt"))

    base_t2i = load_zv_token2id(p_tok)
    cfg = json.loads(p_json.read_text(encoding="utf-8"))["model"]
    cfg["vocab_size"] = len(base_t2i)
    model = ZipVoice(**cfg)
    raw_sd = torch.load(str(p_pt), map_location="cpu", weights_only=False)["model"]
    model.load_state_dict({k.removeprefix("module."): v for k, v in raw_sd.items()}, strict=True)
    inject_hanfire_zh_manifold_embeddings(model, base_t2i)
    model = model.to(device)

    other_trainable, _ = configure_surgical_trainable_params(model)
    # 词表嵌入单独设 weight_decay=0.0，严防权重衰减腐蚀未参与更新的普通话原装词元！
    opt = torch.optim.AdamW(
        [
            {"params": [model.embed.weight], "lr": args.lr * 2.0, "weight_decay": 0.0},
            {"params": other_trainable, "lr": args.lr, "weight_decay": 1e-4},
        ]
    )
    all_trainable = [model.embed.weight] + other_trainable

    ckpt_mgr = AsyncCheckpointManager(
        local_dir=args.local_ckpt_dir,
        gdrive_dir=args.gdrive_dir,
        keep_last_n=4,
    )
    start_epoch, global_step = 0, 0
    if args.resume == "auto":
        resume_path = ckpt_mgr.resolve_resume_path(resume="auto", stage="v3_zh")
        if resume_path is not None:
            resumed = ckpt_mgr.load_checkpoint(resume_path, model=model, optimizer=opt, map_location=str(device))
            start_epoch = int(resumed.get("epoch", 0))
            global_step = int(resumed.get("global_step", 0))
            print(f"[续训恢复] 从 epoch={start_epoch}, step={global_step} 继续训练")

    ds = AlignedPhraseDataset(args.cache_pt)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_aligned)

    model.train()
    for ep in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()
        losses: List[float] = []
        for batch in dl:
            global_step += 1
            set_zipformer_batch_count(model, global_step)
            opt.zero_grad(set_to_none=True)
            loss = forward_character_aligned_fm_loss(
                model=model,
                tokens=batch["tokens"],
                durs=batch["durs"],
                features=batch["mels"].to(device) * FEAT_SCALE,
                features_lens=batch["mel_lens"].to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
            opt.step()
            losses.append(float(loss.item()))

        mean_l = float(np.mean(losses))
        print(f"Epoch {ep:02d}/{args.epochs} | loss={mean_l:.5f} | step={global_step} | {time.time()-t0:.1f}s")
        ckpt_mgr.save_async(
            stage="v3_zh",
            epoch=ep,
            global_step=global_step,
            loss=mean_l,
            model=model,
            optimizer=opt,
            extra_state={"fm_loss": mean_l},
        )

    ckpt_mgr.flush_and_close()
    print("✅ V3.2 华夏声韵流形精准精调全部完成！")


if __name__ == "__main__":
    main()
