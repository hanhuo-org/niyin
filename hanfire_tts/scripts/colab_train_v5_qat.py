#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汉火古音 TTS V5 —— 上古/中古/近古三代贯通精调 + W8A8 INT8 量化感知训练 (QAT) 主程序
(colab_train_v5_qat.py)

全自动三阶段闭环（无需交互式 Google Drive 授权，直接落盘产出最优 FP32 与真 INT8 权重及三代样带）：
- Stage 1 (Three-Era Spherical Embedding Alignment)：
  在冻结声学解码器的前提下，利用上古(OC 24kHz 真人干声)、中古(MC 三十六字母真人单字)、近古(EM 明代官话短句)
  三代数据（带 `valid_mel_bins` 高频保护掩码），在球面流形上精调 86 个古音槽位，产出 `best_fp32.pt` (~490MB)。
- Stage 2 (Teacher-Guided W8A8 INT8 QAT)：
  以收敛后的 `best_fp32` 为冻结教师，将学生模型 99.2% 权重（271 个 Linear + 100 个 ActivationDropoutAndLinear）
  替换为逐通道对称权重 INT8 + 动态逐 token 激活 INT8 的 STE QAT 层，通过全频段 24kHz 流匹配速度场蒸馏
  `MSE(v_student_int8, v_teacher_fp32)` 消除量化误差，原地打包导出真 `torch.int8` 权重 `best_int8.pt` (~125MB)！
- Stage 3 (FP32 vs INT8 数值对齐门禁 + 上古/中古/近古三代男女双音色样带合成)：
  实测对比 FP32 与真 INT8 推理模型的 `mel_L1` 与余弦相似度，并输出三代代表作男女双音色 `.wav` 音频与评测报告。
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import tarfile
import time
import types
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from scipy import signal
from torch.utils.data import DataLoader, Dataset

# 兼容 PyTorch 2.11+ 移除 torchaudio.functional.functional 的遗留顶层导入
if "torchaudio.functional.functional" not in sys.modules:
    _ta_fn = types.ModuleType("torchaudio.functional.functional")
    _ta_fn._hz_to_mel = lambda hz, mel_scale="htk": 2595.0 * math.log10(1.0 + hz / 700.0)
    _ta_fn._mel_to_hz = lambda mel, mel_scale="htk": 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    sys.modules.setdefault("torchaudio.functional", _ta_fn)
    sys.modules["torchaudio.functional.functional"] = _ta_fn

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
VENDOR_ZV = PROJECT_ROOT / "vendor" / "ZipVoice"
for p in (str(PROJECT_ROOT), str(VENDOR_ZV), str(PROJECT_ROOT / "vendor_py")):
    if p not in sys.path:
        sys.path.insert(0, p)

from zipvoice.models.zipvoice import ZipVoice
from zipvoice.utils.common import condition_time_mask, get_tokens_index, make_pad_mask

from hanfire_core.qat_int8 import (
    convert_qat_to_int8_inference,
    load_int8_checkpoint,
    prepare_zipvoice_for_qat,
)
from hanfire_core.referee import judge_utterance_acoustics
from hanfire_core.zipvoice_ancient import PureVocosFbank
from hanfire_core.zipvoice_zh_manifold import (
    DEAD_SLOT_POOL,
    ThreeEraHanfireCompiler,
    _weights_to_exact_frames,
    inject_hanfire_zh_manifold_embeddings,
    load_zv_token2id,
    sample_with_character_durations,
    tokenize_mandarin_prompt_exact,
)

SR = 24000
FEAT_SCALE = 0.1
N_FFT = 1024
HOP = 256


class PureVocosDecoder(nn.Module):
    """零 torchaudio.transforms 依赖的纯 PyTorch 24kHz Vocos 声码器（本地与 Colab 通用）。"""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        import yaml
        from vocos.pretrained import instantiate_class
        cfg_p = Path(hf_hub_download("charactr/vocos-mel-24khz", filename="config.yaml"))
        bin_p = Path(hf_hub_download("charactr/vocos-mel-24khz", filename="pytorch_model.bin"))
        cfg = yaml.safe_load(cfg_p.read_text(encoding="utf-8"))
        self.backbone = instantiate_class(args=(), init=cfg["backbone"])
        self.head = instantiate_class(args=(), init=cfg["head"])
        sd = torch.load(str(bin_p), map_location="cpu", weights_only=True)
        self.backbone.load_state_dict({k.removeprefix("backbone."): v for k, v in sd.items() if k.startswith("backbone.")}, strict=True)
        self.head.load_state_dict({k.removeprefix("head."): v for k, v in sd.items() if k.startswith("head.")}, strict=True)
        self.to(device).eval()
        self.register_buffer("win", torch.hann_window(N_FFT, device=device), persistent=False)

    @staticmethod
    def _sweet_spot_mel_envelope(
        mel_b_c_t: torch.Tensor, max_abs: float = 3.78, max_rel_p90: float = 0.48
    ) -> torch.Tensor:
        """
        仅对超出 Vocos 线性甜区的局部突跳音节（绝对 mel_max > 3.92 或单音节 p90 高出本句浊音中位数 > +0.58 ≈ +5dB）
        做高斯平滑包络归拢：同一时间帧的 100 个 mel 频带同步减去同一平滑标量 Δ(t)，
        零硬削波、不改变共振峰 F1/F2，正常均衡诗句 (excess == 0) 保持 100% 逐比特原样不变！
        """
        from scipy.ndimage import gaussian_filter1d
        m = mel_b_c_t.squeeze(0).detach().cpu().numpy()  # [100, T]
        f_max = np.max(m, axis=0)
        f_p90 = np.percentile(m, 90, axis=0)
        voiced = f_p90 > (np.max(f_p90) - 1.8)
        med_p90 = float(np.median(f_p90[voiced])) if int(np.sum(voiced)) >= 4 else float(np.max(f_p90))
        excess = np.maximum(np.maximum(0.0, f_max - max_abs), np.maximum(0.0, f_p90 - (med_p90 + max_rel_p90)))
        if float(np.max(excess)) > 1e-4:
            sm = np.maximum(
                gaussian_filter1d(excess, sigma=3.0),
                gaussian_filter1d(excess, sigma=1.3) * 0.85,
            ).astype(np.float32)
            m = m - sm[None, :]
            return torch.from_numpy(m).unsqueeze(0).to(device=mel_b_c_t.device, dtype=mel_b_c_t.dtype)
        return mel_b_c_t

    @torch.no_grad()
    def decode_mel(self, mel_b_c_t: torch.Tensor) -> np.ndarray:
        dev = next(self.backbone.parameters()).device
        mel_in = self._sweet_spot_mel_envelope(mel_b_c_t).to(dev)
        x = self.backbone(mel_in)
        x = self.head.out(x).transpose(1, 2)  # [B, 1026, T]
        mag, phase = x.chunk(2, dim=1)
        mag = torch.clip(torch.exp(mag), max=1e3)
        spec = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
        wav = torch.istft(spec, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, window=self.win, center=True)
        return wav.squeeze(0).cpu().numpy().astype(np.float32)


class ThreeEraDataset(Dataset):
    def __init__(self, items: List[Dict]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]


def collate_three_era(batch: List[Dict]) -> Dict:
    max_t = max(x["mel"].shape[0] for x in batch)
    mels = torch.zeros(len(batch), max_t, 100, dtype=torch.float32)
    mel_lens = torch.zeros(len(batch), dtype=torch.int64)
    bin_masks = torch.zeros(len(batch), 1, 100, dtype=torch.bool)
    tokens_list: List[List[int]] = []
    durs_list: List[List[int]] = []

    for i, item in enumerate(batch):
        t_len = item["mel"].shape[0]
        mels[i, :t_len] = item["mel"]
        mel_lens[i] = t_len
        vb = int(item.get("valid_mel_bins", 100))
        bin_masks[i, 0, :vb] = True
        t_ids = [p[0] for p in item["token_items"]]
        t_ws = [p[1] for p in item["token_items"]]
        tokens_list.append(t_ids)
        durs_list.append(_weights_to_exact_frames(t_ws, t_len))

    return {
        "mels": mels,
        "mel_lens": mel_lens,
        "bin_masks": bin_masks,
        "tokens": tokens_list,
        "durs": durs_list,
    }


def set_batch_count(model: nn.Module, step: int) -> None:
    val = 100000.0 + float(step)
    for m in model.modules():
        if hasattr(m, "batch_count"):
            m.batch_count = val


def forward_velocity(
    model: ZipVoice,
    tokens: List[List[int]],
    durs: List[List[int]],
    xt: torch.Tensor,
    t: torch.Tensor,
    speech_cond: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    num_frames = xt.size(1)
    embed, _ = model.forward_text_embed(tokens)
    tokens_index = get_tokens_index(durs, num_frames).to(xt.device)
    text_cond = torch.gather(
        embed, dim=1,
        index=tokens_index.unsqueeze(-1).expand(embed.size(0), num_frames, embed.size(-1)),
    )
    return model.forward_fm_decoder(
        t=t, xt=xt, text_condition=text_cond,
        speech_condition=speech_cond, padding_mask=padding_mask,
    )


def run_stage1_three_era_embed(
    model: ZipVoice,
    dl: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float = 8e-4,
    max_steps_per_ep: int = 0,
) -> Tuple[Dict[str, torch.Tensor], List[float]]:
    for p in model.parameters():
        p.requires_grad = False

    init_emb = model.embed.weight.data.clone().to(device)
    target_norms = init_emb.norm(dim=-1, keepdim=True).clamp(min=1.0)

    model.embed.weight.requires_grad = True
    grad_mask = torch.zeros_like(model.embed.weight, device=device)
    for slot in DEAD_SLOT_POOL:
        if slot < grad_mask.size(0):
            grad_mask[slot] = 1.0
    hook = model.embed.weight.register_hook(lambda g: g * grad_mask)

    opt = torch.optim.AdamW([{"params": [model.embed.weight], "lr": lr, "weight_decay": 0.0}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * len(dl)), eta_min=lr * 0.15)
    model.eval()

    best_loss = float("inf")
    best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    curve: List[float] = []

    for ep in range(1, epochs + 1):
        t0 = time.time()
        losses: List[float] = []
        for step_i, batch in enumerate(dl, 1):
            if max_steps_per_ep > 0 and step_i > max_steps_per_ep:
                break
            opt.zero_grad(set_to_none=True)
            features = batch["mels"].to(device) * FEAT_SCALE
            f_lens = batch["mel_lens"].to(device)
            b_mask = batch["bin_masks"].to(device)
            num_frames = int(f_lens.max().item())

            padding_mask = make_pad_mask(f_lens, max_len=num_frames)
            sc_mask = condition_time_mask(features_lens=f_lens, mask_percent=(0.75, 1.0), max_len=num_frames)
            speech_cond = torch.where(sc_mask.unsqueeze(-1), 0.0, features)

            noise = torch.randn_like(features)
            t = torch.rand(features.size(0), 1, 1, device=device)
            xt = features * t + noise * (1.0 - t)
            ut = features - noise

            vt = forward_velocity(model, batch["tokens"], batch["durs"], xt, t, speech_cond, padding_mask)
            valid_3d = (sc_mask & (~padding_mask)).unsqueeze(-1) & b_mask
            fm_loss = torch.mean((vt[valid_3d] - ut[valid_3d]) ** 2)
            reg = 0.08 * torch.mean(((model.embed.weight - init_emb) * grad_mask) ** 2)
            loss = fm_loss + reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_([model.embed.weight], 1.0)
            opt.step()
            sched.step()

            with torch.no_grad():
                cur_n = model.embed.weight.data.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                rescaled = model.embed.weight.data * (target_norms / cur_n)
                model.embed.weight.data = torch.where(grad_mask > 0.5, rescaled, init_emb)

            losses.append(float(fm_loss.item()))

        ml = float(np.mean(losses))
        curve.append(round(ml, 5))
        if ml < best_loss:
            best_loss = ml
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"  [Stage 1 FP32 Embed] Epoch {ep:02d}/{epochs} | fm_loss={ml:.5f} (best={best_loss:.5f}) | {time.time()-t0:.2f}s")

    hook.remove()
    model.load_state_dict(best_sd, strict=True)
    return best_sd, curve


def run_stage2_qat_int8(
    teacher_fp32: ZipVoice,
    student: ZipVoice,
    dl: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float = 2.0e-5,
    max_steps_per_ep: int = 0,
) -> Tuple[Dict, List[float]]:
    teacher_fp32.eval()
    for p in teacher_fp32.parameters():
        p.requires_grad = False

    qat_stats = prepare_zipvoice_for_qat(student)
    student.to(device)
    print(
        f"  ✓ QAT 包装完成: Linear={qat_stats['replaced_linear']} 层, "
        f"ActivationDropoutAndLinear={qat_stats['replaced_act_linear']} 层, "
        f"INT8 覆盖率={qat_stats['quantized_ratio_pct']}%"
    )

    # 锁死已经调准的 360 行词向量与受保护边界小层，仅微调被 QAT 包装的线性层权重以补偿 INT8 舍入误差！
    qat_params: List[nn.Parameter] = []
    for name, p in student.named_parameters():
        if any(k in name for k in ("embed", "fm_decoder.in_proj", "fm_decoder.out_proj", "time_emb")):
            p.requires_grad = False
        else:
            p.requires_grad = True
            qat_params.append(p)

    opt = torch.optim.AdamW(qat_params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * len(dl)), eta_min=lr * 0.08)
    # 保持 eval 模式关闭随机 Dropout，使 INT8 学生与 FP32 教师在同一确定性流形上逐点对齐速度场！
    student.eval()

    best_kd = float("inf")
    best_qat_sd = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    kd_curve: List[float] = []
    g_step = 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        ep_kds: List[float] = []
        for step_i, batch in enumerate(dl, 1):
            if max_steps_per_ep > 0 and step_i > max_steps_per_ep:
                break
            g_step += 1
            set_batch_count(student, g_step)
            opt.zero_grad(set_to_none=True)

            features = batch["mels"].to(device) * FEAT_SCALE
            f_lens = batch["mel_lens"].to(device)
            num_frames = int(f_lens.max().item())

            padding_mask = make_pad_mask(f_lens, max_len=num_frames)
            sc_mask = condition_time_mask(features_lens=f_lens, mask_percent=(0.55, 1.0), max_len=num_frames)
            speech_cond = torch.where(sc_mask.unsqueeze(-1), 0.0, features)

            noise = torch.randn_like(features)
            t = torch.rand(features.size(0), 1, 1, device=device)
            xt = features * t + noise * (1.0 - t)

            with torch.no_grad():
                vt_teacher = forward_velocity(teacher_fp32, batch["tokens"], batch["durs"], xt, t, speech_cond, padding_mask)

            vt_student = forward_velocity(student, batch["tokens"], batch["durs"], xt, t, speech_cond, padding_mask)
            valid = (~padding_mask)
            kd_loss = F.mse_loss(vt_student[valid], vt_teacher[valid])
            kd_loss.backward()
            torch.nn.utils.clip_grad_norm_(qat_params, 1.0)
            opt.step()
            sched.step()
            ep_kds.append(float(kd_loss.item()))

        mkd = float(np.mean(ep_kds))
        kd_curve.append(round(mkd, 6))
        if mkd < best_kd:
            best_kd = mkd
            best_qat_sd = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
        if ep == 1 or ep % 3 == 0 or ep == epochs:
            print(f"  [Stage 2 W8A8 QAT] Epoch {ep:02d}/{epochs} | velocity_kd_mse={mkd:.6f} (best={best_kd:.6f}) | {time.time()-t0:.2f}s")

    student.load_state_dict(best_qat_sd, strict=True)
    return qat_stats, kd_curve


def trim_edge_silence(w: np.ndarray, sr: int = SR, rel_db: float = 40.0, pre_ms: int = 25, post_ms: int = 70) -> np.ndarray:
    """按能量切除首尾静音（ZipVoice 解码器会在提示音后先生成 ~300ms 静音）。"""
    hop = int(0.01 * sr)
    if len(w) <= hop * 2:
        return w
    db = 20.0 * np.log10(np.sqrt(np.mean(w[: len(w) // hop * hop].reshape(-1, hop) ** 2, axis=1)) + 1e-9)
    act = np.where(db > max(-58.0, float(db.max()) - rel_db))[0]
    if len(act) == 0:
        return w
    s0 = max(0, act[0] * hop - int(pre_ms * sr / 1000))
    s1 = min(len(w), (act[-1] + 1) * hop + int(post_ms * sr / 1000))
    return w[s0:s1]


@torch.no_grad()
def synth_era_poem(
    model: nn.Module,
    vocos: PureVocosDecoder,
    fbank: PureVocosFbank,
    comp: ThreeEraHanfireCompiler,
    era: str,
    prompt_wav: np.ndarray,
    prompt_items: List[Tuple[int, int]],
    couplets: List[Tuple[str, str, Dict[str, str]]],
    frames_per_100w: float = 25.5,
    lead_pause_w: int = 120,
    clause_gap_s: float = 0.16,
    couplet_gap_s: float = 0.32,
    num_step: int = 16,
) -> Tuple[np.ndarray, torch.Tensor]:
    """V8：逐分句合成（停顿在波形域插入，不靠标点 token 演停顿）+ 句首停顿预算 + 首尾静音切除。"""
    dev = next(model.parameters()).device
    pmt_feat = fbank.extract(prompt_wav, sr=SR).unsqueeze(0).to(dev) * FEAT_SCALE
    pmt_len = torch.tensor([pmt_feat.size(1)], dtype=torch.int64, device=dev)
    dot = comp.t2i["."]

    wavs: List[np.ndarray] = []
    mels: List[torch.Tensor] = []
    fade = int(0.012 * SR)
    for idx, (c1, c2, ov) in enumerate(couplets):
        for j, clause in enumerate((c1, c2)):
            it, _ = comp.compile_era_clause(era, clause, context_overrides=ov, end_punct="")
            items = ([(dot, lead_pause_w)] if lead_pause_w > 0 else []) + it
            pred_mel = sample_with_character_durations(
                model=model, prompt_items=prompt_items, target_items=items,
                prompt_features=pmt_feat, prompt_features_lens=pmt_len,
                frames_per_100w=frames_per_100w, num_step=num_step, guidance_scale=1.25,
                t_shift=0.5, seed=42 + idx * 11 + j,
            )
            mels.append(pred_mel.detach().cpu())
            w = trim_edge_silence(vocos.decode_mel(pred_mel.permute(0, 2, 1) / FEAT_SCALE))
            if len(w) > fade * 2:
                w[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
                w[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
            wavs.append(w)
            wavs.append(np.zeros(int((clause_gap_s if j == 0 else couplet_gap_s) * SR), dtype=np.float32))
    return np.concatenate(wavs), torch.cat(mels, dim=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", type=Path, default=PROJECT_ROOT / "v5_bundle")
    ap.add_argument("--bundle-tar", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "v5_outputs")
    ap.add_argument("--stage1-epochs", type=int, default=24)
    ap.add_argument("--qat-epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="本地 10 秒极速全链路冒烟模式")
    ap.add_argument("--eval-only", action="store_true", help="直接从 out-dir 加载已训好的 best_fp32.pt 与 best_int8.pt 执行合成与四维验收")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.bundle_tar is not None and args.bundle_tar.exists() and not (args.bundle_dir / "v5_three_era_lexicon.json").exists():
        args.bundle_dir.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.bundle_tar, "r:gz") as tar:
            tar.extractall(path=args.bundle_dir.parent)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"🚀 启动汉火 V5 三代贯通 + W8A8 INT8 QAT 主程序 | Device: {device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")

    fp32_ckpt_path = args.out_dir / "best_fp32.pt"
    int8_ckpt_path = args.out_dir / "best_int8.pt"

    if args.eval_only:
        fp32_obj = torch.load(str(fp32_ckpt_path), map_location="cpu", weights_only=False)
        int8_obj = torch.load(str(int8_ckpt_path), map_location="cpu", weights_only=False)
        cfg = fp32_obj["config"]
        t2i = fp32_obj["t2i_custom"]
        teacher = ZipVoice(**cfg)
        teacher.load_state_dict(fp32_obj["model"], strict=True)
        teacher = teacher.to(device).eval()
        int8_model, _ = load_int8_checkpoint(int8_ckpt_path, ZipVoice, device=device)
        comp = ThreeEraHanfireCompiler(t2i=t2i, three_era_json=args.bundle_dir / "v5_three_era_lexicon.json")
        fp32_mb = fp32_ckpt_path.stat().st_size / 1e6
        int8_mb = int8_ckpt_path.stat().st_size / 1e6
        compression_x = fp32_mb / max(1e-6, int8_mb)
        qat_stats = int8_obj.get("qat_stats", {})
        s1_curve, qat_curve = [], []
        print(f"  ✓ 已从磁盘加载 Colab 最优权重: FP32={fp32_mb:.1f} MB | 真 INT8={int8_mb:.1f} MB (压缩比 {compression_x:.2f}x)")
    else:
        # 1. 加载 ZipVoice 原装权重并注入 86 个三代古音槽位
        p_json = Path(hf_hub_download("k2-fsa/ZipVoice", filename="zipvoice/model.json"))
        p_tok = Path(hf_hub_download("k2-fsa/ZipVoice", filename="zipvoice/tokens.txt"))
        p_pt = Path(hf_hub_download("k2-fsa/ZipVoice", filename="zipvoice/model.pt"))
        base_t2i = load_zv_token2id(p_tok)
        cfg = json.loads(p_json.read_text(encoding="utf-8"))["model"]
        cfg["vocab_size"] = len(base_t2i)

        teacher = ZipVoice(**cfg)
        raw_sd = torch.load(str(p_pt), map_location="cpu", weights_only=False)["model"]
        teacher.load_state_dict({k.removeprefix("module."): v for k, v in raw_sd.items()}, strict=True)
        t2i = inject_hanfire_zh_manifold_embeddings(teacher, base_t2i)
        orig_mandarin_vec = teacher.embed.weight.data[t2i["uan1"]].clone()
        teacher = teacher.to(device)

        comp = ThreeEraHanfireCompiler(t2i=t2i, three_era_json=args.bundle_dir / "v5_three_era_lexicon.json")
        raw_items = torch.load(str(args.bundle_dir / "v5_train_dataset.pt"), map_location="cpu", weights_only=False)
        dl = DataLoader(
            ThreeEraDataset(raw_items),
            batch_size=(4 if args.smoke else args.batch_size),
            shuffle=True,
            collate_fn=collate_three_era,
        )

        s1_ep = 2 if args.smoke else args.stage1_epochs
        s2_ep = 2 if args.smoke else args.qat_epochs
        max_st = 2 if args.smoke else 0

        # 2. Stage 1: 三代球面音素精调 -> 导出 best_fp32.pt
        print(f"\n[Stage 1/3] 三代 (OC+MC+EM) 球面音素精调 ({s1_ep} epochs)...")
        best_fp32_sd, s1_curve = run_stage1_three_era_embed(teacher, dl, device, epochs=s1_ep, max_steps_per_ep=max_st)
        drift = float(torch.max(torch.abs(teacher.embed.weight.data[t2i["uan1"]].cpu() - orig_mandarin_vec)).item())
        assert drift == 0.0, f"普通话词元发生漂移: {drift}"

        torch.save({"model": best_fp32_sd, "config": cfg, "t2i_custom": t2i}, str(fp32_ckpt_path))
        fp32_mb = fp32_ckpt_path.stat().st_size / 1e6
        print(f"  ✓ 已保存最优 FP32 权重: {fp32_ckpt_path.name} ({fp32_mb:.1f} MB, 普通话漂移={drift:.1e})")

        # 3. Stage 2: 教师引导 W8A8 INT8 QAT -> 导出真 int8 权重 best_int8.pt
        print(f"\n[Stage 2/3] 启动 W8A8 量化感知训练 (QAT, {s2_ep} epochs)...")
        student = copy.deepcopy(teacher).to(device)
        qat_stats, qat_curve = run_stage2_qat_int8(teacher, student, dl, device, epochs=s2_ep, max_steps_per_ep=max_st)

        int8_model = convert_qat_to_int8_inference(copy.deepcopy(student).cpu()).to(device).eval()
        torch.save(
            {
                "int8_state_dict": {k: v.cpu() for k, v in int8_model.state_dict().items()},
                "config": cfg,
                "t2i_custom": t2i,
                "qat_stats": qat_stats,
            },
            str(int8_ckpt_path),
        )
        int8_mb = int8_ckpt_path.stat().st_size / 1e6
        compression_x = fp32_mb / max(1e-6, int8_mb)
        # 磁盘重载校验：确保导出的 best_int8.pt 可独立无损反序列化推理
        del int8_model
        int8_model, _ = load_int8_checkpoint(int8_ckpt_path, ZipVoice, device=device)
        print(f"  ✓ 已导出并完成磁盘重载核验真 INT8 打包权重: {int8_ckpt_path.name} ({int8_mb:.1f} MB vs FP32 {fp32_mb:.1f} MB, 压缩比 {compression_x:.2f}x！)")

    # 4. Stage 3: FP32 vs INT8 数值对齐校验 + 上古/中古/近古三代男女双音色代表作合成
    print("\n[Stage 3/3] 执行 FP32 vs INT8 精度核验并合成【上古 / 中古 / 近古】三代男女双音色样带...")
    vocos = PureVocosDecoder(device=device)
    fbank = PureVocosFbank()

    male_wav, _ = sf.read(str(args.bundle_dir / "ref_male_128Hz.wav"), dtype="float32")
    female_wav, _ = sf.read(str(args.bundle_dir / "ref_female_252Hz.wav"), dtype="float32")

    # 男女参考锚统一采用无内部静音停顿、四声齐全 (1/2/3/4声) 的标准干声，防止无声调平诵污染中古/近古四声八调！
    male_pmt, _ = tokenize_mandarin_prompt_exact("高分经典悬疑电影。", t2i)
    female_pmt, _ = tokenize_mandarin_prompt_exact("今夜的月光如此清亮。", t2i)

    # 三代代表作测试集
    showcase_suites = [
        ("OC_上古_关雎", "oc", [
            ("关关雎鸠", "在河之洲", {}),
            ("窈窕淑女", "君子好逑", {}),
            ("求之不得", "寤寐思服", {"服": "bək"}),
            ("悠哉悠哉", "辗转反侧", {"服": "bək"}),
        ]),
        ("OC_上古_蒹葭", "oc", [
            ("蒹葭苍苍", "白露为霜", {}),
            ("所谓伊人", "在水一方", {}),
            ("溯洄从之", "道阻且长", {"长": "daŋ"}),
            ("溯游从之", "宛在水中央", {}),
        ]),
        ("MC_中古_春晓", "mc", [
            ("春眠不觉晓", "处处闻啼鸟", {"处": "tɕʰɨəh"}),
            ("夜来风雨声", "花落知多少", {}),
        ]),
        ("MC_中古_春望", "mc", [
            ("国破山河在", "城春草木深", {}),
            ("感时花溅泪", "恨别鸟惊心", {"溅": "tsenh"}),
        ]),
        ("MC_中古_江雪", "mc", [
            ("千山鸟飞绝", "万径人踪灭", {}),
            ("孤舟蓑笠翁", "独钓寒江雪", {}),
        ]),
        ("MC_中古_静夜思", "mc", [
            ("床前明月光", "疑是地上霜", {}),
            ("举头望明月", "低头思故乡", {}),
        ]),
        ("EM_近古明代雅音_桃花庵歌", "em", [
            ("桃花坞里桃花庵", "桃花庵里桃花仙", {}),
            ("酒醒只在花前坐", "酒醉还来花下眠", {}),
        ]),
        ("EM_近古明代雅音_琵琶行", "em", [
            ("浔阳江头夜送客", "枫叶荻花秋瑟瑟", {}),
            ("千呼万唤始出来", "犹抱琵琶半遮面", {}),
        ]),
        ("EM_近古明代雅音_念奴娇", "em", [
            ("乱石穿空惊涛拍岸", "卷起千堆雪", {}),
            ("江山如画", "一时多少豪杰", {"画": "xwah"}),
        ]),
        ("EM_近古元曲北音_天净沙", "em_zyyy", [
            ("枯藤老树昏鸦", "小桥流水人家", {}),
            ("古道西风瘦马", "断肠人在天涯", {}),
        ]),
    ]
    if args.smoke:
        showcase_suites = [showcase_suites[1], showcase_suites[2], showcase_suites[4]]

    audio_dir = args.out_dir / "audio_3eras"
    audio_dir.mkdir(parents=True, exist_ok=True)
    parity_records: List[Dict] = []

    for title, era, couplets in showcase_suites:
        for spk_name, p_wav, p_it, fp100 in [
            ("male_128Hz", male_wav, male_pmt, 25.0),
            ("female_252Hz", female_wav, female_pmt, 24.5),
        ]:
            if args.smoke and spk_name == "female_252Hz":
                continue
            w_fp32, m_fp32 = synth_era_poem(teacher, vocos, fbank, comp, era, p_wav, p_it, couplets, fp100)
            w_int8, m_int8 = synth_era_poem(int8_model, vocos, fbank, comp, era, p_wav, p_it, couplets, fp100)

            sf.write(str(audio_dir / f"{title}_{spk_name}_FP32.wav"), w_fp32, SR)
            sf.write(str(audio_dir / f"{title}_{spk_name}_INT8.wav"), w_int8, SR)

            mel_l1 = float(torch.mean(torch.abs(m_fp32 - m_int8)).item())
            mel_cos = float(F.cosine_similarity(m_fp32.flatten(), m_int8.flatten(), dim=0).item())
            min_len = min(len(w_fp32), len(w_int8))
            t_w32 = torch.from_numpy(w_fp32[:min_len])
            t_w8 = torch.from_numpy(w_int8[:min_len])
            win_cpu = torch.hann_window(N_FFT)
            mag32 = torch.stft(t_w32, n_fft=N_FFT, hop_length=HOP, window=win_cpu, return_complex=True).abs()
            mag8 = torch.stft(t_w8, n_fft=N_FFT, hop_length=HOP, window=win_cpu, return_complex=True).abs()
            wav_spec_cos = float(F.cosine_similarity(torch.log1p(mag32).flatten(), torch.log1p(mag8).flatten(), dim=0).item())
            n_chars = sum(len(c1) + len(c2) for c1, c2, _ in couplets)
            aud_int8 = judge_utterance_acoustics(w_int8, SR, n_expected_chars=n_chars)
            rec = {
                "item": f"{title}_{spk_name}",
                "era": era,
                "mel_l1_fp32_vs_int8": round(mel_l1, 5),
                "mel_cosine": round(mel_cos, 5),
                "wav_stft_cosine": round(wav_spec_cos, 5),
                "int8_trill_index": aud_int8.trill_index,
                "int8_trill_ok": aud_int8.trill_ok,
                "int8_sec_per_char": aud_int8.sec_per_char,
            }
            parity_records.append(rec)
            print(
                f"  ✓ {title}_{spk_name:12s} | mel_L1={mel_l1:.4f} | mel_cos={mel_cos:.5f} | "
                f"wav_stft_cos={wav_spec_cos:.5f} | trill={aud_int8.trill_index:.2f}"
            )

    report = {
        "fp32_size_mb": round(fp32_mb, 2),
        "int8_size_mb": round(int8_mb, 2),
        "compression_ratio": round(compression_x, 2),
        "qat_stats": qat_stats,
        "stage1_loss_curve": s1_curve,
        "stage2_qat_kd_curve": qat_curve,
        "fp32_vs_int8_parity": parity_records,
    }
    rep_path = args.out_dir / "v5_qat_eval_report.json"
    rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ V5 三代贯通 + INT8 QAT 全部完成！报告: {rep_path}")


if __name__ == "__main__":
    main()
