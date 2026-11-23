#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
魔改版小米流匹配古音模型：ZipVoiceAncient (zipvoice_ancient.py)

核心改造点（零 k2 / 零 lhotse 外部重依赖，兼容任意 PyTorch 2.x + CUDA / CPU）：
1. 纯 PyTorch 复刻 `PureVocosFbank` (24kHz, 100-mel, n_fft=1024, hop=256)，与官方预训练声学空间逐位对齐；
2. 将原离散词表层 `self.embed` ([vocab, 192]) 替换为 `ArticulatorySlotProjector(42 -> 192)`，
   吃进 [29维口腔发音特征 + 5维单核槽位 + 4维声调轮廓 + 2维音义开合急缓控制 + 2维男女音色基底]；
3. 将原版均分时长的 `prepare_avg_tokens_durations` 升级为 `allocate_mononuclear_durations`：
   - 介音滑音 (j/w/ɥ) 硬锁 <= 3帧 (32ms)，绝不裂变为第二元音；
   - 入声唯闭塞尾 (-p/-t/-k/-ʔ) 硬锁 <= 4帧 (42ms)，绝不拖尾吐出央元音；
   - 剩余时长精准归集至唯一韵腹 (NUCLEUS)，物理锁死“一字一音”！
4. 支持无损加载 HuggingFace `k2-fsa/ZipVoice` 123M 预训练权重 (`embed.weight` 自动剔除，其余层 100% 严丝合缝)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
from scipy import signal as sp_signal

# 挂载本地 vendor/ZipVoice 源码路径
VENDOR_ZIPVOICE = Path(__file__).resolve().parent.parent / "vendor" / "ZipVoice"
if VENDOR_ZIPVOICE.exists() and str(VENDOR_ZIPVOICE) not in sys.path:
    sys.path.insert(0, str(VENDOR_ZIPVOICE))

from zipvoice.models.modules.solver import EulerSolver
from zipvoice.models.modules.zipformer import TTSZipformer

from .phonology import TOTAL_FEAT_DIM, SLOT_NUCLEUS, SLOT_PAUSE


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, int(lengths.max().item()))
    seq_range = torch.arange(0, max_len, device=lengths.device)
    return seq_range.unsqueeze(0).expand(lengths.size(0), max_len) >= lengths.unsqueeze(-1)


def condition_time_mask(
    features_lens: torch.Tensor,
    mask_percent: Tuple[float, float],
    max_len: int = 0,
) -> torch.Tensor:
    mask_size = (
        torch.zeros_like(features_lens, dtype=torch.float32).uniform_(*mask_percent)
        * features_lens
    ).to(torch.int64)
    mask_starts = (
        torch.rand_like(mask_size, dtype=torch.float32) * (features_lens - mask_size)
    ).to(torch.int64)
    mask_ends = mask_starts + mask_size
    max_len = max(max_len, int(features_lens.max().item()))
    seq_range = torch.arange(0, max_len, device=features_lens.device)
    return (seq_range[None, :] >= mask_starts[:, None]) & (seq_range[None, :] < mask_ends[:, None])


def _build_htk_mel_fbanks(n_freqs: int, n_mels: int, sample_rate: int) -> torch.Tensor:
    """精确复刻 torchaudio MelSpectrogram 默认 HTK 三角滤波器组 (norm=None, mel_scale='htk')，零外部依赖。"""
    f_min, f_max = 0.0, float(sample_rate) / 2.0
    m_min = 2595.0 * math.log10(1.0 + f_min / 700.0)
    m_max = 2595.0 * math.log10(1.0 + f_max / 700.0)
    m_pts = torch.linspace(m_min, m_max, n_mels + 2, dtype=torch.float32)
    f_pts = 700.0 * (10.0 ** (m_pts / 2595.0) - 1.0)
    all_freqs = torch.linspace(0.0, f_max, n_freqs, dtype=torch.float32)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_mels + 2)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    fb = torch.maximum(torch.zeros(1, dtype=torch.float32), torch.minimum(down_slopes, up_slopes))
    return fb  # (n_freqs, n_mels)


class PureVocosFbank(nn.Module):
    """纯 PyTorch STFT 实现的官方 ZipVoice 24kHz 100-dim Log-Mel 提取器（免除 torchaudio/lhotse 版本破坏）。"""

    sampling_rate: int = 24000
    n_mels: int = 100
    n_fft: int = 1024
    hop_length: int = 256

    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)
        self.register_buffer(
            "mel_fb",
            _build_htk_mel_fbanks(self.n_fft // 2 + 1, self.n_mels, self.sampling_rate),
            persistent=False,
        )

    @torch.no_grad()
    def extract(self, wav: torch.Tensor | np.ndarray, sr: int = 24000) -> torch.Tensor:
        """输入 1D/2D 波形 -> 返回 (T_frames, 100) float32 Log-Mel 特征。"""
        if torch.is_tensor(wav):
            y_np = wav.detach().cpu().float().numpy()
        else:
            y_np = np.asarray(wav, dtype=np.float32)

        if y_np.ndim == 2:
            # 兼容 (C, N) 与 (N, C)
            if y_np.shape[0] <= 2 and y_np.shape[1] > 2:
                y_np = y_np.mean(axis=0)
            else:
                y_np = y_np.mean(axis=1)

        if sr != self.sampling_rate:
            g = math.gcd(int(sr), self.sampling_rate)
            y_np = sp_signal.resample_poly(y_np.astype(np.float64), self.sampling_rate // g, int(sr) // g).astype(np.float32)

        w_t = torch.from_numpy(y_np).unsqueeze(0).to(self.mel_fb.device)  # (1, N)
        spec_c = torch.stft(
            w_t,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        mag = spec_c.abs().squeeze(0).transpose(0, 1)  # (T, n_freqs)
        mel = torch.matmul(mag, self.mel_fb).clamp(min=1e-7).log()  # (T, 100)

        expected_frames = max(1, int(round(w_t.size(1) / self.hop_length)))
        if mel.size(0) > expected_frames:
            mel = mel[:expected_frames]
        elif mel.size(0) < expected_frames:
            pad_n = expected_frames - mel.size(0)
            mel = torch.cat([mel, mel[-1:].expand(pad_n, -1)], dim=0)
        return mel


class ArticulatorySlotProjector(nn.Module):
    """将 42 维口腔物理/音义/声调向量映射至 Zipformer 192 维输入潜空间。"""

    def __init__(self, in_dim: int = TOTAL_FEAT_DIM, out_dim: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, slot_feats: torch.Tensor) -> torch.Tensor:
        return self.net(slot_feats)


def allocate_mononuclear_durations(
    features_lens: torch.Tensor,     # (B,) int64
    slot_lens: torch.Tensor,         # (B,) int64
    dur_weights: torch.Tensor,       # (B, S) float32
    max_frames: torch.Tensor,        # (B, S) int64
    slot_types: torch.Tensor,        # (B, S) int64 (0:PAUSE, 1:ONSET, 2:MEDIAL, 3:NUCLEUS, 4:CODA)
    max_total_frames: int,
) -> torch.Tensor:
    """
    物理单音节核帧对齐器：返回 `slot_index` 张量 (B, max_total_frames)，供 `torch.gather` 展开条件。
    保证：
      1. 介音 (MEDIAL) 帧数 <= max_frames (3帧)
      2. 入声塞尾 (CODA checked) 帧数 <= max_frames (4帧)
      3. 所有槽位总帧数严格等于 features_lens[b]
    """
    device = features_lens.device
    B = features_lens.size(0)
    out_indices = torch.zeros((B, max_total_frames), dtype=torch.int64, device=device)

    f_lens_cpu = features_lens.detach().cpu().tolist()
    s_lens_cpu = slot_lens.detach().cpu().tolist()
    w_cpu = dur_weights.detach().cpu().numpy()
    cap_cpu = max_frames.detach().cpu().numpy()
    st_cpu = slot_types.detach().cpu().numpy()

    for b in range(B):
        T = max(1, int(f_lens_cpu[b]))
        S = max(1, int(s_lens_cpu[b]))
        if T <= S:
            # 极端短帧（如冒烟测试短帧）：均匀线性映射
            idx = np.linspace(0, S - 1, T).astype(np.int64)
            out_indices[b, :T] = torch.from_numpy(idx).to(device)
            if T < max_total_frames:
                out_indices[b, T:] = S - 1
            continue

        w = np.maximum(w_cpu[b, :S], 0.05)
        caps = np.maximum(cap_cpu[b, :S], 1)
        st = st_cpu[b, :S]

        raw_d = np.maximum(1, np.round(T * (w / np.sum(w))).astype(np.int64))
        # 先对非韵腹槽位施加硬上限锁（掐断双音节源头！）
        non_nuc = st != SLOT_NUCLEUS
        d = np.where(non_nuc, np.minimum(raw_d, caps), raw_d)

        diff = T - int(np.sum(d))
        if diff > 0:
            # 富余帧优先分给唯一主元音核 NUCLEUS，若无则分给 PAUSE
            nuc_pos = np.where(st == SLOT_NUCLEUS)[0]
            targets = nuc_pos if len(nuc_pos) > 0 else np.where(st == SLOT_PAUSE)[0]
            if len(targets) == 0:
                targets = np.arange(S)
            tw = w[targets]
            add = np.floor(diff * (tw / np.sum(tw))).astype(np.int64)
            d[targets] += add
            rem = T - int(np.sum(d))
            for k in range(rem):
                d[targets[k % len(targets)]] += 1
        elif diff < 0:
            # 超出帧从最宽裕的槽位（保留至少 1 帧）均匀扣回
            need = -diff
            for _ in range(need):
                order = np.argsort(-d)
                for idx_max in order:
                    if d[idx_max] > 1:
                        d[idx_max] -= 1
                        break

        expanded = np.repeat(np.arange(S, dtype=np.int64), d)
        out_indices[b, :T] = torch.from_numpy(expanded[:T]).to(device)
        if T < max_total_frames:
            out_indices[b, T:] = S - 1

    return out_indices


class ZipVoiceAncient(nn.Module):
    """基于 Zipformer OT-CFM 的上古/中古/近古统一发音特征语音合成模型。"""

    def __init__(
        self,
        fm_decoder_downsampling_factor: List[int] = [1, 2, 4, 2, 1],
        fm_decoder_num_layers: List[int] = [2, 2, 4, 4, 4],
        fm_decoder_cnn_module_kernel: List[int] = [31, 15, 7, 15, 31],
        fm_decoder_feedforward_dim: int = 1536,
        fm_decoder_num_heads: int = 4,
        fm_decoder_dim: int = 512,
        text_encoder_num_layers: int = 4,
        text_encoder_feedforward_dim: int = 512,
        text_encoder_cnn_module_kernel: int = 9,
        text_encoder_num_heads: int = 4,
        text_encoder_dim: int = 192,
        time_embed_dim: int = 192,
        text_embed_dim: int = 192,
        query_head_dim: int = 32,
        value_head_dim: int = 12,
        pos_head_dim: int = 4,
        pos_dim: int = 48,
        feat_dim: int = 100,
        slot_feat_dim: int = TOTAL_FEAT_DIM,
        **unused_kwargs,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.text_embed_dim = text_embed_dim
        self.slot_feat_dim = slot_feat_dim

        # 名字保持与官方 ZipVoice 完全一致 (`fm_decoder`, `text_encoder`) 以便直接灌入 123M 预训练权重
        self.fm_decoder = TTSZipformer(
            in_dim=feat_dim * 3,
            out_dim=feat_dim,
            downsampling_factor=fm_decoder_downsampling_factor,
            num_encoder_layers=fm_decoder_num_layers,
            cnn_module_kernel=fm_decoder_cnn_module_kernel,
            encoder_dim=fm_decoder_dim,
            feedforward_dim=fm_decoder_feedforward_dim,
            num_heads=fm_decoder_num_heads,
            query_head_dim=query_head_dim,
            pos_head_dim=pos_head_dim,
            value_head_dim=value_head_dim,
            pos_dim=pos_dim,
            use_time_embed=True,
            time_embed_dim=time_embed_dim,
        )

        self.text_encoder = TTSZipformer(
            in_dim=text_embed_dim,
            out_dim=feat_dim,
            downsampling_factor=1,
            num_encoder_layers=text_encoder_num_layers,
            cnn_module_kernel=text_encoder_cnn_module_kernel,
            encoder_dim=text_encoder_dim,
            feedforward_dim=text_encoder_feedforward_dim,
            num_heads=text_encoder_num_heads,
            query_head_dim=query_head_dim,
            pos_head_dim=pos_head_dim,
            value_head_dim=value_head_dim,
            pos_dim=pos_dim,
            use_time_embed=False,
        )

        # 替换离散 self.embed 为连续发音特征映射层
        self.articulatory_proj = ArticulatorySlotProjector(slot_feat_dim, text_embed_dim)
        self.solver = EulerSolver(self, func_name="forward_fm_decoder")

    def forward_fm_decoder(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        text_condition: torch.Tensor,
        speech_condition: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        guidance_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        xt_cat = torch.cat([xt, text_condition, speech_condition], dim=2)
        while t.dim() > 1 and t.size(-1) == 1:
            t = t.squeeze(-1)
        if t.dim() == 0:
            t = t.repeat(xt_cat.shape[0])

        if guidance_scale is not None:
            while guidance_scale.dim() > 1 and guidance_scale.size(-1) == 1:
                guidance_scale = guidance_scale.squeeze(-1)
            if guidance_scale.dim() == 0:
                guidance_scale = guidance_scale.repeat(xt_cat.shape[0])
            return self.fm_decoder(x=xt_cat, t=t, padding_mask=padding_mask, guidance_scale=guidance_scale)
        return self.fm_decoder(x=xt_cat, t=t, padding_mask=padding_mask)

    def encode_slots(
        self,
        slot_feats: torch.Tensor,     # (B, S, 42)
        slot_lens: torch.Tensor,      # (B,)
    ) -> torch.Tensor:
        slot_pad_mask = make_pad_mask(slot_lens, max_len=slot_feats.size(1))
        x = self.articulatory_proj(slot_feats)
        return self.text_encoder(x=x, t=None, padding_mask=slot_pad_mask)  # (B, S, feat_dim=100)

    @staticmethod
    def _extract_slot_types(slot_feats: torch.Tensor) -> torch.Tensor:
        # 29..34 是 5维槽位 One-Hot
        return torch.argmax(slot_feats[:, :, 29:34], dim=-1)

    def build_frame_condition(
        self,
        slot_feats: torch.Tensor,     # (B, S, 42)
        slot_lens: torch.Tensor,      # (B,)
        dur_weights: torch.Tensor,    # (B, S)
        max_frames: torch.Tensor,     # (B, S)
        features_lens: torch.Tensor,  # (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.encode_slots(slot_feats, slot_lens)  # (B, S, F)
        num_frames = int(features_lens.max().item())
        padding_mask = make_pad_mask(features_lens, max_len=num_frames)
        slot_types = self._extract_slot_types(slot_feats)
        slot_idx = allocate_mononuclear_durations(
            features_lens=features_lens,
            slot_lens=slot_lens,
            dur_weights=dur_weights,
            max_frames=max_frames,
            slot_types=slot_types,
            max_total_frames=num_frames,
        )
        text_cond = torch.gather(
            enc,
            dim=1,
            index=slot_idx.unsqueeze(-1).expand(enc.size(0), num_frames, enc.size(-1)),
        )
        return text_cond, padding_mask

    def forward(
        self,
        slot_feats: torch.Tensor,     # (B, S, 42)
        slot_lens: torch.Tensor,      # (B,)
        dur_weights: torch.Tensor,    # (B, S)
        max_frames: torch.Tensor,     # (B, S)
        features: torch.Tensor,       # (B, T, 100) scaled log-mel
        features_lens: torch.Tensor,  # (B,)
        condition_drop_ratio: float = 0.1,
        t_Override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """OT-CFM 训练前向计算流匹配损失。"""
        B, T, F = features.shape
        device = features.device

        text_condition, padding_mask = self.build_frame_condition(
            slot_feats=slot_feats,
            slot_lens=slot_lens,
            dur_weights=dur_weights,
            max_frames=max_frames,
            features_lens=features_lens,
        )

        speech_condition_mask = condition_time_mask(
            features_lens=features_lens,
            mask_percent=(0.7, 1.0),
            max_len=T,
        )
        speech_condition = torch.where(speech_condition_mask.unsqueeze(-1), 0.0, features)

        if condition_drop_ratio > 0.0 and self.training:
            drop_mask = (torch.rand(B, 1, 1, device=device) > condition_drop_ratio).to(text_condition.dtype)
            text_condition = text_condition * drop_mask

        noise = torch.randn_like(features)
        t = t_Override if t_Override is not None else torch.rand(B, 1, 1, device=device, dtype=features.dtype)
        xt = features * t + noise * (1.0 - t)
        ut = features - noise

        vt = self.forward_fm_decoder(
            t=t,
            xt=xt,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
        )

        loss_mask = speech_condition_mask & (~padding_mask)
        if not loss_mask.any():
            loss_mask = ~padding_mask
        return torch.mean((vt[loss_mask] - ut[loss_mask]) ** 2)

    @torch.no_grad()
    def synthesize_features(
        self,
        target_slot_feats: torch.Tensor,      # (1, S_tgt, 42)
        target_dur_weights: torch.Tensor,     # (1, S_tgt)
        target_max_frames: torch.Tensor,      # (1, S_tgt)
        prompt_slot_feats: torch.Tensor,      # (1, S_pmt, 42)
        prompt_dur_weights: torch.Tensor,     # (1, S_pmt)
        prompt_max_frames: torch.Tensor,      # (1, S_pmt)
        prompt_features: torch.Tensor,        # (1, T_pmt, 100) scaled log-mel
        speed: float = 1.0,
        num_step: int = 16,
        guidance_scale: float = 1.0,
        t_shift: float = 0.5,
        frames_per_weight_unit: float = 13.5, # ~144ms per 1.0 weight unit (@93.75Hz), 单字约 30-42 帧 (0.32-0.45s)
    ) -> torch.Tensor:
        """拼接参考干声 Prompt 与目标古音序列，通过 Euler ODE 求解生成目标段 100-dim Log-Mel。"""
        device = prompt_features.device
        T_pmt = prompt_features.size(1)
        total_w = float(target_dur_weights.sum().item())
        T_tgt = max(12, int(round(total_w * frames_per_weight_unit / max(speed, 0.3))))
        T_all = T_pmt + T_tgt

        cat_feats = torch.cat([prompt_slot_feats, target_slot_feats], dim=1)
        cat_weights = torch.cat([prompt_dur_weights, target_dur_weights], dim=1)
        cat_caps = torch.cat([prompt_max_frames, target_max_frames], dim=1)
        cat_slot_lens = torch.tensor([cat_feats.size(1)], dtype=torch.int64, device=device)
        all_feat_lens = torch.tensor([T_all], dtype=torch.int64, device=device)

        # 分别对齐 prompt 区与 target 区，确保边界精确落在 T_pmt！
        enc_all = self.encode_slots(cat_feats, cat_slot_lens)  # (1, S_pmt + S_tgt, F)
        S_pmt = prompt_slot_feats.size(1)
        enc_pmt, enc_tgt = enc_all[:, :S_pmt, :], enc_all[:, S_pmt:, :]

        idx_pmt = allocate_mononuclear_durations(
            features_lens=torch.tensor([T_pmt], dtype=torch.int64, device=device),
            slot_lens=torch.tensor([S_pmt], dtype=torch.int64, device=device),
            dur_weights=prompt_dur_weights,
            max_frames=prompt_max_frames,
            slot_types=self._extract_slot_types(prompt_slot_feats),
            max_total_frames=T_pmt,
        )
        idx_tgt = allocate_mononuclear_durations(
            features_lens=torch.tensor([T_tgt], dtype=torch.int64, device=device),
            slot_lens=torch.tensor([target_slot_feats.size(1)], dtype=torch.int64, device=device),
            dur_weights=target_dur_weights,
            max_frames=target_max_frames,
            slot_types=self._extract_slot_types(target_slot_feats),
            max_total_frames=T_tgt,
        )
        cond_pmt = torch.gather(enc_pmt, 1, idx_pmt.unsqueeze(-1).expand(1, T_pmt, enc_pmt.size(-1)))
        cond_tgt = torch.gather(enc_tgt, 1, idx_tgt.unsqueeze(-1).expand(1, T_tgt, enc_tgt.size(-1)))
        text_condition = torch.cat([cond_pmt, cond_tgt], dim=1)  # (1, T_all, F)
        padding_mask = make_pad_mask(all_feat_lens, max_len=T_all)

        speech_condition = torch.nn.functional.pad(prompt_features, (0, 0, 0, T_tgt), value=0.0)
        x0 = torch.randn(1, T_all, self.feat_dim, device=device, dtype=prompt_features.dtype)

        x1 = self.solver.sample(
            x=x0,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
            num_step=num_step,
            guidance_scale=guidance_scale,
            t_shift=t_shift,
        )
        return x1[:, T_pmt:, :]  # (1, T_tgt, 100)


def build_model_from_pretrained(
    model_json_path: Optional[str | Path] = None,
    pretrained_pt_path: Optional[str | Path] = None,
    tiny_smoke: bool = False,
) -> Tuple[ZipVoiceAncient, Dict]:
    """构建 ZipVoiceAncient，可选注入官方 123M 预训练权重或使用极轻量配置做本地秒级语法冒烟。"""
    if tiny_smoke:
        model = ZipVoiceAncient(
            fm_decoder_downsampling_factor=[1, 2, 1],
            fm_decoder_num_layers=[1, 1, 1],
            fm_decoder_cnn_module_kernel=[7, 7, 7],
            fm_decoder_feedforward_dim=128,
            fm_decoder_num_heads=2,
            fm_decoder_dim=64,
            text_encoder_num_layers=1,
            text_encoder_feedforward_dim=128,
            text_encoder_cnn_module_kernel=5,
            text_encoder_num_heads=2,
            text_encoder_dim=64,
            time_embed_dim=64,
            text_embed_dim=64,
            query_head_dim=16,
            value_head_dim=8,
            pos_head_dim=4,
            pos_dim=16,
            feat_dim=100,
        )
        return model, {"mode": "tiny_smoke", "matched_keys": 0}

    cfg = {}
    if model_json_path and Path(model_json_path).exists():
        raw_cfg = json.loads(Path(model_json_path).read_text(encoding="utf-8"))
        cfg = raw_cfg.get("model", raw_cfg)

    model = ZipVoiceAncient(**cfg)
    info = {"mode": "full_123m", "matched_keys": 0, "missing_keys": [], "dropped_keys": []}

    if pretrained_pt_path and Path(pretrained_pt_path).exists():
        pt_p = Path(pretrained_pt_path)
        if pt_p.suffix == ".safetensors":
            import safetensors.torch
            sd = safetensors.torch.load_file(str(pt_p), device="cpu")
        else:
            raw = torch.load(str(pt_p), map_location="cpu", weights_only=False)
            sd = raw.get("model", raw.get("state_dict", raw))

        cleaned_sd = {}
        for k, v in sd.items():
            k_clean = k[len("module."):] if k.startswith("module.") else k
            if k_clean.startswith("embed."):
                info["dropped_keys"].append(k_clean)
                continue
            cleaned_sd[k_clean] = v

        res = model.load_state_dict(cleaned_sd, strict=False)
        info["matched_keys"] = len(cleaned_sd) - len(res.unexpected_keys)
        info["missing_keys"] = list(res.missing_keys)
        info["unexpected_keys"] = list(res.unexpected_keys)

    return model, info
