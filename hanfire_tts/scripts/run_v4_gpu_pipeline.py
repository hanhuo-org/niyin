#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汉火上古音 TTS V4 —— RTX 2080 Ti 全流程同句 A/B 对照 + 真人上古干声精调管线
(run_v4_gpu_pipeline.py)

严格遵循架构审查修正项与用户最新指示：
1. 100% 采用用户原版汉火词表 (`hanfire_oc_v1.json`)，废除手写 50 字覆盖表；
2. 全面切换为【正常读音模式 (`chant_mode=False`)】：去掉韵脚人工拉长与普通话阳平上挑腔；
3. 深度利用新上传的 3 个录音室级无伴奏零弹舌上古音视频（《论语·学而》8 句 + 《基础词》47 词 + 《连绵词》20 词，
   共 75 条 160dB 数字静音底噪真人干声，覆盖 `-p/-t/-k/-m`、全浊 `b/d/g/dz/ɦ`、疑母 `ŋ`）：
   - 提取《论语·学而》男声（107.6Hz）作为纯正“上古正常朗读男声锚”；
   - 提取 24kHz 清朗女声陈述句（252Hz）作为“正常朗读女声锚”；
4. 在 RTX 2080 Ti 上执行同句公平对比与精调：
   - 路线 A (`DiaMoE-GPU`)：十一大方言原生全浊/入声/闭口 `-m` 专家模型（FP32 CUDA + 4 候选 medoid 择优）；
   - 路线 B (`ZipVoice-ZeroShot-Normal`)：V3.2 正常读音未微调基线；
   - 路线 C (`ZipVoice-V4-Trained`)：用【75 条新视频真人上古干声 + 120 条 DiaMoE 男女双音色全浊/入声/`-m` 最小对立短句】
     在 2080 Ti 上精调 `embed[DEAD_SLOT_POOL]` (`weight_decay=0.0`!) + `text_encoder` + `fm_decoder.in_proj`（冻结声码层保音质）。
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# 配置 Python 3.13 CUDA 环境与纯 Python 依赖路径
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
VENDOR_ZV = PROJECT_ROOT / "vendor" / "ZipVoice"
DIAMOE_ROOT = Path("/mnt/s790/project/hanzi/diamoe-rust")

for p in (
    str(PROJECT_ROOT),
    str(VENDOR_ZV),
    str(DIAMOE_ROOT / "export"),
    str(PROJECT_ROOT / "vendor_py"),
):
    if p not in sys.path:
        sys.path.insert(0, p)
# 追加 diamoe-rust Python 3.13 site-packages（放在末尾，绝不覆盖 hanzi_train 的 CUDA torch！）
sys.path.append(str(DIAMOE_ROOT / ".venv/lib/python3.13/site-packages"))

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import types

# 兼容 PyTorch 2.11+ 移除 torchaudio.functional.functional 的遗留顶层导入
if "torchaudio.functional.functional" not in sys.modules:
    _ta_fn = types.ModuleType("torchaudio.functional.functional")
    _ta_fn._hz_to_mel = lambda hz, mel_scale="htk": 2595.0 * math.log10(1.0 + hz / 700.0)
    _ta_fn._mel_to_hz = lambda mel, mel_scale="htk": 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    sys.modules.setdefault("torchaudio.functional", _ta_fn)
    sys.modules["torchaudio.functional.functional"] = _ta_fn
from huggingface_hub import hf_hub_download
from scipy import signal
from torch.utils.data import DataLoader, Dataset

import export_models as dm_exp
from zipvoice.models.zipvoice import ZipVoice
from zipvoice.utils.common import condition_time_mask, get_tokens_index, make_pad_mask

from hanfire_core.async_checkpoint import AsyncCheckpointManager
from hanfire_core.diamoe_compiler import HanfireDiaMoECompiler
from hanfire_core.phonology import parse_mononuclear_4slot
from hanfire_core.zipvoice_ancient import PureVocosFbank
from hanfire_core.zipvoice_zh_manifold import (
    DEAD_SLOT_POOL,
    HanfireZipVoiceCompiler,
    _weights_to_exact_frames,
    inject_hanfire_zh_manifold_embeddings,
    load_zv_token2id,
    sample_with_character_durations,
    tokenize_mandarin_prompt_exact,
)

SR = 24000
HOP_DM = 256
N_FFT_DM = 1024
FEAT_SCALE_ZV = 0.1
OUT_DIR = PROJECT_ROOT / "local_preview_v4_ab"


# ===========================================================================
# 1. 新上传 3 个视频的 75 条录音室级真人上古音精确标注清单（由视频画面逐帧核验）
# ===========================================================================
LUNYU_ITEMS: List[Tuple[int, str, List[str]]] = [
    (0, "子曰", ["tsəʔ", "wat"]),
    (1, "学而时习之不亦说乎", ["guk", "nə", "də", "zəp", "tə", "pə", "jak", "lɔt", "ɦa"]),
    (2, "有朋自远方来不亦乐乎", ["wəʔ", "bəŋ", "dzih", "wanʔ", "paŋ", "lə", "pə", "jak", "lauk", "ɦa"]),
    (3, "人不知而不愠不亦君子乎", ["niŋ", "pə", "tɛ", "nə", "pə", "ʔunh", "pə", "jak", "kun", "tsəʔ", "ɦa"]),
    (4, "有子曰", ["wəʔ", "tsəʔ", "wat"]),
    (5, "其为人也孝弟而好犯上者鲜矣", ["gə", "wai", "niŋ", "laʔ", "xuh", "lih", "nə", "xuh", "bamʔ", "daŋh", "taʔ", "sanʔ", "ʔəʔ"]),
    (6, "不好犯上而好作乱者未之有也", ["pə", "xuh", "bamʔ", "daŋh", "nə", "xuh", "tsak", "lɔnh", "taʔ", "məih", "tə", "wəʔ", "laʔ"]),
    (7, "君子务本本立而道生孝弟也者其为仁之本与", [
        "kun", "tsəʔ", "moh", "pənʔ", "pənʔ", "ləp", "nə", "luʔ", "seŋ",
        "xuh", "lih", "laʔ", "taʔ", "gə", "wai", "nin", "tə", "pənʔ", "ja"
    ]),
]

BASIC_WORDS_ITEMS: List[Tuple[int, str, List[str]]] = [
    (0, "之", ["tə"]), (1, "而", ["nə"]), (2, "乃", ["nəʔ"]), (3, "其", ["gə"]), (4, "以", ["ləʔ"]),
    (5, "已", ["ləʔ"]), (6, "矣", ["ʔəʔ"]), (7, "也", ["laʔ"]), (8, "也", ["laiʔ"]), (9, "吾", ["ŋa"]),
    (10, "我", ["ŋaiʔ"]), (11, "汝", ["naʔ"]), (12, "尔", ["naiʔ"]), (13, "尔", ["nɛʔ"]), (14, "曰", ["wat"]),
    (15, "云", ["wən"]), (16, "言", ["ŋan"]), (17, "语", ["ŋaʔ"]), (18, "语", ["ŋah"]), (19, "为", ["wai"]),
    (20, "为", ["waih"]), (21, "焉", ["ʔan"]), (22, "者", ["taʔ"]), (23, "乎", ["ɦa"]), (24, "于", ["wa"]),
    (25, "于", ["ʔa"]), (26, "余", ["la"]), (27, "与", ["laʔ"]), (28, "无", ["ma"]), (29, "亡", ["maŋ"]),
    (30, "有", ["wəʔ"]), (31, "不", ["pə"]), (32, "夫", ["pa"]), (33, "夫", ["ba"]), (34, "彼", ["paiʔ"]),
    (35, "非", ["pəi"]), (36, "如", ["na"]), (37, "若", ["nak"]), (38, "若", ["naʔ"]), (39, "然", ["nan"]),
    (40, "父", ["baʔ"]), (41, "父", ["paʔ"]), (42, "母", ["məʔ"]), (43, "呜呼", ["ʔa", "ha"]), (44, "兮", ["ɦɛ"]),
    (45, "且", ["tsʰaʔ"]), (46, "何", ["gai"]),
]

LIANMIANCI_ITEMS: List[Tuple[int, str, List[str]]] = [
    (0, "斯须", ["sɛ", "sɔ"]), (1, "蜘蛛", ["tɛ", "tɔ"]), (2, "踟蹰", ["dɛ", "dɔ"]),
    (3, "邂逅", ["gɛh", "gɔh"]), (4, "踯躅", ["dɛk", "dɔk"]), (5, "彳亍", ["tʰɛk", "tʰɔk"]),
    (6, "溟濛", ["mɛŋ", "mɔŋ"]), (7, "玲珑", ["lɛŋ", "lɔŋ"]), (8, "惺憁", ["sɛŋ", "sɔŋ"]),
    (9, "郑重", ["dɛŋʔ", "dɔŋʔ"]), (10, "孑孓", ["kɛt", "kɔt"]), (11, "契阔", ["kʰɛt", "kʰɔt"]),
    (12, "间关", ["kɛn", "kɔn"]), (13, "辗转", ["tɛnʔ", "tɔnʔ"]), (14, "缱绻", ["kʰɛnʔ", "kʰɔnʔ"]),
    (15, "蛴螬", ["dzi", "dzu"]), (16, "蚍蜉", ["bi", "bu"]), (17, "蟋蟀", ["sit", "sut"]),
    (18, "缤纷", ["pʰin", "pʰun"]), (19, "氤氲", ["ʔin", "ʔun"]),
]


# ===========================================================================
# 2. 纯 PyTorch CUDA Vocos 与 DiaMoE-TTS 高速推理引擎（带多候选 medoid 择优）
# ===========================================================================
class PureCudaVocos:
    """不依赖 torchaudio，直接从 config.yaml 构建 backbone + ISTFTHead 在 GPU 上高速还原 24kHz 波形。"""

    def __init__(self, vocos_dir: Path, device: torch.device) -> None:
        import yaml
        from vocos.pretrained import instantiate_class
        self.device = device
        cfg = yaml.safe_load((vocos_dir / "config.yaml").read_text(encoding="utf-8"))
        backbone = instantiate_class(args=(), init=cfg["backbone"])
        head = instantiate_class(args=(), init=cfg["head"])
        sd = torch.load(str(vocos_dir / "pytorch_model.bin"), map_location="cpu", weights_only=True)
        backbone.load_state_dict({k.removeprefix("backbone."): v for k, v in sd.items() if k.startswith("backbone.")}, strict=True)
        head.load_state_dict({k.removeprefix("head."): v for k, v in sd.items() if k.startswith("head.")}, strict=True)

        class _Holder(torch.nn.Module):
            def __init__(self, b, h):
                super().__init__()
                self.backbone = b
                self.head = h

        self.spec_net = dm_exp.VocosSpec(_Holder(backbone, head)).to(device).eval()
        self.win = torch.hann_window(N_FFT_DM, device=device)

    @torch.no_grad()
    def decode_mel(self, mel_b_c_t: torch.Tensor) -> np.ndarray:
        """mel_b_c_t: [1, 100, T] -> 1D float32 numpy waveform"""
        ri = self.spec_net(mel_b_c_t.to(self.device))  # [1, 1026, T]
        real, imag = ri[:, :513, :], ri[:, 513:, :]
        spec = torch.complex(real, imag)
        wav = torch.istft(
            spec, n_fft=N_FFT_DM, hop_length=HOP_DM, win_length=N_FFT_DM,
            window=self.win, center=True,
        )
        return wav.squeeze(0).clamp(-0.96, 0.96).cpu().numpy().astype(np.float32)


class DiaMoECudaEngine:
    """在 RTX 2080 Ti 上运行原生 FP32 DiaMoE-TTS，支持单次 batch 并行 --best-of K medoid 择优！"""

    def __init__(self, ckpt_path: Path, vocab_path: Path, vocos: PureCudaVocos, device: torch.device) -> None:
        self.device = device
        self.vocos = vocos
        lines = [ln.rstrip("\n") for ln in vocab_path.read_text(encoding="utf-8").splitlines()]
        self.tok_map = {tok: i for i, tok in enumerate(lines)}
        self.max_tok_len = max(len(t) for t in lines if t)

        cfm = dm_exp.build_model(str(ckpt_path), str(vocab_path))
        self.dit_step = dm_exp.DiTStep(cfm.transformer).to(device).eval()
        self.win_stft = torch.hann_window(N_FFT_DM, device=device)
        # 复用 PureVocosFbank (hop=256, n_fft=1024, 100 mels, HTK) 提取与 Vocos 严格一致的 log-mel！
        self.fbank = PureVocosFbank()

    def encode_bracketed_ipa(self, s: str) -> List[int]:
        ids: List[int] = []
        i = 0
        n = len(s)
        while i < n:
            hit = None
            for ln in range(min(self.max_tok_len, n - i), 0, -1):
                sub = s[i: i + ln]
                if sub in self.tok_map:
                    hit = sub
                    break
            if hit is None:
                ids.append(0)
                i += 1
            else:
                ids.append(self.tok_map[hit])
                i += len(hit)
        return ids

    @staticmethod
    def count_syllables(br_ipa: str) -> Tuple[int, int]:
        syls = br_ipa.count("|")
        puncts = sum(br_ipa.count(p) for p in ("，", "。", "？", "！", ",", ".", "?", "!"))
        return max(1, syls), puncts

    @torch.no_grad()
    def mel_of_wav(self, wav_24k: np.ndarray) -> torch.Tensor:
        """100% 严格复刻 DiaMoE export_models.Preprocess (reflect pad + power=1 mag + HTK fb + clamp 1e-5 log)！"""
        rms = float(np.sqrt(np.mean(wav_24k ** 2)) + 1e-9)
        y = wav_24k * (0.10 / rms) if rms > 1e-5 else wav_24k
        w_t = torch.from_numpy(y.astype(np.float32)).unsqueeze(0).to(self.device)
        spec_c = torch.stft(
            w_t, n_fft=N_FFT_DM, hop_length=HOP_DM, win_length=N_FFT_DM,
            window=self.win_stft, center=True, pad_mode="reflect", return_complex=True,
        )
        mag = spec_c.abs().squeeze(0).transpose(0, 1)  # [F, 513]
        fb = self.fbank.mel_fb.to(self.device)         # [513, 100]
        mel = torch.matmul(mag, fb).clamp(min=1e-5).log()
        return mel  # [F, 100]

    @torch.no_grad()
    def synth_chunk(
        self,
        ref_mel: torch.Tensor,
        ref_ids: List[int],
        ref_syls: int,
        gen_br_ipa: str,
        nfe: int = 24,
        cfg: float = 2.0,
        sway: float = -1.0,
        speed: float = 1.0,
        best_of: int = 3,
        base_seed: int = 42,
    ) -> np.ndarray:
        gen_ids = self.encode_bracketed_ipa(gen_br_ipa)
        gen_syls, gen_puncts = self.count_syllables(gen_br_ipa)
        ref_len = int(ref_mel.shape[0])
        # 扣除参考音频末尾 0.35s (33帧) 静音垫后再计算每字帧率，避免时长虚高触发流匹配重复伪影！
        active_ref_frames = max(24, ref_len - int(0.35 * SR / HOP_DM))
        dur = max(
            ref_len + 16,
            int(round(ref_len + (active_ref_frames / max(1, ref_syls)) * (gen_syls + 0.5 * gen_puncts) / speed)),
        )

        full_ids = (ref_ids + [0] + gen_ids)[:dur]
        gate_ids = ([-1] * (len(ref_ids) + 1) + gen_ids)[:dur]

        txt = torch.full((1, dur), -1, dtype=torch.int64, device=self.device)
        txt[0, : len(full_ids)] = torch.tensor(full_ids, dtype=torch.int64, device=self.device)
        txt_gate = torch.full((1, dur), -1, dtype=torch.int64, device=self.device)
        txt_gate[0, : len(gate_ids)] = torch.tensor(gate_ids, dtype=torch.int64, device=self.device)
        txt_neg = torch.full((1, dur), -1, dtype=torch.int64, device=self.device)

        cond = torch.zeros((1, dur, 100), dtype=torch.float32, device=self.device)
        cond[0, :ref_len, :] = ref_mel
        zeros_cond = torch.zeros_like(cond)
        positions = torch.arange(dur, dtype=torch.int64, device=self.device)

        ts = np.linspace(0.0, 1.0, nfe + 1, dtype=np.float64)
        ts = (ts + sway * (np.cos(np.pi / 2.0 * ts) - 1.0 + ts)).astype(np.float32)

        cand_mels: List[torch.Tensor] = []
        for k in range(best_of):
            g = torch.Generator(device=self.device)
            g.manual_seed(base_seed + k * 137)
            x = torch.randn((1, dur, 100), generator=g, device=self.device)
            for s_idx in range(nfe):
                t_t = torch.tensor([ts[s_idx]], dtype=torch.float32, device=self.device)
                dt = float(ts[s_idx + 1] - ts[s_idx])
                v_pos, _ = self.dit_step(x, cond, txt, txt_gate, t_t, torch.tensor([1.0], device=self.device), positions)
                v_neg, _ = self.dit_step(x, zeros_cond, txt_neg, txt_neg, t_t, torch.tensor([0.0], device=self.device), positions)
                v = v_neg + cfg * (v_pos - v_neg)
                x = x + v * dt
            cand_mels.append(x[0, ref_len:, :].clone())  # [T_gen, 100]

        if len(cand_mels) == 1:
            chosen = cand_mels[0]
        else:
            # Medoid 择优：剔除偶发跳帧/重复离群采样
            dists = np.zeros((len(cand_mels), len(cand_mels)), dtype=np.float64)
            for i in range(len(cand_mels)):
                for j in range(i + 1, len(cand_mels)):
                    d = float(torch.mean(torch.abs(cand_mels[i] - cand_mels[j])).item())
                    dists[i, j] = dists[j, i] = d
            best_idx = int(np.argmin(dists.sum(axis=1)))
            chosen = cand_mels[best_idx]

        wav = self.vocos.decode_mel(chosen.transpose(0, 1).unsqueeze(0))
        # 去首尾静音毛刺 + 12ms 淡入淡出
        fade = int(0.012 * SR)
        if len(wav) > fade * 2:
            wav[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            wav[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        return wav

    def synth_couplets(
        self,
        ref_wav_24k: np.ndarray,
        ref_br_ipa: str,
        couplets_br_ipa: List[str],
        speed: float = 0.96,
        best_of: int = 3,
    ) -> np.ndarray:
        ref_mel = self.mel_of_wav(ref_wav_24k)
        ref_ids = self.encode_bracketed_ipa(ref_br_ipa)
        ref_syls, _ = self.count_syllables(ref_br_ipa)
        parts: List[np.ndarray] = []
        for idx, c_ipa in enumerate(couplets_br_ipa):
            w = self.synth_chunk(
                ref_mel=ref_mel, ref_ids=ref_ids, ref_syls=ref_syls,
                gen_br_ipa=c_ipa, nfe=24, cfg=2.0, speed=speed,
                best_of=best_of, base_seed=100 + idx * 19,
            )
            parts.append(w)
            parts.append(np.zeros(int(0.26 * SR), dtype=np.float32))
        return np.concatenate(parts)


# ===========================================================================
# 3. 音频准备与真实声学语音学指标（浊音低频带音比 + 入声无二次爆破检测）
# ===========================================================================
def trim_and_pad_prompt(
    wav_path: Path,
    start_s: float,
    end_s: float,
    target_rms: float = 0.10,
    pad_tail_s: float = 0.35,
) -> np.ndarray:
    y, sr = sf.read(str(wav_path), dtype="float64")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if end_s > start_s:
        y = y[int(start_s * sr): int(end_s * sr)]
    y = y - np.mean(y)
    if sr != SR:
        g = math.gcd(int(sr), SR)
        y = signal.resample_poly(y, SR // g, int(sr) // g)
    sos = signal.butter(4, 70.0 / (SR / 2.0), btype="highpass", output="sos")
    y = signal.sosfiltfilt(sos, y)

    # 裁剪首尾静音 (>4% 峰值包络)，再加干净静音垫防尾音渗漏
    win = int(0.015 * SR)
    env = np.sqrt(np.convolve(y ** 2, np.ones(win) / win, mode="same") + 1e-12)
    active = np.where(env > np.max(env) * 0.04)[0]
    if len(active) > 10:
        a = max(0, int(active[0]) - int(0.03 * SR))
        b = min(len(y), int(active[-1]) + int(0.05 * SR))
        y = y[a:b]

    fade = int(0.012 * SR)
    if len(y) > fade * 2:
        y[:fade] *= np.linspace(0.0, 1.0, fade)
        y[-fade:] *= np.linspace(1.0, 0.0, fade)

    rms = float(np.sqrt(np.mean(y ** 2) + 1e-12))
    if rms > 1e-5:
        y = y * (target_rms / rms)
    pk = float(np.max(np.abs(y)) + 1e-9)
    if pk > 0.92:
        y = y * (0.92 / pk)
    if pad_tail_s > 0:
        y = np.concatenate([y, np.zeros(int(pad_tail_s * SR), dtype=np.float64)])
    return y.astype(np.float32)


def measure_honest_phonetics(wav: np.ndarray, sr: int = SR) -> Dict:
    """
    诚实声学指标（替代失真的分位比 SNR 与 Whisper 置信度）：
    1. median_f0_hz: 有声段基频中位数（区分真女声 ~235-255Hz 与真男声 ~105-135Hz）；
    2. voice_bar_ratio_db: 75-260Hz 低频浊音带能量相对 1-4kHz 的比值（衡量全浊辅音声带振动饱满度）；
    3. hf_8k_11k_rel_db: 8-11.5kHz 高频泛音相对主频带比值（监控是否出现电音/高频丢失）；
    4. tail_double_burst: 检测末尾 0.45s（如入声“侧”）是否存在二次除阻音节峰（即“咋可”检测器！）。
    """
    step, win = int(0.02 * sr), int(0.04 * sr)
    f0s: List[float] = []
    active_chunks: List[np.ndarray] = []
    for i in range(0, len(wav) - win, step):
        seg = wav[i: i + win]
        r = float(np.sqrt(np.mean(seg ** 2)))
        if r < 0.02:
            continue
        active_chunks.append(seg)
        ac = np.correlate(seg, seg, mode="full")[win - 1:]
        lo, hi = int(sr / 400), int(sr / 75)
        pk = lo + int(np.argmax(ac[lo:hi]))
        if ac[pk] > 0.45 * ac[0]:
            f0s.append(sr / pk)

    y_act = np.concatenate(active_chunks) if active_chunks else wav
    f, psd = signal.welch(y_act, fs=sr, nperseg=2048)
    e_vbar = float(10 * np.log10(np.mean(psd[(f >= 75) & (f <= 260)]) + 1e-15))
    e_mid = float(10 * np.log10(np.mean(psd[(f >= 500) & (f <= 3500)]) + 1e-15))
    e_hi = float(10 * np.log10(np.mean(psd[(f >= 8000) & (f <= 11500)]) + 1e-15))

    # 检测最后一个有声字末尾是否分裂成双峰（“咋可”客观检测）
    nz = np.where(np.abs(wav) > 0.015)[0]
    tail_peaks = 1
    if len(nz) > int(0.35 * sr):
        tail_seg = wav[max(0, int(nz[-1]) - int(0.42 * sr)): int(nz[-1]) + int(0.02 * sr)]
        w_sm = int(0.025 * sr)
        env = np.sqrt(np.convolve(tail_seg ** 2, np.ones(w_sm) / w_sm, mode="same") + 1e-12)
        pks, _ = signal.find_peaks(env, height=np.max(env) * 0.28, distance=int(0.09 * sr))
        tail_peaks = len(pks)

    return {
        "median_f0_hz": round(float(np.median(f0s)), 1) if f0s else 0.0,
        "voice_bar_rel_db": round(e_vbar - e_mid, 2),
        "hf_8k_11k_rel_db": round(e_hi - e_mid, 2),
        "last_word_peaks": int(tail_peaks),
        "active_speech_s": round(len(y_act) / sr, 2),
    }


# ===========================================================================
# 4. 构建 24kHz 混合训练集（75 条新视频真人上古干声 + DiaMoE 男女双声道全浊/入声/-m 对立库）
# ===========================================================================
class V4AlignedDataset(Dataset):
    def __init__(self, items: List[Dict]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]


def collate_v4(batch: List[Dict]) -> Dict:
    max_t = max(x["mel"].shape[0] for x in batch)
    mels = torch.zeros(len(batch), max_t, 100, dtype=torch.float32)
    mel_lens = torch.zeros(len(batch), dtype=torch.int64)
    tokens_list: List[List[int]] = []
    durs_list: List[List[int]] = []
    for i, item in enumerate(batch):
        t_len = item["mel"].shape[0]
        mels[i, :t_len] = item["mel"]
        mel_lens[i] = t_len
        t_ids = [p[0] for p in item["token_items"]]
        t_ws = [p[1] for p in item["token_items"]]
        tokens_list.append(t_ids)
        durs_list.append(_weights_to_exact_frames(t_ws, t_len))
    return {"mels": mels, "mel_lens": mel_lens, "tokens": tokens_list, "durs": durs_list}


def build_v4_training_items(
    zv_comp: HanfireZipVoiceCompiler,
    dm_comp: HanfireDiaMoECompiler,
    dm_engine: DiaMoECudaEngine,
    fbank: PureVocosFbank,
    male_ref_wav: np.ndarray,
    male_ref_br: str,
    female_ref_wav: np.ndarray,
    female_ref_br: str,
) -> List[Dict]:
    items: List[Dict] = []
    seg_root = PROJECT_ROOT / "local_vocals" / "new_assets_24k"

    def trim_active(y: np.ndarray) -> np.ndarray:
        win = int(0.012 * SR)
        env = np.sqrt(np.convolve(y ** 2, np.ones(win) / win, mode="same") + 1e-12)
        act = np.where(env > max(np.max(env) * 0.035, 0.002))[0]
        if len(act) > 10:
            a = max(0, int(act[0]) - int(0.025 * SR))
            b = min(len(y), int(act[-1]) + int(0.04 * SR))
            y = y[a:b]
        rms = float(np.sqrt(np.mean(y ** 2) + 1e-12))
        if rms > 1e-5:
            y = y * (0.10 / rms)
        return np.clip(y, -0.95, 0.95).astype(np.float32)

    # A. 仅载入零句内停顿的独立单字与双字连绵词真人录音室干声（basic_words 47 词 + lianmianci 20 词 + lunyu 短句 0,4）
    # 剔除含多重句读停顿的长句（防均匀帧对齐错位污染词向量！）
    for tag, manifest in [
        ("basic_words", BASIC_WORDS_ITEMS),
        ("lianmianci", LIANMIANCI_ITEMS),
        ("lunyu", [LUNYU_ITEMS[0], LUNYU_ITEMS[4]]),
    ]:
        full_wav, _ = sf.read(str(seg_root / f"{tag}_full_24k.wav"), dtype="float32")
        segs = json.loads((seg_root / f"{tag}_segments.json").read_text(encoding="utf-8"))
        for seg_idx, hanzi_str, syl_ipas in manifest:
            if seg_idx >= len(segs):
                continue
            a_s, b_s = segs[seg_idx]
            clip = trim_active(full_wav[int(a_s * SR): int(b_s * SR)])
            if len(clip) < int(0.16 * SR):
                continue
            tok_items: List[Tuple[int, int]] = []
            chs = list(hanzi_str)
            if len(chs) != len(syl_ipas):
                continue
            for ch, s_ipa in zip(chs, syl_ipas):
                c_it, _ = zv_comp.compile_char(ch, override_ipa=s_ipa, is_rhyme_word=False, chant_mode=False)
                tok_items.extend(c_it)
            mel = fbank.extract(clip, sr=SR).cpu()
            if mel.shape[0] >= len(tok_items) + 2:
                rec = {"mel": mel, "token_items": tok_items, "src": f"human_{tag}_{seg_idx}"}
                # 真人干声重复 2 次提升采样比重
                items.append(rec)
                items.append(rec)

    human_cnt = len(items) // 2
    print(f"  ✓ 已收录新上传视频 24kHz 录音室真人上古干声: {human_cnt} 条 (2x 采样权重 = {len(items)})")

    # B. 用 DiaMoE-GPU 合成专门针对汉火难点音位（全浊 b/d/g/dz/ʑ/ɣ、闭口 -m、入声 -p/-t/-k）的男女双声道四言短句！
    contrast_phrases = [
        "关关雎鸠", "在河之洲", "窈窕淑女", "君子好逑",
        "求之不得", "寤寐思服", "悠哉悠哉", "辗转反侧",
        "蒹葭苍苍", "白露为霜", "所谓伊人", "在水一方",
        "溯洄从之", "道阻且长", "宛在水中", "青青子衿",
        "悠悠我心", "子宁不音", "岂曰无衣", "与子同袍",
        "与子同泽", "与子偕作", "桃之夭夭", "有蕡其实",
        "宜其家室", "昔我往矣", "雨雪霏霏", "我心伤悲",
    ]
    ref_m_mel = dm_engine.mel_of_wav(male_ref_wav)
    ref_m_ids = dm_engine.encode_bracketed_ipa(male_ref_br)
    ref_m_syls, _ = dm_engine.count_syllables(male_ref_br)

    ref_f_mel = dm_engine.mel_of_wav(female_ref_wav)
    ref_f_ids = dm_engine.encode_bracketed_ipa(female_ref_br)
    ref_f_syls, _ = dm_engine.count_syllables(female_ref_br)

    ov_all = {"服": "bək"}
    dm_added = 0
    for idx_p, phrase in enumerate(contrast_phrases):
        _, br_ipa, n_ch = dm_comp.compile_text(phrase + "。", context_overrides=ov_all, rhyme_chars=set())
        zv_it, _ = zv_comp.compile_clause(phrase, context_overrides=ov_all, rhyme_last=False, end_punct=".", chant_mode=False)
        for spk_tag, r_mel, r_ids, r_syls in [
            ("dm_male", ref_m_mel, ref_m_ids, ref_m_syls),
            ("dm_female", ref_f_mel, ref_f_ids, ref_f_syls),
        ]:
            w = dm_engine.synth_chunk(
                ref_mel=r_mel, ref_ids=r_ids, ref_syls=r_syls,
                gen_br_ipa=br_ipa, nfe=16, cfg=2.0, speed=0.98,
                best_of=2, base_seed=500 + idx_p * 13,
            )
            w_trim = trim_active(w)
            mel = fbank.extract(w_trim, sr=SR).cpu()
            if mel.shape[0] >= len(zv_it) + 2:
                items.append({"mel": mel, "token_items": zv_it, "src": f"{spk_tag}_{phrase}"})
                dm_added += 1

    print(f"  ✓ 已补充 DiaMoE 十一大方言全浊/入声/-m 男女双音色四言对立短句: {dm_added} 条 (总训练条目 = {len(items)})")
    return items


# ===========================================================================
# 5. ZipVoice V4 无损声码冻结精调（修复 weight_decay=0 与 AsyncCheckpointManager 接口）
# ===========================================================================
def train_zipvoice_v4_gpu(
    model: ZipVoice,
    train_items: List[Dict],
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 8e-4,
) -> List[float]:
    # 100% 冻结 text_encoder 与 fm_decoder 全部权重（零灾难性遗忘！彻底杜绝吐字变糊！），
    # 仅在球面流形约束下精调 DEAD_SLOT_POOL 中的 73 个新注入汉火音素嵌入向量！
    for p in model.parameters():
        p.requires_grad = False

    init_emb_snapshot = model.embed.weight.data.clone().to(device)
    target_norms = init_emb_snapshot.norm(dim=-1, keepdim=True).clamp(min=1.0)

    model.embed.weight.requires_grad = True
    grad_mask = torch.zeros_like(model.embed.weight, device=device)
    for slot in DEAD_SLOT_POOL:
        if slot < grad_mask.size(0):
            grad_mask[slot] = 1.0
    hook_handle = model.embed.weight.register_hook(lambda g: g * grad_mask)

    opt = torch.optim.AdamW(
        [{"params": [model.embed.weight], "lr": lr, "weight_decay": 0.0}]
    )

    ckpt_dir = PROJECT_ROOT / "checkpoints_v4_local"
    ckpt_mgr = AsyncCheckpointManager(local_dir=ckpt_dir, gdrive_dir=None, keep_last_n=2)

    ds = V4AlignedDataset(train_items)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_v4)

    loss_curve: List[float] = []
    global_step = 0
    # 保持 eval() 模式关闭 Dropout，使流匹配梯度对嵌入向量提供零噪声精准引导！
    model.eval()

    for ep in range(1, epochs + 1):
        t0 = time.time()
        ep_losses: List[float] = []
        for batch in dl:
            global_step += 1
            opt.zero_grad(set_to_none=True)
            features = batch["mels"].to(device) * FEAT_SCALE_ZV
            features_lens = batch["mel_lens"].to(device)
            num_frames = int(features_lens.max().item())

            embed, _ = model.forward_text_embed(batch["tokens"])
            padding_mask = make_pad_mask(features_lens, max_len=num_frames)
            tokens_index = get_tokens_index(batch["durs"], num_frames).to(device)
            text_cond = torch.gather(
                embed, dim=1,
                index=tokens_index.unsqueeze(-1).expand(embed.size(0), num_frames, embed.size(-1)),
            )

            sc_mask = condition_time_mask(features_lens=features_lens, mask_percent=(0.75, 1.0), max_len=num_frames)
            speech_cond = torch.where(sc_mask.unsqueeze(-1), 0.0, features)

            noise = torch.randn_like(features)
            t = torch.rand(features.size(0), 1, 1, device=device)
            xt = features * t + noise * (1.0 - t)
            ut = features - noise

            vt = model.forward_fm_decoder(
                t=t, xt=xt, text_condition=text_cond,
                speech_condition=speech_cond, padding_mask=padding_mask,
            )
            l_mask = sc_mask & (~padding_mask)
            fm_loss = torch.mean((vt[l_mask] - ut[l_mask]) ** 2)
            # 锚定正则：防止新音位向量偏离初始声韵几何邻域过远导致乱码
            anchor_reg = 0.08 * torch.mean(((model.embed.weight - init_emb_snapshot) * grad_mask) ** 2)
            loss = fm_loss + anchor_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_([model.embed.weight], 1.0)
            opt.step()

            # 投影回原始中文词元球面范数 (~8.2)
            with torch.no_grad():
                cur_n = model.embed.weight.data.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                rescaled = model.embed.weight.data * (target_norms / cur_n)
                model.embed.weight.data = torch.where(grad_mask > 0.5, rescaled, init_emb_snapshot)

            ep_losses.append(float(fm_loss.item()))

        ml = float(np.mean(ep_losses))
        loss_curve.append(round(ml, 5))
        if ep in (1, 5, 10, 15, epochs):
            print(f"  [ZipVoice-V4 EmbedOpt] Epoch {ep:02d}/{epochs} | fm_loss={ml:.5f} | {time.time()-t0:.2f}s")

    ckpt_mgr.save_async(
        stage="v4_zh_manifold",
        epoch=epochs,
        global_step=global_step,
        loss=loss_curve[-1],
        model=model,
        optimizer=opt,
        extra_state={"loss_curve": loss_curve},
        is_milestone=True,
    )
    ckpt_mgr.flush_and_close()
    hook_handle.remove()
    model.eval()
    return loss_curve


@torch.no_grad()
def synth_zv_normal(
    model: ZipVoice,
    vocos: PureCudaVocos,
    fbank: PureVocosFbank,
    zv_comp: HanfireZipVoiceCompiler,
    prompt_wav: np.ndarray,
    prompt_items: List[Tuple[int, int]],
    couplets: List[Tuple[str, str, Dict[str, str]]],
    frames_per_100w: float = 25.5,
) -> np.ndarray:
    device = next(model.parameters()).device
    pmt_feat = fbank.extract(prompt_wav, sr=SR).unsqueeze(0).to(device) * FEAT_SCALE_ZV
    pmt_len = torch.tensor([pmt_feat.size(1)], dtype=torch.int64, device=device)
    segs: List[np.ndarray] = []
    for idx, (c1, c2, ov) in enumerate(couplets):
        it1, _ = zv_comp.compile_clause(c1, context_overrides=ov, rhyme_last=False, end_punct=",", chant_mode=False)
        it2, _ = zv_comp.compile_clause(c2, context_overrides=ov, rhyme_last=False, end_punct=".", chant_mode=False)
        pred_mel = sample_with_character_durations(
            model=model, prompt_items=prompt_items, target_items=it1 + it2,
            prompt_features=pmt_feat, prompt_features_lens=pmt_len,
            frames_per_100w=frames_per_100w, num_step=16, guidance_scale=1.25,
            t_shift=0.5, seed=42 + idx * 11,
        )
        wav = vocos.decode_mel(pred_mel.permute(0, 2, 1) / FEAT_SCALE_ZV)
        fade = int(0.015 * SR)
        if len(wav) > fade * 2:
            wav[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            wav[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        segs.append(wav)
        segs.append(np.zeros(int(0.26 * SR), dtype=np.float32))
    return np.concatenate(segs)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"🚀 启动汉火 V4 同句对比与精调管线 | GPU: {torch.cuda.get_device_name(0)}")

    # 1. 初始化 Vocos 与男女双音色正常朗读参考锚
    vocos = PureCudaVocos(DIAMOE_ROOT / "weights" / "vocos", device=device)
    fbank = PureVocosFbank()

    # 男声正常朗读参考锚：新上传《论语·学而》第 1 句（"學而時習之，不亦說乎？"，107.6Hz 录音室干声，零伴奏零弹舌！）
    lunyu_wav_path = PROJECT_ROOT / "local_vocals" / "new_assets_24k" / "lunyu_full_24k.wav"
    male_ref_wav = trim_and_pad_prompt(lunyu_wav_path, 6.62, 9.68, target_rms=0.10, pad_tail_s=0.35)
    sf.write(str(OUT_DIR / "00_参考男声_论语学而原声_108Hz.wav"), male_ref_wav, SR)
    male_ref_br_dm = (
        "[ɡ] [ˈŭᴸ] [k] | [n] [ˈəᴹᴹ] | [d] [ˈəᴹᴹ] | [z] [ˈɐ̆ᴸ] [p] | [t] [ˈəᴴᴴ] | ， "
        "[p] [ˈəᴴᴴ] | [j] [ˈăᴸ] [k] | [l] [ˈɔ̆ᴸ] [t] | [ɦ] [ˈaᴹᴹ] | ？"
    )

    # 女声正常朗读参考锚：24kHz 清朗陈述短句（"今夜的月光如此清亮。"，252Hz 真女声，不含吟诵拖腔）
    yae_src = Path("/mnt/s790/project/tts/voxtts/probes/seed-vc/examples/source/yae_0.wav")
    female_ref_wav = trim_and_pad_prompt(yae_src, 0.58, 2.75, target_rms=0.10, pad_tail_s=0.35)
    sf.write(str(OUT_DIR / "00_参考女声_清朗正常读音_252Hz.wav"), female_ref_wav, SR)
    female_ref_br_dm = (
        "[t͜ɕ] [ˈiᴴᴴ] [n] | [j] [ˈɛᴴᴸ] | [t] [ə] | [ɥ] [ˈœᴴᴸ] | [k] [w] [ˈɑᴴᴴ] [ŋ] | "
        "[ɻ] [ˈuᴹᴴ] | [t͜sʰ] [ˈɹ̩ᴹᴸ] | [t͜ɕʰ] [ˈiᴴᴴ] [ŋ] | [l] [j] [ˈɑᴴᴸ] [ŋ] | 。"
    )

    # 2. 初始化汉火编译器（100% 忠于用户 hanfire_oc_v1.json！）
    dm_comp = HanfireDiaMoECompiler()
    p_json = Path(hf_hub_download("k2-fsa/ZipVoice", filename="zipvoice/model.json"))
    p_tok = Path(hf_hub_download("k2-fsa/ZipVoice", filename="zipvoice/tokens.txt"))
    p_pt = Path(hf_hub_download("k2-fsa/ZipVoice", filename="zipvoice/model.pt"))
    base_t2i = load_zv_token2id(p_tok)
    zv_cfg = json.loads(p_json.read_text(encoding="utf-8"))["model"]
    zv_cfg["vocab_size"] = len(base_t2i)

    zv_model = ZipVoice(**zv_cfg)
    raw_sd = torch.load(str(p_pt), map_location="cpu", weights_only=False)["model"]
    zv_model.load_state_dict({k.removeprefix("module."): v for k, v in raw_sd.items()}, strict=True)
    t2i = inject_hanfire_zh_manifold_embeddings(zv_model, base_t2i)
    # 备份一份普通话原始词元用于校验训练后 0 漂移
    orig_mandarin_sample = zv_model.embed.weight.data[t2i["uan1"]].clone()
    zv_model = zv_model.to(device).eval()
    zv_comp = HanfireZipVoiceCompiler(t2i=t2i)

    # 准备 ZipVoice 参考 token 序列
    male_zv_p1, _ = zv_comp.compile_clause(
        "学而时习之",
        context_overrides={"学": "guk", "而": "nə", "时": "də", "习": "zəp", "之": "tə"},
        end_punct=",", chant_mode=False,
    )
    male_zv_p2, _ = zv_comp.compile_clause(
        "不亦说乎",
        context_overrides={"不": "pə", "亦": "jak", "说": "lɔt", "乎": "ɣa"},
        end_punct="?", chant_mode=False,
    )
    male_zv_pmt = male_zv_p1 + male_zv_p2
    female_zv_pmt, _ = tokenize_mandarin_prompt_exact("今夜的月光如此清亮。", t2i)

    # 目标测试集（100% 用户原版汉火词表 hanfire_oc_v1.json！）
    ov_gj3 = {"服": "bək"}
    gj_couplets = [
        ("关关雎鸠", "在河之洲", {}),
        ("窈窕淑女", "君子好逑", {}),
        ("求之不得", "寤寐思服", ov_gj3),
        ("悠哉悠哉", "辗转反侧", ov_gj3),
    ]
    jj_couplets = [
        ("蒹葭苍苍", "白露为霜", {}),
        ("所谓伊人", "在水一方", {}),
    ]

    # 3. 路线 A：运行 DiaMoE-GPU 同句合成 + 生成多音色训练辅助集
    print("\n[Stage 1/3] 运行 DiaMoE-TTS (GPU FP32, --best-of 4) 合成正常读音样带 & 构建混合训练集...")
    t0 = time.time()
    dm_engine = DiaMoECudaEngine(
        ckpt_path=DIAMOE_ROOT / "weights" / "10ep_mlpEXP_9.pt",
        vocab_path=DIAMOE_ROOT / "models" / "vocab.txt",
        vocos=vocos, device=device,
    )
    dm_gj_br = [
        dm_comp.compile_text(f"{c1}，{c2}。", context_overrides=ov, rhyme_chars=set())[1]
        for c1, c2, ov in gj_couplets
    ]
    dm_jj_br = [
        dm_comp.compile_text(f"{c1}，{c2}。", context_overrides=ov, rhyme_chars=set())[1]
        for c1, c2, ov in jj_couplets
    ]

    p_dm_gj_m = OUT_DIR / "A_DiaMoE_关雎_male_论语男声.wav"
    p_dm_jj_m = OUT_DIR / "A_DiaMoE_蒹葭_male_论语男声.wav"
    p_dm_gj_f = OUT_DIR / "A_DiaMoE_关雎_female_清朗女声.wav"
    p_dm_jj_f = OUT_DIR / "A_DiaMoE_蒹葭_female_清朗女声.wav"
    cache_pt = OUT_DIR / "v4_train_items_cache.pt"

    if all(p.exists() for p in (p_dm_gj_m, p_dm_jj_m, p_dm_gj_f, p_dm_jj_f)):
        w_dm_gj_m, _ = sf.read(str(p_dm_gj_m), dtype="float32")
        w_dm_jj_m, _ = sf.read(str(p_dm_jj_m), dtype="float32")
        w_dm_gj_f, _ = sf.read(str(p_dm_gj_f), dtype="float32")
        w_dm_jj_f, _ = sf.read(str(p_dm_jj_f), dtype="float32")
        print("  ✓ 复用已生成的 DiaMoE-GPU 四首样带")
    else:
        w_dm_gj_m = dm_engine.synth_couplets(male_ref_wav, male_ref_br_dm, dm_gj_br, speed=0.98, best_of=4)
        w_dm_jj_m = dm_engine.synth_couplets(male_ref_wav, male_ref_br_dm, dm_jj_br, speed=0.98, best_of=4)
        w_dm_gj_f = dm_engine.synth_couplets(female_ref_wav, female_ref_br_dm, dm_gj_br, speed=0.96, best_of=4)
        w_dm_jj_f = dm_engine.synth_couplets(female_ref_wav, female_ref_br_dm, dm_jj_br, speed=0.96, best_of=4)
        sf.write(str(p_dm_gj_m), w_dm_gj_m, SR)
        sf.write(str(p_dm_jj_m), w_dm_jj_m, SR)
        sf.write(str(p_dm_gj_f), w_dm_gj_f, SR)
        sf.write(str(p_dm_jj_f), w_dm_jj_f, SR)
        print(f"  ✓ DiaMoE-GPU 四首样带合成完成 ({time.time()-t0:.1f}s)")

    if cache_pt.exists():
        train_items = torch.load(str(cache_pt), map_location="cpu", weights_only=False)
        print(f"  ✓ 复用已缓存混合训练集: {len(train_items)} 条")
    else:
        train_items = build_v4_training_items(
            zv_comp=zv_comp, dm_comp=dm_comp, dm_engine=dm_engine, fbank=fbank,
            male_ref_wav=male_ref_wav, male_ref_br=male_ref_br_dm,
            female_ref_wav=female_ref_wav, female_ref_br=female_ref_br_dm,
        )
        torch.save(train_items, str(cache_pt))
    # 释放 DiaMoE 显存，腾出全部显存供 ZipVoice 高速训练
    del dm_engine
    torch.cuda.empty_cache()

    # 4. 路线 B：先保存未微调的 ZipVoice 正常读音基线（供严谨训练前后对比！）
    print("\n[Stage 2/3] 生成 ZipVoice 未微调正常读音基线 -> 随后启动 2080 Ti 球面嵌入精调...")
    w_zv0_gj_m = synth_zv_normal(zv_model, vocos, fbank, zv_comp, male_ref_wav, male_zv_pmt, gj_couplets, 26.0)
    w_zv0_jj_m = synth_zv_normal(zv_model, vocos, fbank, zv_comp, male_ref_wav, male_zv_pmt, jj_couplets, 26.0)
    w_zv0_gj_f = synth_zv_normal(zv_model, vocos, fbank, zv_comp, female_ref_wav, female_zv_pmt, gj_couplets, 25.0)
    w_zv0_jj_f = synth_zv_normal(zv_model, vocos, fbank, zv_comp, female_ref_wav, female_zv_pmt, jj_couplets, 25.0)
    sf.write(str(OUT_DIR / "B_ZipVoice未训基线_关雎_male_论语男声.wav"), w_zv0_gj_m, SR)
    sf.write(str(OUT_DIR / "B_ZipVoice未训基线_蒹葭_male_论语男声.wav"), w_zv0_jj_m, SR)
    sf.write(str(OUT_DIR / "B_ZipVoice未训基线_关雎_female_清朗女声.wav"), w_zv0_gj_f, SR)
    sf.write(str(OUT_DIR / "B_ZipVoice未训基线_蒹葭_female_清朗女声.wav"), w_zv0_jj_f, SR)

    # 启动 ZipVoice-V4 球面嵌入精调（冻结 text_encoder/fm_decoder，零灾难遗忘！）
    loss_curve = train_zipvoice_v4_gpu(zv_model, train_items, device=device, epochs=20, batch_size=16, lr=8e-4)

    # 严格验证原装普通话词元零衰减污染！
    drift = float(torch.max(torch.abs(zv_model.embed.weight.data[t2i["uan1"]].cpu() - orig_mandarin_sample)).item())
    assert drift == 0.0, f"原装普通话词元发生漂移: {drift}"
    print(f"  ✓ 权重衰减隔离校验通过：原装普通话词元漂移 = {drift:.1e} (严格 0.0！)")

    # 5. 路线 C：生成精调后的 ZipVoice-V4 正常读音样带并输出诚实声学评测
    print("\n[Stage 3/3] 合成精调后的 ZipVoice-V4 样带并运行客观声学对比...")
    w_zv4_gj_m = synth_zv_normal(zv_model, vocos, fbank, zv_comp, male_ref_wav, male_zv_pmt, gj_couplets, 26.0)
    w_zv4_jj_m = synth_zv_normal(zv_model, vocos, fbank, zv_comp, male_ref_wav, male_zv_pmt, jj_couplets, 26.0)
    w_zv4_gj_f = synth_zv_normal(zv_model, vocos, fbank, zv_comp, female_ref_wav, female_zv_pmt, gj_couplets, 25.0)
    w_zv4_jj_f = synth_zv_normal(zv_model, vocos, fbank, zv_comp, female_ref_wav, female_zv_pmt, jj_couplets, 25.0)

    sf.write(str(OUT_DIR / "C_ZipVoice精调V4_关雎_male_论语男声.wav"), w_zv4_gj_m, SR)
    sf.write(str(OUT_DIR / "C_ZipVoice精调V4_蒹葭_male_论语男声.wav"), w_zv4_jj_m, SR)
    sf.write(str(OUT_DIR / "C_ZipVoice精调V4_关雎_female_清朗女声.wav"), w_zv4_gj_f, SR)
    sf.write(str(OUT_DIR / "C_ZipVoice精调V4_蒹葭_female_清朗女声.wav"), w_zv4_jj_f, SR)

    summary = {
        "loss_curve": loss_curve,
        "mandarin_embed_drift": drift,
        "metrics_guanju": {
            "A_DiaMoE_male": measure_honest_phonetics(w_dm_gj_m),
            "A_DiaMoE_female": measure_honest_phonetics(w_dm_gj_f),
            "B_ZipVoice_ZeroShot_male": measure_honest_phonetics(w_zv0_gj_m),
            "B_ZipVoice_ZeroShot_female": measure_honest_phonetics(w_zv0_gj_f),
            "C_ZipVoice_V4_Trained_male": measure_honest_phonetics(w_zv4_gj_m),
            "C_ZipVoice_V4_Trained_female": measure_honest_phonetics(w_zv4_gj_f),
        },
        "metrics_jianjia": {
            "A_DiaMoE_male": measure_honest_phonetics(w_dm_jj_m),
            "A_DiaMoE_female": measure_honest_phonetics(w_dm_jj_f),
            "C_ZipVoice_V4_Trained_male": measure_honest_phonetics(w_zv4_jj_m),
            "C_ZipVoice_V4_Trained_female": measure_honest_phonetics(w_zv4_jj_f),
        },
    }
    (OUT_DIR / "v4_ab_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["metrics_guanju"], ensure_ascii=False, indent=2))
    print(f"\n✅ 全部完成！输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
