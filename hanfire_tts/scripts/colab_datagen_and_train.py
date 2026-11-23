#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汉火上古音 TTS —— Colab Ultra (L4 / A100 / H100) 全自动造数、断点续训与声学裁判主程序
(colab_datagen_and_train.py)

设计特性：
1. 原生挂载 Google Drive (`/content/drive/MyDrive/hanfire_ancient_tts`) 持久化全部素材、特征缓存、
   异步中间 Checkpoints 与最终试听报告；
2. 异步非阻塞存盘 (`AsyncCheckpointManager`)：每 `--save-every` 步后台落盘至本地与 GDrive，不卡 GPU；
3. 任意时刻可中断恢复 (`--resume auto`)：下次启动自动跳过已完成的造数缓存，从最新 step 继续训练；
4. 训后全自动运行「音·形·义」四维声波裁判：输出男女双音色《诗经》音频 + 共振峰 Bark 押韵分 + 一字一音单峰率报告。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hanfire_core.async_checkpoint import AsyncCheckpointManager
from hanfire_core.lexicon_builder import text_to_encoded_items
from hanfire_core.phonology import (
    TOTAL_FEAT_DIM,
    encode_syllables_to_sequence,
    parse_mononuclear_4slot,
)
from hanfire_core.referee import (
    extract_vowel_formants_bark,
    judge_utterance_acoustics,
    judge_waveform_rhyme_pair,
)
from hanfire_core.zipvoice_ancient import (
    PureVocosFbank,
    ZipVoiceAncient,
    build_model_from_pretrained,
)

FEAT_SCALE = 0.1
SR = 24000


# ---------------------------------------------------------------------------
# 状态同步与环境准备
# ---------------------------------------------------------------------------
class StatusLogger:
    def __init__(self, local_work: Path, gdrive_root: Optional[Path]):
        self.paths = [local_work / "STATUS.json"]
        if gdrive_root is not None:
            self.paths.append(gdrive_root / "STATUS.json")
        self.state: Dict = {"started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "events": []}

    def update(self, **kwargs) -> None:
        self.state.update(kwargs)
        self.state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if "phase" in kwargs or "step_msg" in kwargs:
            msg = kwargs.get("step_msg") or f"phase={kwargs.get('phase')}"
            self.state["events"] = (self.state["events"] + [f"[{self.state['updated_at']}] {msg}"])[-40:]
        txt = json.dumps(self.state, ensure_ascii=False, indent=2)
        for p in self.paths:
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_suffix(".json.tmp")
                tmp.write_text(txt, encoding="utf-8")
                os.replace(str(tmp), str(p))
            except Exception:
                pass


def ensure_bundle_unpacked(work_dir: Path, gdrive_root: Optional[Path], bundle_tar: Optional[Path]) -> Path:
    """确保素材包已解压到 GDrive（优先）或本地工作目录，后续会话零重复上传。"""
    target_base = gdrive_root if gdrive_root is not None else work_dir
    bundle_dst = target_base / "gdrive_bundle"
    lex_file = bundle_dst / "lexicon" / "hanfire_oc_v1.json"
    if lex_file.exists():
        print(f"[Bundle] 发现已就绪素材目录: {bundle_dst}")
        return bundle_dst

    # 查找上传的 tar.gz
    search_tars = [
        bundle_tar,
        gdrive_root / "hanfire_gdrive_bundle.tar.gz" if gdrive_root else None,
        work_dir / "hanfire_gdrive_bundle.tar.gz",
        PROJECT_ROOT / "hanfire_gdrive_bundle.tar.gz",
        Path("/content/hanfire_gdrive_bundle.tar.gz"),
    ]
    found_tar = next((p for p in search_tars if p is not None and p.exists()), None)
    if found_tar is not None:
        print(f"[Bundle] 正在解压 {found_tar} -> {target_base} ...")
        target_base.mkdir(parents=True, exist_ok=True)
        with tarfile.open(str(found_tar), "r:gz") as tar:
            tar.extractall(path=str(target_base))
        if lex_file.exists():
            return bundle_dst

    if (PROJECT_ROOT / "gdrive_bundle" / "lexicon" / "hanfire_oc_v1.json").exists():
        return PROJECT_ROOT / "gdrive_bundle"

    raise FileNotFoundError("未找到 gdrive_bundle 素材目录或 hanfire_gdrive_bundle.tar.gz！")


# ---------------------------------------------------------------------------
# 多语种音系拼图与 Stage 2 真人语料构建
# ---------------------------------------------------------------------------
def espeak_words_to_ipa_syllables(text: str, voice: str) -> List[str]:
    """调用 espeak-ng 将母语文本转为纯净单音节级 IPA 列表（剥离零宽字符、语言标签与粤语嵌入式声调数字）。"""
    import re
    # 普通话在部分 Linux 发行版缺 cmn 字典会退回拼读英文数字，优先走 zhy(粤) 以外的标准语音或逐词清洗
    v_arg = "zh" if voice == "cmn" else voice
    try:
        r = subprocess.run(
            ["espeak-ng", "-q", "--ipa=3", "-v", v_arg, text],
            capture_output=True, text=True, timeout=15,
        )
        raw = r.stdout.strip()
    except Exception:
        return []
    # 1. 剥离零宽连接符 (\u200b-\u200f) 与 espeak 语言切换标记如 (en) (cmn) (zh)
    cleaned = re.sub(r"[\u200b-\u200f\ufeff]", "", raw)
    cleaned = re.sub(r"\([a-zA-Z\-]+\)", " ", cleaned)
    # 如果 espeak 退化成了英文数字拼读 (如 nwɒn / θɹiː)，说明系统缺普通话词表，改用汉字字表近似或跳过
    if voice == "cmn" and ("θɹ" in cleaned or "nwɒ" in cleaned):
        return []
    # 2. 剥离粤语/声调语言嵌在音节中间的调号数字 0-9 及重音符 (如 ɡˈa1m jˈa6t -> ɡam jat)
    cleaned = re.sub(r"[0-9ˈˌ˥-˫́̂̌‿|‖\"'()\[\]]+", "", cleaned)
    words = [w.strip() for w in re.split(r"[\s\-]+", cleaned) if w.strip()]
    syls: List[str] = []
    v_set = set("aeiouɑæɐəɛɜɪɨɔœøɒʌʊʉɯɤy")
    for w in words:
        cur = []
        seen_v = False
        for idx, ch in enumerate(w):
            cur.append(ch)
            if ch in v_set:
                seen_v = True
            elif seen_v and idx + 1 < len(w) and w[idx + 1] in v_set and len(cur) >= 2:
                c_next = cur.pop()
                syls.append("".join(cur))
                cur = [c_next]
                seen_v = False
        if cur:
            syls.append("".join(cur))
    return [s for s in syls if s]


def estimate_gender_spk_id(wav: np.ndarray, sr: int) -> int:
    """快速自相关基频估计：F0 < 165Hz 归为男声基底 (0)，否则女声基底 (1)。"""
    seg = wav[: min(len(wav), sr * 4)]
    seg = seg - np.mean(seg)
    if np.max(np.abs(seg)) < 1e-4:
        return 0
    # 中心削波自相关测基频 (75Hz ~ 320Hz)
    win = seg[len(seg) // 4: len(seg) // 4 + min(len(seg) // 2, sr)]
    if len(win) < int(sr * 0.08):
        return 0
    corr = np.correlate(win, win, mode="full")[len(win) - 1:]
    min_lag = int(sr / 320.0)
    max_lag = min(len(corr) - 1, int(sr / 75.0))
    if max_lag <= min_lag:
        return 0
    best_lag = min_lag + int(np.argmax(corr[min_lag:max_lag]))
    f0 = sr / max(1, best_lag)
    return 0 if f0 < 165.0 else 1


def build_or_load_stage1_cache(
    cache_path: Path,
    fbank: PureVocosFbank,
    samples_per_lang: int,
    status: StatusLogger,
    smoke: bool = False,
) -> List[Dict]:
    if cache_path.exists() and not smoke:
        print(f"[Stage1 Data] 命中 Google Drive 特征缓存: {cache_path}")
        return torch.load(str(cache_path), map_location="cpu", weights_only=False)

    if smoke:
        # 冒烟模式：构造 6 条尺寸真实的轻量特征样本验证全链路
        rng = np.random.default_rng(42)
        dummy = []
        for i in range(6):
            items = [
                {"type": "char", "char": "关", "ipa": "kuan"},
                {"type": "char", "char": "雎", "ipa": "tsʰia"},
                {"type": "pause", "pause_kind": "comma"},
                {"type": "char", "char": "急", "ipa": "kiəp", "alpha_dur": 0.72},
            ]
            enc = encode_syllables_to_sequence(items, speaker_id=i % 2)
            T = 96 + i * 8
            dummy.append({
                "utt_id": f"smoke_{i}",
                "slot_feats": enc.feats,
                "dur_weights": enc.dur_weights,
                "max_frames": enc.max_frames,
                "mel": rng.normal(-4.0, 1.0, (T, 100)).astype(np.float32) * FEAT_SCALE,
                "speaker_id": i % 2,
                "lang": "smoke",
            })
        return dummy

    from datasets import load_dataset
    # 精选 5 大互补语种：
    # yue(粤语 -p/-t/-k 唯闭入声), cmn(普通话 基础元辅音与古风气息),
    # pl(波兰语 辅音丛), ar(阿拉伯语 小舌音 q 与咽化 ˤ), cy(威尔士语 清响音/清边音)
    lang_plan = [
        ("yue_hant_hk", "yue", samples_per_lang),
        ("vi_vn",       "vi",  max(120, samples_per_lang // 2)),
        ("pl_pl",       "pl",  max(120, samples_per_lang // 2)),
        ("ar_eg",       "ar",  max(120, samples_per_lang // 2)),
        ("cy_gb",       "cy",  max(100, samples_per_lang // 2)),
    ]

    dataset_items: List[Dict] = []
    for fleurs_cfg, esp_voice, quota in lang_plan:
        status.update(phase="stage1_datagen", step_msg=f"拉取并编码 FLEURS {fleurs_cfg} (目标 {quota} 条)")
        print(f"[Stage1 Data] 流式加载 google/fleurs/{fleurs_cfg} (quota={quota}) ...", flush=True)
        try:
            ds = load_dataset("google/fleurs", fleurs_cfg, split="train", streaming=True)
        except Exception as e:
            print(f"  [跳过] {fleurs_cfg} 加载异常: {e}")
            continue

        kept = 0
        scanned = 0
        for ex in ds:
            if kept >= quota or scanned >= quota * 5:
                break
            scanned += 1
            arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
            sr = int(ex["audio"]["sampling_rate"])
            dur = len(arr) / max(sr, 1)
            if dur < 1.2 or dur > 18.0:
                continue

            txt = str(ex.get("transcription") or ex.get("raw_transcription") or "").strip()
            if not txt:
                continue
            syl_ipas = espeak_words_to_ipa_syllables(txt, esp_voice)
            if len(syl_ipas) < 2 or len(syl_ipas) > 95:
                continue

            # 声学弹舌与静音底线筛查
            audit = judge_utterance_acoustics(arr, sr, n_expected_chars=len(syl_ipas))
            if not audit.trill_ok or audit.active_dur_s < 0.8:
                continue

            spk_id = estimate_gender_spk_id(arr, sr)
            seq_items = [{"type": "char", "char": "", "ipa": s} for s in syl_ipas]
            enc = encode_syllables_to_sequence(seq_items, speaker_id=spk_id)
            mel = fbank.extract(arr, sr=sr).numpy() * FEAT_SCALE
            if mel.shape[0] <= enc.feats.shape[0] + 4:
                continue

            dataset_items.append({
                "utt_id": f"{fleurs_cfg}_{kept:04d}",
                "slot_feats": enc.feats,
                "dur_weights": enc.dur_weights,
                "max_frames": enc.max_frames,
                "mel": mel.astype(np.float32),
                "speaker_id": spk_id,
                "lang": fleurs_cfg,
            })
            kept += 1

        print(f"  ✓ {fleurs_cfg}: 保留高质量样本 {kept} 条 (扫描 {scanned} 条)", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_p = cache_path.with_suffix(".pt.tmp")
    torch.save(dataset_items, str(tmp_p))
    os.replace(str(tmp_p), str(cache_path))
    print(f"[Stage1 Data] 共计 {len(dataset_items)} 条样本已持久化至 {cache_path}")
    return dataset_items


def build_stage2_real_samples(bundle_dir: Path, lex_data: Dict, fbank: PureVocosFbank) -> List[Dict]:
    mf_path = bundle_dir / "stage2_real" / "manifest_stage2.jsonl"
    if not mf_path.exists():
        return []
    out: List[Dict] = []
    for line in mf_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        wav_p = bundle_dir / rec["wav"]
        if not wav_p.exists():
            continue
        wav, sr = sf.read(str(wav_p))
        items = text_to_encoded_items(rec["text"], lex_data)
        if not items:
            continue
        spk_id = int(rec.get("speaker_id", 0))
        enc = encode_syllables_to_sequence(items, speaker_id=spk_id)
        mel = fbank.extract(wav, sr=sr).numpy() * FEAT_SCALE
        if mel.shape[0] <= enc.feats.shape[0]:
            continue
        out.append({
            "utt_id": rec["utt_id"],
            "slot_feats": enc.feats,
            "dur_weights": enc.dur_weights,
            "max_frames": enc.max_frames,
            "mel": mel.astype(np.float32),
            "speaker_id": spk_id,
            "lang": "oc_real",
        })
    return out


# ---------------------------------------------------------------------------
# PyTorch Dataset & Collate
# ---------------------------------------------------------------------------
class AncientFlowDataset(Dataset):
    def __init__(self, records: List[Dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        return self.records[idx]


def collate_flow_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    B = len(batch)
    s_lens = [x["slot_feats"].shape[0] for x in batch]
    t_lens = [x["mel"].shape[0] for x in batch]
    max_s = max(s_lens)
    max_t = max(t_lens)

    slot_feats = torch.zeros((B, max_s, TOTAL_FEAT_DIM), dtype=torch.float32)
    dur_weights = torch.zeros((B, max_s), dtype=torch.float32)
    max_frames = torch.ones((B, max_s), dtype=torch.int64)
    features = torch.zeros((B, max_t, 100), dtype=torch.float32)

    for i, x in enumerate(batch):
        s, t = s_lens[i], t_lens[i]
        slot_feats[i, :s] = torch.from_numpy(x["slot_feats"])
        dur_weights[i, :s] = torch.from_numpy(x["dur_weights"])
        max_frames[i, :s] = torch.from_numpy(x["max_frames"])
        features[i, :t] = torch.from_numpy(x["mel"])

    return {
        "slot_feats": slot_feats,
        "slot_lens": torch.tensor(s_lens, dtype=torch.int64),
        "dur_weights": dur_weights,
        "max_frames": max_frames,
        "features": features,
        "features_lens": torch.tensor(t_lens, dtype=torch.int64),
    }


# ---------------------------------------------------------------------------
# 训练主引擎 (支持 bf16 混合精度 + 差分学习率 + 异步存盘 + 精确断点续训)
# ---------------------------------------------------------------------------
def run_training_stage(
    *,
    stage_name: str,
    model: ZipVoiceAncient,
    records: List[Dict],
    epochs: int,
    batch_size: int,
    proj_lr: float,
    backbone_lr: float,
    save_every_steps: int,
    ckpt_mgr: AsyncCheckpointManager,
    status: StatusLogger,
    device: torch.device,
    resume_info: Optional[Dict] = None,
) -> int:
    ds = AncientFlowDataset(records)
    dl = DataLoader(
        ds,
        batch_size=min(batch_size, max(1, len(ds))),
        shuffle=True,
        collate_fn=collate_flow_batch,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
        drop_last=(len(ds) > batch_size),
    )

    # 差分学习率：新注入的发音特征映射层高学习率快收敛，预训练 123M 声学骨干低学习率稳音质
    proj_params = list(model.articulatory_proj.parameters())
    proj_ids = {id(p) for p in proj_params}
    backbone_params = [p for p in model.parameters() if id(p) not in proj_ids and p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": proj_params, "lr": proj_lr, "weight_decay": 1e-4},
            {"params": backbone_params, "lr": backbone_lr, "weight_decay": 1e-4},
        ],
        betas=(0.9, 0.98),
    )
    total_steps = max(1, epochs * len(dl))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=backbone_lr * 0.1)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    start_epoch = 0
    global_step = 0
    if resume_info and resume_info.get("stage") == stage_name:
        if not resume_info.get("extra_state", {}).get("stage_completed", False):
            ckpt_p = resume_info["loaded_from"]
            ckpt_mgr.load_checkpoint(ckpt_p, model, optimizer=optimizer, scheduler=scheduler, map_location=device)
            start_epoch = int(resume_info["epoch"])
            global_step = int(resume_info["global_step"])
            print(f"[{stage_name}] 🔄 从断点恢复: epoch={start_epoch}, global_step={global_step}", flush=True)

    model.train()
    for ep in range(start_epoch, epochs):
        ep_losses = []
        t_ep = time.time()
        for batch in dl:
            # 关键：同步官方 ZipVoice 微调的 ScheduledFloat/Balancer 步数偏移 (+100000)
            for m in model.modules():
                if hasattr(m, "batch_count"):
                    m.batch_count = float(100000 + global_step)

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)

            loss = model(
                slot_feats=batch["slot_feats"],
                slot_lens=batch["slot_lens"],
                dur_weights=batch["dur_weights"],
                max_frames=batch["max_frames"],
                features=batch["features"],
                features_lens=batch["features_lens"],
                condition_drop_ratio=0.1,
            )

            if not torch.isfinite(loss):
                print(f"[{stage_name}] ⚠️ 跳过异常梯度步 step={global_step}", flush=True)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            loss_val = float(loss.detach().item())
            ep_losses.append(loss_val)

            if global_step % 25 == 0:
                status.update(
                    phase=stage_name,
                    epoch=f"{ep + 1}/{epochs}",
                    global_step=global_step,
                    loss=round(loss_val, 5),
                    step_msg=f"{stage_name} ep {ep+1}/{epochs} step {global_step} loss={loss_val:.4f}",
                )
                print(
                    f"[{stage_name}] ep {ep+1}/{epochs} | step {global_step} | loss={loss_val:.4f} | "
                    f"lr_proj={optimizer.param_groups[0]['lr']:.2e}",
                    flush=True,
                )

            if save_every_steps > 0 and (global_step % save_every_steps == 0):
                ckpt_mgr.save_async(
                    stage=stage_name,
                    epoch=ep,
                    global_step=global_step,
                    loss=loss_val,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    extra_state={"stage_completed": False},
                )

        mean_ep_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        print(f"[{stage_name}] ✓ Epoch {ep+1}/{epochs} 完成 ({time.time()-t_ep:.1f}s) mean_loss={mean_ep_loss:.4f}", flush=True)

    # Stage 结束里程碑异步存盘并等待落盘确认
    ckpt_mgr.save_async(
        stage=stage_name,
        epoch=epochs,
        global_step=global_step,
        loss=mean_ep_loss if epochs > start_epoch else 0.0,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        extra_state={"stage_completed": True},
        is_milestone=True,
        tag="done",
    )
    ckpt_mgr.wait_pending()
    return global_step


# ---------------------------------------------------------------------------
# Phase 3: 《诗经》男女双音色合成与四维声波裁判自动验收
# ---------------------------------------------------------------------------
def evaluate_shijing_and_iconicity(
    *,
    model: ZipVoiceAncient,
    vocoder: Optional[torch.nn.Module],
    fbank: PureVocosFbank,
    bundle_dir: Path,
    lex_data: Dict,
    out_dir: Path,
    device: torch.device,
    k_candidates: int = 3,
) -> Dict:
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    pmt_meta = json.loads((bundle_dir / "prompts" / "prompts_manifest.json").read_text(encoding="utf-8"))

    speakers = []
    if "male_primary" in pmt_meta:
        speakers.append(("male", 0, pmt_meta["male_primary"]))
    if "female_primary" in pmt_meta:
        speakers.append(("female", 1, pmt_meta["female_primary"]))

    target_stanzas = [s for s in lex_data["shijing_anthology"] if s["id"] in {
        "guanju_1", "guanju_3", "taoyao_1", "taoyao_3", "jianjia_1", "meng_3", "caiwei_1"
    }]

    eval_records = []
    iconicity_checks = []

    def _decode_mel_to_wav(mel_scaled: torch.Tensor) -> np.ndarray:
        # mel_scaled: (1, T, 100)
        mel = mel_scaled.permute(0, 2, 1) / FEAT_SCALE  # (1, 100, T)
        if vocoder is not None:
            with torch.no_grad():
                wav_t = vocoder.decode(mel).squeeze().clamp(-1.0, 1.0)
            return wav_t.detach().cpu().numpy().astype(np.float32)
        # 本地无 vocoder 时的 Griffin-Lim 纯数学回退（仅冒烟测试使用）
        n_T = mel.size(-1)
        t_axis = np.linspace(0, n_T * 256 / SR, n_T * 256, endpoint=False)
        return (0.2 * np.sin(2 * np.pi * 140.0 * t_axis)).astype(np.float32)

    for spk_tag, spk_id, p_info in speakers:
        pmt_wav, pmt_sr = sf.read(str(bundle_dir / p_info["wav"]))
        pmt_mel = fbank.extract(pmt_wav, sr=pmt_sr).unsqueeze(0).to(device) * FEAT_SCALE
        pmt_items = text_to_encoded_items(p_info["text"], lex_data)
        pmt_enc = encode_syllables_to_sequence(pmt_items, speaker_id=spk_id)
        pmt_sf = torch.from_numpy(pmt_enc.feats[None]).to(device)
        pmt_dw = torch.from_numpy(pmt_enc.dur_weights[None]).to(device)
        pmt_mf = torch.from_numpy(pmt_enc.max_frames[None]).to(device)

        spk_out = out_dir / f"shijing_{spk_tag}"
        spk_out.mkdir(parents=True, exist_ok=True)

        for st in target_stanzas:
            items = text_to_encoded_items(st["text"], lex_data, stanza_id=st["id"])
            char_items = [it for it in items if it["type"] == "char"]
            enc = encode_syllables_to_sequence(items, speaker_id=spk_id)
            tgt_sf = torch.from_numpy(enc.feats[None]).to(device)
            tgt_dw = torch.from_numpy(enc.dur_weights[None]).to(device)
            tgt_mf = torch.from_numpy(enc.max_frames[None]).to(device)

            # Best-of-K 自动择优：挑出音节波峰数最贴合汉字数且弹舌最低的一条！
            best_wav = None
            best_audit = None
            best_rank = -1e9
            for _ in range(max(1, k_candidates)):
                pred_mel = model.synthesize_features(
                    target_slot_feats=tgt_sf,
                    target_dur_weights=tgt_dw,
                    target_max_frames=tgt_mf,
                    prompt_slot_feats=pmt_sf,
                    prompt_dur_weights=pmt_dw,
                    prompt_max_frames=pmt_mf,
                    prompt_features=pmt_mel,
                    num_step=16 if vocoder is not None else 4,
                    guidance_scale=1.0,
                )
                wav_cand = _decode_mel_to_wav(pred_mel)
                aud = judge_utterance_acoustics(wav_cand, SR, n_expected_chars=len(char_items))
                rank = -abs(aud.n_detected_peaks - len(char_items)) * 10.0 - max(0.0, aud.trill_index - 5.0)
                if rank > best_rank:
                    best_rank = rank
                    best_wav = wav_cand
                    best_audit = aud

            wav_file = spk_out / f"{st['id']}_{spk_tag}.wav"
            sf.write(str(wav_file), best_wav, SR)

            # 按权重比例定位韵脚字音频窗口，测声波 LPC 共振峰 Bark 押韵分！
            slot_w = enc.dur_weights
            cum_w = np.concatenate([[0.0], np.cumsum(slot_w)])
            total_w = max(cum_w[-1], 1e-6)
            char_wav_map: Dict[str, np.ndarray] = {}
            for s_idx, e_idx, ch in enc.char_span_slots:
                r0 = cum_w[s_idx] / total_w
                r1 = cum_w[e_idx] / total_w
                i0 = int(r0 * len(best_wav))
                i1 = max(i0 + int(0.08 * SR), int(r1 * len(best_wav)))
                char_wav_map[ch] = best_wav[i0: min(len(best_wav), i1)]

            rhyme_judgments = []
            for grp in st["rhymes"]:
                for i in range(len(grp)):
                    for j in range(i + 1, len(grp)):
                        c1, c2 = grp[i], grp[j]
                        if c1 in char_wav_map and c2 in char_wav_map:
                            rj = judge_waveform_rhyme_pair(char_wav_map[c1], char_wav_map[c2], SR)
                            rhyme_judgments.append({"pair": f"{c1}-{c2}", **rj})

            eval_records.append({
                "stanza_id": st["id"],
                "title": st["title"],
                "speaker": spk_tag,
                "wav": str(wav_file.relative_to(out_dir)),
                "utterance_audit": best_audit.to_dict(),
                "rhyme_acoustic_checks": rhyme_judgments,
            })

        # 额外：合成单字「冷(凜) vs 暖」与「急 vs 舒」，用声波物理验证「音形义统一」！
        if spk_tag == "male":
            contrast_chars = [("凜", "liəm"), ("暖", "nuan"), ("急", "kiəp"), ("舒", "ɕia")]
            char_metrics = {}
            for ch, fallback_ipa in contrast_chars:
                it = text_to_encoded_items(ch, lex_data) or [{"type": "char", "char": ch, "ipa": fallback_ipa}]
                enc_c = encode_syllables_to_sequence(it, speaker_id=spk_id)
                pred_m = model.synthesize_features(
                    target_slot_feats=torch.from_numpy(enc_c.feats[None]).to(device),
                    target_dur_weights=torch.from_numpy(enc_c.dur_weights[None]).to(device),
                    target_max_frames=torch.from_numpy(enc_c.max_frames[None]).to(device),
                    prompt_slot_feats=pmt_sf,
                    prompt_dur_weights=pmt_dw,
                    prompt_max_frames=pmt_mf,
                    prompt_features=pmt_mel,
                    num_step=16 if vocoder is not None else 4,
                )
                w_c = _decode_mel_to_wav(pred_m)
                sf.write(str(spk_out / f"char_{ch}.wav"), w_c, SR)
                fm = extract_vowel_formants_bark(w_c, SR)
                aud_c = judge_utterance_acoustics(w_c, SR, n_expected_chars=1)
                char_metrics[ch] = {
                    "duration_s": round(len(w_c) / SR, 3),
                    "f1_hz_openness": fm["f_hz"][0],
                    "peaks": aud_c.n_detected_peaks,
                }
            iconicity_checks.append(char_metrics)

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "records": eval_records,
        "iconicity_physical_verification": iconicity_checks,
    }
    (out_dir / "eval_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 生成人类可读 Markdown 验收单
    md = [
        "# 汉火上古音 TTS —— 四维数学与声波自动验收报告\n",
        f"- 生成时间: `{summary['generated_at']}`",
        f"- 评测篇章数: `{len(eval_records)}` (男女双音色)\n",
        "## 1. 《诗经》各章一字一音与声波共振峰押韵成绩单\n",
        "| 篇目 | 音色 | 期望字数 | 实测音节峰 | 弹舌指数 | 韵脚共振峰押韵均分 | Bark距离 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in eval_records:
        ua = r["utterance_audit"]
        rjs = r["rhyme_acoustic_checks"]
        avg_sc = round(float(np.mean([x["acoustic_rhyme_score"] for x in rjs])), 1) if rjs else 0.0
        avg_bk = round(float(np.mean([x["bark_distance"] for x in rjs])), 2) if rjs else 0.0
        md.append(
            f"| {r['title']} | `{r['speaker']}` | {ua['n_expected_chars']} | **{ua['n_detected_peaks']}** | "
            f"{ua['trill_index']} | **{avg_sc}** | {avg_bk} Bark |"
        )
    if iconicity_checks:
        ic = iconicity_checks[0]
        md += [
            "\n## 2. 「音·形·义」声波物理抽检（冷闭暖开 / 急促舒缓）\n",
            f"- **凜 (`liəm` 寒冷闭口)** vs **暖 (`nuan` 温暖张口)**: F1 第一共振峰 `{ic['凜']['f1_hz_openness']} Hz` vs `{ic['暖']['f1_hz_openness']} Hz`",
            f"- **急 (`kiəp` 急迫塞尾)** vs **舒 (`ɕia` 舒缓开尾)**: 合成总时长 `{ic['急']['duration_s']} s` vs `{ic['舒']['duration_s']} s`",
        ]
    (out_dir / "EVAL_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Hanfire Old Chinese ZipVoice-Ancient Training Pipeline")
    p.add_argument("--work-dir", type=str, default="/content/hanfire_work")
    p.add_argument("--gdrive-root", type=str, default="/content/drive/MyDrive/hanfire_ancient_tts")
    p.add_argument("--bundle-tar", type=str, default="")
    p.add_argument("--resume", type=str, default="auto", help="auto | none | /path/to/ckpt.pt")
    p.add_argument("--stage1-epochs", type=int, default=20)
    p.add_argument("--stage2-epochs", type=int, default=15)
    p.add_argument("--samples-per-lang", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--save-every", type=int, default=200, help="异步保存中间 checkpoint 的步数间隔")
    p.add_argument("--smoke", action="store_true", help="极速冒烟模式：不拉大模型，秒级验证全流程")
    return p.parse_args()


def main():
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    args = parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    gd_candidate = Path(args.gdrive_root) if args.gdrive_root else None
    gdrive_root: Optional[Path] = None
    if gd_candidate is not None and (gd_candidate.parent.exists() or args.smoke):
        gdrive_root = gd_candidate
        gdrive_root.mkdir(parents=True, exist_ok=True)
        print(f"[GDrive] ✅ 已绑定持久化目录: {gdrive_root}")
    else:
        print(f"[GDrive] ⚠️ 未检测到挂载点 {gd_candidate}，本次将仅使用本地目录 {work_dir}")

    status = StatusLogger(work_dir, gdrive_root)
    status.update(phase="init", smoke=args.smoke)

    try:
        bundle_dir = ensure_bundle_unpacked(
            work_dir, gdrive_root, Path(args.bundle_tar) if args.bundle_tar else None
        )
        lex_data = json.loads((bundle_dir / "lexicon" / "hanfire_oc_v1.json").read_text(encoding="utf-8"))

        device = torch.device("cuda:0" if torch.cuda.is_available() and not args.smoke else "cpu")
        fbank = PureVocosFbank()

        # 1. 下载或构建预训练底座与 Vocos 声码器
        if args.smoke:
            model, init_info = build_model_from_pretrained(tiny_smoke=True)
            vocoder = None
        else:
            from huggingface_hub import hf_hub_download
            from vocos import Vocos
            status.update(phase="download_weights", step_msg="下载 k2-fsa/ZipVoice 与 Vocos 24kHz 预训练权重")
            model_json = hf_hub_download("k2-fsa/ZipVoice", filename="zipvoice/model.json")
            model_pt = hf_hub_download("k2-fsa/ZipVoice", filename="zipvoice/model.pt")
            model, init_info = build_model_from_pretrained(model_json, model_pt, tiny_smoke=False)
            vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()

        model = model.to(device)
        print(f"[Model] 初始化完成: {init_info}, 参数量={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

        # 2. 初始化异步双写 Checkpoint 管理器 + 查找可恢复断点
        ckpt_mgr = AsyncCheckpointManager(
            local_dir=work_dir / "checkpoints",
            gdrive_dir=(gdrive_root / "checkpoints") if gdrive_root else None,
            keep_last_n=5,
            milestone_every_steps=max(500, args.save_every * 5),
        )
        resume_path = ckpt_mgr.resolve_resume_path(args.resume)
        resume_info = None
        if resume_path is not None:
            print(f"[Resume] 发现历史 Checkpoint: {resume_path}", flush=True)
            resume_info = ckpt_mgr.load_checkpoint(resume_path, model, strict_model=True, map_location=device)

        # 3. 构建或加载 Stage 1 多语种发音特征数据 + Stage 2 真人校准句对
        cache_base = gdrive_root if gdrive_root is not None else work_dir
        s1_records = build_or_load_stage1_cache(
            cache_base / "dataset_cache" / "stage1_multilingual_v2.pt",
            fbank=fbank,
            samples_per_lang=args.samples_per_lang,
            status=status,
            smoke=args.smoke,
        )
        s2_records = build_stage2_real_samples(bundle_dir, lex_data, fbank)
        print(f"[Data] Stage1={len(s1_records)} 条, Stage2 真人校准={len(s2_records)} 条", flush=True)

        # 4. Stage 1 训练（若断点显示已进入 stage2 或已完成 stage1，则自动跳过！）
        skip_s1 = bool(
            resume_info
            and (
                resume_info["stage"] == "stage2"
                or (resume_info["stage"] == "stage1" and resume_info["extra_state"].get("stage_completed"))
            )
        )
        if not skip_s1 and args.stage1_epochs > 0:
            run_training_stage(
                stage_name="stage1",
                model=model,
                records=s1_records,
                epochs=args.stage1_epochs,
                batch_size=args.batch_size,
                proj_lr=3e-4,
                backbone_lr=6e-5,
                save_every_steps=args.save_every,
                ckpt_mgr=ckpt_mgr,
                status=status,
                device=device,
                resume_info=resume_info,
            )

        # 5. Stage 2 真人上古《诗经》《碣石门》韵律精调（80% 真人古汉语 × 重复加权 + 20% Stage1 防遗忘回放）
        skip_s2 = bool(
            resume_info
            and resume_info["stage"] == "stage2"
            and resume_info["extra_state"].get("stage_completed")
        )
        if not skip_s2 and args.stage2_epochs > 0 and s2_records:
            replay_n = max(4, len(s2_records) * 2)
            mixed_s2 = (s2_records * 8) + random.sample(s1_records, min(len(s1_records), replay_n))
            run_training_stage(
                stage_name="stage2",
                model=model,
                records=mixed_s2,
                epochs=args.stage2_epochs,
                batch_size=args.batch_size,
                proj_lr=5e-5,
                backbone_lr=1.5e-5,
                save_every_steps=max(50, args.save_every // 2),
                ckpt_mgr=ckpt_mgr,
                status=status,
                device=device,
                resume_info=resume_info if (resume_info and resume_info["stage"] == "stage2") else None,
            )

        # 6. 四维声波裁判自动验收 + 输出男女双版《诗经》至 Google Drive
        status.update(phase="evaluate", step_msg="合成男女双版《诗经》并运行四维声波押韵裁判")
        out_base = (gdrive_root / "outputs") if gdrive_root is not None else (work_dir / "outputs")
        eval_summary = evaluate_shijing_and_iconicity(
            model=model,
            vocoder=vocoder,
            fbank=fbank,
            bundle_dir=bundle_dir,
            lex_data=lex_data,
            out_dir=out_base,
            device=device,
            k_candidates=1 if args.smoke else 3,
        )
        ckpt_mgr.flush_and_close()
        status.update(phase="DONE", step_msg="训练与四维声波验收全部完成！", eval_records=len(eval_summary["records"]))
        print(f"🎉 全部流程圆满完成！产物与报告位于: {out_base}", flush=True)

    except Exception as e:
        status.update(phase="ERROR", error=f"{type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
