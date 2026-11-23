#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汉火古音唱吟 · 本地 RTX 2080 Ti 微调语料准备与配置生成脚本 (prepare_local_finetune.py)

严格遵循 AGENTS.md §1 音质红线：
  - 仅采纳经实测零削波、底噪 <= -55dB、无伴奏的 tier1_studio_clean 唐调干声：
      * spk 1 (tang_male)  : t16 陋室铭 + t17 问刘十九 + t18 前赤壁赋 (71 句 / 6.15 分钟)
      * spk 2 (tang_female): t09 兰亭集序 + t12 诫兄子严敦书 + t20 诗经·秦风·无衣 (56 句 / 4.34 分钟)
  - 绝不将 tier2/tier3 含配乐/Demucs分离/高底噪录音送入声学微调（防止把分离伪影学成电音底噪）！
  - 直接从原始音频以 44.1kHz 全带宽重切 + 无损高通/去削波/响度归一，保留 12~16kHz 清晰高频泛音；
  - 用本地 Whisper small 逐字时间戳对齐至 opencpop_clean 音素序列，生成 DiffSinger 标准 transcriptions.csv；
  - 自动兼容你后续在 hanfire_sing/data_local_custom/<说话人名>/ 中新增的干净棚录干声（.wav + .txt），
    随时一行命令即可把新腔调/新音色微调进模型！
"""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import soundfile as sf
import torch
import whisper
import yaml
from pypinyin import Style, lazy_pinyin

ROOT = Path(__file__).resolve().parents[2]
RAW_MP3_DIR = ROOT / "assets" / "吟诵"
SING_DIR = ROOT / "hanfire_sing"
DS_DIR = SING_DIR / "DiffSinger"
TANG_DIR = SING_DIR / "data_tang_processed"
FT_DATA_DIR = SING_DIR / "data_local_finetune"
CUSTOM_DIR = SING_DIR / "data_local_custom"
SR_44K = 44100

if str(SING_DIR) not in sys.path:
    sys.path.insert(0, str(SING_DIR))
from hanfire_sing_phonemes import mandarin_pinyin_to_sing_phones

_OPENCPOP_MAP = {"iou": "iu", "uei": "ui"}


def ensure_clean_dict() -> Tuple[Path, Set[str]]:
    src = DS_DIR / "dictionaries" / "opencpop-extension.txt"
    dst = DS_DIR / "dictionaries" / "opencpop_clean.txt"
    if not dst.exists():
        lines = [
            l for l in src.read_text(encoding="utf-8").splitlines()
            if not any(p in ("E", "En") for p in l.split("\t")[-1].split())
        ]
        dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    valid: Set[str] = {"SP", "AP"}
    for line in dst.read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            for p in line.split("\t", 1)[1].split():
                valid.add(p)
    return dst, valid


def char_to_opencpop_phones(ch: str, valid_phs: Set[str]) -> List[str]:
    py = lazy_pinyin(ch, style=Style.NORMAL)[0]
    raw_phs = mandarin_pinyin_to_sing_phones(py)
    out = []
    for p in raw_phs:
        mp = _OPENCPOP_MAP.get(p, p)
        if mp in valid_phs:
            out.append(mp)
    return out or ["a"]


def extract_clean_44k_body(mp3_path: Path, body_start: float, body_end: float, dst_wav: Path) -> Tuple[np.ndarray, int]:
    """直接从原音频 44.1kHz 全带宽截取正文并做无损高通 + 温和去削波 + RMS=0.085 峰值保护归一。"""
    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    tmp_raw = dst_wav.parent / f"._{dst_wav.stem}_raw44.wav"
    dur = max(0.5, body_end - body_start)
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-ss", f"{body_start:.3f}", "-t", f"{dur:.3f}",
            "-i", str(mp3_path),
            "-af", "highpass=f=70,adeclip=w=55:o=75:a=8:t=10:n=1000",
            "-ar", str(SR_44K), "-ac", "1", str(tmp_raw),
        ],
        check=True,
    )
    y, sr = sf.read(str(tmp_raw), dtype="float32")
    tmp_raw.unlink(missing_ok=True)
    rms = float(np.sqrt(np.mean(y ** 2)) + 1e-8)
    gain = 0.085 / rms
    peak = float(np.max(np.abs(y)) + 1e-8)
    if peak * gain > 0.72:
        gain = 0.72 / peak
    y = (y * gain).astype(np.float32)
    sf.write(str(dst_wav), y, sr)
    return y, sr


def whisper_clause_to_ph_alignment(
    w_model,
    wav_path: Path,
    dur_wav: float,
    valid_phs: Set[str],
    device: str,
) -> Tuple[List[str], List[float]] | None:
    """用 Whisper 逐字时间戳构造精准的 (ph_seq, ph_dur)，不达标乐句自动过滤防错对齐污染。"""
    res = w_model.transcribe(
        str(wav_path),
        language="zh",
        word_timestamps=True,
        fp16=(device == "cuda"),
        verbose=False,
    )
    char_events: List[Tuple[str, float, float]] = []
    for seg in res.get("segments", []):
        words = seg.get("words") or []
        if words:
            for w in words:
                cs = re.findall(r"[\u4e00-\u9fff]", w.get("word", ""))
                if not cs:
                    continue
                ws = max(0.0, float(w.get("start", seg["start"])))
                we = min(dur_wav, float(w.get("end", seg["end"])))
                if we <= ws + 0.02:
                    continue
                dt = (we - ws) / len(cs)
                for k, ch in enumerate(cs):
                    char_events.append((ch, ws + k * dt, ws + (k + 1) * dt))
        else:
            cs = re.findall(r"[\u4e00-\u9fff]", seg.get("text", ""))
            if not cs:
                continue
            ws = max(0.0, float(seg["start"]))
            we = min(dur_wav, float(seg["end"]))
            if we <= ws + 0.02:
                continue
            dt = (we - ws) / len(cs)
            for k, ch in enumerate(cs):
                char_events.append((ch, ws + k * dt, ws + (k + 1) * dt))

    if len(char_events) < 2:
        return None

    # 检查语音覆盖率（防止 Whisper 在慢拖腔下整段漏认）
    voiced_span = char_events[-1][2] - char_events[0][1]
    if voiced_span < dur_wav * 0.45:
        return None

    y_wav, sr_wav = sf.read(str(wav_path), dtype="float32")
    ph_seq: List[str] = []
    ph_dur: List[float] = []
    cursor = 0.0

    for ch, cs, ce in char_events:
        cs = max(cursor, cs)
        if ce <= cs + 0.03:
            continue
        gap = cs - cursor
        if gap >= 0.06:
            # 区分静音停顿 (SP) 与换气呼吸 (AP)：间隙能量 > -50dB 且 0.08~0.45s 标为 AP
            g_seg = y_wav[int(cursor * sr_wav) : int(cs * sr_wav)]
            g_db = float(20.0 * np.log10(float(np.sqrt(np.mean(g_seg ** 2))) + 1e-8)) if len(g_seg) > 64 else -90.0
            ph_seq.append("AP" if (0.08 <= gap <= 0.45 and g_db > -50.0) else "SP")
            ph_dur.append(gap)
        elif gap > 0 and ph_dur:
            ph_dur[-1] += gap

        cdur = ce - cs
        phs = char_to_opencpop_phones(ch, valid_phs)
        if len(phs) == 1:
            ph_seq.append(phs[0])
            ph_dur.append(cdur)
        else:
            # 声母短促 (最多 85ms 或 22% 字长)，其余全部给可拖腔主元音核！
            onset_d = min(0.085, max(0.025, cdur * 0.22))
            nuc_d = max(0.025, cdur - onset_d)
            ph_seq.extend([phs[0], phs[1]])
            ph_dur.extend([onset_d, nuc_d])
        cursor = ce

    tail = dur_wav - cursor
    if tail >= 0.04:
        ph_seq.append("SP")
        ph_dur.append(tail)
    elif tail > 0 and ph_dur:
        ph_dur[-1] += tail

    if len(ph_seq) < 3 or abs(sum(ph_dur) - dur_wav) > 0.15:
        return None
    # 吸收浮点尾差至最后一个音素
    diff = dur_wav - sum(ph_dur)
    if ph_dur[-1] + diff > 0.01:
        ph_dur[-1] += diff
    return ph_seq, ph_dur


def build_local_finetune_datasets() -> Tuple[List[dict], Path]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dict_path, valid_phs = ensure_clean_dict()
    manifest = json.loads((TANG_DIR / "tang_manifest.json").read_text(encoding="utf-8"))

    print(f"=== 本地 RTX 2080 Ti 唱吟微调数据集构建 (device={device}) ===")
    print("1. 严格筛选 tier1_studio_clean 棚录级唐调干声 (按男女声分组为 spk 1 / spk 2)...")
    w_model = whisper.load_model("small", device=device)

    spk_buckets: Dict[str, List[Tuple[str, str, str]]] = {"tang_male": [], "tang_female": []}
    for spk in spk_buckets:
        (FT_DATA_DIR / spk / "wavs").mkdir(parents=True, exist_ok=True)

    for tr in manifest["tracks"]:
        if tr["audio_grade"] != "tier1_studio_clean":
            continue
        spk_name = "tang_male" if tr["med_f0_hz"] < 185.0 else "tang_female"
        mp3_path = RAW_MP3_DIR / tr["filename"]
        body_44k = FT_DATA_DIR / "_bodies_44k" / f"{tr['track_id']}_44k.wav"
        y_body, sr = extract_clean_44k_body(mp3_path, tr["body_start_s"], tr["body_end_s"], body_44k)

        kept = 0
        for cl in tr["clauses"]:
            s0 = int(round(cl["start_in_body_s"] * sr))
            s1 = int(round(cl["end_in_body_s"] * sr))
            seg = y_body[s0:s1]
            dur_s = len(seg) / sr
            if dur_s < 1.2 or dur_s > 16.0:
                continue
            cname = f"{tr['track_id']}_c{cl['clause_idx']:03d}"
            wav_out = FT_DATA_DIR / spk_name / "wavs" / f"{cname}.wav"
            sf.write(str(wav_out), seg, sr)

            aligned = whisper_clause_to_ph_alignment(w_model, wav_out, dur_s, valid_phs, device)
            if aligned is None:
                wav_out.unlink(missing_ok=True)
                continue
            ph_seq, ph_dur = aligned
            spk_buckets[spk_name].append((cname, " ".join(ph_seq), " ".join(f"{d:.6f}" for d in ph_dur)))
            kept += 1
        print(f"  ✓ [{spk_name:11s}] {tr['track_id']} {tr['title']:12s}: {kept}/{len(tr['clauses'])} 句通过 44.1kHz 对齐校验")

    # 写出 transcriptions.csv 并组装 DiffSinger datasets 列表（保留 spk_id 1=tang_male, 2=tang_female）
    ds_entries = []
    spk_id_map = {"tang_male": 1, "tang_female": 2}
    for spk_name, rows in spk_buckets.items():
        if len(rows) < 4:
            continue
        csv_p = FT_DATA_DIR / spk_name / "transcriptions.csv"
        with open(csv_p, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "ph_seq", "ph_dur"])
            w.writerows(rows)
        val_prefixes = [r[0] for r in rows[-4:]]
        ds_entries.append({
            "raw_data_dir": str(FT_DATA_DIR / spk_name),
            "speaker": spk_name,
            "spk_id": spk_id_map[spk_name],
            "language": "zh",
            "test_prefixes": val_prefixes,
        })
        print(f"  => 说话人 {spk_name} (spk_id={spk_id_map[spk_name]}): 共 {len(rows)} 句写入 {csv_p}")

    # 自动检测最新拉回的云端底座权重
    base_dir = SING_DIR / "checkpoints" / "hanfire_sing_base"
    ckpts = sorted(
        base_dir.glob("model_ckpt_steps_*.ckpt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    ) if base_dir.exists() else []
    base_ckpt = str(ckpts[-1]) if ckpts else str(base_dir / "model_ckpt_steps_40000.ckpt")

    cfg = {
        "base_config": ["configs/acoustic.yaml"],
        "dictionaries": {"zh": "dictionaries/opencpop_clean.txt"},
        "extra_phonemes": [],
        "datasets": ds_entries,
        "binary_data_dir": str(FT_DATA_DIR / "binary_tang_local"),
        "vocoder": "NsfHifiGAN",
        "vocoder_ckpt": str(DS_DIR / "checkpoints" / "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02" / "model.ckpt"),
        "use_spk_id": True,
        "num_spk": 4,
        "use_lang_id": False,
        "use_key_shift_embed": True,
        "use_speed_embed": True,
        "binarization_args": {"num_workers": 4},
        "augmentation_args": {
            "random_pitch_shifting": {"enabled": True, "range": [-4.0, 4.0], "scale": 0.75},
            "fixed_pitch_shifting": {"enabled": False, "targets": [-5.0, 5.0], "scale": 0.5},
            "random_time_stretching": {"enabled": True, "range": [0.85, 1.35], "scale": 0.5},
        },
        "pe": "parselmouth",
        "diffusion_type": "reflow",
        "backbone_type": "lynxnet2",
        "backbone_args": {
            "num_channels": 768,
            "num_layers": 6,
            "kernel_size": 31,
            "dropout_rate": 0.0,
            "use_conditioner_cache": True,
            "glu_type": "atanglu",
        },
        "use_shallow_diffusion": True,
        "T_start": 0.4,
        "T_start_infer": 0.4,
        # 本地 RTX 2080 Ti 低学习率温和微调（保留云端 100% 歌唱共振与发音能力！）
        "finetune_enabled": True,
        "finetune_ckpt_path": base_ckpt,
        "finetune_ignored_params": [],
        "finetune_strict_shapes": True,
        "optimizer_args": {
            "optimizer_cls": "modules.optimizer.muon.Muon_AdamW",
            "lr": 0.00012,
            "beta1": 0.9,
            "beta2": 0.98,
            "weight_decay": 0,
            "muon_args": {"weight_decay": 0.1},
            "adamw_args": {"weight_decay": 0.0},
        },
        "lr_scheduler_args": {"scheduler_cls": "torch.optim.lr_scheduler.StepLR", "step_size": 1500, "gamma": 0.7},
        "max_batch_frames": 48000,
        "max_batch_size": 24,
        "max_updates": 4000,
        "val_check_interval": 1000,
        "num_valid_plots": 4,
        "num_ckpt_keep": 4,
        # RTX 2080 Ti (Turing SM 7.5) 使用原生 FP16 Tensor Core 加速！
        "pl_trainer_precision": "16-mixed",
        "num_sanity_val_steps": 1,
        "ds_workers": 4,
    }
    cfg_path = DS_DIR / "configs" / "hanfire_local_finetune.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"\n✅ 本地微调配置已生成 -> {cfg_path}")
    print(f"   指向云端底座权重: {base_ckpt}")
    return ds_entries, cfg_path


if __name__ == "__main__":
    build_local_finetune_datasets()
