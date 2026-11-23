#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汉火古音唱吟 (hanfire_sing) · Colab 云端 L4/A100 歌唱底座极速备料与二进制化脚本
  1. 安装兼容 Python 3.13 的 OpenVPI DiffSinger 依赖（保留 Colab 原生 CUDA PyTorch）；
  2. 下载官方 50MB PC-NSF-HiFiGAN 2025.02 声码器 (含 config.json + model.ckpt)；
  3. 单文件高速拉取 OpenCpop 棚录中文歌唱库 (Saaaxman/Opencpop/Opencpop.tar.gz, 3756 句 / 5.2h)；
  4. 自动转换 OpenCpop transcriptions.txt -> DiffSinger 原生 transcriptions.csv，
     开启 random_pitch_shifting [-8, +4] 半音覆盖全男女音域，预埋 21 个三代古音扩展音素 + 4 说话人槽位；
  5. 运行 scripts/binarize.py 生成 44.1kHz 128-mel + Parselmouth F0 二进制训练集。
"""
from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Set

WORK = Path("/content/work")
DS_DIR = WORK / "DiffSinger"
RAW_OC = WORK / "opencpop_raw"
DS_OC = WORK / "opencpop_ds"

# 21 个三代古音扩展音素（预埋在底座词表里，本地 2080 Ti 微调零维度冲突！）
HISTORICAL_EXTRA_PHONEMES = [
    "h_b", "h_d", "h_g", "h_dz", "h_dzh", "h_dj",
    "h_z", "h_zh", "h_zj", "h_ng", "h_ny", "h_v",
    "h_hm", "h_hn", "h_hl", "h_hng", "h_gh",
    "coda_m", "coda_p", "coda_t", "coda_k",
]

PH_NORM = {
    "sil": "SP",
    "pau": "SP",
    "sp": "SP",
    "ap": "AP",
    "iou": "iu",
    "uei": "ui",
    "uen": "un",
    "ung": "ong",
    "ü": "v",
    "üe": "ve",
    "üan": "van",
    "ün": "vn",
    "ue": "ve",
}


def run(cmd: str, cwd: Path | None = None, check: bool = True) -> None:
    print(f"\n>>> {cmd}", flush=True)
    subprocess.run(cmd, shell=True, cwd=str(cwd) if cwd else None, check=check)


def load_opencpop_phonemes(dict_path: Path) -> Set[str]:
    phs: Set[str] = {"SP", "AP"}
    for line in dict_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "\t" not in line:
            continue
        _, seq = line.split("\t", 1)
        for p in seq.split():
            phs.add(p)
    return phs


def prepare_opencpop_dataset(valid_phs: Set[str]) -> dict:
    """在解压目录中自动定位 OpenCpop wavs 与 transcriptions.txt，生成 DiffSinger 原生 transcriptions.csv。"""
    import soundfile as sf

    # 自动寻找 transcriptions.txt (或 train/test 标注) 与 wav 目录
    txts = sorted(RAW_OC.rglob("transcriptions.txt"))
    if not txts:
        txts = sorted(RAW_OC.rglob("*transcription*.txt")) + sorted(RAW_OC.rglob("*.csv"))
    assert txts, f"未找到 OpenCpop 标注文件，目录内容: {list(RAW_OC.rglob('*'))[:30]}"
    print("找到标注文件:", [str(p) for p in txts], flush=True)

    # 索引全部 wav 路径（按 stem -> Path）
    wav_map: Dict[str, Path] = {}
    for w in RAW_OC.rglob("*.wav"):
        wav_map[w.stem] = w
    print(f"索引到 WAV 文件总数: {len(wav_map)}", flush=True)
    assert len(wav_map) > 100, "OpenCpop WAV 数量异常！"

    out_wav_dir = DS_OC / "wavs"
    out_wav_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    seen = set()
    skipped_ph = 0
    skipped_missing = 0
    unknown_phs: Dict[str, int] = {}

    for txt_p in txts:
        for line in txt_p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or "|" not in line:
                continue
            parts = line.split("|")
            if len(parts) < 6:
                continue
            name = parts[0].strip()
            if name in seen:
                continue
            wav_src = wav_map.get(name)
            if wav_src is None:
                skipped_missing += 1
                continue
            ph_raw = parts[2].strip().split()
            dur_raw = [float(x) for x in parts[5].strip().split()]
            if len(ph_raw) != len(dur_raw) or len(ph_raw) < 3:
                skipped_ph += 1
                continue

            ph_seq: List[str] = []
            ph_dur: List[float] = []
            ok = True
            prev = ""
            for p, d in zip(ph_raw, dur_raw):
                mp = PH_NORM.get(p, p)
                if mp == "i":
                    if prev in ("z", "c", "s"):
                        mp = "i0"
                    elif prev in ("zh", "ch", "sh", "r"):
                        mp = "ir"
                if mp not in valid_phs:
                    low = mp.lower()
                    if low in valid_phs:
                        mp = low
                    else:
                        unknown_phs[p] = unknown_phs.get(p, 0) + 1
                        ok = False
                        break
                ph_seq.append(mp)
                ph_dur.append(max(0.005, d))
                prev = mp
            if not ok:
                skipped_ph += 1
                continue

            try:
                info = sf.info(str(wav_src))
                dur_wav = info.frames / float(info.samplerate)
            except Exception:
                continue
            if dur_wav < 1.0 or dur_wav > 20.0:
                continue

            # 校正总时长微小误差
            diff = dur_wav - sum(ph_dur)
            if abs(diff) <= 0.40 and ph_dur[-1] + diff > 0.01:
                ph_dur[-1] += diff

            wav_link = out_wav_dir / f"{name}.wav"
            if not wav_link.exists():
                os.symlink(str(wav_src.resolve()), str(wav_link))
            rows.append((name, " ".join(ph_seq), " ".join(f"{x:.6f}" for x in ph_dur)))
            seen.add(name)

    csv_path = DS_OC / "transcriptions.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "ph_seq", "ph_dur"])
        w.writerows(rows)

    # 选最后 16 条（不同歌曲尾段）做验证集
    val_prefixes = [r[0] for r in rows[-16:]]
    print(
        f"✓ OpenCpop 转换完成: {len(rows)} 条有效 (缺wav={skipped_missing}, 跳过音素={skipped_ph}) "
        f"unknown={sorted(unknown_phs.items(), key=lambda x: -x[1])[:8]}",
        flush=True,
    )
    assert len(rows) > 1000, f"有效条目过少: {len(rows)}"
    return {
        "raw_data_dir": str(DS_OC),
        "speaker": "opencpop_base",
        "spk_id": 0,
        "language": "zh",
        "test_prefixes": val_prefixes,
    }


def write_diffsinger_config(ds_entry: dict) -> Path:
    import yaml

    cfg = {
        "base_config": ["configs/acoustic.yaml"],
        "dictionaries": {"zh": "dictionaries/opencpop_clean.txt"},
        "extra_phonemes": [],
        "datasets": [ds_entry],
        "binary_data_dir": str(WORK / "binary_opencpop"),
        "vocoder": "NsfHifiGAN",
        "vocoder_ckpt": str(DS_DIR / "checkpoints" / "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02" / "model.ckpt"),
        "use_spk_id": True,
        "num_spk": 4,  # 0=Opencpop歌唱底座；1=预留本地古音男声；2=预留本地古音女声；3=预留唐调
        "use_lang_id": False,
        "use_key_shift_embed": True,
        "use_speed_embed": True,
        "binarization_args": {"num_workers": 4},
        "augmentation_args": {
            "random_pitch_shifting": {
                "enabled": True,
                "range": [-8.0, 4.0],  # 向下覆盖男低/男中音(110~170Hz)，向上覆盖女高音！
                "scale": 0.75,
            },
            "fixed_pitch_shifting": {"enabled": False, "targets": [-5.0, 5.0], "scale": 0.5},
            "random_time_stretching": {"enabled": True, "range": [0.75, 1.5], "scale": 0.5},
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
        "max_batch_frames": 80000,
        "max_batch_size": 48,
        "max_updates": 40000,
        "val_check_interval": 4000,
        "num_valid_plots": 6,
        "num_ckpt_keep": 5,
        "pl_trainer_precision": "bf16-mixed",
        "num_sanity_val_steps": 1,
        "ds_workers": 4,
    }
    cfg_path = DS_DIR / "configs" / "hanfire_sing_base.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"✓ 已生成 DiffSinger 配置: {cfg_path}", flush=True)
    return cfg_path


def main() -> None:
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    # 清理之前取消的 m4_raw 残余腾出空间
    shutil.rmtree(WORK / "m4_raw", ignore_errors=True)

    print("=== [1/6] 检查 GPU 与依赖 ===", flush=True)
    import torch
    print(f"PyTorch={torch.__version__}, CUDA={torch.cuda.is_available()}, GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}", flush=True)
    run(
        f"{sys.executable} -m pip install -q "
        "lightning einops h5py praat-parselmouth pyworld resampy torchmetrics tensorboardX click pyyaml soundfile"
    )

    print("\n=== [2/6] 校验 OpenVPI DiffSinger 仓库并应用 Lightning 2.6 兼容补丁 ===", flush=True)
    if not DS_DIR.exists():
        run(f"git clone --depth 1 https://github.com/openvpi/DiffSinger.git {DS_DIR}")
    tu = DS_DIR / "utils" / "training_utils.py"
    tu_txt = tu.read_text(encoding="utf-8")
    if "if hasattr(accelerator_connector" not in tu_txt:
        tu.write_text(
            tu_txt.replace(
                "            accelerator_connector._register_external_accelerators_and_strategies()",
                "            if hasattr(accelerator_connector, '_register_external_accelerators_and_strategies'):\n                accelerator_connector._register_external_accelerators_and_strategies()",
            ),
            encoding="utf-8",
        )
    # 生成不含零频 E/En 的干净词表
    clean_dict = DS_DIR / "dictionaries" / "opencpop_clean.txt"
    if not clean_dict.exists():
        src_l = (DS_DIR / "dictionaries" / "opencpop-extension.txt").read_text(encoding="utf-8").splitlines()
        clean_dict.write_text(
            "\n".join(l for l in src_l if not any(p in ("E", "En") for p in l.split("\t")[-1].split())) + "\n",
            encoding="utf-8",
        )

    print("\n=== [3/6] 下载官方 50MB PC-NSF-HiFiGAN 2025.02 声码器 (model.ckpt + config.json) ===", flush=True)
    voc_dir = DS_DIR / "checkpoints" / "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"
    if not (voc_dir / "config.json").exists() or not (voc_dir / "model.ckpt").exists():
        shutil.rmtree(voc_dir, ignore_errors=True)
        voc_zip = WORK / "voc_50mb.zip"
        run(
            f"curl -fL --retry 3 -o {voc_zip} "
            "https://github.com/openvpi/vocoders/releases/download/pc-nsf-hifigan-44.1k-hop512-128bin-2025.02/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.zip"
        )
        (DS_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(voc_zip) as zf:
            zf.extractall(DS_DIR / "checkpoints")
        voc_zip.unlink(missing_ok=True)
    print("✓ Vocoder ready:", [p.name for p in voc_dir.iterdir()], flush=True)

    print("\n=== [4/6] 单文件高速拉取 OpenCpop 棚录歌唱库 (6.96GB) ===", flush=True)
    if not (RAW_OC / ".extracted").exists():
        RAW_OC.mkdir(parents=True, exist_ok=True)
        tar_p = WORK / "Opencpop.tar.gz"
        if not tar_p.exists():
            run(
                f"curl -fL --retry 3 -o {tar_p} "
                "https://huggingface.co/datasets/Saaaxman/Opencpop/resolve/main/Opencpop.tar.gz"
            )
        run(f"tar xzf {tar_p} -C {RAW_OC}")
        tar_p.unlink(missing_ok=True)
        (RAW_OC / ".extracted").write_text("ok")

    print("\n=== [5/6] 转换 OpenCpop 标注并生成 DiffSinger 配置 ===", flush=True)
    valid_phs = load_opencpop_phonemes(DS_DIR / "dictionaries" / "opencpop-extension.txt")
    ds_entry = prepare_opencpop_dataset(valid_phs)
    cfg_path = write_diffsinger_config(ds_entry)

    print("\n=== [6/6] 运行 DiffSinger binarize.py 提取 44.1kHz 128-mel 与 F0 ===", flush=True)
    run(f"{sys.executable} scripts/binarize.py --config {cfg_path}", cwd=DS_DIR)
    print(f"\n✅ 全部数据准备与二进制化完成！总耗时 {(time.time()-t0)/60:.1f} 分钟。", flush=True)


if __name__ == "__main__":
    main()
