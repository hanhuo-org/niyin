#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汉火古音唱吟 (hanfire_sing_male) · Colab 云端 A100/L4 纯正男声多说话人底座备料与二进制化脚本
  1. 安装兼容 OpenVPI DiffSinger 依赖（保留 Colab 原生 CUDA PyTorch）；
  2. 下载官方 50MB PC-NSF-HiFiGAN 2025.02 声码器 (44.1kHz, hop512, 128bin)；
  3. 从 cyjin-yl/m4singerForDiffSingerSVS 高速拉取 6 位顶级棚录男声（10.35 小时 / 7,281 句 44.1kHz 干声 + 毫秒级 TextGrid）：
       - spk 0: m4_bass1  (Bass-1,  148.7 min, 78~131Hz  醇厚学者男低中音)
       - spk 1: m4_tenor7 (Tenor-7, 144.6 min, 117~247Hz 纯净清朗抒情男声，男版 OpenCpop)
       - spk 2: m4_tenor5 (Tenor-5, 105.6 min, 123~233Hz 温润儒雅男中音)
       - spk 3: m4_bass3  (Bass-3,   85.9 min, 92~123Hz  沉稳磁性男低音)
       - spk 4: m4_tenor6 (Tenor-6,  67.1 min, 110~247Hz 明亮昂扬男高音，适配秦风豪放调)
       - spk 5: m4_tenor2 (Tenor-2,  68.8 min, 139~311Hz 清越男高音，扩展高音区泛化)
       - spk 6..7: 预留本地唐调男声 / 自定义男声微调槽位
  4. 严格将全部 TextGrid 音素归一至与女声底座 100% 一致的 opencpop_clean.txt 词表（零 OOD 音素，无缝复用 shijing_v9_engine.py！）；
  5. 运行 scripts/binarize.py 提取 44.1kHz 128-mel + Parselmouth F0 (f0_min=55Hz) 二进制训练集。
"""
from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Set, Tuple

WORK = Path("/content/work")
DS_DIR = WORK / "DiffSinger"
RAW_M4 = WORK / "m4_male_raw"
DS_M4 = WORK / "m4_male_ds"

MALE_SPEAKERS: List[Tuple[str, str, int]] = [
    ("Bass-1",  "m4_bass1",  0),
    ("Tenor-7", "m4_tenor7", 1),
    ("Tenor-5", "m4_tenor5", 2),
    ("Bass-3",  "m4_bass3",  3),
    ("Tenor-6", "m4_tenor6", 4),
    ("Tenor-2", "m4_tenor2", 5),
]

PH_NORM = {
    "<SP>": "SP", "<AP>": "AP",
    "sil": "SP", "pau": "SP", "sp": "SP", "ap": "AP", "": "SP",
    "iou": "iu", "uei": "ui", "uen": "un", "ung": "ong",
    "ü": "v", "üe": "ve", "üan": "van", "ün": "vn", "ue": "ve",
    # opencpop_clean.txt 不含独立 E/En，按前后语境安全归并
    "E": "e", "En": "an",
}


def run(cmd: str, cwd: Path | None = None, check: bool = True) -> None:
    print(f"\n>>> {cmd}", flush=True)
    subprocess.run(cmd, shell=True, cwd=str(cwd) if cwd else None, check=check)


def load_clean_phonemes(dict_path: Path) -> Set[str]:
    phs: Set[str] = {"SP", "AP"}
    for line in dict_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "\t" not in line:
            continue
        _, seq = line.split("\t", 1)
        for p in seq.split():
            phs.add(p)
    return phs


def parse_phones_tier(tg_path: Path) -> List[Tuple[float, float, str]]:
    """纯正则高速解析 TextGrid 的 phones 音频对齐层，零第三方库格式兼容隐患。"""
    txt = tg_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r'name\s*=\s*"phones"', txt)
    if not m:
        return []
    sub = txt[m.start():]
    # 截断到下一个 item（如有）
    nxt = re.search(r'\n\s*item\s*\[\d+\]:', sub[10:])
    if nxt:
        sub = sub[: 10 + nxt.start()]
    pat = re.compile(
        r'xmin\s*=\s*([0-9.eE+-]+)\s+xmax\s*=\s*([0-9.eE+-]+)\s+text\s*=\s*"([^"]*)"'
    )
    out: List[Tuple[float, float, str]] = []
    for xmin_s, xmax_s, label in pat.findall(sub):
        x0, x1 = float(xmin_s), float(xmax_s)
        if x1 > x0 + 1e-4:
            out.append((x0, x1, label.strip()))
    return out


def prepare_male_datasets(valid_phs: Set[str]) -> List[dict]:
    import soundfile as sf

    ds_entries: List[dict] = []
    total_clips = 0
    total_sec = 0.0

    for raw_spk, spk_name, spk_id in MALE_SPEAKERS:
        wav_dir_in = RAW_M4 / raw_spk / "wav"
        tg_dir_in = RAW_M4 / raw_spk / "TextGrid"
        if not wav_dir_in.is_dir() or not tg_dir_in.is_dir():
            print(f"⚠️ 跳过缺失目录: {raw_spk}", flush=True)
            continue

        spk_out = DS_M4 / spk_name
        wav_out = spk_out / "wavs"
        wav_out.mkdir(parents=True, exist_ok=True)

        rows = []
        skipped_ph = 0
        skipped_dur = 0
        unknown_phs: Dict[str, int] = {}
        spk_sec = 0.0

        wav_files = sorted(wav_dir_in.glob("*.wav"))
        for wf in wav_files:
            stem = wf.stem
            tg_f = tg_dir_in / f"{stem}.TextGrid"
            if not tg_f.exists():
                continue
            intervals = parse_phones_tier(tg_f)
            if len(intervals) < 3:
                continue

            try:
                info = sf.info(str(wf))
                dur_wav = info.frames / float(info.samplerate)
            except Exception:
                continue
            if dur_wav < 1.2 or dur_wav > 18.0:
                skipped_dur += 1
                continue

            ph_seq: List[str] = []
            ph_dur: List[float] = []
            ok = True
            prev = ""
            for x0, x1, raw_p in intervals:
                d = max(0.005, x1 - x0)
                if raw_p == "E" and prev == "y":
                    mp = "e"
                elif raw_p == "E" and prev in ("b", "p", "m", "d", "t", "n", "l", "j", "q", "x"):
                    mp = "ie"
                elif raw_p == "En" and prev == "y":
                    mp = "an"
                elif raw_p == "En" and prev in ("b", "p", "m", "d", "t", "n", "l", "j", "q", "x"):
                    mp = "ian"
                else:
                    mp = PH_NORM.get(raw_p, raw_p)

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
                        unknown_phs[raw_p] = unknown_phs.get(raw_p, 0) + 1
                        ok = False
                        break
                # 合并连续相邻的相同 SP
                if mp == "SP" and ph_seq and ph_seq[-1] == "SP":
                    ph_dur[-1] += d
                else:
                    ph_seq.append(mp)
                    ph_dur.append(d)
                prev = mp

            if not ok or len(ph_seq) < 3:
                skipped_ph += 1
                continue

            # 校正总时长微小误差
            diff = dur_wav - sum(ph_dur)
            if abs(diff) > 0.35:
                skipped_dur += 1
                continue
            if ph_dur[-1] + diff > 0.01:
                ph_dur[-1] += diff
            else:
                skipped_dur += 1
                continue

            cname = f"{spk_name}_{stem}"
            dst_link = wav_out / f"{cname}.wav"
            if not dst_link.exists():
                os.symlink(str(wf.resolve()), str(dst_link))
            rows.append((cname, " ".join(ph_seq), " ".join(f"{x:.6f}" for x in ph_dur)))
            spk_sec += dur_wav

        csv_path = spk_out / "transcriptions.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "ph_seq", "ph_dur"])
            w.writerows(rows)

        # 每位男声留最后 4 条作验证集（共 24 条跨音域验证集）
        val_prefixes = [r[0] for r in rows[-4:]]
        print(
            f"✓ [{spk_name:9s} | spk_id={spk_id}] {len(rows):4d} 有效 ({spk_sec/60:5.1f} min) "
            f"跳过音素={skipped_ph} 时长={skipped_dur} unk={sorted(unknown_phs.items(), key=lambda x: -x[1])[:5]}",
            flush=True,
        )
        assert len(rows) > 300, f"{spk_name} 有效条目过少: {len(rows)}"
        total_clips += len(rows)
        total_sec += spk_sec
        ds_entries.append({
            "raw_data_dir": str(spk_out),
            "speaker": spk_name,
            "spk_id": spk_id,
            "language": "zh",
            "test_prefixes": val_prefixes,
        })

    print(f"\n🎯 6 位纯棚录男声总计: {total_clips} 句 / {total_sec/3600:.2f} 小时 (100% 44.1kHz 干声)", flush=True)
    return ds_entries


def write_diffsinger_config(ds_entries: List[dict]) -> Path:
    import yaml

    cfg = {
        "base_config": ["configs/acoustic.yaml"],
        "dictionaries": {"zh": "dictionaries/opencpop_clean.txt"},
        "extra_phonemes": [],
        "datasets": ds_entries,
        "binary_data_dir": str(WORK / "binary_m4_male"),
        "vocoder": "NsfHifiGAN",
        "vocoder_ckpt": str(DS_DIR / "checkpoints" / "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02" / "model.ckpt"),
        "use_spk_id": True,
        "num_spk": 8,  # 0..5=6位M4Singer顶级棚录男声; 6..7=预留本地唐调男声微调槽位
        "use_lang_id": False,
        "use_key_shift_embed": True,
        "use_speed_embed": True,
        "use_stretch_embed": True,
        "f0_min": 55,  # 男低音胸声下探至 A1(55Hz)
        "f0_max": 1100,
        "binarization_args": {"num_workers": 6},
        "augmentation_args": {
            "random_pitch_shifting": {
                "enabled": True,
                "range": [-4.0, 5.0],  # 贯通男低/男中/男高全音域（65Hz~350Hz）
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
        "num_ckpt_keep": 4,
        "pl_trainer_precision": "bf16-mixed",
        "num_sanity_val_steps": 1,
        "ds_workers": 4,
    }
    cfg_path = DS_DIR / "configs" / "hanfire_sing_male.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"✓ 已生成 DiffSinger 男声配置: {cfg_path}", flush=True)
    return cfg_path


def main() -> None:
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)

    print("=== [1/6] 检查 GPU 与安装依赖 ===", flush=True)
    import torch
    print(f"PyTorch={torch.__version__}, CUDA={torch.cuda.is_available()}, GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}", flush=True)
    run(
        f"{sys.executable} -m pip install -q "
        "lightning einops h5py praat-parselmouth pyworld resampy torchmetrics tensorboardX click pyyaml soundfile huggingface_hub hf_transfer"
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
    clean_dict = DS_DIR / "dictionaries" / "opencpop_clean.txt"
    if not clean_dict.exists():
        src_l = (DS_DIR / "dictionaries" / "opencpop-extension.txt").read_text(encoding="utf-8").splitlines()
        clean_dict.write_text(
            "\n".join(l for l in src_l if not any(p in ("E", "En") for p in l.split("\t")[-1].split())) + "\n",
            encoding="utf-8",
        )

    print("\n=== [3/6] 下载官方 50MB PC-NSF-HiFiGAN 2025.02 声码器 ===", flush=True)
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

    print("\n=== [4/6] 32 线程直连高速拉取 M4Singer 6 位棚录男声 (10.35 小时 44.1kHz 干声 + TextGrid) ===", flush=True)
    if not (RAW_M4 / ".downloaded").exists():
        import concurrent.futures
        import requests
        from huggingface_hub import HfApi

        t_dl = time.time()
        api = HfApi()
        repo = "cyjin-yl/m4singerForDiffSingerSVS"
        base_url = f"https://huggingface.co/datasets/{repo}/resolve/main/"
        todo: List[str] = []
        for raw_spk, _, _ in MALE_SPEAKERS:
            for sub in ("wav", "TextGrid"):
                for item in api.list_repo_tree(repo, path_in_repo=f"{raw_spk}/{sub}", repo_type="dataset"):
                    if item.path.endswith((".wav", ".TextGrid")):
                        dst = RAW_M4 / item.path
                        if not (dst.exists() and dst.stat().st_size > 100):
                            todo.append(item.path)
        print(f"待拉取文件总数: {len(todo)} (已自动跳过断点续传已完成文件)", flush=True)

        sess = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=48, pool_maxsize=48, max_retries=3)
        sess.mount("https://", adapter)

        done_cnt = 0
        fail_cnt = 0

        def _fetch(rel_p: str) -> bool:
            dst = RAW_M4 / rel_p
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(dst.suffix + ".part")
            for attempt in range(4):
                try:
                    resp = sess.get(base_url + rel_p, timeout=30)
                    if resp.status_code == 200 and len(resp.content) > 100:
                        tmp.write_bytes(resp.content)
                        tmp.replace(dst)
                        return True
                except Exception:
                    pass
                time.sleep(0.5 * (attempt + 1))
            return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
            for ok in ex.map(_fetch, todo):
                if ok:
                    done_cnt += 1
                else:
                    fail_cnt += 1
                if (done_cnt + fail_cnt) % 1000 == 0:
                    el = time.time() - t_dl
                    print(f"  进度: {done_cnt + fail_cnt}/{len(todo)} ({done_cnt / max(1e-3, el):.1f} files/s, fail={fail_cnt})", flush=True)

        print(f"✓ 6 位男声拉取完成: ok={done_cnt}, fail={fail_cnt}, 耗时 {(time.time()-t_dl)/60:.1f} 分钟", flush=True)
        assert fail_cnt < 100, f"下载失败过多: {fail_cnt}"
        (RAW_M4 / ".downloaded").write_text("ok")

    print("\n=== [5/6] 转换 6 位男声 TextGrid 标注并生成 DiffSinger 配置 ===", flush=True)
    valid_phs = load_clean_phonemes(clean_dict)
    ds_entries = prepare_male_datasets(valid_phs)
    cfg_path = write_diffsinger_config(ds_entries)

    print("\n=== [6/6] 运行 DiffSinger binarize.py 提取 44.1kHz 128-mel 与 F0 ===", flush=True)
    run(f"{sys.executable} scripts/binarize.py --config {cfg_path}", cwd=DS_DIR)
    print(f"\n✅ 男声全部数据准备与二进制化完成！总耗时 {(time.time()-t0)/60:.1f} 分钟。", flush=True)


if __name__ == "__main__":
    main()
