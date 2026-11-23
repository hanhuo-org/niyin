#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键从 Hugging Face 数据集 (`hanhuo-org/hanhuo-niyin-dataset`) 按需下载并解压「汉火拟音」训练素材与唱吟母带。

支持模块化按需拉取：
  --raw-assets  (~472 MB) 上古/中古/近古原始音韵与示范音频素材 + 《上古汉语音节表.xlsx》 -> assets/
  --tts-train   (~650 MB) 三代 TTS 训练集 (v5/v10_train_dataset.pt, 24kHz 人声干声, 小切韵/西游记对齐) -> hanfire_tts/
  --sing-tang   (~695 MB) 唐调唱吟处理后母带 (bodies/clauses/tracks/manifests + local_finetune) -> hanfire_sing/
  --showcase    (~121 MB) 《诗经》V9.6 + 古文/辞赋 V16 男女双声部离线对比试听音频 -> hanfire_tts/web_wasm/dist_cf/_sing_showcase/
  --all                   拉取上述全部模块

用法：
  python scripts/download_dataset.py --sing-tang
  python scripts/download_dataset.py --tts-train --raw-assets
  python scripts/download_dataset.py --all
"""
from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "hanhuo-org/hanhuo-niyin-dataset"


def extract_tar(tar_path: Path, dst_dir: Path, remove_tar: bool = False) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    marker = dst_dir / f".extracted_{tar_path.name}"
    if marker.exists():
        print(f"  [SKIP] 已解压: {tar_path.name}", flush=True)
        return
    print(f"  [EXT ] 解压 {tar_path.name} -> {dst_dir} ...", flush=True)
    mode = "r:gz" if tar_path.name.endswith(".gz") else "r:"
    with tarfile.open(tar_path, mode) as tf:
        tf.extractall(dst_dir, filter="data")
    marker.write_text("ok\n", encoding="utf-8")
    if remove_tar:
        tar_path.unlink()


def main() -> None:
    ap = argparse.ArgumentParser(description="从 Hugging Face 下载汉火拟音模块化训练数据集")
    ap.add_argument("--repo", default=DEFAULT_REPO, help=f"HF Dataset 仓库 (默认 {DEFAULT_REPO})")
    ap.add_argument("--raw-assets", action="store_true", help="下载原始三代音韵/吟诵音频素材 (~472 MB)")
    ap.add_argument("--tts-train", action="store_true", help="下载三代 TTS 精切训练集与词表 (~650 MB)")
    ap.add_argument("--sing-tang", action="store_true", help="下载唐调唱吟母带与微调语料 (~695 MB)")
    ap.add_argument("--showcase", action="store_true", help="下载离线试听台展示音频 (~121 MB)")
    ap.add_argument("--all", action="store_true", help="下载全部模块")
    ap.add_argument("--no-extract", action="store_true", help="仅下载压缩包不自动解压")
    ap.add_argument("--remove-tar", action="store_true", help="解压完成后删除缓存压缩包")
    args = ap.parse_args()

    if not any([args.raw_assets, args.tts_train, args.sing_tang, args.showcase, args.all]):
        ap.error("请至少指定一个模块: --raw-assets / --tts-train / --sing-tang / --showcase / --all")

    cache = ROOT / "data" / "_hf_cache"

    if args.all or args.raw_assets:
        print("[1] 下载三代历史汉语原始素材 -> assets/", flush=True)
        dl = Path(snapshot_download(repo_id=args.repo, repo_type="dataset", allow_patterns=["raw_assets/**"], local_dir=str(cache)))
        if not args.no_extract:
            for tp in sorted((dl / "raw_assets").glob("*.tar*")):
                extract_tar(tp, ROOT / "assets", remove_tar=args.remove_tar)

    if args.all or args.tts_train:
        print("[2] 下载三代 TTS 训练集与对齐清单 -> hanfire_tts/", flush=True)
        dl = Path(snapshot_download(repo_id=args.repo, repo_type="dataset", allow_patterns=["tts_training/**"], local_dir=str(cache)))
        src = dl / "tts_training"
        if not args.no_extract:
            for tp in sorted(src.glob("*.tar*")):
                extract_tar(tp, ROOT / "hanfire_tts", remove_tar=args.remove_tar)
            if (src / "v10_train_dataset.pt").exists():
                dst_v10 = ROOT / "hanfire_tts" / "v10_bundle" / "v10_train_dataset.pt"
                dst_v10.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src / "v10_train_dataset.pt", dst_v10)

    if args.all or args.sing_tang:
        print("[3] 下载唐调唱吟母带与语料 -> hanfire_sing/", flush=True)
        dl = Path(snapshot_download(repo_id=args.repo, repo_type="dataset", allow_patterns=["sing_tang_corpus/**"], local_dir=str(cache)))
        if not args.no_extract:
            for tp in sorted((dl / "sing_tang_corpus").glob("*.tar*")):
                extract_tar(tp, ROOT / "hanfire_sing", remove_tar=args.remove_tar)

    if args.all or args.showcase:
        print("[4] 下载离线对比试听台音频 -> hanfire_tts/web_wasm/dist_cf/_sing_showcase/", flush=True)
        dl = Path(snapshot_download(repo_id=args.repo, repo_type="dataset", allow_patterns=["showcase_audio/**"], local_dir=str(cache)))
        if not args.no_extract:
            for tp in sorted((dl / "showcase_audio").glob("*.tar*")):
                extract_tar(tp, ROOT / "hanfire_tts" / "web_wasm" / "dist_cf", remove_tar=args.remove_tar)

    print("\n✅ 全部所选训练素材就绪！", flush=True)


if __name__ == "__main__":
    main()
