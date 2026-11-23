#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键下载并安装「汉火拟音 (Hanfire Niyin v1.0)」预训练模型与运行时权重包。

默认从 GitHub Release (`hanhuo-org/niyin` tag `v1.0`) 拉取，也支持：
  - `--source hf` 从 Hugging Face (`hanhuo-org/hanhuo-niyin-models`) 拉取；
  - `--local-dir release_artifacts` 从本地打包目录离线安装。

支持按需模块化安装：
  (默认)          Web 浏览器离线三代古音 TTS 运行包 (~168 MB) -> `hanfire_tts/web_wasm/dist_cf/`
  --tts          三代古音 TTS 完整 PyTorch/ONNX 权重 + 词表与参考锚 (~756 MB)
  --sing         唐调《诗经》/古文/辞赋唱吟男女双声部 DiffSinger + 44.1kHz 声码器 + 活腔母带 (~877 MB)
  --all          一次性安装全部 Web + TTS + 唱吟模型 (~1.8 GB)

用法：
  python scripts/download_weights.py                       # 仅装 Web 离线运行包 (~168 MB)
  python scripts/download_weights.py --tts                 # Web + TTS PyTorch/ONNX 权重
  python scripts/download_weights.py --sing                # 唱吟双声部底座 + 声码器 + 唐调活腔母带
  python scripts/download_weights.py --all                 # 全部安装
  python scripts/download_weights.py --local-dir release_artifacts --all
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "hanhuo-org/niyin"
DEFAULT_HF_REPO = "hanhuo-org/hanhuo-niyin-models"
DEFAULT_TAG = "v1.0"

WEB_ASSETS = ["niyin-tts-v10-web-models.tar.gz"]
TTS_ASSETS = [
    "niyin_tts_v10_studio_fp32.pt",
    "niyin_tts_v10_studio_int8.pt",
    "niyin_tts_v10_onnx_int8.tar.gz",
    "niyin_tts_lexicon_and_refs.tar.gz",
]
SING_ASSETS = [
    "niyin_sing_female_base_40k.tar.gz",
    "niyin_sing_male_m4_40k.tar.gz",
    "niyin_sing_vocoder_pc_nsf_hifigan_44k.tar.gz",
    "niyin_sing_tang_bodies.tar.gz",
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_url(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"  [GET] {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "HanfireNiyinInstaller/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            f.write(buf)
            done += len(buf)
            if total > 0:
                print(
                    f"\r        {done/1048576:6.1f} / {total/1048576:6.1f} MB ({100*done/total:5.1f}%)",
                    end="",
                    flush=True,
                )
        print("", flush=True)
    tmp.replace(dst)


def fetch_asset(name: str, repo: str, hf_repo: str, tag: str, source: str, cache_dir: Path, local_dir: Path | None) -> Path:
    if local_dir is not None:
        src = local_dir / name
        if not src.exists():
            raise FileNotFoundError(f"本地目录中未找到 {src}")
        return src
    dst = cache_dir / name
    if dst.exists() and dst.stat().st_size > 0:
        print(f"  [HIT] {dst} ({dst.stat().st_size/1048576:.1f} MB)", flush=True)
        return dst
    if source == "hf":
        from huggingface_hub import hf_hub_download
        p = Path(hf_hub_download(repo_id=hf_repo, filename=f"release/{name}", repo_type="model"))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        return dst
    url = f"https://github.com/{repo}/releases/download/{tag}/{name}"
    download_url(url, dst)
    return dst


def safe_extract(tar_path: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(dst_dir, filter="data")


def patch_sing_configs() -> None:
    """将 DiffSinger 男女声底座 config.yaml 中的路径自动指向当前仓库绝对路径，并创建符号链接。"""
    sing_dir = ROOT / "hanfire_sing"
    ds_ckpt = sing_dir / "DiffSinger" / "checkpoints"
    ds_ckpt.mkdir(parents=True, exist_ok=True)
    vocoder_ckpt = (ds_ckpt / "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02" / "model.ckpt").resolve()

    for name in ("hanfire_sing_base", "hanfire_sing_male"):
        cdir = sing_dir / "checkpoints" / name
        if not cdir.is_dir():
            continue
        cfg = cdir / "config.yaml"
        if cfg.exists():
            txt = cfg.read_text(encoding="utf-8")
            txt = re.sub(r"(?m)^binary_data_dir:.*$", f"binary_data_dir: {cdir.resolve()}", txt)
            txt = re.sub(r"(?m)^vocoder_ckpt:.*$", f"vocoder_ckpt: {vocoder_ckpt}", txt)
            cfg.write_text(txt, encoding="utf-8")
        link = ds_ckpt / name
        if link.is_symlink() or link.exists():
            if link.is_symlink():
                link.unlink()
        if not link.exists():
            os.symlink(cdir.resolve(), link)


def main() -> None:
    ap = argparse.ArgumentParser(description="下载并安装汉火拟音 (Hanfire Niyin) 预训练权重")
    ap.add_argument("--repo", default=DEFAULT_REPO, help=f"GitHub 仓库 (默认 {DEFAULT_REPO})")
    ap.add_argument("--hf-repo", default=DEFAULT_HF_REPO, help=f"HF 模型仓库 (默认 {DEFAULT_HF_REPO})")
    ap.add_argument("--tag", default=DEFAULT_TAG, help=f"GitHub Release 标签 (默认 {DEFAULT_TAG})")
    ap.add_argument("--source", choices=("github", "hf"), default="github", help="下载源 (github 或 hf)")
    ap.add_argument("--tts", action="store_true", help="安装三代古音 TTS 完整 PyTorch/ONNX 权重与词表")
    ap.add_argument("--sing", action="store_true", help="安装唐调唱吟男女双声部 DiffSinger + 声码器 + 活腔母带")
    ap.add_argument("--all", action="store_true", help="一次性安装全部 Web + TTS + 唱吟模型")
    ap.add_argument("--local-dir", type=Path, default=None, help="从本地 release_artifacts/ 目录离线安装")
    args = ap.parse_args()

    cache_dir = ROOT / "release_artifacts"
    sums: dict[str, str] = {}
    try:
        sp = fetch_asset("SHA256SUMS.txt", args.repo, args.hf_repo, args.tag, args.source, cache_dir, args.local_dir)
        for line in sp.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) == 2:
                sums[parts[1]] = parts[0]
    except Exception:
        pass

    targets: list[str] = []
    if args.all:
        targets = WEB_ASSETS + TTS_ASSETS + SING_ASSETS
    else:
        if not args.sing or args.tts:
            targets.extend(WEB_ASSETS)
        if args.tts:
            targets.extend(TTS_ASSETS)
        if args.sing:
            targets.extend(SING_ASSETS)

    for name in targets:
        p = fetch_asset(name, args.repo, args.hf_repo, args.tag, args.source, cache_dir, args.local_dir)
        if name in sums:
            got = sha256_file(p)
            if got != sums[name]:
                raise RuntimeError(f"SHA256 校验失败: {name} (got={got}, expected={sums[name]})")
            print(f"  [OK ] SHA256 verified: {name}", flush=True)

        if name == "niyin-tts-v10-web-models.tar.gz":
            dst = ROOT / "hanfire_tts" / "web_wasm" / "dist_cf"
            print(f"  [EXT] 解压 Web 运行时模型包 -> {dst}", flush=True)
            safe_extract(p, dst)
        elif name == "niyin_tts_v10_studio_fp32.pt":
            dst = ROOT / "hanfire_tts" / "v10_outputs" / "best_fp32_v10_studio.pt"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            print(f"  [CP ] 安装 TTS FP32 权重 -> {dst}", flush=True)
        elif name == "niyin_tts_v10_studio_int8.pt":
            dst = ROOT / "hanfire_tts" / "v10_outputs" / "best_int8_v10_studio.pt"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            print(f"  [CP ] 安装 TTS INT8 权重 -> {dst}", flush=True)
        elif name == "niyin_tts_v10_onnx_int8.tar.gz":
            dst = ROOT / "hanfire_tts" / "web_wasm" / "models"
            print(f"  [EXT] 解压 TTS ONNX INT8 -> {dst}", flush=True)
            safe_extract(p, dst)
        elif name == "niyin_tts_lexicon_and_refs.tar.gz":
            dst = ROOT / "hanfire_tts"
            print(f"  [EXT] 解压三代词表与参考锚 -> {dst}", flush=True)
            safe_extract(p, dst)
        elif name in ("niyin_sing_female_base_40k.tar.gz", "niyin_sing_male_m4_40k.tar.gz"):
            dst = ROOT / "hanfire_sing" / "checkpoints"
            print(f"  [EXT] 解压唱吟声部底座 -> {dst}", flush=True)
            safe_extract(p, dst)
        elif name == "niyin_sing_vocoder_pc_nsf_hifigan_44k.tar.gz":
            dst = ROOT / "hanfire_sing" / "DiffSinger" / "checkpoints"
            print(f"  [EXT] 解压 44.1kHz 声码器 -> {dst}", flush=True)
            safe_extract(p, dst)
        elif name == "niyin_sing_tang_bodies.tar.gz":
            dst = ROOT / "hanfire_sing" / "data_tang_processed"
            print(f"  [EXT] 解压唐调真人活腔母带 -> {dst}", flush=True)
            safe_extract(p, dst)

    if any(x in targets for x in SING_ASSETS):
        patch_sing_configs()
        print("  [CFG] 已自动校准 DiffSinger config.yaml 路径与符号链接", flush=True)

    print("\n✅ 汉火拟音权重安装完成！", flush=True)
    if WEB_ASSETS[0] in targets:
        print("  ▸ 启动本地离线 Web 工作台: python hanfire_tts/web_wasm/serve_web.py --port 8080", flush=True)
    if args.sing or args.all:
        print("  ▸ 渲染《诗经》/古文/辞赋唱吟: python hanfire_sing/scripts/render_guwen_lab.py", flush=True)


if __name__ == "__main__":
    main()
