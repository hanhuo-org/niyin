#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汉火古音唱吟 · 本地 RTX 2080 Ti 推理试唱脚本 (infer_tang_chant.py)

支持直接用【云端歌唱底座】或【本地唐调微调权重】，将任意古诗文以
「上古 OC / 中古 MC / 近古 EM / 现代 ZH」+「唐调五大体裁（诗经四言/五绝/七绝/古文/辞赋）」
合成为 44.1kHz 录音室级清唱/吟诵 WAV！

示例：
  /mnt/s790/project/hanzi_train/.venv/bin/python hanfire_sing/scripts/infer_tang_chant.py \
    --text "蒹葭苍苍，白露为霜。所谓伊人，在水一方。" \
    --era oc --genre shijing_4 --key_hz 145 --spk opencpop_base \
    --out hanfire_sing/demo_jianjia_oc.wav
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf

SING_DIR = Path(__file__).resolve().parents[1]
DS_DIR = SING_DIR / "DiffSinger"

if str(SING_DIR) not in sys.path:
    sys.path.insert(0, str(SING_DIR))
from tang_melody_prior import compile_ancient_tang_ds


def auto_find_exp_and_ckpt(exp: str | None) -> Tuple[str, Path]:
    """自动定位最新可用的实验目录（默认使用清晰度最高的云端 40k 底座 hanfire_sing_base）。"""
    cands = [exp] if exp else ["hanfire_sing_base", "hanfire_tang_local"]
    for name in cands:
        for base in (DS_DIR / "checkpoints" / name, SING_DIR / "checkpoints" / name):
            if not base.exists():
                continue
            ckpts = sorted(
                base.glob("model_ckpt_steps_*.ckpt"),
                key=lambda p: int(p.stem.split("_")[-1]),
            )
            if ckpts:
                ds_exp = DS_DIR / "checkpoints" / name
                if not ds_exp.exists():
                    ds_exp.symlink_to(base.resolve())
                return name, ckpts[-1]
    raise FileNotFoundError("未找到任何可用的 DiffSinger 权重，请先拉取云端底座或完成本地微调！")


VOICE_PRESETS = {
    # 严格落在 40k 底座原生纯净甜区（绝不把女声底座强压至 130Hz 引发“嗓子有痰”破音！）
    "female": {"key_hz": 294.0, "gender": 0.0},     # D4 原生清雅女声（最纯净！）
    "bright": {"key_hz": 330.0, "gender": 0.08},    # E4 清亮古风女声
    "alto":   {"key_hz": 220.0, "gender": -0.40},   # A3 温润中低音/青衣书生腔（零痰音！）
    "male":   {"key_hz": 196.0, "gender": -0.52},   # G3 温和男中音上限（保底不破音）
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="古诗文内容（标点分句）")
    ap.add_argument("--era", default="mc", choices=["oc", "mc", "em", "zh"])
    ap.add_argument("--genre", default="shijing_4", choices=["shijing_4", "wujue_5", "qijue_7", "guwen", "fu"])
    ap.add_argument("--voice", default="female", choices=["female", "bright", "alto", "male", "custom"], help="预置纯净人声（默认清雅女声 female）")
    ap.add_argument("--key_hz", type=float, default=None, help="自定义基准音高 Hz")
    ap.add_argument("--gender", type=float, default=None, help="声道共振峰移调 [-1.0~+1.0]，男声约 -0.75，女声约 -0.12")
    ap.add_argument("--spk", default="opencpop_base", help="说话人名 (默认 opencpop_base)")
    ap.add_argument("--tempo", type=float, default=1.0, help="唱速缩放 (>1 更慢更舒展, <1 更轻快)")
    ap.add_argument("--exp", default="hanfire_sing_base", help="实验名 (默认 hanfire_sing_base)")
    ap.add_argument("--ckpt", type=int, default=None, help="指定步数 checkpoint")
    ap.add_argument("--steps", type=int, default=20, help="Rectified Flow 采样步数 (12~25)")
    ap.add_argument("--out", required=True, help="输出 44.1kHz WAV 路径")
    args = ap.parse_args()

    preset = VOICE_PRESETS.get(args.voice, VOICE_PRESETS["male"])
    key_hz = args.key_hz if args.key_hz is not None else preset["key_hz"]
    gender = args.gender if args.gender is not None else preset["gender"]

    exp_name, ckpt_path = auto_find_exp_and_ckpt(args.exp)
    ckpt_step = args.ckpt if args.ckpt is not None else int(ckpt_path.stem.split("_")[-1])
    print(f"使用实验: {exp_name} (step={ckpt_step}) | 时代={args.era} | 体裁={args.genre} | 基音={key_hz}Hz | gender={gender} | 音色={args.spk}")

    ds_items = compile_ancient_tang_ds(
        text=args.text,
        era=args.era,
        genre=args.genre,
        key_hz=key_hz,
        speaker=args.spk,
        tempo_scale=args.tempo,
    )
    assert ds_items, "未能编译出有效唱吟乐句！"

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        ds_file = td_p / "score.ds"
        out_dir = td_p / "wavs"
        out_dir.mkdir(parents=True, exist_ok=True)
        ds_file.write_text(json.dumps(ds_items, ensure_ascii=False, indent=2), encoding="utf-8")

        cmd = [
            sys.executable, "scripts/infer.py", "acoustic",
            str(ds_file),
            "--exp", exp_name,
            "--ckpt", str(ckpt_step),
            "--spk", args.spk,
            f"--gender={gender}",
            "--steps", str(args.steps),
            "--out", str(out_dir),
        ]
        subprocess.run(cmd, cwd=str(DS_DIR), check=True)

        wav_files = sorted(out_dir.glob("*.wav"))
        assert wav_files, "DiffSinger 未生成 WAV 文件！"
        chunks = []
        sr = 44100
        for wf in wav_files:
            y, sr = sf.read(str(wf), dtype="float32")
            chunks.append(y)
            chunks.append(np.zeros(int(0.28 * sr), dtype=np.float32))
        full = np.concatenate(chunks[:-1]) if len(chunks) > 1 else chunks[0]

        # 无损峰值保护归一化 (RMS=0.085, Peak<=0.72)
        rms = float(np.sqrt(np.mean(full ** 2)) + 1e-8)
        gain = 0.085 / rms
        peak = float(np.max(np.abs(full)) + 1e-8)
        if peak * gain > 0.72:
            gain = 0.72 / peak
        full = (full * gain).astype(np.float32)

        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_p), full, sr)
        print(f"✅ 古音唐调吟诵合成完成 -> {out_p} ({len(full)/sr:.2f}s, {sr}Hz)")


if __name__ == "__main__":
    main()
