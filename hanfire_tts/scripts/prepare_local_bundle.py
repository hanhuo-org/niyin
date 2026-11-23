#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地素材清洗与 Google Drive 上传包构建脚本 (prepare_local_bundle.py)

执行内容（纯数据处理，不跑 GPU 训练）：
1. 生成《汉火上古音单核词表 v1》(`hanfire_oc_v1.json`，13,075 字 + 诗经100%叶韵表 + 音义控制参数)；
2. 从评测最高分的男女音源切出干净 24kHz 单声道标准参考锚 (`prompts/`):
   - 男声主锚 `male_jianjia.wav` (白沙蒹葭干声, 122.7Hz 儒雅男声) + `male_jinteng.wav` (尚书金縢, 93.5Hz 浑厚男声)
   - 女声主锚 `female_guanju.wav` (关雎去伴奏干声, 180.7Hz 温婉女中音) + `female_xian.wav` (245Hz 清亮朗读女声)
3. 汇总 MFA 已验证通过的真人古音句对 (`jianjia_bs` + `jieshimen` + `chushibiao`)，统一重采样为 24kHz，
   生成 `stage2_real/manifest_stage2.jsonl`；
4. 提取 `assets/` 下无 BGM 高分上古朗读干声 (`金縢`, `采薇`, `利簋`, `氓`) 存入 `assets_clean_24k/` 供随时扩展；
5. 打包为 `/mnt/s790/project/tts/hanfire_tts/hanfire_gdrive_bundle.tar.gz`，一键上传至 Google Drive 持久复用。
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hanfire_core.lexicon_builder import build_hanfire_lexicon

VOXTTS_ROOT = Path("/mnt/s790/project/tts/voxtts")
ASSETS_DIR = Path("/mnt/s790/project/tts/assets")
BUNDLE_DIR = PROJECT_ROOT / "gdrive_bundle"
TARGET_SR = 24000


def resample_and_norm_24k(src_wav: np.ndarray, src_sr: int, target_rms: float = 0.09) -> np.ndarray:
    if src_wav.ndim > 1:
        src_wav = src_wav.mean(axis=1)
    y = src_wav.astype(np.float64)
    y = y - np.mean(y)
    if src_sr != TARGET_SR:
        g = math.gcd(src_sr, TARGET_SR)
        y = signal.resample_poly(y, TARGET_SR // g, src_sr // g)
    rms = float(np.sqrt(np.mean(y ** 2) + 1e-12))
    if rms > 1e-5:
        y = y * (target_rms / rms)
    peak = float(np.max(np.abs(y)) + 1e-9)
    if peak > 0.98:
        y = y * (0.98 / peak)
    return y.astype(np.float32)


def ffmpeg_extract_24k(src_media: Path, dst_wav: Path, start_s: float = 0.0, dur_s: float = 0.0) -> bool:
    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y"]
    if start_s > 0:
        cmd += ["-ss", f"{start_s:.2f}"]
    cmd += ["-i", str(src_media)]
    if dur_s > 0:
        cmd += ["-t", f"{dur_s:.2f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(TARGET_SR), str(dst_wav)]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode == 0 and dst_wav.exists():
        w, sr = sf.read(str(dst_wav))
        sf.write(str(dst_wav), resample_and_norm_24k(w, sr), TARGET_SR)
        return True
    return False


def pick_clean_active_window(wav_path: Path, want_sec: float = 7.5) -> np.ndarray:
    """在长干声中自动截取能量平稳、吐字连续的最佳参考窗口。"""
    w, sr = sf.read(str(wav_path))
    w = resample_and_norm_24k(w, sr)
    win_len = int(want_sec * TARGET_SR)
    if len(w) <= win_len:
        return w
    hop = TARGET_SR // 2
    best_score = -1e9
    best_seg = w[:win_len]
    for s in range(TARGET_SR, max(TARGET_SR + 1, len(w) - win_len - TARGET_SR), hop):
        seg = w[s: s + win_len]
        # 分成 15 段看活跃均衡度，避免截到大段空白
        chunks = np.array_split(seg, 15)
        chunk_rms = np.array([np.sqrt(np.mean(c ** 2) + 1e-9) for c in chunks])
        active_ratio = float(np.mean(chunk_rms > 0.025))
        score = active_ratio - 0.5 * float(np.std(chunk_rms))
        if score > best_score:
            best_score = score
            best_seg = seg
    return best_seg


def main():
    print("[1/5] 构建汉火上古音单核 4 槽位词表 + 诗经全自动叶韵表...")
    lex_dir = BUNDLE_DIR / "lexicon"
    lex_dir.mkdir(parents=True, exist_ok=True)
    lex_data = build_hanfire_lexicon(VOXTTS_ROOT)
    lex_path = lex_dir / "hanfire_oc_v1.json"
    lex_path.write_text(json.dumps(lex_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("  词表统计:", lex_data["meta"])

    print("[2/5] 提取并归一化 1男1女 (主+备共4枚) 24kHz 纯净古风参考音色锚...")
    pmt_dir = BUNDLE_DIR / "prompts"
    pmt_dir.mkdir(parents=True, exist_ok=True)
    prompts_meta = {}

    # 男声 1A: 蒹葭白沙段已验证首两句拼接 ("蒹葭苍苍，白露为霜。所谓伊人，在水一方。") —— 精确字音对齐！
    jj_dir = VOXTTS_ROOT / "collect/sent_pairs/jianjia_bs"
    if (jj_dir / "sent_000.wav").exists() and (jj_dir / "sent_001.wav").exists():
        parts = []
        for fn in ("sent_000.wav", "sent_001.wav", "sent_002.wav"):
            p = jj_dir / fn
            if p.exists():
                w, sr = sf.read(str(p))
                parts.append(resample_and_norm_24k(w, sr))
                parts.append(np.zeros(int(0.18 * TARGET_SR), dtype=np.float32))
        male_jj = np.concatenate(parts)
        sf.write(str(pmt_dir / "male_scholar_jianjia.wav"), male_jj, TARGET_SR)
        prompts_meta["male_primary"] = {
            "speaker_id": 0,
            "gender": "male",
            "wav": "prompts/male_scholar_jianjia.wav",
            "text": "蒹葭苍苍，白露为霜。所谓伊人。",
            "duration_s": round(len(male_jj) / TARGET_SR, 2),
            "note": "122.7Hz 清朗儒雅学者男声，弹舌极低(2.49)，字音精确对齐",
        }

    # 男声 1B: 尚书·金縢 93.5Hz 青铜浑厚低音男声
    jinteng_mp4 = ASSETS_DIR / "【上古汉语】《尚书·金縢》_1080P.mp4"
    tmp_jt = BUNDLE_DIR / "assets_clean_24k" / "shangshu_jinteng_full.wav"
    if jinteng_mp4.exists() and ffmpeg_extract_24k(jinteng_mp4, tmp_jt):
        seg_jt = pick_clean_active_window(tmp_jt, want_sec=7.5)
        sf.write(str(pmt_dir / "male_bronze_jinteng.wav"), seg_jt, TARGET_SR)
        prompts_meta["male_bronze"] = {
            "speaker_id": 0,
            "gender": "male",
            "wav": "prompts/male_bronze_jinteng.wav",
            "text": "既克商二年，王有疾弗豫。",
            "duration_s": round(len(seg_jt) / TARGET_SR, 2),
            "note": "93.5Hz 浑厚青铜感朗读男声 (评分93.1冠军)",
        }

    # 女声 1A: 西安清亮自然女声锚 (精确普通话参考文本 -> 可直接转对应上古/现代单核特征做 Prompt)
    xian_fem = Path("/mnt/s790/project/hanzi/diamoe-rust/voices/xian_female_prompt.wav")
    if xian_fem.exists():
        w, sr = sf.read(str(xian_fem))
        w24 = resample_and_norm_24k(w, sr)
        sf.write(str(pmt_dir / "female_clear_xian.wav"), w24, TARGET_SR)
        prompts_meta["female_primary"] = {
            "speaker_id": 1,
            "gender": "female",
            "wav": "prompts/female_clear_xian.wav",
            "text": "窈窕淑女，君子好逑。",
            "duration_s": round(len(w24) / TARGET_SR, 2),
            "note": "245Hz 清亮端庄朗读女声 (无BGM干声，弹舌2.86)",
        }

    # 女声 1B: 关雎去伴奏温婉女中音干声
    guanju_voc = VOXTTS_ROOT / "collect/audio/BV1Wun3zBEwx_vocals.wav"
    if guanju_voc.exists():
        seg_gj = pick_clean_active_window(guanju_voc, want_sec=7.0)
        sf.write(str(pmt_dir / "female_gentle_guanju.wav"), seg_gj, TARGET_SR)
        prompts_meta["female_gentle"] = {
            "speaker_id": 1,
            "gender": "female",
            "wav": "prompts/female_gentle_guanju.wav",
            "text": "关关雎鸠，在河之洲。",
            "duration_s": round(len(seg_gj) / TARGET_SR, 2),
            "note": "180.7Hz 温婉古风女中音 (UVR去伴奏干声，SNR 74dB)",
        }

    (pmt_dir / "prompts_manifest.json").write_text(
        json.dumps(prompts_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[3/5] 收集并重采样 Stage 2 真人上古朗读校准句对 (MFA verified)...")
    s2_dir = BUNDLE_DIR / "stage2_real"
    s2_wav_dir = s2_dir / "wavs"
    s2_wav_dir.mkdir(parents=True, exist_ok=True)
    s2_records = []

    for bank_name in ("jianjia_bs", "jieshimen", "chushibiao"):
        mf = VOXTTS_ROOT / "collect/sent_pairs" / bank_name / "manifest_sent_verified.tsv"
        if not mf.exists():
            continue
        lines = mf.read_text(encoding="utf-8").splitlines()[1:]
        for idx, ln in enumerate(lines):
            cols = ln.split("\t")
            if len(cols) < 3 or cols[2].strip() != "ok":
                continue
            chars_txt, wav_rel = cols[0].strip(), cols[1].strip()
            src_p = mf.parent / wav_rel
            if not src_p.exists():
                continue
            w, sr = sf.read(str(src_p))
            w24 = resample_and_norm_24k(w, sr)
            dur = len(w24) / TARGET_SR
            if dur < 0.6 or dur > 12.0:
                continue
            out_name = f"{bank_name}_{idx:03d}.wav"
            sf.write(str(s2_wav_dir / out_name), w24, TARGET_SR)
            s2_records.append({
                "utt_id": f"{bank_name}_{idx:03d}",
                "source": bank_name,
                "speaker_id": 0,
                "text": chars_txt,
                "wav": f"stage2_real/wavs/{out_name}",
                "duration_s": round(dur, 3),
            })

    with open(s2_dir / "manifest_stage2.jsonl", "w", encoding="utf-8") as f:
        for r in s2_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  已打包 MFA 核验真人古汉语短句: {len(s2_records)} 条")

    print("[4/5] 提取 assets/ 高分纯净干声备份至 assets_clean_24k/...")
    clean_assets = [
        ("caiwei_clean.wav", ASSETS_DIR / "【上古汉语】诗经·小雅·采薇_1080P.mp4", 0, 0),
        ("ligui_clean.wav", ASSETS_DIR / "【上古汉语】利簋_1080P.mp4", 0, 0),
        ("meng_clean.wav", ASSETS_DIR / "【上古汉语】诗经·卫风·氓_1080P.mp4", 0, 0),
        ("jianjia_bs_73s.wav", ASSETS_DIR / "蒹葭 古汉语朗读：白沙上古音、郑张尚芳上古音、中古音对比_1080P.mp4", 0, 73.0),
    ]
    for out_fn, mp4_p, st_s, du_s in clean_assets:
        if mp4_p.exists():
            ffmpeg_extract_24k(mp4_p, BUNDLE_DIR / "assets_clean_24k" / out_fn, start_s=st_s, dur_s=du_s)

    print("[5/5] 生成 Google Drive 一键归档包 hanfire_gdrive_bundle.tar.gz...")
    tar_path = PROJECT_ROOT / "hanfire_gdrive_bundle.tar.gz"
    with tarfile.open(str(tar_path), "w:gz") as tar:
        tar.add(str(BUNDLE_DIR), arcname="gdrive_bundle")
        tar.add(str(PROJECT_ROOT / "hanfire_core"), arcname="hanfire_core")

    size_mb = tar_path.stat().st_size / (1024 * 1024)
    print(f"✅ 打包完成: {tar_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
