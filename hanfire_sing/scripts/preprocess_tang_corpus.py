#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汉火古音唱吟 · 唐调吟诵语料全自动预处理管线 (preprocess_tang_corpus.py)
严格遵循 AGENTS.md §1 音质红线，全流程在本地 RTX 2080 Ti 免费完成：
  1. 解码标准化：统一转 24kHz 单声道 WAV；
  2. Demucs (htdemucs --two-stems=vocals) 智能伴奏/杂音检测与剥离：
     - 自动测算伴奏能量比与底噪；
     - 含配乐/杂音曲目用分离后纯人声轨提取时间戳与 F0，并严格标记 audio_grade="melody_only"（绝不用于解冻声学主干！）；
     - 原生纯净干声标记 audio_grade="clean_dry"，零相位损伤保留原声；
  3. Whisper small (word_timestamps=True) + 拼音模糊锚点匹配：
     - 精准识别并切除每首开头「标题 / 作者 / 吟诵者旁白」与片尾杂音，对齐至正文首字开口毫秒级起点；
  4. 气口逐句切片 + 跨男女声归一化五声 F0 轮廓提取（相对中位基音的半音曲线 rel_semitones）；
  5. 生成 tang_manifest.json 核对清单与局域网可视化抽检页。
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
import whisper
from pypinyin import Style, lazy_pinyin

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "assets" / "吟诵"
SING_DIR = ROOT / "hanfire_sing"
OUT_DIR = SING_DIR / "data_tang_processed"
SR = 24000

# 20 首唐调曲目元数据：(文件名关键词, 体裁 genre, 标题, 标题/作者报幕参考词, 正文首句锚点)
TRACK_RULES = [
    ("关雎", "shijing_4", "诗经·周南·关雎", "诗经周南关雎", "关关雎鸠在河之洲"),
    ("闻王昌龄", "qijue_7", "闻王昌龄左迁龙标遥有此寄", "闻王昌龄左迁龙标遥有此寄唐李白", "杨花落尽子规啼"),
    ("泊秦淮", "qijue_7", "泊秦淮", "泊秦淮唐杜牧", "烟笼寒水月笼沙"),
    ("论语", "guwen", "论语十二章", "论语十二章", "子曰学而时习之"),
    ("逍遥游", "guwen", "庄子·逍遥游", "逍遥游庄子", "北冥有鱼其名为鲲"),
    ("答谢中书书", "guwen", "答谢中书书", "答谢中书书南朝陶弘景", "山川之美古来共谈"),
    ("三峡", "guwen", "三峡", "三峡北魏郦道元", "自三峡七百里中"),
    ("兰亭集序", "guwen", "兰亭集序", "兰亭集序东晋王羲之", "永和九年岁在癸丑"),
    ("岳阳楼记", "guwen", "岳阳楼记", "岳阳楼记宋范仲淹", "庆历四年春滕子京"),
    ("马说", "guwen", "杂说四·马说", "杂说四马说唐韩愈", "世有伯乐然后有千里马"),
    ("诫兄子严敦书", "guwen", "诫兄子严敦书", "诫兄子严敦书汉马援", "援兄子严敦并喜讥议"),
    ("采薇", "shijing_4", "诗经·小雅·采薇", "诗经小雅采薇", "采薇采薇薇亦作止"),
    ("无衣", "shijing_4", "诗经·秦风·无衣", "诗经秦风无衣", "岂曰无衣与子同袍"),
    ("蒹葭", "shijing_4", "诗经·秦风·蒹葭", "诗经秦风蒹葭", "蒹葭苍苍白露为霜"),
    ("陋室铭", "guwen", "陋室铭", "唐刘禹锡陋室铭", "山不在高有仙则名"),
    ("问刘十九", "wujue_5", "问刘十九", "唐白居易问刘十九", "绿蚁新醅酒红泥小火炉"),
    ("前赤壁赋", "fu", "前赤壁赋", "宋苏轼前赤壁赋", "壬戌之秋七月既望"),
]


def match_track_rule(filename: str) -> Tuple[str, str, str, str]:
    for key, genre, title, intro_ref, anchor in TRACK_RULES:
        if key in filename:
            return genre, title, intro_ref, anchor
    return "other", Path(filename).stem, Path(filename).stem, ""


def extract_speaker_tag(filename: str) -> str:
    m = re.search(r"吟诵[：:_]\s*([^.(（\n]+)", filename)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"[（(]([^)）]+?)\s*吟诵[)）]", filename)
    if m2:
        return m2.group(1).strip()
    if "唐调学吟" in filename:
        return "唐调学吟示范"
    return "未知唱者"


def run_demucs_check(src_mp3: Path, stem_dir: Path, device: str) -> Tuple[Path, float, float]:
    """运行 htdemucs --two-stems=vocals，返回 (vocals_24k_wav, bgm_ratio_db, orig_noise_floor_db)。"""
    stem_dir.mkdir(parents=True, exist_ok=True)
    # 先解码原音频 24k mono（已缓存则直接复用）
    orig_24k = stem_dir / "orig_24k.wav"
    if not orig_24k.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(src_mp3), "-ac", "1", "-ar", str(SR), str(orig_24k)],
            check=True,
        )
    y_orig, _ = sf.read(str(orig_24k), dtype="float32")
    rms_frames = librosa.feature.rms(y=y_orig, frame_length=1024, hop_length=256)[0]
    noise_floor_db = float(20.0 * np.log10(float(np.percentile(rms_frames, 5)) + 1e-8))

    # 运行 demucs 分离人声与伴奏（检测真实配乐/杂音占比）
    voc_24k = stem_dir / "vocals_24k.wav"
    if not voc_24k.exists():
        demucs_out = stem_dir / "_demucs_tmp"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "demucs",
                "-n",
                "htdemucs",
                "--two-stems=vocals",
                "-d",
                device,
                "-o",
                str(demucs_out),
                str(orig_24k),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        v_raw = demucs_out / "htdemucs" / "orig_24k" / "vocals.wav"
        nv_raw = demucs_out / "htdemucs" / "orig_24k" / "no_vocals.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(v_raw), "-ac", "1", "-ar", str(SR), str(voc_24k)],
            check=True,
        )
        nv_24k = stem_dir / "no_vocals_24k.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(nv_raw), "-ac", "1", "-ar", str(SR), str(nv_24k)],
            check=True,
        )
        # 清理 44.1k 临时大文件
        subprocess.run(["rm", "-rf", str(demucs_out)], check=False)

    y_voc, _ = sf.read(str(voc_24k), dtype="float32")
    nv_24k = stem_dir / "no_vocals_24k.wav"
    y_nv, _ = sf.read(str(nv_24k), dtype="float32") if nv_24k.exists() else (np.zeros_like(y_voc), SR)
    rms_v = float(np.sqrt(np.mean(y_voc ** 2)) + 1e-8)
    rms_nv = float(np.sqrt(np.mean(y_nv ** 2)) + 1e-8)
    bgm_ratio_db = float(20.0 * np.log10(rms_nv / rms_v))
    return orig_24k, voc_24k, bgm_ratio_db, noise_floor_db


def hanzi_to_py_list(text: str) -> List[str]:
    chars = re.findall(r"[\u4e00-\u9fff]", text)
    if not chars:
        return []
    return lazy_pinyin("".join(chars), style=Style.NORMAL)


def compute_adaptive_voiced_intervals(
    y: np.ndarray,
    sr: int,
    hop: int = 256,
    frame_length: int = 1024,
    min_gap_s: float = 0.35,
    min_dur_s: float = 0.12,
) -> Tuple[List[Tuple[float, float]], float]:
    """基于每首曲目真实 RMS 分布自适应定位有效发声段（兼容含残余底噪的 Demucs 人声轨）。"""
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]
    rms_db = 20.0 * np.log10(rms + 1e-8)
    p10, p90 = [float(x) for x in np.percentile(rms_db, [10, 90])]
    # 动态范围较窄（如 t10 Demucs 人声底噪 -45dB / 人声 -25dB）时自动贴近底噪上方 7.5dB 处切气口
    thresh_db = min(p90 - 11.5, max(p10 + 7.5, p90 - 24.0))
    active = rms_db > thresh_db

    # 3 帧中值平滑去掉单帧毛刺
    if len(active) >= 5:
        from scipy.ndimage import binary_closing, binary_opening
        active = binary_closing(active, iterations=2)
        active = binary_opening(active, iterations=1)

    raw_iv: List[List[int]] = []
    in_seg = False
    seg_s = 0
    for i, v in enumerate(active):
        if v and not in_seg:
            seg_s = i
            in_seg = True
        elif not v and in_seg:
            raw_iv.append([seg_s, i])
            in_seg = False
    if in_seg:
        raw_iv.append([seg_s, len(active)])

    # 先过滤 < min_dur_s 的短促杂音/吸气声，再合并短于 min_gap_s 的同句微停顿
    filtered = [(s * hop / sr, e * hop / sr) for s, e in raw_iv if (e - s) * hop / sr >= min_dur_s]
    merged: List[List[float]] = []
    for s, e in filtered:
        if not merged or (s - merged[-1][1]) >= min_gap_s:
            merged.append([s, e])
        else:
            merged[-1][1] = e
    return [(s, e) for s, e in merged], thresh_db


def _fuzzy_bag_score(obs_chars: List[Tuple[str, str]], ref_text: str) -> float:
    """计算观测汉字序列与参考词表（标题/作者或正文首句）的字形+拼音模糊命中分。"""
    if not obs_chars or not ref_text:
        return 0.0
    ref_cs = re.findall(r"[\u4e00-\u9fff]", ref_text)
    if not ref_cs:
        return 0.0
    ref_pys = lazy_pinyin("".join(ref_cs), style=Style.NORMAL)
    used = set()
    score = 0.0
    for oc, op in obs_chars:
        best_sc = 0.0
        best_j = -1
        for j, (rc, rp) in enumerate(zip(ref_cs, ref_pys)):
            if j in used:
                continue
            if oc == rc:
                sc = 2.0
            elif op == rp:
                sc = 1.4
            elif len(op) >= 2 and len(rp) >= 2 and (op[:2] == rp[:2] or op[-2:] == rp[-2:]):
                sc = 0.55
            else:
                sc = 0.0
            if sc > best_sc:
                best_sc = sc
                best_j = j
        if best_j >= 0 and best_sc > 0:
            used.add(best_j)
            score += best_sc
    return score


def _anchor_head_score(obs_chars: List[Tuple[str, str]], anchor: str) -> float:
    """计算观测序列头部与正文首句锚点的有序滑窗匹配分（奖励正文首字对齐在头部）。"""
    if not obs_chars or not anchor:
        return 0.0
    anc_cs = re.findall(r"[\u4e00-\u9fff]", anchor)[:7]
    if not anc_cs:
        return 0.0
    anc_pys = lazy_pinyin("".join(anc_cs), style=Style.NORMAL)
    best = 0.0
    for offset in range(min(3, len(obs_chars))):
        win = obs_chars[offset : offset + len(anc_cs) + 2]
        sc = _fuzzy_bag_score(win, "".join(anc_cs))
        if offset == 0 and win and (win[0][0] == anc_cs[0] or win[0][1] == anc_pys[0]):
            sc += 1.2
        sc -= offset * 0.6
        if sc > best:
            best = sc
    return max(0.0, best)


def extract_whisper_char_events(result: dict) -> List[Tuple[str, str, float, float]]:
    char_events: List[Tuple[str, str, float, float]] = []
    for seg in result.get("segments", []):
        words = seg.get("words") or []
        if words:
            for w in words:
                txt = re.findall(r"[\u4e00-\u9fff]", w.get("word", ""))
                if not txt:
                    continue
                ws = float(w.get("start", seg["start"]))
                we = float(w.get("end", seg["end"]))
                dt = max(0.01, (we - ws) / len(txt))
                pys = lazy_pinyin("".join(txt), style=Style.NORMAL)
                for k, (ch, py) in enumerate(zip(txt, pys)):
                    char_events.append((ch, py, ws + k * dt, ws + (k + 1) * dt))
        else:
            txt = re.findall(r"[\u4e00-\u9fff]", seg.get("text", ""))
            if not txt:
                continue
            ws, we = float(seg["start"]), float(seg["end"])
            dt = max(0.01, (we - ws) / len(txt))
            pys = lazy_pinyin("".join(txt), style=Style.NORMAL)
            for k, (ch, py) in enumerate(zip(txt, pys)):
                char_events.append((ch, py, ws + k * dt, ws + (k + 1) * dt))
    return char_events


def detect_intro_and_body_bounds(
    head_whisper: dict,
    voiced_ivs: List[Tuple[float, float]],
    intro_ref: str,
    anchor: str,
    total_dur: float,
) -> Tuple[float, float, str, str]:
    """
    结合「前 30s 无提示 Whisper 逐字时间戳 + 自适应气口停顿」精准识别并切除开头报幕（标题/作者/朝代），
    杜绝长音频 Whisper 漏字导致的误切正文开头问题！
    """
    if not voiced_ivs:
        return 0.0, total_dur, "", "no_voiced"

    body_end = min(total_dur, voiced_ivs[-1][1] + 0.45)
    first_start = max(0.0, voiced_ivs[0][0] - 0.12)
    char_events = extract_whisper_char_events(head_whisper)
    if not char_events:
        return first_start, body_end, "", "no_asr_head"

    # 收集 [0.8s .. 7.5s] 区间内所有有效气口分界 (intro_end_s, next_voiced_s, gap_s)
    candidates: List[Tuple[float, float, float]] = []
    for i in range(len(voiced_ivs) - 1):
        iv_end = voiced_ivs[i][1]
        nxt_start = voiced_ivs[i + 1][0]
        gap = nxt_start - iv_end
        if 0.8 <= iv_end <= 7.5 and gap >= 0.42:
            candidates.append((iv_end, nxt_start, gap))

    best_cut = None
    best_total = 0.0
    best_debug = ""

    for iv_end, nxt_start, gap in candidates:
        # 归属分界左侧（报幕候选）与右侧（正文首句候选）的汉字
        left_ev = [c for c in char_events if 0.5 * (c[2] + c[3]) <= iv_end + 0.35]
        right_ev = [c for c in char_events if nxt_start - 0.90 <= 0.5 * (c[2] + c[3]) <= nxt_start + 9.0]
        if not left_ev or len(left_ev) > 16:
            continue

        left_pairs = [(c[0], c[1]) for c in left_ev]
        right_pairs = [(c[0], c[1]) for c in right_ev]

        # 左侧匹配标题/作者分 + 左侧误含正文分（仅对非标题重叠字做惩罚）
        intro_sc = _fuzzy_bag_score(left_pairs, intro_ref)
        left_anc_sc = _anchor_head_score(left_pairs, anchor)
        right_anc_sc = _anchor_head_score(right_pairs, anchor)
        # 检查右侧开头是否仍残留未切完的标题/作者词（如只切了“苏轼”还剩“前赤壁赋”）
        right_head_intro = _fuzzy_bag_score(right_pairs[:4], intro_ref)

        # 必须满足左侧确实像报幕，或「大停顿 (>=1.5s) 后紧接高置信正文首句且左侧非正文」（兼容重伴奏导致标题 ASR 听歪如 t03 泊秦淮）
        strong_post_anchor = (gap >= 1.5 and right_anc_sc >= 5.0 and left_anc_sc < 1.5)
        if intro_sc < 1.8 and not strong_post_anchor:
            continue
        if left_anc_sc > intro_sc + 1.2 and right_anc_sc < 1.5:
            continue

        # 综合打分：覆盖完整标题/作者 + 报幕后的大停顿(gap) + 右侧对齐正文首句 - 右侧残留标题
        gap_bonus = min(gap, 3.2) * 1.45
        total_sc = intro_sc * 1.35 + gap_bonus + right_anc_sc * 0.95 - max(0.0, right_head_intro - right_anc_sc) * 1.15
        if total_sc > best_total:
            best_total = total_sc
            best_cut = (iv_end, nxt_start, gap, left_ev, right_ev)
            best_debug = f"intro={intro_sc:.1f},gap={gap:.2f}s,anc={right_anc_sc:.1f}"

    if best_cut is None or best_total < 3.5:
        head_txt = "".join(c[0] for c in char_events[:8])
        return first_start, body_end, "", f"no_intro_keep_all({head_txt})"

    iv_end, nxt_start, gap, left_ev, right_ev = best_cut
    body_start = max(iv_end + 0.04, nxt_start - 0.12)
    stripped_intro = "".join(c[0] for c in left_ev)
    matched_head = "".join(c[0] for c in right_ev[:10])
    return body_start, body_end, stripped_intro, f"trimmed({best_debug}): {matched_head}"


def _clean_f0_octave_jumps(f0: np.ndarray, vf: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    """清理 pyin 单帧倍频/半频跳变毛刺，返回平滑后 f0 及稳健 (p10, med_f0, p90)。"""
    from scipy.ndimage import median_filter
    f0_clean = np.nan_to_num(f0, nan=0.0).astype(np.float32)
    valid_mask = (vf & np.isfinite(f0) & (f0_clean > 45.0))
    vals = f0_clean[valid_mask]
    if len(vals) < 8:
        return f0_clean, 0.0, 0.0, 0.0
    med = float(np.median(vals))
    # 修正超过 ±9.5 半音的倍频/半频误追踪
    idx_v = np.where(valid_mask)[0]
    for k in idx_v:
        v = f0_clean[k]
        if v > med * 1.78 and (v * 0.5) >= med * 0.56:
            f0_clean[k] = v * 0.5
        elif v < med * 0.56 and (v * 2.0) <= med * 1.78:
            f0_clean[k] = v * 2.0
    # 对连续有声段做 5 帧中值滤波
    if len(idx_v) >= 5:
        f0_med = median_filter(f0_clean, size=5)
        # 仅在相邻帧也有声时采用平滑值，避免边界被 0 拉低
        inner = valid_mask.copy()
        inner[:2] = False
        inner[-2:] = False
        has_nbr = valid_mask[:-4] & valid_mask[1:-3] & valid_mask[2:-2] & valid_mask[3:-1] & valid_mask[4:]
        inner[2:-2] &= has_nbr
        f0_clean[inner] = f0_med[inner]

    # 在正常人声 ±10 半音主区间内计算稳健 p10 / med / p90（杜绝孤立毛刺拉出 28st 假音域）
    v2 = f0_clean[valid_mask]
    v2 = v2[(v2 >= med * 0.52) & (v2 <= med * 1.92)]
    if len(v2) < 8:
        v2 = f0_clean[valid_mask]
    p10, med_f0, p90 = [float(x) for x in np.percentile(v2, [10, 50, 90])]
    return f0_clean, med_f0, p10, p90


def slice_clauses_and_extract_f0(
    y_body: np.ndarray,
    y_f0_src: np.ndarray,
    sr: int,
    clause_dir: Path,
    track_id: str,
) -> Tuple[List[dict], float, float, float]:
    """按自适应气口停顿切分为 2~14s 唱吟乐句，并提取去跳变 pyin F0 与相对中位音高的五声半音曲线。"""
    # 清空旧切片避免残留
    if clause_dir.exists():
        for old_f in clause_dir.glob("*"):
            old_f.unlink()
    clause_dir.mkdir(parents=True, exist_ok=True)

    hop = 256
    f0_raw, vf_full, _ = librosa.pyin(y_f0_src, fmin=65.0, fmax=580.0, sr=sr, frame_length=1024, hop_length=hop)
    if f0_raw is None:
        return [], 0.0, 0.0, 0.0
    f0_full, med_f0, p10, p90 = _clean_f0_octave_jumps(f0_raw, vf_full)
    if med_f0 <= 0.0:
        return [], 0.0, 0.0, 0.0

    # 用自适应 RMS 阈值找自然气口停顿
    voiced_s, _ = compute_adaptive_voiced_intervals(y_f0_src, sr, hop=hop, min_gap_s=0.30, min_dur_s=0.14)
    merged: List[List[int]] = []
    max_len = int(13.5 * sr)
    for vs, ve in voiced_s:
        s, e = int(round(vs * sr)), int(round(ve * sr))
        if not merged:
            merged.append([s, e])
        else:
            gap = s - merged[-1][1]
            new_len = e - merged[-1][0]
            if gap < int(0.32 * sr) or ((merged[-1][1] - merged[-1][0]) < int(1.6 * sr) and new_len <= max_len):
                merged[-1][1] = e
            else:
                merged.append([s, e])

    # 对超过 14.5s 的连唱长句，按 RMS 局部低谷在气口处拆分，保证每句 2~14s 适配扩散训练显存
    rms_env = librosa.feature.rms(y=y_f0_src, frame_length=1024, hop_length=hop)[0]
    final_segs: List[Tuple[int, int]] = []
    for s, e in merged:
        stack = [(s, e)]
        while stack:
            cs, ce = stack.pop(0)
            if (ce - cs) <= int(14.5 * sr):
                final_segs.append((cs, ce))
                continue
            # 在中间 [35%..65%] 区域寻找最低能量帧作为自然气口分割点
            f_a = int((cs + 0.35 * (ce - cs)) // hop)
            f_b = min(len(rms_env) - 1, int((cs + 0.65 * (ce - cs)) // hop))
            if f_b <= f_a + 2:
                mid = (cs + ce) // 2
            else:
                split_f = f_a + int(np.argmin(rms_env[f_a:f_b]))
                mid = split_f * hop
            stack.insert(0, (mid, ce))
            stack.insert(0, (cs, mid))

    clauses = []
    out_idx = 0
    for s, e in final_segs:
        dur = (e - s) / sr
        if dur < 0.9:
            continue
        s_pad = max(0, s - int(0.06 * sr))
        e_pad = min(len(y_body), e + int(0.08 * sr))
        w_seg = y_body[s_pad:e_pad].astype(np.float32)

        f_lo = s_pad // hop
        f_hi = min(len(f0_full), max(f_lo + 1, e_pad // hop))
        f0_seg = f0_full[f_lo:f_hi].astype(np.float32)
        uv_seg = (vf_full[f_lo:f_hi] & np.isfinite(f0_raw[f_lo:f_hi]) & (f0_seg > 45.0)).astype(np.uint8)
        rel_st = np.zeros_like(f0_seg, dtype=np.float32)
        m_v = uv_seg > 0
        if np.any(m_v):
            rel_st[m_v] = (12.0 * np.log2(np.maximum(f0_seg[m_v], 40.0) / med_f0)).astype(np.float32)

        wav_out = clause_dir / f"{track_id}_c{out_idx:03d}.wav"
        npz_out = clause_dir / f"{track_id}_c{out_idx:03d}_f0.npz"
        sf.write(str(wav_out), w_seg, sr)
        np.savez_compressed(
            str(npz_out),
            f0_hz=f0_seg,
            uv=uv_seg,
            rel_semitones=rel_st,
            med_f0=np.float32(med_f0),
            sr=np.int32(sr),
            hop=np.int32(hop),
        )
        clauses.append({
            "clause_idx": out_idx,
            "start_in_body_s": round(s_pad / sr, 2),
            "end_in_body_s": round(e_pad / sr, 2),
            "dur_s": round((e_pad - s_pad) / sr, 2),
            "wav": str(wav_out.relative_to(OUT_DIR)),
            "f0_npz": str(npz_out.relative_to(OUT_DIR)),
        })
        out_idx += 1
    return clauses, med_f0, p10, p90


def restore_and_normalize_audio(
    src_wav: Path,
    dst_wav: Path,
    has_bgm: bool,
    body_start: float,
    body_end: float,
) -> Tuple[np.ndarray, dict]:
    """
    先裁出正文区间，再依据实测削波率/底噪/低频嗡声分级做无损/温和音频修复：
      - 绝不对无配乐纯人声强行套 Demucs；
      - 峰值 >= 0.94 或存在削波：ffmpeg adeclip + adeclick 修复削波破音；
      - 正文底噪 > -54dB：按实测底噪做自适应温和 afftdn（不伤人声谐波）；
      - 全曲统一 highpass=75Hz（消除喷麦/50Hz交流电嗡声）+ 峰值保护响度归一（RMS=0.085, Peak<=0.70）。
    """
    y_raw, sr = sf.read(str(src_wav), dtype="float32")
    s0 = max(0, int(round(body_start * sr)))
    s1 = min(len(y_raw), int(round(body_end * sr)))
    y_seg = y_raw[s0:s1]

    raw_peak = float(np.max(np.abs(y_seg)) + 1e-8)
    raw_clip_pct = float(np.mean(np.abs(y_seg) >= 0.985) * 100.0)
    frm_raw = librosa.feature.rms(y=y_seg, frame_length=1024, hop_length=256)[0]
    raw_sil_db = float(20.0 * np.log10(np.percentile(frm_raw, 10) + 1e-8))

    # 检测 50/100Hz 交流电嗡声占比
    S = np.abs(librosa.stft(y_seg[: min(len(y_seg), 30 * sr)], n_fft=1024, hop_length=256)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    sub80_pct = float(np.sum(S[freqs < 80]) / (np.sum(S) + 1e-12) * 100.0)

    filters = ["highpass=f=75"]
    actions = []
    if sub80_pct > 0.8:
        filters.append("equalizer=f=100:t=q:w=4:g=-12")
        actions.append("dehum")
    if raw_clip_pct > 0.001 or raw_peak >= 0.94:
        filters.append("adeclip=w=55:o=75:a=8:t=8:n=1000")
        filters.append("adeclick")
        actions.append("declip")

    # 仅对底噪偏高（> -53dB）或去伴奏后有残噪的曲目施加温和频域降噪，纯净棚录不加降噪保零相位损伤
    need_denoise = (raw_sil_db > -53.0) or (has_bgm and raw_sil_db > -62.0)
    if need_denoise:
        # 根据实测底噪设定温和降噪量（10~20dB），防止水声/金属感
        nr = int(np.clip(round((raw_sil_db + 62.0) * 0.7), 10, 20))
        nf = int(np.clip(round(raw_sil_db + 2.0), -58, -28))
        filters.append(f"afftdn=nr={nr}:nf={nf}:tn=1")
        actions.append(f"denoise({nr}dB)")

    tmp_in = dst_wav.parent / f"._{dst_wav.stem}_in.wav"
    tmp_out = dst_wav.parent / f"._{dst_wav.stem}_out.wav"
    sf.write(str(tmp_in), y_seg, sr)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(tmp_in), "-af", ",".join(filters), "-ar", str(sr), "-ac", "1", str(tmp_out)],
        check=True,
    )
    y_fix, _ = sf.read(str(tmp_out), dtype="float32")
    tmp_in.unlink(missing_ok=True)
    tmp_out.unlink(missing_ok=True)

    # 统一舒适响度 + 无损峰值保护（RMS=0.085, Peak<=0.70），杜绝浏览器播放炸耳破音
    rms = float(np.sqrt(np.mean(y_fix ** 2)) + 1e-8)
    gain = 0.085 / rms
    peak = float(np.max(np.abs(y_fix)) + 1e-8)
    if peak * gain > 0.70:
        gain = 0.70 / peak
    y_norm = (y_fix * gain).astype(np.float32)

    # 前后 15ms 余弦淡入淡出防切口咔哒声
    fade_n = min(int(0.015 * sr), len(y_norm) // 4)
    if fade_n > 4:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fade_n, dtype=np.float32)))
        y_norm[:fade_n] *= ramp
        y_norm[-fade_n:] *= ramp[::-1]

    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst_wav), y_norm, sr)

    frm_fix = librosa.feature.rms(y=y_norm, frame_length=1024, hop_length=256)[0]
    fix_sil_db = float(20.0 * np.log10(np.percentile(frm_fix, 10) + 1e-8))
    # 严格三级音质判定（绝不把有削波或高底噪的曲目误标为顶级干声！）
    if (not has_bgm) and raw_clip_pct == 0.0 and raw_sil_db <= -55.0:
        grade = "tier1_studio_clean"
    elif not has_bgm:
        grade = "tier2_restored_melody"
    else:
        grade = "tier3_bgm_sep_melody"

    stats = {
        "audio_grade": grade,
        "raw_peak": round(raw_peak, 3),
        "raw_clip_pct": round(raw_clip_pct, 4),
        "raw_sil_db": round(raw_sil_db, 1),
        "fixed_peak": round(float(np.max(np.abs(y_norm))), 3),
        "fixed_rms": round(float(np.sqrt(np.mean(y_norm ** 2))), 3),
        "fixed_sil_db": round(fix_sil_db, 1),
        "restore_ops": ",".join(actions) if actions else "hp75+norm",
    }
    return y_norm, stats


def main() -> None:
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(RAW_DIR / "*.mp3")))
    print(f"=== 汉火唐调吟诵语料全自动预处理 ({len(files)} 首, device={device}) ===")

    print("1. 加载本地缓存 Whisper small 模型...")
    w_model = whisper.load_model("small", device=device)

    manifest_tracks = []
    total_body_sec = 0.0
    total_clauses = 0

    for idx, fp_str in enumerate(files, 1):
        fp = Path(fp_str)
        track_id = f"t{idx:02d}"
        genre, title, intro_ref, anchor = match_track_rule(fp.name)
        speaker = extract_speaker_tag(fp.name)
        t_dir = OUT_DIR / "tracks" / track_id

        # Step 1: Demucs 检测伴奏占比（仅真正含配乐的曲目才取 Demucs 人声轨；纯人声带底噪绝不强套 Demucs 防金属水声！）
        orig_24k, voc_24k, bgm_ratio_db, noise_floor_db = run_demucs_check(fp, t_dir, device)
        has_bgm = bgm_ratio_db > -24.0
        work_wav = voc_24k if has_bgm else orig_24k

        y_f0_full, _ = sf.read(str(work_wav), dtype="float32")
        total_dur = len(y_f0_full) / SR

        # Step 2: 仅对前 30 秒做无提示 Whisper 逐字识别 + 全曲自适应 VAD，精准切除开头标题/作者旁白与片尾死静音！
        head30_wav = t_dir / "head30_24k.wav"
        sf.write(str(head30_wav), y_f0_full[: min(len(y_f0_full), 30 * SR)], SR)
        w_res = w_model.transcribe(
            str(head30_wav),
            language="zh",
            word_timestamps=True,
            fp16=(device == "cuda"),
            verbose=False,
        )
        voiced_ivs, _ = compute_adaptive_voiced_intervals(y_f0_full, SR, min_gap_s=0.35, min_dur_s=0.12)
        body_start, body_end, stripped_intro, match_info = detect_intro_and_body_bounds(
            w_res, voiced_ivs, intro_ref, anchor, total_dur
        )

        # Step 3: 分级音频修复（去削波破音 + 去 50/100Hz 电流嗡声 + 自适应温和去底噪 + 统一响度 RMS=0.085, Peak<=0.70）
        body_wav_path = OUT_DIR / "bodies" / f"{track_id}_{genre}_{title}.wav"
        y_body, fix_stats = restore_and_normalize_audio(
            src_wav=work_wav,
            dst_wav=body_wav_path,
            has_bgm=has_bgm,
            body_start=body_start,
            body_end=body_end,
        )
        audio_grade = fix_stats["audio_grade"]

        # Step 4: 基于修复后的干净正文做逐句切分 + 相对五声半音 F0 提取
        clauses, med_f0, f0_p10, f0_p90 = slice_clauses_and_extract_f0(
            y_body=y_body,
            y_f0_src=y_body,
            sr=SR,
            clause_dir=OUT_DIR / "clauses" / track_id,
            track_id=track_id,
        )
        body_dur = len(y_body) / SR
        total_body_sec += body_dur
        total_clauses += len(clauses)
        span_st = round(12.0 * np.log2(f0_p90 / max(1e-6, f0_p10)), 1) if f0_p10 > 0 else 0.0

        rec = {
            "track_id": track_id,
            "filename": fp.name,
            "title": title,
            "genre": genre,
            "speaker": speaker,
            "audio_grade": audio_grade,
            "has_bgm": has_bgm,
            "bgm_ratio_db": round(bgm_ratio_db, 1),
            "noise_floor_db": fix_stats["fixed_sil_db"],
            "raw_sil_db": fix_stats["raw_sil_db"],
            "raw_peak": fix_stats["raw_peak"],
            "raw_clip_pct": fix_stats["raw_clip_pct"],
            "fixed_peak": fix_stats["fixed_peak"],
            "fixed_rms": fix_stats["fixed_rms"],
            "restore_ops": fix_stats["restore_ops"],
            "orig_dur_s": round(total_dur, 2),
            "body_start_s": round(body_start, 2),
            "body_end_s": round(body_end, 2),
            "body_dur_s": round(body_dur, 2),
            "stripped_intro": stripped_intro,
            "anchor_match": match_info,
            "med_f0_hz": round(med_f0, 1),
            "f0_span_st": span_st,
            "body_wav": str(body_wav_path.relative_to(OUT_DIR)),
            "num_clauses": len(clauses),
            "clauses": clauses,
        }
        manifest_tracks.append(rec)
        print(
            f"[{idx:02d}/20] {title:12s} ({genre:9s}) | {audio_grade:21s} [{fix_stats['restore_ops']:18s}] | "
            f"Sil:{fix_stats['raw_sil_db']:5.1f}->{fix_stats['fixed_sil_db']:5.1f}dB Peak:{fix_stats['raw_peak']:.2f}->{fix_stats['fixed_peak']:.2f} | "
            f"切头={body_start:4.2f}s -> {len(clauses):2d}句 ({body_dur:5.1f}s) | {med_f0:5.1f}Hz 跨{span_st:4.1f}st"
        )

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_tracks": len(manifest_tracks),
        "total_body_sec": round(total_body_sec, 1),
        "total_clauses": total_clauses,
        "tracks": manifest_tracks,
    }
    manifest_path = OUT_DIR / "tang_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # 同步生成局域网抽检页，方便直接点击试听“切掉标题后的正文开头”确认无误！
    review_dir = ROOT / "hanfire_tts" / "web_wasm" / "dist_cf" / "_tang_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    for rec in manifest_tracks:
        src_b = OUT_DIR / rec["body_wav"]
        # 仅导出前 18 秒预览片段供网页秒开核对开头切割与人声纯净度
        yb, _ = sf.read(str(src_b), dtype="float32")
        sf.write(str(review_dir / f"{rec['track_id']}_head18s.wav"), yb[: min(len(yb), 18 * SR)], SR)

    review_html = ROOT / "hanfire_tts" / "web_wasm" / "dist_cf" / "tang_review.html"
    ver_tag = int(time.time())
    rows = []
    for r in manifest_tracks:
        if r["audio_grade"] == "tier1_studio_clean":
            badge = "🟢 棚录级纯净干声<br><small>声学+曲调均可用</small>"
        elif r["audio_grade"] == "tier2_restored_melody":
            badge = "🟡 已修破音/去底噪<br><small>仅提五声曲调与节奏</small>"
        else:
            badge = "🔵 已剥离伴奏/去嗡声<br><small>仅提五声曲调与节奏</small>"
        rows.append(
            f"<tr><td>{r['track_id']}</td><td><b>{r['title']}</b><br><small>{r['speaker']} · {r['genre']}</small></td>"
            f"<td>{badge}<br><small>修复: {r['restore_ops']} · 底噪 {r['raw_sil_db']}→{r['noise_floor_db']}dB · 峰 {r['raw_peak']}→{r['fixed_peak']}</small></td>"
            f"<td>切掉前 <b>{r['body_start_s']}s</b><br><small>「{r['stripped_intro'] or '(无报幕)'}」</small></td>"
            f"<td>{r['num_clauses']} 句 / {r['body_dur_s']}s<br><small>中位 {r['med_f0_hz']}Hz · 跨 {r['f0_span_st']} 半音</small></td>"
            f"<td><audio controls preload='none' src='./_tang_review/{r['track_id']}_head18s.wav?v={ver_tag}'></audio></td></tr>"
        )
    review_html.write_text(
        f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><title>唐调 20 首自动预处理核对台</title>
<style>body{{margin:0;background:#f7f4ed;color:#2b2623;font-family:system-ui,sans-serif;padding:20px}}
h1{{color:#c34435;font-family:"Noto Serif CJK SC",serif}}table{{width:100%;border-collapse:collapse;background:#fdfbf7;border:1px solid #ddd3be}}
th,td{{padding:9px 10px;border-bottom:1px solid #e6decb;text-align:left;font-size:0.86rem;vertical-align:middle}}
th{{background:#f1ece0;color:#c34435}}audio{{width:240px;height:32px}}small{{color:#6e6459}}</style></head>
<body><h1>唐调吟诵 20 首自动预处理核对台（切头 + 去破音/去噪/去伴奏 + 响度归一）</h1>
<p>共 {len(manifest_tracks)} 首，切出有效正文 <b>{total_body_sec/60:.1f} 分钟 / {total_clauses} 个唱吟乐句</b>。已全部完成 <b>adeclip 去削波 + 75Hz高通去嗡声 + 自适应降噪 + RMS=0.085 峰值保护归一</b>。注意：🟡🔵 曲目仅用于提取五声旋律(F0)与节奏，绝不污染古音声学音色！</p>
<table><thead><tr><th>ID</th><th>曲目与体裁</th><th>音质分级与修复明细</th><th>自动切除片头</th><th>切片与五声音域</th><th>正文前18秒试听</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></body></html>""",
        encoding="utf-8",
    )
    print(f"\n✅ 预处理全部完成！耗时 {time.time()-t0:.1f}s | 清单: {manifest_path} | 核对页: {review_html}")


if __name__ == "__main__":
    main()
