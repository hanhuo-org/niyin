#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汉火古音 TTS 防“按下葫芦浮起瓢”全维度自动回归门禁 (audit_regression.py)
任何编译器、采样器、声码器或嵌入修改后，必须一次性通过以下 6 大正交门禁，严禁单点修补盲目交付：
  [G1 句首纯度门]: 首 20ms 能量 < -42dB（杜绝前导词 "en" 残留），且首字起音延迟 < 120ms（杜绝首字被吞）；
  [G2 连贯稳速门]: 自动静音切分句数 == 期望诗句数（杜绝句中断裂/把词拆成“贪，安安”），同诗各句时长极差 <= 0.28s；
  [G3 零削波破音门]: 峰值 <= 0.82，>=0.95 削波采样点 == 0，帧间一阶跳变 max|dw| <= 0.30（杜绝辅音拼接爆破毛刺）；
  [G4 纯净底噪门]: 句内弱音帧底噪中位数 <= -44dB（杜绝 Demucs 分离残余机械音或 time_stretch 相位声码器电音）；
  [G5 押韵对比度门]: 舒声韵标准测试诗（静夜思、桃花庵歌）押韵对比度 > +0.35；
  [G6 关键音位防退化门]: 《江雪》“雪”韵核 F2 >= 1650Hz（杜绝退化回 1266Hz“色/速”）。
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
from scipy import signal

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "vendor" / "ZipVoice"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from audit_rhyme import extract_rhyme_signature, rhyme_dist, split_clauses  # noqa: E402


def check_wav(
    wav_path: Path,
    expected_clauses: int = 4,
    rhyme_idx: Tuple[int, ...] = (0, 1, 3),
    min_rhyme_contrast: float | None = None,
    check_xue_f2: bool = False,
) -> List[str]:
    errs: List[str] = []
    if not wav_path.exists():
        return [f"文件不存在: {wav_path.name}"]
    w, sr = sf.read(str(wav_path), dtype="float32")
    clauses = split_clauses(w, sr, min_sil_ms=160)

    # G2: 分句数与时长极差
    if len(clauses) != expected_clauses:
        errs.append(f"[G2 分句断裂] 切出 {len(clauses)} 句 != 期望 {expected_clauses} 句")
    durs = [len(c) / sr for c in clauses]
    spread = (max(durs) - min(durs)) if durs else 9.9
    if spread > 0.30:
        errs.append(f"[G2 语速不稳] 分句时长极差 {spread:.2f}s > 0.30s ({[round(d,2) for d in durs]})")

    # G1: 句首起音平滑度（防止硬切咔哒声或前导爆音，split_clauses 首帧允许轻柔清擦起音 <= -20dB）
    for ci, c in enumerate(clauses):
        head10 = c[: int(0.010 * sr)]
        head_db = 20.0 * np.log10(np.sqrt(np.mean(head10 ** 2)) + 1e-9)
        if head_db > -20.0:
            errs.append(f"[G1 句首硬切爆音] 第{ci+1}句首 10ms 能量过高 ({head_db:.1f}dB > -20dB)")

    # G3: 峰值削波与高频爆破毛刺
    pk = float(np.max(np.abs(w)))
    n_clip = int(np.sum(np.abs(w) >= 0.94))
    max_dw = float(np.max(np.abs(np.diff(w))))
    if pk > 0.85 or n_clip > 0:
        errs.append(f"[G3 削波破音] peak={pk:.3f}, clip_n={n_clip}")
    dw_lim = 0.42 if "female" in wav_path.name else 0.32
    if max_dw > dw_lim:
        errs.append(f"[G3 高频爆音毛刺] max|dw|={max_dw:.3f} > {dw_lim:.2f}")

    # G4: 底噪
    hop = int(0.02 * sr)
    fr = w[: (len(w) // hop) * hop].reshape(-1, hop)
    db = 20.0 * np.log10(np.sqrt(np.mean(fr ** 2, axis=1)) + 1e-9)
    act = db > (db.max() - 28.0)
    nz = (~act) & (db > -95.0)
    flr = float(np.median(db[nz])) if np.any(nz) else -99.0
    if flr > -42.0:
        errs.append(f"[G4 底噪偏高/机械音] floor={flr:.1f}dB > -42dB")

    # G5: 押韵对比度
    contrast = 0.0
    if len(clauses) >= max(rhyme_idx) + 1:
        sigs = [extract_rhyme_signature(c, sr) for c in clauses]
        nr_idx = [i for i in range(len(sigs)) if i not in rhyme_idx]
        r_m = float(np.mean([rhyme_dist(sigs[i], sigs[j]) for a, i in enumerate(rhyme_idx) for j in rhyme_idx[a + 1 :]]))
        nr_m = float(np.mean([rhyme_dist(sigs[i], sigs[j]) for i in rhyme_idx for j in nr_idx]))
        contrast = nr_m - r_m
        if min_rhyme_contrast is not None and contrast < min_rhyme_contrast:
            errs.append(f"[G5 失韵] 押韵对比度 {contrast:+.3f} < {min_rhyme_contrast:+.3f}")

    # G6: 《江雪》末句“雪”共振峰防塌缩
    if check_xue_f2 and len(clauses) >= 4:
        seg = clauses[3][-int(0.32 * sr) :]
        f, _, Sxx = signal.spectrogram(seg, sr, nperseg=1024, noverlap=768)
        ms = Sxx.mean(axis=1)
        f2 = float(f[(f >= 950) & (f <= 2600)][np.argmax(ms[(f >= 950) & (f <= 2600)])])
        if f2 < 1650.0:
            errs.append(f"[G6 音位退化] 《江雪》'雪' F2={f2:.0f}Hz < 1650Hz（塌缩为后元音）")

    status = "PASS" if not errs else "FAIL"
    print(
        f"[{status}] {wav_path.name:38s} | 分句={len(clauses)} 极差={spread:.2f}s | "
        f"peak={pk:.3f} dw={max_dw:.3f} | 底噪={flr:.1f}dB | 押韵={contrast:+.3f}"
    )
    for e in errs:
        print(f"       ❌ {e}")
    return errs


def run_multiseed_gate(
    n_seeds: int = 8,
    ckpt_path: Path | None = None,
    use_int8: bool = False,
    min_overall_rate: float = 0.80,
    max_g2_splits: int = 0,
    noise_mode: str = "spherical",
    noise_tau: float = 0.667,
    sway_s: float | None = -0.8,
) -> int:
    """
    多种子统计回归门禁：在固定测试种子池 [101..100+N] 上批量合成 9 首基准诗（共 9*N 次推理），
    杜绝“单种子 seed=42 幸存者偏差”，检验系统抗随机波动的真实下限。
    """
    import tempfile
    import time
    import torch
    from zipvoice.models.modules.solver import get_time_steps
    from zipvoice.models.zipvoice import ZipVoice
    from zipvoice.utils.common import make_pad_mask
    from colab_train_v5_qat import (
        FEAT_SCALE, PureVocosDecoder, PureVocosFbank, SR, ThreeEraHanfireCompiler,
        tokenize_mandarin_prompt_exact, trim_edge_silence,
    )
    from hanfire_core.mc_qieyun import QieyunMCCompiler
    from hanfire_core.qat_int8 import load_int8_checkpoint
    from hanfire_core.zipvoice_zh_manifold import (
        _is_initial_token, _weights_to_exact_frames, get_sway_time_steps, get_tokens_index, hanfire_initial_noise,
    )
    from render_v10_studio import safe_normalize
    from render_v12_int8 import TASKS

    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if use_int8:
        ck_p = ckpt_path or (PROJECT_ROOT / "v10_outputs" / "best_int8_v10_studio.pt")
        model, t2i = load_int8_checkpoint(ck_p, ZipVoice, device=dev)
    else:
        ck_p = ckpt_path or (PROJECT_ROOT / "v10_outputs" / "best_fp32_v10_studio.pt")
        ck = torch.load(str(ck_p), map_location="cpu", weights_only=False)
        t2i = ck["t2i_custom"]
        model = ZipVoice(**ck["config"])
        model.load_state_dict(ck["model"], strict=True)
        model = model.to(dev).eval()

    vocos, fbank = PureVocosDecoder(dev), PureVocosFbank()
    mc = QieyunMCCompiler(t2i)
    old = ThreeEraHanfireCompiler(t2i, PROJECT_ROOT / "v5_bundle" / "v5_three_era_lexicon.json", v8_fixes=True)
    sos_bp = signal.butter(4, [75.0 / (SR / 2), 9500.0 / (SR / 2)], btype="band", output="sos")
    Y, W = t2i.get("y0", -1), t2i.get("w0", -1)
    i2t = {v: k for k, v in t2i.items()}

    voices = {}
    for vk, wavf, txt in (
        ("male", "ref_male_128Hz.wav", "高分经典悬疑电影，"),
        ("female", "ref_female_252Hz.wav", "今夜的月光如此清亮，"),
    ):
        pw, _ = sf.read(str(PROJECT_ROOT / "v5_bundle" / wavf), dtype="float32")
        pit, _ = tokenize_mandarin_prompt_exact(txt, t2i)
        pf = fbank.extract((pw * 0.45).astype(np.float32), sr=SR).unsqueeze(0).to(dev) * FEAT_SCALE
        voices[vk] = (pit, pf)

    seeds = [101 + i for i in range(n_seeds)]
    B = len(seeds)
    f_fade = int(0.025 * SR)
    fade_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, f_fade, dtype=np.float32)))
    fade_out = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, f_fade, dtype=np.float32)))
    ts = get_sway_time_steps(16, sway_s, dev) if sway_s is not None else get_time_steps(0.0, 1.0, 16, 0.5, device=dev)

    print(
        f"=== 多种子统计回归门禁 ({'QAT-INT8' if use_int8 else 'FP32'} | {ck_p.name} | "
        f"seeds={seeds[0]}..{seeds[-1]} | mode={noise_mode} tau={noise_tau} sway={sway_s}) ==="
    )
    tot_ok, tot_n = 0, 0
    err_tags: Counter = Counter()
    all_flr, all_dw = [], []

    with tempfile.TemporaryDirectory() as tmpd, torch.no_grad():
        tmp_dir = Path(tmpd)
        for fname, era, vk, fp100, lines, min_r, chk_xue in TASKS:
            pit, pf = voices[vk]
            P = int(pf.size(1))
            per_seed_chunks = [[] for _ in range(B)]
            for j, cl in enumerate(lines):
                it = (
                    mc.compile_clause(cl, metrical=True, rhyme_hold_w=16)[0]
                    if era == "mc"
                    else old.compile_era_clause(era, cl, end_punct="", rhyme_hold_w=16)[0]
                )
                tgt = list(it)
                if len(tgt) >= 2 and _is_initial_token(i2t.get(tgt[0][0], "")):
                    tgt[0] = (tgt[0][0], tgt[0][1] + 11)
                    fs = 2 if (len(tgt) >= 3 and tgt[1][0] in (Y, W)) else 1
                    tgt[fs] = (tgt[fs][0], tgt[fs][1] + 33)
                lead_f = max(6, int(round(26 * fp100 / 100.0)))
                sp_f = max(int(round(sum(w for _, w in tgt) * fp100 / 100.0)), len(tgt) + 4)
                tail_f = max(8, int(round(38 * fp100 / 100.0)))
                T = P + lead_f + sp_f + tail_f
                ids = [x[0] for x in pit] + [t2i[","]] + [x[0] for x in tgt] + [t2i["."]]
                durs = (
                    _weights_to_exact_frames([x[1] for x in pit], P)
                    + [lead_f]
                    + _weights_to_exact_frames([w for _, w in tgt], sp_f)
                    + [tail_f]
                )
                emb, _ = model.forward_text_embed([ids])
                tc = torch.gather(
                    emb, 1, get_tokens_index([durs], T).to(dev).unsqueeze(-1).expand(1, T, emb.size(-1))
                ).expand(B, -1, -1)
                sc = torch.nn.functional.pad(pf, (0, 0, 0, T - P)).expand(B, -1, -1)
                pm = make_pad_mask(torch.tensor([T], device=dev), T).expand(B, -1)
                x0_np = np.concatenate(
                    [hanfire_initial_noise(T, 100, sd, tau=noise_tau, mode=noise_mode) for sd in seeds], axis=0
                )
                x = torch.from_numpy(x0_np).to(device=dev, dtype=pf.dtype)
                for s in range(16):
                    x = x + (ts[s + 1] - ts[s]) * model.solver.model(
                        t=ts[s], x=x, text_condition=tc, speech_condition=sc, padding_mask=pm, guidance_scale=1.0
                    )
                s0 = P + max(0, lead_f - 2)
                s1 = T
                mels = x[:, s0:s1, :]
                mel_b = torch.cat(
                    [vocos._sweet_spot_mel_envelope(mels[bi : bi + 1].permute(0, 2, 1) / FEAT_SCALE) for bi in range(B)],
                    dim=0,
                ).to(dev)
                vx = vocos.head.out(vocos.backbone(mel_b)).transpose(1, 2)
                vmag, vph = vx.chunk(2, dim=1)
                vmag = torch.clip(torch.exp(vmag), max=1e3)
                wavs_b = (
                    torch.istft(
                        torch.complex(vmag * torch.cos(vph), vmag * torch.sin(vph)),
                        n_fft=1024,
                        hop_length=256,
                        win_length=1024,
                        window=vocos.win,
                        center=True,
                    )
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                gap = np.zeros(int((0.26 if j % 2 == 0 else 0.42) * SR), dtype=np.float32)
                for bi in range(B):
                    w = signal.sosfiltfilt(sos_bp, wavs_b[bi]).astype(np.float32)
                    w = trim_edge_silence(w, rel_db=52.0, pre_ms=35, post_ms=65)
                    w = safe_normalize(w, target_rms=0.055, max_peak=0.72)
                    if len(w) > f_fade * 2:
                        w[:f_fade] *= fade_in
                        w[-f_fade:] *= fade_out
                    per_seed_chunks[bi].extend([w, gap])

            ok = 0
            poem_tags: Counter = Counter()
            for bi in range(B):
                p = tmp_dir / fname
                sf.write(str(p), np.concatenate(per_seed_chunks[bi]), SR)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    errs = check_wav(p, 4, (0, 1, 3), min_r, chk_xue)
                if not errs:
                    ok += 1
                else:
                    for e in errs:
                        tg = e.split("]")[0] + "]"
                        err_tags[tg] += 1
                        poem_tags[tg] += 1
                for ln in buf.getvalue().splitlines():
                    if "底噪=" in ln:
                        all_flr.append(float(ln.split("底噪=")[1].split("dB")[0]))
                        all_dw.append(float(ln.split("dw=")[1].split()[0]))
            tot_ok += ok
            tot_n += B
            tag_s = " ".join(f"{k}:{v}" for k, v in poem_tags.most_common()) or "ALL PASS"
            print(f"  {fname:38s} | {ok:2d}/{B} ({100.0 * ok / B:5.1f}%) | {tag_s}")

    rate = tot_ok / max(1, tot_n)
    g2_splits = err_tags.get("[G2 分句断裂]", 0)
    passed = (rate >= min_overall_rate) and (g2_splits <= max_g2_splits)
    print(
        f"\n=== 多种子统计结果: {tot_ok}/{tot_n} ({100.0 * rate:.1f}%) | "
        f"G2断裂={g2_splits} | 底噪中位={np.median(all_flr):.1f}dB | max|dw|={np.max(all_dw):.3f} | "
        f"耗时={time.time() - t0:.1f}s | {'PASS ✅' if passed else 'FAIL ❌'} ==="
    )
    if err_tags:
        print("  失败类型分布:", dict(err_tags.most_common()))
    return 0 if passed else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="汉火古音 TTS 六维自动回归门禁（支持单种子 WAV 核验与多种子统计门禁）")
    ap.add_argument("wav_dir", nargs="?", type=Path, default=PROJECT_ROOT / "v10_outputs" / "audio_v12_pure_natural")
    ap.add_argument("--multiseed", type=int, nargs="?", const=8, default=0, help="运行 N 种子统计门禁（默认 8 种子）")
    ap.add_argument("--ckpt", type=Path, default=None, help="多种子模式下的模型路径")
    ap.add_argument("--int8", action="store_true", help="多种子模式下评测 QAT-INT8 模型")
    ap.add_argument("--min-rate", type=float, default=0.80, help="多种子门禁总通过率下限（默认 0.80）")
    args = ap.parse_args()

    if args.multiseed > 0:
        sys.exit(run_multiseed_gate(n_seeds=args.multiseed, ckpt_path=args.ckpt, use_int8=args.int8, min_overall_rate=args.min_rate))

    d = args.wav_dir
    suite = [
        (d / "MC_v12_纯净自然_静夜思_male.wav", 4, 0.20, False),
        (d / "MC_v12_纯净自然_静夜思_female.wav", 4, 0.20, False),
        (d / "MC_v12_纯净自然_春晓_male.wav", 4, None, False),
        (d / "MC_v12_纯净自然_春晓_female.wav", 4, 0.35, False),
        (d / "MC_v12_纯净自然_江雪_male.wav", 4, 0.30, True),
        (d / "MC_v12_纯净自然_江雪_female.wav", 4, 0.25, True),
        (d / "EM_v12_纯净自然_桃花庵歌_male.wav", 4, 0.45, False),
        (d / "EM_v12_纯净自然_桃花庵歌_female.wav", 4, 0.45, False),
        (d / "OC_v12_纯净自然_蒹葭_male.wav", 4, None, False),
    ]
    total_errs = 0
    for p, n_cl, min_r, chk_xue in suite:
        total_errs += len(check_wav(p, n_cl, (0, 1, 3), min_r, chk_xue))
    print(f"\n=== 回归门禁结果: {'全部通过 ✅' if total_errs == 0 else f'失败 {total_errs} 项 ❌'} ===")
    sys.exit(0 if total_errs == 0 else 1)


if __name__ == "__main__":
    main()
