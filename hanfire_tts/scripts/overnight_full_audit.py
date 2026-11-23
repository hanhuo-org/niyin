#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三代全音系盘点、代表诗篇合成与六维审美自动巡检脚本 (overnight_full_audit.py)
覆盖：
  1. 语料缺口盘点：统计上古(OC)、中古(MC)、近古(EM)全音系中缺真人样本或低频薄弱的声母/韵部/调类；
  2. 全音系代表字谱合成与检验：三十八母、平上去入八调、上古清鼻流音与入声塞尾、近古尖团五声（按四字格律组块合成，锁死一字一音）；
  3. 三代 12 首经典诗词双音色合成与六维审美门禁（一字一音、句首无杂音、稳速零断裂、零削波毛刺、纯净底噪、押韵与平上去入声调轮廓）。
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
from scipy import signal

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "vendor" / "ZipVoice"), str(HERE)]

from audit_rhyme import extract_rhyme_signature, rhyme_dist, split_clauses  # noqa: E402
from colab_train_v5_qat import (  # noqa: E402
    FEAT_SCALE, PureVocosDecoder, PureVocosFbank, SR, ThreeEraHanfireCompiler,
    ZipVoice, tokenize_mandarin_prompt_exact, trim_edge_silence,
)
from hanfire_core.mc_qieyun import MU_INIT, QieyunMCCompiler  # noqa: E402
from hanfire_core.phonology import parse_mononuclear_4slot  # noqa: E402
from hanfire_core.zipvoice_zh_manifold import sample_clean_zero_carrier  # noqa: E402
from render_v10_studio import calibrate_studio_embeddings, safe_normalize  # noqa: E402


def estimate_f0_contour(wav: np.ndarray, sr: int = SR) -> Tuple[float, float, float]:
    """用自相关法估计浊音段起/中/末基频 (Hz) 与斜率，检验平/上/去/入声调特征。"""
    hop = int(0.01 * sr)
    win = int(0.03 * sr)
    f0s: List[float] = []
    min_lag, max_lag = int(sr / 420), int(sr / 75)
    for i in range(0, max(1, len(wav) - win), hop):
        seg = wav[i : i + win]
        if np.sqrt(np.mean(seg ** 2)) < 0.015:
            continue
        seg = seg - np.mean(seg)
        corr = np.correlate(seg, seg, mode="full")[len(seg) - 1 :]
        if corr[0] <= 1e-8:
            continue
        corr = corr / corr[0]
        pk_lag = min_lag + int(np.argmax(corr[min_lag:max_lag]))
        if corr[pk_lag] > 0.32:
            f0s.append(float(sr / pk_lag))
    if len(f0s) < 3:
        return (0.0, 0.0, 0.0)
    k = max(1, len(f0s) // 3)
    f_start = float(np.median(f0s[:k]))
    f_end = float(np.median(f0s[-k:]))
    return (round(f_start, 1), round(f_end, 1), round(f_end - f_start, 1))


def audit_missing_phonemes(mc_comp: QieyunMCCompiler, old_comp: ThreeEraHanfireCompiler) -> dict:
    """盘点三代音系在词表与真实训练录音中的覆盖度与缺失音素清单。"""
    report: dict = {}

    # 1. MC 中古音盘点 (38 声母 + 249 韵类 + 8 调类)
    xqy_p = PROJECT_ROOT / "data_mc_raw" / "xiaoqieyun_labels_v3.json"
    zn_p = PROJECT_ROOT / "data_mc_raw" / "mc_zn_verified_manifest.json"
    v5_p = PROJECT_ROOT / "v5_bundle" / "v5_train_dataset.pt"
    v10_p = PROJECT_ROOT / "v10_bundle" / "v10_train_dataset.pt"

    all_inis, all_fins = mc_comp.all_token_names()
    all_yls = sorted({f[2:].rsplit("_", 1)[0] for f in all_fins})

    seen_tok_counts: Counter = Counter()
    if v10_p.exists():
        ds = torch.load(str(v10_p), map_location="cpu", weights_only=False)
        i2t = {v: k for k, v in ds["t2i_custom"].items()}
        for it in ds["items"]:
            for tid, _ in it["token_items"]:
                if tid in i2t:
                    seen_tok_counts[i2t[tid]] += 1

    mc_missing_inis = [ini[2:] for ini in all_inis if seen_tok_counts[ini] == 0]
    mc_low_inis = [(ini[2:], seen_tok_counts[ini]) for ini in all_inis if 0 < seen_tok_counts[ini] < 8]

    # 按韵类统计样本数，并附上代表字
    yl_rep: Dict[str, str] = {}
    for ch, rs in mc_comp.readings.items():
        if len(ch) == 1 and "\u4e00" <= ch <= "\u9fff":
            yl_rep.setdefault(rs[0].yunlei, f"{ch}({rs[0].desc})")
    yl_counts = {
        yl: sum(seen_tok_counts[f] for f in all_fins if f.startswith(f"Q_{yl}_"))
        for yl in all_yls
    }
    mc_zero_yls = [f"{yl}[例:{yl_rep.get(yl,'?')}]" for yl, c in yl_counts.items() if c == 0]
    mc_low_yls = [f"{yl}({c}次)[例:{yl_rep.get(yl,'?')}]" for yl, c in sorted(yl_counts.items(), key=lambda x: x[1]) if 0 < c < 5]

    report["mc"] = {
        "total_initials": len(all_inis),
        "missing_initials": mc_missing_inis,
        "low_sample_initials": mc_low_inis,
        "total_yunleis": len(all_yls),
        "zero_audio_yunleis_count": len(mc_zero_yls),
        "zero_audio_yunleis": mc_zero_yls,
        "low_audio_yunleis": mc_low_yls[:30],
    }

    # 2. OC 上古音盘点（声母含清鼻流音 hm/hn/hŋ/hl、全浊 b/d/g/dz、小舌/喉音 q/ʔ/ɣ、塞尾 -p/-t/-k/-m）
    oc_onsets: Counter = Counter()
    oc_codas: Counter = Counter()
    oc_examples: Dict[str, str] = {}
    for ch, (ipa, ttag) in old_comp.eras["oc"].items():
        clean = ipa[:-1] if len(ipa) >= 2 and ipa.endswith(("ʔ", "h", "s")) and not ipa.endswith(("ts", "tʂ", "tʃ")) else ipa
        s = parse_mononuclear_4slot(clean)
        if s.onset:
            oc_onsets[s.onset] += 1
            oc_examples.setdefault(f"onset:{s.onset}", f"{ch}({ipa})")
        if s.coda:
            oc_codas[s.coda] += 1
            oc_examples.setdefault(f"coda:{s.coda}", f"{ch}({ipa})")

    # 检查上古哪些特殊音位目前仅靠普通话近似回退而无独立棚录嵌入槽
    oc_approx_onsets = []
    for ons, cnt in oc_onsets.most_common():
        zv_tok = old_comp.ONSET_TO_ZV.get(ons, "")
        if not zv_tok.startswith("OC_") and ons not in ("p", "pʰ", "m", "t", "tʰ", "n", "l", "k", "kʰ", "x", "s", "ts", "tsʰ", "tɕ", "tɕʰ", "ɕ", "tʂ", "tʂʰ", "ʂ", "w", "j"):
            oc_approx_onsets.append(f"{ons}->{zv_tok or '零'}({cnt}字, 例:{oc_examples.get('onset:'+ons,'')})")
    oc_dead_zero = [k for k in seen_tok_counts if k.startswith("OC_") and seen_tok_counts[k] == 0]
    oc_untrained = [k for k in old_comp.t2i if k.startswith("OC_") and seen_tok_counts[k] < 5]

    report["oc"] = {
        "lexicon_chars": len(old_comp.eras["oc"]),
        "distinct_onsets": len(oc_onsets),
        "distinct_codas": dict(oc_codas),
        "approximated_onsets_needing_native_recording": oc_approx_onsets,
        "low_or_zero_audio_oc_tokens": [(k, seen_tok_counts[k]) for k in sorted(oc_untrained)],
    }

    # 3. EM 近古官话盘点（微母 ʋ、疑母 ŋ、尖音 tsj/sj、入声 -ʔ、闭口韵 -m）
    em_onsets: Counter = Counter()
    em_tones: Counter = Counter()
    em_examples: Dict[str, str] = {}
    for ch, (ipa, ttag) in old_comp.eras["em"].items():
        s = parse_mononuclear_4slot(ipa)
        em_onsets[s.onset or "∅"] += 1
        em_tones[ttag] += 1
        em_examples.setdefault(s.onset or "∅", f"{ch}({ipa})")

    em_untrained = [k for k in old_comp.t2i if k.startswith("EM_") and seen_tok_counts[k] < 5]
    report["em"] = {
        "lexicon_chars": len(old_comp.eras["em"]),
        "distinct_onsets": dict(em_onsets.most_common()),
        "tone_distribution": dict(em_tones),
        "special_em_slots_low_audio": [(k, seen_tok_counts[k]) for k in em_untrained],
        "notes_for_recording": [
            "近古棚录干声缺口：目前近古新语料多来自 B 站视频（不能用于解冻声学解码器），若能补充 20~30 句无伴奏 24kHz 录音室级明清官话干声（覆盖尖音 仙/酒/錢/小、微母 ʋ 文/無/萬、疑母 ŋ 我/岸/牛、短促入声 -ʔ 竹/石/白/月），效果将产生质变！"
        ],
    }
    return report


def synth_lines(
    model: ZipVoice,
    vocos: PureVocosDecoder,
    t2i: dict,
    mc_comp: QieyunMCCompiler,
    old_comp: ThreeEraHanfireCompiler,
    era: str,
    lines: List[str],
    pit: list,
    pf: torch.Tensor,
    pl: torch.Tensor,
    fp100: float,
    sos_bp: np.ndarray,
    out_wav: Path,
) -> dict:
    chunks, durs, peaks, dws = [], [], [], []
    char_counts = []
    for j, cl in enumerate(lines):
        clean_chars = re.sub(r"\[([^\[\]=]+)=([^\[\]]+)\]", r"\1", cl)
        clean_chars = re.sub(r"[^\u4e00-\u9fff]", "", clean_chars)
        char_counts.append(len(clean_chars))
        if era == "mc":
            it, _ = mc_comp.compile_clause(cl, metrical=True, rhyme_hold_w=16)
        else:
            it, _ = old_comp.compile_era_clause(era, cl, end_punct="", rhyme_hold_w=(24 if era == "oc" else 16))
        mel = sample_clean_zero_carrier(
            model=model,
            prompt_items=pit,
            target_items=it,
            t2i=t2i,
            prompt_features=pf,
            prompt_features_lens=pl,
            frames_per_100w=fp100,
            char0_boost_w=44,
            num_step=16,
            guidance_scale=1.0,
            t_shift=0.5,
            seed=42,
        )
        w = vocos.decode_mel(mel.permute(0, 2, 1) / FEAT_SCALE)
        w = signal.sosfiltfilt(sos_bp, w).astype(np.float32)
        w = trim_edge_silence(w, rel_db=52.0, pre_ms=35, post_ms=65)
        w = safe_normalize(w, target_rms=0.055, max_peak=0.72)
        f = int(0.025 * SR)
        if len(w) > f * 2:
            w[:f] *= 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, f, dtype=np.float32)))
            w[-f:] *= 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, f, dtype=np.float32)))
        durs.append(round(len(w) / SR, 2))
        peaks.append(round(float(np.max(np.abs(w))), 3))
        dws.append(round(float(np.max(np.abs(np.diff(w)))), 3))
        chunks += [w, np.zeros(int((0.26 if j % 2 == 0 else 0.42) * SR), dtype=np.float32)]

    full = np.concatenate(chunks)
    sf.write(str(out_wav), full, SR)
    clauses = split_clauses(full, SR, min_sil_ms=160)

    # 底噪
    w_sp = np.concatenate(chunks[::2])
    hop = int(0.02 * SR)
    fr = w_sp[: (len(w_sp) // hop) * hop].reshape(-1, hop)
    db = 20 * np.log10(np.sqrt(np.mean(fr ** 2, axis=1)) + 1e-9)
    act = db > (db.max() - 28.0)
    nz = (~act) & (db > -95.0)
    flr = round(float(np.median(db[nz])) if np.any(nz) else -99.0, 1)

    # 一字一音字均时长 (s/char)
    spc = [round(d / max(1, nc), 3) for d, nc in zip(durs, char_counts)]
    equal_len = len(set(char_counts)) == 1
    spread = round(max(durs) - min(durs), 2) if equal_len else 0.0

    return {
        "wav": str(out_wav),
        "expected_clauses": len(lines),
        "detected_clauses": len(clauses),
        "durs": durs,
        "sec_per_char": spc,
        "spread": spread,
        "max_peak": max(peaks),
        "max_dw": max(dws),
        "noise_floor_db": flr,
        "raw_clause_wavs": chunks[::2],
    }


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck_p = PROJECT_ROOT / "v10_outputs" / "best_fp32_v10_studio.pt"
    if not ck_p.exists():
        ck_p = PROJECT_ROOT / "v9_outputs" / "best_fp32_v9mc.pt"
    ck = torch.load(str(ck_p), map_location="cpu", weights_only=False)
    t2i = ck["t2i_custom"]

    model = ZipVoice(**ck["config"])
    model.load_state_dict(ck["model"], strict=True)
    if ck_p.name != "best_fp32_v10_studio.pt":
        calibrate_studio_embeddings(model.embed.weight.data, t2i)
    model = model.to(dev).eval()

    vocos, fbank = PureVocosDecoder(dev), PureVocosFbank()
    mc_comp = QieyunMCCompiler(t2i)
    old_comp = ThreeEraHanfireCompiler(t2i, PROJECT_ROOT / "v5_bundle" / "v5_three_era_lexicon.json", v8_fixes=True)

    out_dir = PROJECT_ROOT / "v10_outputs" / "overnight_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] 盘点上古/中古/近古三代音系覆盖度与缺口...")
    missing_rep = audit_missing_phonemes(mc_comp, old_comp)
    print(f"  ✓ MC 零音频韵类={missing_rep['mc']['zero_audio_yunleis_count']}/{missing_rep['mc']['total_yunleis']} | OC 低样本槽={len(missing_rep['oc']['low_or_zero_oc_tokens']) if 'low_or_zero_oc_tokens' in missing_rep['oc'] else len(missing_rep['oc']['low_or_zero_audio_oc_tokens'])}")

    pw_m, _ = sf.read(str(PROJECT_ROOT / "v5_bundle" / "ref_male_128Hz.wav"), dtype="float32")
    pw_m = (pw_m * 0.45).astype(np.float32)
    pit_m, _ = tokenize_mandarin_prompt_exact("高分经典悬疑电影，", t2i)
    pf_m = fbank.extract(pw_m, sr=SR).unsqueeze(0).to(dev) * FEAT_SCALE
    pl_m = torch.tensor([pf_m.size(1)], device=dev)

    pw_f, _ = sf.read(str(PROJECT_ROOT / "v5_bundle" / "ref_female_252Hz.wav"), dtype="float32")
    pw_f = (pw_f * 0.45).astype(np.float32)
    pit_f, _ = tokenize_mandarin_prompt_exact("今夜的月光如此清亮，", t2i)
    pf_f = fbank.extract(pw_f, sr=SR).unsqueeze(0).to(dev) * FEAT_SCALE
    pl_f = torch.tensor([pf_f.size(1)], device=dev)

    sos_bp = signal.butter(4, [75.0 / (SR / 2), 9500.0 / (SR / 2)], btype="band", output="sos")

    # ==================== [2/4] 三代全音系代表字谱合成 ====================
    print("[2/4] 合成三代全音系代表字谱（三十八母、平上去入八调、上古塞尾清鼻音、近古五声尖团）...")
    phoneme_sheets = [
        # 中古三十八母全覆盖代表字谱（按发音部位分组，每组四/五字，一字一母）
        ("Sheet_MC_三十八母全表_male.wav", "mc", 24.5, [
            "幫滂並明",      # 唇音四母
            "端透定泥來",    # 舌头半舌五母
            "知徹澄孃",      # 舌上四母
            "精清從心邪",    # 齿头精组五母
            "莊初崇生俟",    # 正齿庄组五母
            "章昌船書常日",  # 正齿章组+半齿六母
            "見溪羣疑",      # 牙音四母
            "影曉匣云以",    # 喉音五母（合 38 母！）
        ]),
        # 中古平上去入四声八调同韵最小对立谱（东/冬/阳/仙四摄平上去入）
        ("Sheet_MC_平上去入八调对立_male.wav", "mc", 24.5, [
            "東董送屋",      # 阴调：平上去入 (-uŋ / -uk)
            "同動洞獨",      # 阳调：平上去入 (-uŋ / -uk)
            "方紡放縛",      # 阳摄：平上去入 (-aŋ / -ak)
            "仙獮線薛",      # 仙摄：平上去入 (-jɛn / -jɛt)
        ]),
        # 上古音核心特征音系谱（全浊塞音、清鼻流音、复元音/圆唇舌根、-p/-t/-k/-m 四塞鼻尾）
        ("Sheet_OC_上古核心音系与塞尾_male.wav", "oc", 25.0, [
            "關雎窈窕",      # 圆唇舌根与幽部
            "蒹葭白露",      # 鱼铎开合对转 (-a / -ak)
            "參差荇菜",      # 侵部 -m 与之部
            "習習谷風",      # 缉部 -p、屋部 -k、侵冬部
            "陟彼高岡",      # 职部 -k 与阳部 -ŋ
            "昔我往矣",      # 铎锡入声与上声 *-ʔ
        ]),
        # 近古明代官话核心音系谱（尖团对立、平分阴阳、入声短促收束）
        ("Sheet_EM_近古尖团五声全谱_male.wav", "em", 24.5, [
            "清風明月",      # 尖音清 vs 阳平明 vs 入声月
            "秋江夜雪",      # 尖音秋/雪 vs 团音江
            "東風吹酒",      # 尖音酒 vs 阴平东/风/吹
            "桃花仙人",      # 尖音仙 vs 阳平桃/人
        ]),
    ]

    sheet_results = []
    sheet_failures = []
    for fname, era, fp100, lines in phoneme_sheets:
        res = synth_lines(model, vocos, t2i, mc_comp, old_comp, era, lines, pit_m, pf_m, pl_m, fp100, sos_bp, out_dir / fname)
        # 若为平上去入对立表，额外检验第1句 (東董送屋) 四字时长与基频轮廓！
        tone_audit = {}
        if "平上去入" in fname:
            c0 = res["raw_clause_wavs"][0]
            L = len(c0)
            # 切为4等份近似观测 東(平) 董(上) 送(去) 屋(入) 的 F0 斜率
            for idx_t, t_label in enumerate(["平(東)", "上(董)", "去(送)", "入(屋)"]):
                sub = c0[int(L * idx_t / 4) : int(L * (idx_t + 1) / 4)]
                f_s, f_e, df = estimate_f0_contour(sub, SR)
                tone_audit[t_label] = {"f0_start": f_s, "f0_end": f_e, "f0_delta": df}
        del res["raw_clause_wavs"]
        res["tone_audit"] = tone_audit
        s_errs = []
        if res["detected_clauses"] != res["expected_clauses"]:
            s_errs.append(f"句中断裂({res['detected_clauses']}!={res['expected_clauses']})")
        if res["max_peak"] > 0.80 or res["max_dw"] > 0.32:
            s_errs.append(f"过载破音或爆破毛刺(pk={res['max_peak']}, dw={res['max_dw']})")
        if res["noise_floor_db"] > -42.0:
            s_errs.append(f"底噪偏高({res['noise_floor_db']}dB)")
        res["aesthetic_pass"] = len(s_errs) == 0
        res["errors"] = s_errs
        if s_errs:
            sheet_failures.append((fname, s_errs))
        sheet_results.append(res)
        s_tag = "PASS" if not s_errs else "FAIL"
        print(f"  [{s_tag}] {fname:36s} | 分句={res['detected_clauses']}/{res['expected_clauses']} | pk={res['max_peak']} dw={res['max_dw']} | 字速={res['sec_per_char']}s/字 | 底噪={res['noise_floor_db']}dB {s_errs if s_errs else ''}")
        if tone_audit:
            print(f"    四声基频轨迹: {tone_audit}")

    # ==================== [3/4] 三代 12 首经典代表诗篇全检验 ====================
    print("\n[3/4] 合成三代代表诗篇并运行六维审美与押韵检验...")
    poems = [
        # 上古音 (OC) 4首（幽部、阳部、鱼部、宵部）
        ("OC_诗经_关雎_male.wav", "oc", "male", 25.0, ["關關雎鳩", "在河之洲", "窈窕淑女", "君子好逑"], (0, 1, 3)),
        ("OC_诗经_蒹葭_male.wav", "oc", "male", 25.5, ["蒹葭蒼蒼", "白露為霜", "所謂伊人", "在水一方"], (0, 1, 3)),
        ("OC_诗经_桃夭_female.wav", "oc", "female", 26.0, ["桃之夭夭", "灼灼其華", "之子于歸", "宜其室家"], (1, 3)),
        ("OC_诗经_鹿鸣_male.wav", "oc", "male", 25.0, ["呦呦鹿鳴", "食野之苹", "我有嘉賓", "鼓瑟吹笙"], (0, 1, 3)),
        # 中古音 (MC) 7首（含入声韵、阳声韵、阴声韵、上声韵、尤侯合韵）
        ("MC_唐诗_静夜思_male.wav", "mc", "male", 24.5, ["床前明月光", "疑是地上霜", "舉頭望明月", "低頭思故鄉"], (0, 1, 3)),
        ("MC_唐诗_静夜思_female.wav", "mc", "female", 26.5, ["床前明月光", "疑是地上霜", "舉頭望明月", "低頭思故鄉"], (0, 1, 3)),
        ("MC_唐诗_春晓_male.wav", "mc", "male", 24.5, ["春眠不覺曉", "[處=昌開三魚去]處聞啼鳥", "夜來風雨聲", "花落知多少"], (0, 1, 3)),
        ("MC_唐诗_江雪_male.wav", "mc", "male", 24.5, ["千山鳥飛絕", "萬徑人蹤滅", "孤舟蓑笠翁", "獨釣寒江雪"], (0, 1, 3)),
        ("MC_唐诗_登鹳雀楼_male.wav", "mc", "male", 24.5, ["白日依山盡", "黃河入海流", "欲窮千里目", "更上一層樓"], (1, 3)),
        ("MC_唐诗_相思_female.wav", "mc", "female", 25.5, ["紅豆生南國", "春來發幾枝", "願君多采擷", "此物最相思"], (1, 3)),
        ("MC_唐诗_枫桥夜泊_male.wav", "mc", "male", 24.5, ["月落烏啼霜滿天", "江楓漁火對愁眠", "姑蘇城外寒山寺", "夜半鐘聲到客船"], (0, 1, 3)),
        # 近古音 (EM) 4首（明代七绝、明代长短句词）
        ("EM_明诗_桃花庵歌_male.wav", "em", "male", 24.0, ["桃花塢裏桃花庵", "桃花庵下桃花仙", "桃花仙人[種=tʂuŋh]桃樹", "又摘桃花換酒錢"], (0, 1, 3)),
        ("EM_明诗_桃花庵歌_female.wav", "em", "female", 25.0, ["桃花塢裏桃花庵", "桃花庵下桃花仙", "桃花仙人[種=tʂuŋh]桃樹", "又摘桃花換酒錢"], (0, 1, 3)),
        ("EM_明词_临江仙_male.wav", "em", "male", 24.0, ["滾滾長江東逝水", "浪花淘盡英雄", "是非成敗轉頭空", "青山依舊在", "幾度夕陽紅"], (1, 2, 4)),
        ("EM_明诗_石灰吟_male.wav", "em", "male", 24.0, ["千錘萬鑿出深山", "烈火焚燒若等閒", "粉骨碎身渾不怕", "要留清白在人間"], (0, 1, 3)),
    ]

    poem_results = []
    failures = []
    for fname, era, vtag, fp100, lines, r_idx in poems:
        pit, pf, pl = (pit_m, pf_m, pl_m) if vtag == "male" else (pit_f, pf_f, pl_f)
        res = synth_lines(model, vocos, t2i, mc_comp, old_comp, era, lines, pit, pf, pl, fp100, sos_bp, out_dir / fname)
        raw_cs = res.pop("raw_clause_wavs")

        # 与 audit_regression.py / audit_poem 完全一致：在去除首尾静音后的纯净分句上计算押韵对比度
        w_saved, _ = sf.read(str(out_dir / fname), dtype="float32")
        eval_cs = split_clauses(w_saved, SR, min_sil_ms=160)
        if len(eval_cs) != len(lines):
            eval_cs = raw_cs
        sigs = [extract_rhyme_signature(c, SR) for c in eval_cs]
        nr_idx = [i for i in range(len(sigs)) if i not in r_idx]
        r_m = float(np.mean([rhyme_dist(sigs[i], sigs[j]) for a, i in enumerate(r_idx) for j in r_idx[a + 1 :]]))
        nr_m = float(np.mean([rhyme_dist(sigs[i], sigs[j]) for i in r_idx for j in nr_idx]))
        contrast = round(nr_m - r_m, 3)
        res["rhyme_contrast"] = contrast

        # 六维审美门禁判定
        errs = []
        if res["detected_clauses"] != res["expected_clauses"]:
            errs.append(f"句中断裂({res['detected_clauses']}!={res['expected_clauses']})")
        if any(s < 0.17 or s > 0.36 for s in res["sec_per_char"]):
            errs.append(f"字速越界({res['sec_per_char']})")
        if res["spread"] > 0.28:
            errs.append(f"句长极差过大({res['spread']}s)")
        dw_lim = 0.42 if vtag == "female" else 0.32
        if res["max_peak"] > 0.80 or res["max_dw"] > dw_lim:
            errs.append(f"削波或爆音(pk={res['max_peak']}, dw={res['max_dw']})")
        if res["noise_floor_db"] > -42.0:
            errs.append(f"底噪偏高({res['noise_floor_db']}dB)")
        if contrast <= 0.0:
            errs.append(f"押韵对比度不足({contrast:+.3f})")

        res["aesthetic_pass"] = len(errs) == 0
        res["errors"] = errs
        poem_results.append(res)
        if errs:
            failures.append((fname, errs))
        tag = "PASS" if not errs else "FAIL"
        print(f"  [{tag}] {fname:32s} | 分句={res['detected_clauses']}/{res['expected_clauses']} 字速={res['sec_per_char']} | pk={res['max_peak']} dw={res['max_dw']} | 底噪={res['noise_floor_db']}dB | 押韵={contrast:+.3f} {errs if errs else ''}")

    full_report = {
        "missing_phonemes": missing_rep,
        "phoneme_sheets": sheet_results,
        "sheet_failures": sheet_failures,
        "poems": poem_results,
        "total_poems": len(poems),
        "passed_poems": len(poems) - len(failures),
        "failures": failures,
    }
    rep_p = out_dir / "audit_report.json"
    rep_p.write_text(json.dumps(full_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[4/4] 巡检报告已保存: {rep_p} | 字谱通过={len(phoneme_sheets)-len(sheet_failures)}/{len(phoneme_sheets)} | 诗词通过={len(poems)-len(failures)}/{len(poems)}")
    sys.exit(0 if (not failures and not sheet_failures) else 1)


if __name__ == "__main__":
    main()
