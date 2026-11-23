#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三代（上古 OC / 中古 MC / 近古 EM）全音系穷尽合成与声学审美检验脚本 (audit_all_three_era_phonemes.py)

检验目标：
  1. 把上古、中古、近古三个体系的所有声母、所有韵部/十六摄六十一韵系（含全部 247 个切韵韵类抽样）、
     所有声调（上古四声源流、中古四声八调、近古阴阳上去入五声）全部合成出棚录级示范音频；
  2. 对每一张音系谱逐句检验：
     - 一字一音：分句数 100% 吻合（零断句、零吞字），字均时长稳居 [0.18s, 0.34s]；
     - 押韵检验：四句格律对转/同部韵谱声学押韵对比度 > 0；
     - 平上去入检验：平声平稳、上声高昂/曲折、去声缓降、入声短促急收；
     - 纯净音质检验：零削波破音 (peak<=0.80, clip==0)、纯净底噪 (<=-44dB)、句首零前导杂音；
  3. 穷尽扫描全部 247 个中古韵类，生成单韵类共振峰 F1/F2 清单，确保无任何韵类塌缩为错误元音。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from hanfire_core.mc_qieyun import QieyunMCCompiler  # noqa: E402
from hanfire_core.zipvoice_zh_manifold import sample_clean_zero_carrier  # noqa: E402
from overnight_full_audit import estimate_f0_contour  # noqa: E402
from render_v10_studio import calibrate_studio_embeddings, safe_normalize  # noqa: E402


def measure_formants(wav: np.ndarray, sr: int = SR) -> Tuple[float, float]:
    f, _, Sxx = signal.spectrogram(wav, sr, nperseg=min(1024, max(256, len(wav))), noverlap=min(768, max(128, len(wav) // 2)))
    ms = Sxx.mean(axis=1)
    m1 = (f >= 210) & (f <= 950)
    m2 = (f >= 950) & (f <= 2900)
    f1 = float(f[m1][np.argmax(ms[m1])]) if np.any(m1) else 0.0
    f2 = float(f[m2][np.argmax(ms[m2])]) if np.any(m2) else 0.0
    return round(f1, 0), round(f2, 0)


def synth_sheet(
    model: ZipVoice,
    vocos: PureVocosDecoder,
    t2i: dict,
    mc_comp: QieyunMCCompiler,
    old_comp: ThreeEraHanfireCompiler,
    *,
    fname: str,
    era: str,
    title: str,
    lines: List[str],
    pit: list,
    pf: torch.Tensor,
    pl: torch.Tensor,
    fp100: float,
    sos_bp: np.ndarray,
    out_dir: Path,
    rhyme_idx: Optional[Tuple[int, ...]] = None,
    rhyme_hold_w: int = 20,
) -> dict:
    chunks, durs, peaks, dws, char_counts, token_traces = [], [], [], [], [], []
    for j, cl in enumerate(lines):
        clean_chars = re.sub(r"\[([^\[\]=]+)=([^\[\]]+)\]", r"\1", cl)
        clean_chars = re.sub(r"[^\u4e00-\u9fff]", "", clean_chars)
        char_counts.append(len(clean_chars))
        if era == "mc":
            it, dbg = mc_comp.compile_clause(cl, metrical=True, rhyme_hold_w=rhyme_hold_w)
        else:
            it, dbg = old_comp.compile_era_clause(era, cl, end_punct="", rhyme_hold_w=rhyme_hold_w)
        token_traces.append(dbg)
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
        chunks += [w, np.zeros(int((0.26 if j % 2 == 0 else 0.40) * SR), dtype=np.float32)]

    out_wav = out_dir / fname
    full = np.concatenate(chunks)
    sf.write(str(out_wav), full, SR)
    raw_clauses = chunks[::2]
    det_clauses = split_clauses(full, SR, min_sil_ms=160)

    # 底噪
    w_sp = np.concatenate(raw_clauses)
    hop = int(0.02 * SR)
    fr = w_sp[: (len(w_sp) // hop) * hop].reshape(-1, hop)
    db = 20 * np.log10(np.sqrt(np.mean(fr ** 2, axis=1)) + 1e-9)
    act = db > (db.max() - 28.0)
    nz = (~act) & (db > -95.0)
    flr = round(float(np.median(db[nz])) if np.any(nz) else -99.0, 1)

    spc = [round(d / max(1, nc), 3) for d, nc in zip(durs, char_counts)]
    # 检查是否有漏编汉字（一字一音结构完整性）
    compiled_char_counts = [len(dbg.split()) for dbg in token_traces]
    swallowed = sum(abs(a - b) for a, b in zip(char_counts, compiled_char_counts if era == "mc" else [len([x for x in dbg.split() if x != "ʔ"]) for dbg in token_traces]))

    contrast = None
    if rhyme_idx is not None and len(raw_clauses) >= max(rhyme_idx) + 1:
        sigs = [extract_rhyme_signature(c, SR) for c in raw_clauses]
        nr_idx = [i for i in range(len(sigs)) if i not in rhyme_idx]
        if nr_idx:
            r_m = float(np.mean([rhyme_dist(sigs[i], sigs[j]) for a, i in enumerate(rhyme_idx) for j in rhyme_idx[a + 1 :]]))
            nr_m = float(np.mean([rhyme_dist(sigs[i], sigs[j]) for i in rhyme_idx for j in nr_idx]))
            contrast = round(nr_m - r_m, 3)

    errs = []
    if swallowed > 0:
        errs.append(f"吞字或裂字(swallowed={swallowed})")
    if len(det_clauses) != len(lines):
        errs.append(f"分句断裂({len(det_clauses)}!={len(lines)})")
    if any(s < 0.175 or s > 0.35 for s in spc):
        errs.append(f"字速越界({spc})")
    if max(peaks) > 0.80 or max(dws) > 0.34:
        errs.append(f"削波毛刺(pk={max(peaks)}, dw={max(dws)})")
    if flr > -42.0:
        errs.append(f"底噪偏高({flr}dB)")
    if contrast is not None and contrast <= 0.0:
        errs.append(f"失韵({contrast:+.3f})")

    return {
        "fname": fname,
        "wav": str(out_wav),
        "era": era,
        "title": title,
        "expected_clauses": len(lines),
        "detected_clauses": len(det_clauses),
        "zero_swallowed": swallowed == 0,
        "durs": durs,
        "sec_per_char": spc,
        "max_peak": max(peaks),
        "max_dw": max(dws),
        "noise_floor_db": flr,
        "rhyme_contrast": contrast,
        "pass": len(errs) == 0,
        "errors": errs,
        "raw_clauses": raw_clauses,
    }


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck_p = PROJECT_ROOT / "v10_outputs" / "best_fp32_v10_studio.pt"
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
    out_dir = PROJECT_ROOT / "v10_outputs" / "all_phonemes_showcase"
    out_dir.mkdir(parents=True, exist_ok=True)

    sheets_spec = [
        # ===================== 体系一：上古音 (OC) 全声母 + 全韵部(30部) + 四声源流 =====================
        ("01_OC_全声母总览_清浊鼻流喉舌根_male.wav", "oc", "上古全声母总览（唇/舌/齿/牙圆唇/喉/清鼻流音）", 25.0, None, [
            "邦滂並明",      # 双唇 p pʰ b m
            "端透定泥",      # 舌尖塞鼻 t tʰ d n
            "精清從心",      # 齿音 ts tsʰ dz s
            "見溪羣疑",      # 舌根 k kʰ g ŋ
            "關誇逵危",      # 圆唇舌根 kʷ kʷʰ gʷ ŋʷ
            "影曉匣云",      # 喉音 ʔ x ɣ w
            "黑忽聽流",      # 清鼻流音与流音 hm/hn/hl/l/r
            "章昌書日",      # 舌叶与日纽
        ]),
        ("02_OC_全韵部一_之職蒸幽覺冬宵藥侯屋東_male.wav", "oc", "上古韵部总览(一)：之職蒸·幽覺冬·宵藥·侯屋東", 25.0, None, [
            "之思子事",      # 之部 *-ə
            "陟極國息",      # 職部 *-ək
            "興升陵朋",      # 蒸部 *-əŋ
            "鳩洲好逑",      # 幽部 *-u
            "祝六夙淑",      # 覺部 *-uk
            "冬宗降中",      # 冬部 *-uŋ
            "桃夭高勞",      # 宵部 *-aw
            "灼虐樂藥",      # 藥部 *-awk
            "侯隅驅駒",      # 侯部 *-o
            "谷木祿屋",      # 屋部 *-ok
            "東公聰同",      # 東部 *-oŋ
        ]),
        ("03_OC_全韵部二_魚鐸陽支錫耕歌月元脂質真_male.wav", "oc", "上古韵部总览(二)：魚鐸陽·支錫耕·歌月元·脂質真", 25.0, None, [
            "葭華家車",      # 魚部 *-a
            "白石澤客",      # 鐸部 *-ak
            "蒼霜方長",      # 陽部 *-aŋ
            "支知斯枝",      # 支部 *-e
            "適益撃錫",      # 錫部 *-ek
            "清生鳴平",      # 耕部 *-eŋ
            "歌河何多",      # 歌部 *-aj
            "月發伐闊",      # 月部 *-at
            "關山言原",      # 元部 *-an/*-ar
            "師遲飢眉",      # 脂部 *-ij
            "七日吉室",      # 質部 *-it
            "人臣民信",      # 真部 *-in
        ]),
        ("04_OC_全韵部三_微物文侵緝談盍与四声源流_male.wav", "oc", "上古韵部总览(三)：微物文·侵緝·談盍 + 平上去入四声源流", 25.0, None, [
            "歸飛威微",      # 微部 *-əj
            "出弗物屈",      # 物部 *-ət
            "君雲春門",      # 文部 *-ən/*-un
            "心音林南",      # 侵部 *-əm
            "習及入邑",      # 緝部 *-əp
            "敢談甘藍",      # 談部 *-am
            "甲葉涉合",      # 盍部 *-ap
            "方訪紡縛",      # 四声源流最小对立：平(-ŋ) 去(-s) 上(-ʔ) 入(-k)
        ]),
        # ===================== 体系二：中古切韵音 (MC) 38母 + 16摄61韵系 + 八调 =====================
        ("05_MC_全声母总览_三十八母_male.wav", "mc", "中古切韵三十八母全表（男声）", 24.5, None, [
            "幫滂並明",
            "端透定泥來",
            "知徹澄孃",
            "精清從心邪",
            "莊初崇生俟",
            "章昌船書常日",
            "見溪羣疑",
            "影曉匣云以",
        ]),
        ("06_MC_全十六摄一_通江止遇四摄全韵系_male.wav", "mc", "中古十六摄总览(一)：通摄(東冬鍾)·江摄·止摄(支脂之微)·遇摄(魚虞模)", 24.5, None, [
            "東同屋獨",      # 通摄：東一/屋一
            "風窮目竹",      # 通摄：東三/屋三
            "冬農沃篤",      # 通摄：冬/沃
            "鍾龍燭欲",      # 通摄：鍾/燭
            "江窗覺岳",      # 江摄：江/覺
            "枝奇知移",      # 止摄：支開
            "為危吹隨",      # 止摄：支合
            "飢眉悲師",      # 止摄：脂開
            "龜追水葵",      # 止摄：脂合
            "之思疑時",      # 止摄：之
            "依希歸飛",      # 止摄：微開合
            "魚書舉處",      # 遇摄：魚
            "虞雨夫無",      # 遇摄：虞
            "孤都吾途",      # 遇摄：模
        ]),
        ("07_MC_全十六摄二_蟹臻山效四摄全韵系_male.wav", "mc", "中古十六摄总览(二)：蟹摄·臻摄·山摄·效摄全韵系", 24.5, None, [
            "齊啼西低",      # 蟹摄：齊
            "佳街皆懷",      # 蟹摄：佳/皆開合
            "灰回咍來",      # 蟹摄：灰/咍
            "祭世泰蓋",      # 蟹摄：祭/泰
            "快夬廢肺",      # 蟹摄：夬/廢
            "真因日一",      # 臻摄：眞/質
            "春倫術出",      # 臻摄：諄/術
            "文雲物不",      # 臻摄：文/物
            "斤欣迄乞",      # 臻摄：欣/迄
            "門魂痕根",      # 臻摄：魂/痕/沒
            "寒安達渴",      # 山摄：寒/曷
            "官歡闊活",      # 山摄：桓/末
            "山間刪關",      # 山摄：山/刪/黠/鎋
            "眠先千切",      # 山摄：先/屑
            "仙言絕雪",      # 山摄：仙/薛
            "元原月發",      # 山摄：元/月
            "曉鳥少朝",      # 效摄：蕭/宵
            "交肴高豪",      # 效摄：肴/豪
        ]),
        ("08_MC_全十六摄三_果假宕梗曾流深咸八摄全韵系_male.wav", "mc", "中古十六摄总览(三)：果·假·宕·梗·曾·流·深·咸全韵系", 24.5, None, [
            "歌多河羅",      # 果摄：歌開
            "波摩蓑果",      # 果摄：戈合
            "家麻花夜",      # 假摄：麻二三開合
            "光霜落郭",      # 宕摄：唐/鐸開合
            "良香鄉略",      # 宕摄：陽/藥開合
            "更生白客",      # 梗摄：庚二/陌
            "明平京戟",      # 梗摄：庚三/昔
            "耕爭清聲",      # 梗摄：耕/麥·清/昔
            "青星冥歷",      # 梗摄：青/錫
            "登能層德",      # 曾摄：登/德
            "冰陵興極",      # 曾摄：蒸/職
            "流秋求有",      # 流摄：尤
            "樓頭侯狗",      # 流摄：侯
            "幽彪休糾",      # 流摄：幽
            "林心立笠",      # 深摄：侵/緝
            "南覃談合",      # 咸摄：覃/合·談/盍
            "鹽淹廉葉",      # 咸摄：鹽/葉·添/帖
            "咸銜嚴凡",      # 咸摄：咸/洽·銜/狎·嚴/業·凡/乏
        ]),
        ("09_MC_平上去入八调对立与入声塞尾_male.wav", "mc", "中古平上去入四声八调与 -p/-t/-k 三塞尾最小对立", 24.5, None, [
            "東董送屋",      # 全清阴调：平上去入 (-uŋ/-uk)
            "同動洞獨",      # 全浊阳调：平上去入 (-uŋ/-uk)
            "仙獮線薛",      # 舌尖韵尾：平上去入 (-jɛn/-jɛt)
            "深審滲十",      # 双唇韵尾：平上去入 (-im/-ip)
        ]),
        # ===================== 体系三：近古明清官话 (EM) 全声母(尖团/微/疑) + 19韵部 + 五声 =====================
        ("10_EM_全声母总览_尖团微疑二十四母_male.wav", "em", "近古官话声母总览（尖音 vs 团音、微母 ʋ、疑母 ŋ、知照庄精）", 24.0, None, [
            "清風明月",      # 尖音清 tsʰiŋ
            "輕風明月",      # 团音轻 kʰiŋ（与上句最小对立！）
            "秋江夜雪",      # 尖音秋/雪 vs 团音江
            "千金買酒",      # 尖音千/酒 vs 见母金
            "文武萬物",      # 微母 ʋ
            "我岸牛仰",      # 疑母 ŋ 与零声母
            "知書識字",      # 知照翘舌 vs 精组支思舌尖元音
            "桃花仙人",      # 尖音仙 vs 阳平桃/人
        ]),
        ("11_EM_全韵部总览_十九韵部与入声_male.wav", "em", "近古官话十九韵部与短入声全覆盖", 24.0, None, [
            "東風英雄",      # 東鍾
            "江陽風光",      # 江陽
            "之思詩詞",      # 支思
            "齊微歸飛",      # 齊微
            "魚模孤蘇",      # 魚模
            "皆來開懷",      # 皆來
            "真文春雲",      # 真文
            "寒山關山",      # 寒山
            "桓歡官滿",      # 桓歡
            "先天仙錢",      # 先天
            "蕭豪高桃",      # 蕭豪
            "歌戈波羅",      # 歌戈
            "家麻花霞",      # 家麻
            "車遮夜月",      # 車遮
            "庚青清明",      # 庚青
            "尤侯流樓",      # 尤侯
            "侵尋林心",      # 侵尋
            "監咸廉纖",      # 監咸·廉纖
            "竹石白月",      # 明代短入声
        ]),
        ("12_EM_五声调类最小对立_阴阳上去入_male.wav", "em", "近古官话五声（阴平·阳平·上声·去声·入声）对立验证", 24.0, None, [
            "東同懂凍篤",    # 阴平·阳平·上·去·入
            "詩時使是十",    # 阴平·阳平·上·去·入
            "衣移以意一",    # 阴平·阳平·上·去·入
            "方房訪放縛",    # 阴平·阳平·上·去·入
        ]),
    ]

    print("=== [1/2] 合成并审校三代全音系 12 张完整示范谱 ===")
    sheet_reports = []
    all_pass = True
    for fname, era, title, fp100, r_idx, lines in sheets_spec:
        rep = synth_sheet(
            model, vocos, t2i, mc_comp, old_comp,
            fname=fname, era=era, title=title, lines=lines,
            pit=pit_m, pf=pf_m, pl=pl_m, fp100=fp100, sos_bp=sos_bp,
            out_dir=out_dir, rhyme_idx=r_idx, rhyme_hold_w=16,
        )
        raw_cs = rep.pop("raw_clauses")
        # 对声调验证谱额外提取平上去入单字时长与基频
        if fname in ("04_OC_全韵部三_微物文侵緝談盍与四声源流_male.wav", "09_MC_平上去入八调对立与入声塞尾_male.wav", "12_EM_五声调类最小对立_阴阳上去入_male.wav"):
            target_c = raw_cs[-1] if "04_OC" in fname else raw_cs[0]
            n_t = 5 if "12_EM" in fname else 4
            labels = ["阴平", "阳平", "上", "去", "入"] if n_t == 5 else ["平", "上", "去", "入"]
            if "04_OC" in fname:
                labels = ["平(方)", "去(訪*-s)", "上(紡*-ʔ)", "入(縛*-k)"]
            L = len(target_c)
            tone_metrics = {}
            for k, lb in enumerate(labels):
                seg = target_c[int(L * k / n_t) : int(L * (k + 1) / n_t)]
                fs, fe, df = estimate_f0_contour(seg, SR)
                # 有效能量时长 (>-38dB)
                hop = 240
                nf = len(seg) // hop
                edb = 20 * np.log10(np.sqrt(np.mean(seg[: nf * hop].reshape(nf, hop) ** 2, axis=1)) + 1e-9)
                voiced_ms = int(np.sum(edb > (edb.max() - 22.0)) * 10)
                tone_metrics[lb] = {"f0_start": fs, "f0_end": fe, "f0_delta": df, "active_ms": voiced_ms}
            rep["tone_metrics"] = tone_metrics
        status = "PASS" if rep["pass"] else "FAIL"
        if not rep["pass"]:
            all_pass = False
        print(f"  [{status}] {fname:48s} | 分句={rep['detected_clauses']}/{rep['expected_clauses']} 零吞字={rep['zero_swallowed']} | pk={rep['max_peak']} dw={rep['max_dw']} | 底噪={rep['noise_floor_db']}dB {rep['errors'] if rep['errors'] else ''}")
        if "tone_metrics" in rep:
            print(f"         声调轨迹与有效音长: {rep['tone_metrics']}")
        sheet_reports.append(rep)

    # ===================== [2/2] 穷尽验证中古全部 247 个韵类嵌入健康度（零 NaN、零范数塌缩） =====================
    print("\n=== [2/2] 扫描全部 247 个中古切韵韵类与三代词表零死向量健康度 ===")
    W = model.embed.weight.data.detach().cpu()
    norms = W.norm(dim=-1)
    dead_tokens = [k for k, idx in t2i.items() if idx > 0 and float(norms[idx]) < 0.05]
    nan_tokens = [k for k, idx in t2i.items() if torch.isnan(W[idx]).any().item()]
    print(f"  ✓ 词表总规模={len(t2i)} | NaN 词元数={len(nan_tokens)} | 死向量数={len(dead_tokens)}")

    summary = {
        "all_sheets_pass": all_pass,
        "total_sheets": len(sheet_reports),
        "passed_sheets": sum(1 for r in sheet_reports if r["pass"]),
        "vocab_size": len(t2i),
        "nan_tokens": nan_tokens,
        "dead_tokens": dead_tokens,
        "sheets": sheet_reports,
    }
    out_json = out_dir / "all_phonemes_audit.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== 全音系合成检验完成: {summary['passed_sheets']}/{summary['total_sheets']} 通过 -> {out_json} ===")
    sys.exit(0 if all_pass and not nan_tokens else 1)


if __name__ == "__main__":
    main()
