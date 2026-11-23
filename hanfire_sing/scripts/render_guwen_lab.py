#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汉火古音唱吟 · V9.6 古文/辞赋三大调式精雕实验室 (render_guwen_lab.py)
  - 调式 G1（书斋清雅骈散调，母带 t09《兰亭集序》）：原版对照 + 《兰亭集序》现代/中古 + 盲测《爱莲说》
  - 调式 G2（唐调铭文沉雄顿挫调，母带 t16《陋室铭》）：原版对照 + 《陋室铭》现代/中古 + 盲测《诫子书》
  - 调式 G3（唐调辞赋大开大合长短句调，母带 t18《前赤壁赋》）：原版对照 + 《前赤壁赋》现代/中古 + 盲测《岳阳楼记》
自动检测本地是否已有纯正男声底座 (hanfire_sing_male)，有则同步渲染纯男声对照卡片！
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
SING_DIR = ROOT / "hanfire_sing"
DS_DIR = SING_DIR / "DiffSinger"
TANG_DIR = SING_DIR / "data_tang_processed"
WEB_DIST = ROOT / "hanfire_tts" / "web_wasm" / "dist_cf"
AUDIO_DIR = WEB_DIST / "_sing_showcase"
SCRATCH = ROOT / "work" / "sing_cache"
SCRATCH.mkdir(parents=True, exist_ok=True)

if str(SING_DIR) not in sys.path:
    sys.path.insert(0, str(SING_DIR))
from guwen_v14_engine import GUWEN_V14_MODES as GUWEN_V9_MODES, apply_gain_plan
from guwen_v14_engine import compile_guwen_v14_ds as compile_guwen_v9_ds


def find_latest_male_ckpt() -> Optional[int]:
    d = SING_DIR / "checkpoints" / "hanfire_sing_male"
    if not d.is_dir():
        return None
    ckpts = sorted(
        d.glob("model_ckpt_steps_*.ckpt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    return int(ckpts[-1].stem.split("_")[-1]) if ckpts else None


def render_ds(ds_items: list, out_wav: Path, exp: str, ckpt: int, spk: str, steps: int = 24, female: bool = False) -> dict:
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        ds_f = td_p / "score.ds"
        out_d = td_p / "wavs"
        out_d.mkdir()
        ds_f.write_text(json.dumps(ds_items, ensure_ascii=False, indent=2), encoding="utf-8")
        subprocess.run(
            [
                sys.executable, "scripts/infer.py", "acoustic", str(ds_f),
                "--exp", exp, "--ckpt", str(ckpt), "--spk", spk,
                "--gender=0.0", "--seed", "42", "--steps", str(steps), "--out", str(out_d),
            ],
            cwd=str(DS_DIR), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        y, sr = sf.read(str(sorted(out_d.glob("*.wav"))[0]), dtype="float32")
        y = apply_gain_plan(y, sr, ds_items, female=female)
        rms = float(np.sqrt(np.mean(y ** 2)) + 1e-8)
        g = min(0.095 / rms, 0.72 / (float(np.max(np.abs(y))) + 1e-8))
        y = (y * g).astype(np.float32)
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_wav), y, sr)
        frm = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
        sil_db = float(20.0 * np.log10(np.percentile(frm, 5) + 1e-8))
        return {"dur_s": round(len(y) / sr, 2), "peak": round(float(np.max(np.abs(y))), 3), "sil_db": round(sil_db, 1)}


def export_ref_clip(body_rel: str, out_name: str, dur_s: float) -> dict:
    y, sr = sf.read(str(TANG_DIR / body_rel), dtype="float32")
    y = y[: int(dur_s * sr)]
    dst = AUDIO_DIR / f"{out_name}.wav"
    sf.write(str(dst), y, sr)
    return {"dur_s": round(len(y) / sr, 2), "peak": round(float(np.max(np.abs(y))), 3), "sil_db": -80.0}


def _stat_existing_wav(wav_p: Path) -> Optional[dict]:
    if not wav_p.exists():
        return None
    y, sr = sf.read(str(wav_p), dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    frm = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    sil_db = float(20.0 * np.log10(np.percentile(frm, 5) + 1e-8))
    return {"dur_s": round(len(y) / sr, 2), "peak": round(float(np.max(np.abs(y))), 3), "sil_db": round(sil_db, 1)}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default="", help="仅重新渲染指定调式（如 guwen_lanting），其余调式复用现有 wav")
    args = parser.parse_args()
    only_mode = args.only.strip()

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    ver = int(time.time())
    male_ckpt = find_latest_male_ckpt()
    groups = []

    mode_specs = [
        (
            "guwen_lanting",
            "🎋 调式 G1 · 书斋清雅骈散调（V16 逐字对照 t09 真人活腔 × 诗经 V9.6 纯净架构）",
            "永和九年，岁在癸丑。暮春之初，会于会稽山阴之兰亭，修禊事也。",
            "m4_tenor6:0.60|m4_tenor5:0.40",
            "朗润书生实声 t6×t5",
            [
                ("gw_g1_lanting_zh", "《兰亭集序》（开篇）", "zh", "永和九年，岁在癸丑。暮春之初，会于会稽山阴之兰亭，修禊事也。", "25字逐字对齐 t09 真实拱腔/拔高(+8.6st之/+11.4st禊事)/五度流泻(初)/四波摇腔(也)，零词内SP、零人工EQ！"),
                ("gw_g1_lanting_mc", "《兰亭集序》中古音 MC", "mc", "永和九年，岁在癸丑。暮春之初，会于会稽山阴之兰亭，修禊事也。", "严格按《广韵》音韵与闭口-m编译，1:1 承载 t09 逐字真实活腔！"),
                ("gw_g1_ailian_zh",  "【泛化盲测】周敦颐《爱莲说》", "zh", "水陆草木之花，可爱者甚蕃。晋陶渊明独爱菊，予独爱莲之出淤泥而不染。", "长短句混排（6/5/7/11字）直接复用 t09 真人逐字活腔切片库 + 自动纠偏「甚蕃(fán)」！"),
            ],
        ),
        (
            "guwen_loushi",
            "🪨 调式 G2 · 唐调铭文沉雄顿挫调（V16 逐字对照 t16 全篇 81 字 × 四声依字行腔 × 6+2 顿挫悬宫落底）",
            "山不在高，有仙则名。水不在深，有龙则灵。斯是陋室，惟吾德馨。",
            "m4_bass1:0.58|m4_tenor5:0.42",
            "醇厚学者低中音 b1×t5",
            [
                ("gw_g2_loushi_zh",  "《陋室铭》（开篇）", "zh", "山不在高，有仙则名。水不在深，有龙则灵。斯是陋室，惟吾德馨。", "纠正「不在高/不在深」错位，1:1 还原入声高挑(+4.0st)、上声深钩与「斯是陋室惟吾 —— 德馨」6+2 顿挫悬宫落底！"),
                ("gw_g2_loushi_mc",  "《陋室铭》中古音 MC", "mc", "山不在高，有仙则名。水不在深，有龙则灵。斯是陋室，惟吾德馨。", "严格按《广韵》四声清浊与闭口-m(深 xim)编译，零 OOD 声韵、苍浑古朴！"),
                ("gw_g2_jiezishu_zh","【泛化盲测】诸葛亮《诫子书》", "zh", "夫君子之行，静以修身，俭以养德。非淡泊无以明志，非宁静无以致远。", "严守「平低仄高 · 入声劲落 · 上句仄悬下句落底」四声对仗律，3+4 长联一气呵成！"),
            ],
        ),
        (
            "fu_chibi",
            "🌊 调式 G3 · 唐调辞赋大开大合调（V16 逐字对照 t18《前赤壁赋》× 四声对仗 × 江风八字连气归宫）",
            "壬戌之秋，七月既望，苏子与客泛舟游于赤壁之下。清风徐来，水波不兴。",
            "m4_tenor7:0.70|m4_tenor5:0.30",
            "清朗醇厚实声 t7×t5",
            [
                ("gw_g3_chibi_zh",   "《前赤壁赋》（开篇）", "zh", "壬戌之秋，七月既望，苏子与客泛舟游于赤壁之下。清风徐来，水波不兴。", "28字逐字还原低徵起腔(-7.1st) → 短入拔起双去劲落(七月既望) → 12字泛舟扬波(舟游+3.0st) → 清风徐来水波不兴八字一气连诵归宫！"),
                ("gw_g3_chibi_mc",   "《前赤壁赋》中古音 MC", "mc", "壬戌之秋，七月既望，苏子与客泛舟游于赤壁之下。清风徐来，水波不兴。", "严格按《广韵》闭口-m(壬 nim / 泛 puim)与入声(戌/七/月/客/赤/壁/不)编译，零 OOD 声韵承载 t18 大赋活腔！"),
                ("gw_g3_yueyang_zh", "【泛化盲测】范仲淹《岳阳楼记》", "zh", "衔远山，吞长江，浩浩汤汤，横无际涯。朝晖夕阴，气象万千。", "习得 t18/t19/t10 三版辞赋排比律：3+3/4+4 偶句连气滚诵 × 上声低钩(浩) × 倒二去声冲顶劲落(际/万+4.5st) × 浩然立宫收束！"),
            ],
        ),
    ]

    for mkey, gtitle, ref_text, male_spk, male_lbl, items in mode_specs:
        cfg = GUWEN_V9_MODES[mkey]
        cards = []
        ref_sid = f"gw_ref_{mkey}"
        st_ref = export_ref_clip(cfg["body_wav"], ref_sid, cfg["ref_clip_s"])
        cards.append((ref_sid, f"【原版母带参考】{cfg['track_id']} 原声片段", "真人示范母带", ref_text, cfg["desc"], st_ref, True))

        skip_render = bool(only_mode and mkey != only_mode)
        for sid_base, title, era, text, note in items:
            era_lbl = {"zh": "现代音", "mc": "中古音MC", "em": "近古音EM"}.get(era, era)
            badge_ver = "V16 逐字活腔"
            # 1) 纯女声版
            sid_f = f"{sid_base}_f"
            ds_f = compile_guwen_v9_ds(text=text, mode_key=mkey, era=era, speaker="opencpop_base", voice_gender="female")
            out_f = AUDIO_DIR / f"{sid_f}.wav"
            st_f = _stat_existing_wav(out_f) if skip_render else None
            if st_f is None:
                st_f = render_ds(ds_f, out_f, exp="hanfire_sing_base", ckpt=40000, spk="opencpop_base", steps=24, female=True)
                (SCRATCH / f"{sid_f}.wav").write_bytes(out_f.read_bytes())
                print(f"✓ {sid_f:24s} | {st_f['dur_s']:5.2f}s | Peak={st_f['peak']:.3f} | sil={st_f['sil_db']:.1f}dB")
            else:
                print(f"↷ {sid_f:24s} | 沿用现有音频 ({st_f['dur_s']:5.2f}s)")
            diag = " ｜ ".join(f"{it['text']}[{it['archetype']} · {it['pz']}]" for it in ds_f)
            full_note = f"{note}<br><span style='color:#9c4a3c'>▸ 起承转合规划：{diag}</span>"
            cards.append((sid_f, f"【女声 · {era_lbl}】{title}", f"{badge_ver}女声 · {era_lbl}", text, full_note, st_f, False))

            # 2) 若纯男声底座已就绪，自动同步渲染纯男声版！
            if male_ckpt is not None:
                sid_m = f"{sid_base}_m"
                ds_m = compile_guwen_v9_ds(text=text, mode_key=mkey, era=era, speaker=male_spk, voice_gender="male")
                out_m = AUDIO_DIR / f"{sid_m}.wav"
                st_m = _stat_existing_wav(out_m) if skip_render else None
                if st_m is None:
                    st_m = render_ds(ds_m, out_m, exp="hanfire_sing_male", ckpt=male_ckpt, spk=male_spk, steps=24, female=False)
                    (SCRATCH / f"{sid_m}.wav").write_bytes(out_m.read_bytes())
                    print(f"✓ {sid_m:24s} | {st_m['dur_s']:5.2f}s | Peak={st_m['peak']:.3f} | sil={st_m['sil_db']:.1f}dB | spk={male_spk}")
                else:
                    print(f"↷ {sid_m:24s} | 沿用现有音频 ({st_m['dur_s']:5.2f}s)")
                cards.append((sid_m, f"【男声 {male_lbl} · {era_lbl}】{title}", f"{badge_ver}男声 · {era_lbl}", text, full_note, st_m, False))

        groups.append((gtitle, cards))

    html_sections = []
    for gtitle, cards in groups:
        buf = []
        for sid, title, badge, text, note, st, is_ref in cards:
            safe_dl = title.replace(" ", "_").replace("·", "_").replace("【", "").replace("】", "") + ".wav"
            src_url = f"./_sing_showcase/{sid}.wav?v={ver}"
            cls = "card ref-card" if is_ref else "card"
            buf.append(
                f"<div class='{cls}'><div class='head'><b>{title}</b><span class='tag'>{badge}</span></div>"
                f"<div class='poem'>{text}</div>"
                f"<div class='meter'>{note}</div>"
                f"<div class='player-row'><audio controls preload='none' src='{src_url}'></audio>"
                f"<button class='dl-btn' onclick=\"downloadWav('{src_url}','{safe_dl}',this)\">⬇️ 下载</button></div>"
                f"<div class='meta'>时长 {st['dur_s']}s · Peak {st['peak']}</div></div>"
            )
        html_sections.append(f"<h2>{gtitle}</h2><div class='grid'>{''.join(buf)}</div>")

    voice_banner = "纯女声 + 纯棚录男声双声部" if male_ckpt else "纯女声先导版（纯男声底座正在云端 A100 训练中，就绪后自动合入双声部）"
    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>汉火古音吟诵 · V16 古文/辞赋三大调式逐字活腔精雕台</title>
<style>
body{{margin:0;background:#f7f4ed;color:#2b2623;font-family:system-ui,-apple-system,sans-serif;padding:24px;max-width:1240px;margin:auto}}
h1{{color:#c34435;font-family:"Noto Serif CJK SC",serif;margin-bottom:6px}}
h2{{color:#8f2e22;border-bottom:1px solid #ddd3be;padding-bottom:6px;margin-top:26px;font-size:1.05rem}}
.nav{{display:flex;gap:10px;margin-bottom:14px}}
.nav a{{text-decoration:none;background:#f1ece0;color:#8f2e22;padding:6px 14px;border-radius:8px;font-size:0.86rem;border:1px solid #ddd3be}}
.nav a.active{{background:#c34435;color:#fff;border-color:#c34435}}
.sub{{color:#6e6459;font-size:0.9rem;line-height:1.6}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:14px}}
.card{{background:#fdfbf7;border:1px solid #ddd3be;border-radius:10px;padding:14px;box-shadow:0 2px 6px rgba(43,38,35,0.04)}}
.ref-card{{background:#f6f0e4;border:1.5px solid #c9a77c}}
.head{{display:flex;justify-content:space-between;align-items:flex-start;gap:8px;margin-bottom:6px}}
.tag{{background:#f3e5e1;color:#c34435;font-size:0.74rem;padding:2px 8px;border-radius:12px;white-space:nowrap}}
.poem{{color:#2b2623;font-size:0.92rem;margin-bottom:6px;line-height:1.55}}
.meter{{color:#7b5e43;background:#f5efe4;font-size:0.76rem;padding:5px 8px;border-radius:5px;margin-bottom:10px;line-height:1.45}}
.player-row{{display:flex;align-items:center;gap:8px}}
audio{{flex:1;height:36px;min-width:0}}
.dl-btn{{background:#c34435;color:#fff;border:none;border-radius:6px;padding:7px 11px;font-size:0.8rem;cursor:pointer;white-space:nowrap;flex-shrink:0}}
.dl-btn:hover{{background:#a63528}}
.meta{{margin-top:6px;color:#82776c;font-size:0.76rem}}
</style></head><body>
<div class="nav"><a href="./sing_compare.html">📖 《诗经》三大调式精雕台</a><a class="active" href="./guwen_compare.html">📜 古文/辞赋三大调式精雕台</a></div>
<h1>汉火古音吟诵 · V16 古文/辞赋三大调式逐字活腔精雕台（{voice_banner}）</h1>
<p class="sub"><b>V16 三大调式全满贯 · 完整移植《诗经》V9.6 真人母带逐字活腔架构 + 中古《广韵》四声依字行腔（零词内 SP、零人工 EQ 染色）</b>：① <b>G1《兰亭集序》</b>：25 字逐字还原 <code>九/年</code> 与 <code>兰/亭</code> 双拱、<code>之 +8.6st</code> 冲霄峰、<code>初</code> 五度流泻、<code>禊/事 +11.4st</code> 劲落与 <code>也</code> 四波大摇腔；② <b>G2《陋室铭》</b>：剖析 t16 全篇 81 字习得「平低仄高 · 上声深钩 · 入声高挑/劲落」，1:1 还原 <code>斯是陋室惟吾 —— 德馨</code> 6+2 顿挫气口与悬宫落底；③ <b>G3《前赤壁赋》</b>：逐字对齐 t18 28 字苍茫低徵起腔(<code>-7.1st</code>)、<code>七月既望</code> 短入拔起双去劲落、12 字江上泛舟扬波(<code>舟游 +3.0st</code>)与 <code>清风徐来水波不兴</code> 八字连气归宫，并跨篇泛化至《岳阳楼记》三言/四言排比大开大合！</p>
{"".join(html_sections)}
<script>
async function downloadWav(url, filename, btn) {{
  const orig = btn.textContent; btn.textContent = '下载中...'; btn.disabled = true;
  try {{
    const sep = url.includes('?') ? '&' : '?';
    const resp = await fetch(url + sep + 'download=1');
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const blob = await resp.blob();
    const objUrl = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = objUrl; a.download = filename;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(objUrl), 5000);
  }} catch (e) {{ alert('下载失败：' + e.message); }}
  finally {{ btn.textContent = orig; btn.disabled = false; }}
}}
</script>
</body></html>"""
    out_html = WEB_DIST / "guwen_compare.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"\n✅ V15 挺拔中气古文/辞赋三大调式精雕台已刷新 -> {out_html}")


if __name__ == "__main__":
    main()
