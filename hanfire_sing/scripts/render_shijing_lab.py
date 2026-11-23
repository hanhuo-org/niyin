#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汉火古音唱吟 · V9.6 《诗经》三大调式精雕实验室（纯女声 + 纯棚录男声双声部对照）
  - 纯女声底座: hanfire_sing_base (opencpop_base, 40,000步)
  - 纯男声底座: hanfire_sing_male (M4Singer 6位棚录男声 10.35h, 40,000步)
    * 调式 A（雅正苍凉咏叹调，母带 t15《蒹葭》/ t13《采薇》）：男声首选 m4_bass1 (122Hz 醇厚学者男低中音)
    * 调式 B（周南清丽起兴调，母带 t01《关雎》）：男声首选 m4_tenor7 (142Hz 清朗儒雅抒情男声)
    * 调式 C（秦风豪放铿锵调，母带 t20/t14《无衣》）：男声首选 m4_tenor6 (152Hz 昂扬雄浑男高音) & m4_bass1 (128Hz 沉雄统帅)
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
from shijing_v9_engine import SHIJING_V9_MODES, compile_shijing_v9_ds


def find_latest_male_ckpt() -> Optional[int]:
    d = SING_DIR / "checkpoints" / "hanfire_sing_male"
    if not d.is_dir():
        return None
    ckpts = sorted(d.glob("model_ckpt_steps_*.ckpt"), key=lambda p: int(p.stem.split("_")[-1]))
    return int(ckpts[-1].stem.split("_")[-1]) if ckpts else None


def render_ds(ds_items: list, out_wav: Path, exp: str, ckpt: int, spk: str, steps: int = 24) -> dict:
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
        rms = float(np.sqrt(np.mean(y ** 2)) + 1e-8)
        g = min(0.088 / rms, 0.72 / (float(np.max(np.abs(y))) + 1e-8))
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


def main() -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    ver = int(time.time())
    male_ckpt = find_latest_male_ckpt()
    groups = []

    mode_specs = [
        (
            "shijing_guanju",
            "🌸 调式 B · 诗经周南清丽起兴调（V9.6 真人活腔 × 零词内SP物理成阻 · 男女双声部）",
            "关关雎鸠，在河之洲。窈窕淑女，君子好逑。",
            [
                # (sid, title, era, gender, spk, key_override, text, note)
                ("v9_b1_guanju_zh_f", "【女声 · 现代音】《诗经·关雎》", "zh", "female", "opencpop_base", None,
                 "关关雎鸠，在河之洲。窈窕淑女，君子好逑。",
                 "已验收定版：238Hz 实音区 + 零词内 SP + 口腔成阻基频微谷，6/6 多种子稳过「窈窕淑女」！"),
                ("v9_b1_guanju_zh_m", "【男声 m4_tenor7 · 现代音】《诗经·关雎》", "zh", "male", "m4_tenor7", 142.0,
                 "关关雎鸠，在河之洲。窈窕淑女，君子好逑。",
                 "全新 10.35h 纯棚录男声底座（Tenor-7 清朗儒雅男中音）：零调参一次迁移女声 V9.6 全部规律！"),
                ("v9_b2_guanju_mc_f", "【女声 · 中古音MC】《诗经·关雎》", "mc", "female", "opencpop_base", None,
                 "关关雎鸠，在河之洲。窈窕淑女，君子好逑。",
                 "严格按《广韵》平上去入与清浊编译（「淑」常三東入利落收紧）。"),
                ("v9_b2_guanju_mc_m", "【男声 m4_tenor7 · 中古音MC】《诗经·关雎》", "mc", "male", "m4_tenor7", 142.0,
                 "关关雎鸠，在河之洲。窈窕淑女，君子好逑。",
                 "纯棚录男声中古《广韵》吟诵，儒雅古朴！"),
                ("v9_b2_guanju_em_f", "【女声 · 近古音EM】《诗经·关雎》", "em", "female", "opencpop_base", None,
                 "关关雎鸠，在河之洲。窈窕淑女，君子好逑。",
                 "按《中原音韵》入派三声（「淑」全浊入作阳平、「在」浊上变去）。"),
                ("v9_b3_taoyao_zh_f", "【女声 · 泛化验证】《诗经·桃夭》", "zh", "female", "opencpop_base", None,
                 "桃之夭夭，灼灼其华。之子于归，宜其室家。",
                 "自动识别「灼灼/室」入声与「之子于归」平顺收托。"),
                ("v9_b3_taoyao_zh_m", "【男声 m4_tenor7 · 泛化验证】《诗经·桃夭》", "zh", "male", "m4_tenor7", 142.0,
                 "桃之夭夭，灼灼其华。之子于归，宜其室家。",
                 "男声泛化盲测《桃夭》：检验「之子于归」清晰塞音成阻与平顺衔接！"),
                ("v9_b4_jingnv_zh_f", "【女声 · 全新盲测】《诗经·邶风·静女》", "zh", "female", "opencpop_base", None,
                 "静女其姝，俟我于城隅。爱而不见，搔首踟蹰。",
                 "含五言句「俟我于城隅」与去声转句「爱而不见」。"),
            ],
        ),
        (
            "shijing_jianjia",
            "🌿 调式 A · 诗经雅正苍凉咏叹调（V9.6 先扬后抑 × 低五度咏叹 · 男女双声部）",
            "蒹葭苍苍，白露为霜。所谓伊人，在水一方。",
            [
                ("v9_a1_jianjia_zh_f", "【女声 · 现代音】《诗经·蒹葭》", "zh", "female", "opencpop_base", None,
                 "蒹葭苍苍，白露为霜。所谓伊人，在水一方。",
                 "已验收定版：起句中宫舒展 → 承句高扬沉徵 → 转合低五度苍凉咏叹。"),
                ("v9_a1_jianjia_zh_m", "【男声 m4_bass1 · 现代音】《诗经·蒹葭》", "zh", "male", "m4_bass1", 122.0,
                 "蒹葭苍苍，白露为霜。所谓伊人，在水一方。",
                 "全新纯棚录男低中音 Bass-1 (122Hz，与原版男声母带 t15 同高！)：醇厚沧桑、书斋学者质感！"),
                ("v9_a2_jianjia_mc_f", "【女声 · 中古音MC】《诗经·蒹葭》", "mc", "female", "opencpop_base", None,
                 "蒹葭苍苍，白露为霜。所谓伊人，在水一方。",
                 "中古音版（白/一入声干脆 + 节拍守恒代偿）。"),
                ("v9_a2_jianjia_mc_m", "【男声 m4_bass1 · 中古音MC】《诗经·蒹葭》", "mc", "male", "m4_bass1", 122.0,
                 "蒹葭苍苍，白露为霜。所谓伊人，在水一方。",
                 "纯棚录醇厚男低中音 × 中古《广韵》吟诵《蒹葭》！"),
                ("v9_a3_caiwei_zh_f",  "【女声 · 泛化验证】《诗经·采薇》", "zh", "female", "opencpop_base", None,
                 "采薇采薇，薇亦作止。曰归曰归，岁亦莫止。",
                 "「薇亦作止」重拍时长精准对齐「白露为霜」，仄收自动护调！"),
                ("v9_a3_caiwei_zh_m",  "【男声 m4_bass1 · 泛化验证】《诗经·采薇》", "zh", "male", "m4_bass1", 122.0,
                 "采薇采薇，薇亦作止。曰归曰归，岁亦莫止。",
                 "男低中音 Bass-1 泛化《采薇》：还原 t13 原版男声苍凉悲壮气韵！"),
                ("v9_a4_luming_zh_f",  "【女声 · 全新盲测】《诗经·小雅·鹿鸣》", "zh", "female", "opencpop_base", None,
                 "呦呦鹿鸣，食野之苹。我有嘉宾，鼓瑟吹笙。",
                 "全新未见过雅诗盲测。"),
            ],
        ),
        (
            "shijing_wuyi",
            "⚔️ 调式 C · 诗经秦风豪放铿锵调（V9.6 10.4s 从容大度 × 起承转扬合沉雄 · 男女双声部）",
            "岂曰无衣，与子同袍。王于兴师，修我戈矛。",
            [
                ("v9_c1_wuyi_zh_f",       "【女声 · 现代音】《诗经·无衣》", "zh", "female", "opencpop_base", None,
                 "岂曰无衣，与子同袍。王于兴师，修我戈矛。",
                 "已验收定版：10.4s 原版从容步调 + 转句昂扬、合句沉雄。"),
                ("v9_c1_wuyi_zh_m",       "【男声 m4_tenor6 · 现代音】《诗经·无衣》", "zh", "male", "m4_tenor6", 152.0,
                 "岂曰无衣，与子同袍。王于兴师，修我戈矛。",
                 "昂扬男高音 Tenor-6 (152Hz，与男声母带 t14 同高！)：慷慨激越、秦军出征气势拉满！"),
                ("v9_c1_wuyi_bass1_zh_m", "【男声 m4_bass1 · 沉雄统帅版】《诗经·无衣》", "zh", "male", "m4_bass1", 128.0,
                 "岂曰无衣，与子同袍。王于兴师，修我戈矛。",
                 "深沉男低中音 Bass-1 (128Hz)：浑厚威严、大将点兵之声！"),
                ("v9_c2_wuyi_mc_f",       "【女声 · 中古音MC】《诗经·无衣》", "mc", "female", "opencpop_base", None,
                 "岂曰无衣，与子同袍。王于兴师，修我戈矛。",
                 "中古音版（曰入声铿锵顿挫）。"),
                ("v9_c2_wuyi_mc_m",       "【男声 m4_tenor6 · 中古音MC】《诗经·无衣》", "mc", "male", "m4_tenor6", 152.0,
                 "岂曰无衣，与子同袍。王于兴师，修我戈矛。",
                 "纯棚录男声 × 中古《广韵》豪放吟诵《无衣》！"),
                ("v9_c3_duangexing_zh_f", "【女声 · 泛化验证】曹操《短歌行》", "zh", "female", "opencpop_base", None,
                 "对酒当歌，人生几何。譬如朝露，去日苦多。",
                 "全新汉末四言乐府套用 V9.6 调式 C。"),
                ("v9_c3_duangexing_zh_m", "【男声 m4_bass1 · 泛化验证】曹操《短歌行》", "zh", "male", "m4_bass1", 128.0,
                 "对酒当歌，人生几何。譬如朝露，去日苦多。",
                 "曹孟德横槊赋诗本色！纯棚录男低中音 Bass-1 苍凉慷慨！"),
            ],
        ),
    ]

    for mkey, gtitle, ref_text, items in mode_specs:
        cfg = SHIJING_V9_MODES[mkey]
        cards = []
        ref_sid = f"v8_ref_{mkey}"
        st_ref = export_ref_clip(cfg["body_wav"], ref_sid, cfg["ref_clip_s"])
        cards.append((ref_sid, f"【原版母带参考】{cfg['track_id']} 原声片段", "真人示范母带", ref_text, cfg["desc"], st_ref, True, False))

        for sid, title, era, gender, spk, khz, text, note in items:
            if gender == "male" and male_ckpt is None:
                continue
            ds = compile_shijing_v9_ds(
                text=text, mode_key=mkey, era=era, key_hz=khz, speaker=spk, voice_gender=gender
            )
            out_w = AUDIO_DIR / f"{sid}.wav"
            exp = "hanfire_sing_male" if gender == "male" else "hanfire_sing_base"
            ckpt = male_ckpt if gender == "male" else 40000
            st = render_ds(ds, out_w, exp=exp, ckpt=ckpt, spk=spk, steps=24)
            (SCRATCH / f"{sid}.wav").write_bytes(out_w.read_bytes())
            diag = " ｜ ".join(f"{it['text']}[{it['pz']}·{it['archetype']}]" for it in ds)
            full_note = f"{note}<br><span style='color:#9c4a3c'>▸ 实时平仄与移阶：{diag}</span>"
            print(f"✓ {sid:26s} | {st['dur_s']:5.2f}s | Peak={st['peak']:.3f} | sil={st['sil_db']:.1f}dB | {spk}")
            era_lbl = {"zh": "现代音", "mc": "中古音MC", "em": "近古音EM"}.get(era, era)
            badge = f"V9.6 {'纯男声' if gender=='male' else '纯女声'} · {era_lbl}"
            cards.append((sid, title, badge, text, full_note, st, False, gender == "male"))
        groups.append((gtitle, cards))

    html_sections = []
    for gtitle, cards in groups:
        buf = []
        for sid, title, badge, text, note, st, is_ref, is_male in cards:
            safe_dl = title.replace(" ", "_").replace("·", "_").replace("【", "").replace("】", "") + ".wav"
            src_url = f"./_sing_showcase/{sid}.wav?v={ver}"
            cls = "card ref-card" if is_ref else ("card male-card" if is_male else "card")
            tag_cls = "tag male-tag" if is_male else "tag"
            buf.append(
                f"<div class='{cls}'><div class='head'><b>{title}</b><span class='{tag_cls}'>{badge}</span></div>"
                f"<div class='poem'>{text}</div>"
                f"<div class='meter'>{note}</div>"
                f"<div class='player-row'><audio controls preload='none' src='{src_url}'></audio>"
                f"<button class='dl-btn' onclick=\"downloadWav('{src_url}','{safe_dl}',this)\">⬇️ 下载</button></div>"
                f"<div class='meta'>时长 {st['dur_s']}s · Peak {st['peak']}</div></div>"
            )
        html_sections.append(f"<h2>{gtitle}</h2><div class='grid'>{''.join(buf)}</div>")

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>汉火古音吟诵 · V9.6 《诗经》三大调式精雕台（纯女声 + 纯棚录男声双声部）</title>
<style>
body{{margin:0;background:#f7f4ed;color:#2b2623;font-family:system-ui,-apple-system,sans-serif;padding:24px;max-width:1240px;margin:auto}}
h1{{color:#c34435;font-family:"Noto Serif CJK SC",serif;margin-bottom:6px}}
h2{{color:#8f2e22;border-bottom:1px solid #ddd3be;padding-bottom:6px;margin-top:26px;font-size:1.05rem}}
.nav{{display:flex;gap:10px;margin-bottom:14px}}
.nav a{{text-decoration:none;background:#f1ece0;color:#8f2e22;padding:6px 14px;border-radius:8px;font-size:0.86rem;border:1px solid #ddd3be}}
.nav a.active{{background:#c34435;color:#fff;border-color:#c34435}}
.sub{{color:#6e6459;font-size:0.9rem;line-height:1.6}}
.box{{background:#f1ece0;border-left:4px solid #c34435;padding:12px 14px;border-radius:6px;margin:14px 0;font-size:0.87rem;line-height:1.65}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:14px}}
.card{{background:#fdfbf7;border:1px solid #ddd3be;border-radius:10px;padding:14px;box-shadow:0 2px 6px rgba(43,38,35,0.04)}}
.ref-card{{background:#f6f0e4;border:1.5px solid #c9a77c}}
.male-card{{background:#f4f7fa;border:1.5px solid #9db4c8}}
.head{{display:flex;justify-content:space-between;align-items:flex-start;gap:8px;margin-bottom:6px}}
.tag{{background:#f3e5e1;color:#c34435;font-size:0.74rem;padding:2px 8px;border-radius:12px;white-space:nowrap}}
.male-tag{{background:#e1ecf5;color:#235784}}
.poem{{color:#2b2623;font-size:0.92rem;margin-bottom:6px;line-height:1.55}}
.meter{{color:#7b5e43;background:#f5efe4;font-size:0.76rem;padding:5px 8px;border-radius:5px;margin-bottom:10px;line-height:1.45}}
.player-row{{display:flex;align-items:center;gap:8px}}
audio{{flex:1;height:36px;min-width:0}}
.dl-btn{{background:#c34435;color:#fff;border:none;border-radius:6px;padding:7px 11px;font-size:0.8rem;cursor:pointer;white-space:nowrap;flex-shrink:0}}
.dl-btn:hover{{background:#a63528}}
.meta{{margin-top:6px;color:#82776c;font-size:0.76rem}}
</style></head><body>
<div class="nav"><a class="active" href="./sing_compare.html">📖 《诗经》三大调式精雕台（男女双声部）</a><a href="./guwen_compare.html">📜 古文/辞赋三大调式精雕台（男女双声部）</a></div>
<h1>汉火古音吟诵 · V9.6 《诗经》三大调式精雕台（纯女声 + 全新 10.35h 纯棚录男声底座）</h1>
<p class="sub">全新训成的 <b>hanfire_sing_male (40,000步 · M4Singer 6位棚录44.1kHz干声男歌手 10.35h)</b> 已上线！蓝色卡片为纯正原生男声（零伪男声压喉感），直接零改动继承女声已过关的 V9.6 全部声学规律！</p>
<div class="box">
<b>纯棚录男声底座声部选型：</b><br>
1. <b>m4_bass1（醇厚学者男低中音，122~128Hz）</b>：专精调式 A《蒹葭》《采薇》与深沉统帅版《无衣》/曹操《短歌行》，胸声浑厚沧桑；<br>
2. <b>m4_tenor7（清朗儒雅抒情男声，142Hz）</b>：男版 OpenCpop，专精调式 B《关雎》《桃夭》清丽吟诵，零换声区发虚；<br>
3. <b>m4_tenor6（昂扬雄浑男高音，152Hz）</b>：专精调式 C 秦风豪放《无衣》，气势沛然！
</div>
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
    out_html = WEB_DIST / "sing_compare.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"\n✅ V9.6 《诗经》三大调式双声部精雕台已刷新 -> {out_html}")


if __name__ == "__main__":
    main()
