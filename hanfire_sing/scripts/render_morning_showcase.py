#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汉火古音唱吟 · V7 全能吟诵验收台生成器
  - 顶部置顶：【非近体古诗文 · 四大真人实录调式专场】（原版母带直听 + 4 大调式逐一合成对比！）
      1. 调式一 · 诗经雅正抒情调（母带 t15《蒹葭》）→ 合成《蒹葭》 + 新诗《关雎》
      2. 调式二 · 诗经秦风豪放调（母带 t20《无衣》）→ 合成《无衣》 + 新诗《观沧海》
      3. 调式三 · 书斋清雅文言调（母带 t09《兰亭集序》）→ 合成《兰亭集序》 + 新文《爱莲说》
      4. 调式四 · 唐调沉雄古文赋调（母带 t16《陋室铭》/ t18《前赤壁赋》）→ 合成《陋室铭》 + 《前赤壁赋》 + 新文《岳阳楼记》
  - 下半区保留：V6 近体格律诗（五绝/七绝/五律 仄起/平起 四式自动辨体吟诵）全部成果！
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

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
from tang_melody_prior import NONREG_MODES, analyze_poem_meter, compile_ancient_tang_ds

VOICE_PRESETS = {
    "female": (261.6, 0.0),    # C4 清雅沉稳女声吟诵
    "alto":   (220.0, -0.40),  # A3 温润书生中低音
}


def render_ds_items(ds_items: list, out_wav: Path, gender: float, steps: int = 20) -> dict:
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        ds_f = td_p / "score.ds"
        out_d = td_p / "wavs"
        out_d.mkdir()
        ds_f.write_text(json.dumps(ds_items, ensure_ascii=False, indent=2), encoding="utf-8")
        subprocess.run(
            [
                sys.executable, "scripts/infer.py", "acoustic", str(ds_f),
                "--exp", "hanfire_sing_base", "--ckpt", "40000", "--spk", "opencpop_base",
                f"--gender={gender}", "--steps", str(steps), "--out", str(out_d),
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


def export_master_clip(body_rel: str, out_name: str, max_sec: float = 15.0) -> dict:
    src = TANG_DIR / body_rel
    dst = AUDIO_DIR / f"{out_name}.wav"
    y, sr = sf.read(str(src), dtype="float32")
    y = y[: int(max_sec * sr)]
    sf.write(str(dst), y, sr)
    return {"dur_s": round(len(y) / sr, 2), "peak": round(float(np.max(np.abs(y))), 3), "sil_db": -80.0}


def main() -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    ver = int(time.time())
    cards = []

    # =========================================================================
    # A. 非近体古诗文 · 四大真人实录调式专场（每调式：原版母带直听 + 合成实测！）
    # =========================================================================
    m1_grp = "🎵 调式一 · 诗经雅正抒情调（母带 t15《蒹葭》：大弧度咏叹、偶句深沉落韵）"
    st_r1 = export_master_clip("bodies/t15_shijing_4_诗经·秦风·蒹葭.wav", "v7_m1_ref_t15_jianjia", 13.5)
    cards.append((m1_grp, "v7_m1_ref_t15_jianjia", "【原版母带】t15《诗经·蒹葭》片段", "真人示范母带",
                  "蒹葭苍苍，白露为霜。所谓伊人，在水一方。", "大五度深沉落韵 + 4+4 联句大弧度咏叹", st_r1))

    m2_grp = "🥁 调式二 · 诗经秦风豪放调（母带 t20《无衣》：节奏明快、字字铿锵、高亢顿挫）"
    st_r2 = export_master_clip("bodies/t20_shijing_4_诗经·秦风·无衣.wav", "v7_m2_ref_t20_wuyi", 11.0)
    cards.append((m2_grp, "v7_m2_ref_t20_wuyi", "【原版母带】t20《诗经·无衣》片段", "真人示范母带",
                  "岂曰无衣，与子同袍。王于兴师，修我戈矛。", "明快顿挫（~2.4s/句）+ 高亢扬收", st_r2))

    m3_grp = "🍵 调式三 · 书斋清雅文言调（母带 t09《兰亭集序》：江南温润平叙、句尾轻扬留白）"
    st_r3 = export_master_clip("bodies/t09_guwen_兰亭集序.wav", "v7_m3_ref_t09_lanting", 13.0)
    cards.append((m3_grp, "v7_m3_ref_t09_lanting", "【原版母带】t09《兰亭集序》片段", "真人示范母带",
                  "永和九年，岁在癸丑，暮春之初，会于会稽山阴之兰亭。", "江南读书调：平叙温润 + 句尾轻扬悬收", st_r3))

    m4_grp = "🏛️ 调式四 · 唐调沉雄古文赋调（母带 t16《陋室铭》/ t18《前赤壁赋》：句前快诵、句尾大沉落）"
    st_r4 = export_master_clip("bodies/t18_fu_前赤壁赋.wav", "v7_m4_ref_t18_chibifu", 15.0)
    cards.append((m4_grp, "v7_m4_ref_t18_chibifu", "【原版母带】t18《前赤壁赋》片段", "真人示范母带",
                  "壬戌之秋，七月既望。苏子与客泛舟游于赤壁之下。清风徐来，水波不兴。", "唐调古文赋：前段紧凑疾诵 + 句尾重锤大沉落", st_r4))

    nonreg_jobs = [
        # 调式一
        ("v7_m1_01_jianjia_mc_female", m1_grp, "【调式一合成】《蒹葭》中古音 · 女声", "mc", "shijing_lyrical", "female",
         "蒹葭苍苍，白露为霜。所谓伊人，在水一方。", True),
        ("v7_m1_02_guanju_mc_alto", m1_grp, "【调式一·新诗迁移】《关雎》中古音 · 中低音", "mc", "shijing_lyrical", "alto",
         "关关雎鸠，在河之洲。窈窕淑女，君子好逑。", True),
        # 调式二
        ("v7_m2_01_wuyi_mc_female", m2_grp, "【调式二合成】《无衣》中古音 · 女声", "mc", "shijing_heroic", "female",
         "岂曰无衣，与子同袍。王于兴师，修我戈矛。", True),
        ("v7_m2_02_guancanghai_mc_alto", m2_grp, "【调式二·新诗迁移】曹操《观沧海》中古音 · 中低音", "mc", "shijing_heroic", "alto",
         "东临碣石，以观沧海。水何澹澹，山岛竦峙。", True),
        # 调式三
        ("v7_m3_01_lanting_mc_female", m3_grp, "【调式三合成】《兰亭集序》中古音 · 女声", "mc", "guwen_elegant", "female",
         "永和九年，岁在癸丑。暮春之初，会于会稽山阴之兰亭。", True),
        ("v7_m3_02_ailianshuo_em_female", m3_grp, "【调式三·新文迁移】周敦颐《爱莲说》近古音 · 女声", "em", "guwen_elegant", "female",
         "予独爱莲之出淤泥而不染，濯清涟而不妖。中通外直，不蔓不枝。", True),
        # 调式四
        ("v7_m4_01_loushiming_mc_alto", m4_grp, "【调式四合成】刘禹锡《陋室铭》中古音 · 中低音", "mc", "guwen_tang", "alto",
         "山不在高，有仙则名。水不在深，有龙则灵。斯是陋室，惟吾德馨。", True),
        ("v7_m4_02_chibifu_em_alto", m4_grp, "【调式四合成】苏轼《前赤壁赋》近古音 · 中低音", "em", "guwen_tang", "alto",
         "壬戌之秋，七月既望。苏子与客泛舟游于赤壁之下。清风徐来，水波不兴。", True),
        ("v7_m4_03_yueyanglou_em_alto", m4_grp, "【调式四·新文迁移】范仲淹《岳阳楼记》近古音 · 中低音", "em", "guwen_tang", "alto",
         "庆历四年春，滕子京谪守巴陵郡。越明年，政通人和，百废具兴。", True),
    ]

    for sid, grp, title, era, mode_key, voice, text, save_scratch in nonreg_jobs:
        khz, gen = VOICE_PRESETS[voice]
        ds = compile_ancient_tang_ds(text=text, era=era, genre=mode_key, key_hz=khz)
        out_w = AUDIO_DIR / f"{sid}.wav"
        st = render_ds_items(ds, out_w, gender=gen)
        if save_scratch:
            (SCRATCH / f"{sid}.wav").write_bytes(out_w.read_bytes())
        badge = NONREG_MODES[mode_key]["name"]
        note = NONREG_MODES[mode_key]["desc"]
        print(f"✓ {sid:32s} | {st['dur_s']:5.2f}s | {badge}")
        cards.append((grp, sid, title, badge, text, note, st))

    # =========================================================================
    # B. 保留 V6 近体格律诗（直接复用已渲染好的 v6_00..v6_07 WAV，零重复耗时！）
    # =========================================================================
    reg_grp = "📜 已验收 · 近体格律诗（五绝/七绝/五律 仄起/平起 自动辨体四式）"
    reg_existing = [
        ("v6_00_orig_t17_wenliu", "【原版参考】真人唐调《问刘十九》", "真人原声",
         "绿蚁新醅酒，红泥小火炉。晚来天欲雪，能饮一杯无。"),
        ("v6_01_wenliu_auto_zh_female", "【V6自动辨体】《问刘十九》现代音 · 女声", "五绝·仄起首句不入韵",
         "绿蚁新醅酒，红泥小火炉。晚来天欲雪，能饮一杯无。"),
        ("v6_03_sujiandejiang_mc_female", "孟浩然《宿建德江》中古音 · 女声", "五绝·平起首句不入韵",
         "移舟泊烟渚，日暮客愁新。野旷天低树，江清月近人。"),
        ("v6_04_dengguanquelou_mc_alto", "王之涣《登鹳雀楼》中古音 · 中低音", "五绝·仄起首句不入韵",
         "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。"),
        ("v6_06_boqinhuai_mc_female", "杜牧《泊秦淮》中古音 · 女声", "七绝·平起首句入韵",
         "烟笼寒水月笼沙，夜泊秦淮近酒家。商女不知亡国恨，隔江犹唱后庭花。"),
        ("v6_07_zaofabaidi_mc_alto", "李白《早发白帝城》中古音 · 中低音", "七绝·平起首句入韵",
         "朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。"),
    ]
    for sid, title, badge, text in reg_existing:
        p = AUDIO_DIR / f"{sid}.wav"
        if p.exists():
            y, sr = sf.read(str(p), dtype="float32")
            mt = analyze_poem_meter(text)
            note = f"自动判式：{mt.form_name}｜调式：{' → '.join(mt.line_patterns)}" if mt.is_regulated else ""
            cards.append((reg_grp, sid, title, badge, text, note, {
                "dur_s": round(len(y) / sr, 2), "peak": round(float(np.max(np.abs(y))), 3), "sil_db": -80.0
            }))

    # 按分组渲染卡片（确保同一调式的原版母带与合成样本在同一组并排！）
    grouped: dict[str, list] = {}
    for item in cards:
        grouped.setdefault(item[0], []).append(item)

    html_groups = []
    for grp, items in grouped.items():
        buf = []
        for _, sid, title, badge, text, note, st in items:
            safe_dl = title.replace(" ", "_").replace("·", "_").replace("【", "").replace("】", "") + ".wav"
            src_url = f"./_sing_showcase/{sid}.wav?v={ver}"
            is_ref = "原版" in title
            cls = "card ref-card" if is_ref else "card"
            buf.append(
                f"<div class='{cls}'><div class='head'><b>{title}</b><span class='tag'>{badge}</span></div>"
                f"<div class='poem'>{text}</div>"
                f"<div class='meter'>{note}</div>"
                f"<div class='player-row'><audio controls preload='none' src='{src_url}'></audio>"
                f"<button class='dl-btn' onclick=\"downloadWav('{src_url}','{safe_dl}',this)\">⬇️ 下载</button></div>"
                f"<div class='meta'>时长 {st['dur_s']}s · Peak {st['peak']}</div></div>"
            )
        html_groups.append(f"<h2>{grp}</h2><div class='grid'>{''.join(buf)}</div>")

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>汉火古音吟诵 · V7 非近体四大实录调式 + 近体四式全能验收台</title>
<style>
body{{margin:0;background:#f7f4ed;color:#2b2623;font-family:system-ui,-apple-system,sans-serif;padding:24px;max-width:1220px;margin:auto}}
h1{{color:#c34435;font-family:"Noto Serif CJK SC",serif;margin-bottom:6px}}
h2{{color:#8f2e22;border-bottom:1px solid #ddd3be;padding-bottom:6px;margin-top:26px;font-size:1.06rem}}
.sub{{color:#6e6459;font-size:0.9rem;line-height:1.6}}
.box{{background:#f1ece0;border-left:4px solid #c34435;padding:12px 14px;border-radius:6px;margin:14px 0;font-size:0.87rem;line-height:1.65}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(370px,1fr));gap:14px}}
.card{{background:#fdfbf7;border:1px solid #ddd3be;border-radius:10px;padding:14px;box-shadow:0 2px 6px rgba(43,38,35,0.04)}}
.ref-card{{background:#f6f0e4;border:1.5px solid #c9a77c}}
.head{{display:flex;justify-content:space-between;align-items:flex-start;gap:8px;margin-bottom:6px}}
.tag{{background:#f3e5e1;color:#c34435;font-size:0.74rem;padding:2px 8px;border-radius:12px;white-space:nowrap}}
.poem{{color:#2b2623;font-size:0.9rem;margin-bottom:6px;line-height:1.55}}
.meter{{color:#7b5e43;background:#f5efe4;font-size:0.75rem;padding:5px 8px;border-radius:5px;margin-bottom:10px;line-height:1.45}}
.player-row{{display:flex;align-items:center;gap:8px}}
audio{{flex:1;height:36px;min-width:0}}
.dl-btn{{background:#c34435;color:#fff;border:none;border-radius:6px;padding:7px 11px;font-size:0.8rem;cursor:pointer;white-space:nowrap;flex-shrink:0}}
.dl-btn:hover{{background:#a63528}}
.meta{{margin-top:6px;color:#82776c;font-size:0.76rem}}
</style></head><body>
<h1>汉火古音吟诵 · V7 非近体四大实录调式试听台</h1>
<p class="sub">全面分析 17 首非近体素材后，提炼出 <b>4 大真实人类调式</b>——彻底抛弃公式扁平曲线，100% 直传对应母带的真人曲调弧度！每组第一张卡片为【原版真人母带】，后两张为用该调式合成的原篇与全新迁移篇，方便逐一对比挑调。</p>
<div class="box">
<b>17 首非近体素材聚类出的 4 大调式：</b><br>
1. <b>调式一 · 诗经雅正抒情调</b>（t15《蒹葭》/ t13《采薇》/ t01《关雎》）：大弧度咏叹，高起回旋、偶句大五度深沉落韵（-5~-7st）；<br>
2. <b>调式二 · 诗经秦风豪放调</b>（t20/t14《无衣》）：节奏明快（~2.4s/句）、字字铿锵、高亢顿挫；<br>
3. <b>调式三 · 书斋清雅文言调</b>（t09《兰亭集序》/ t04《论语》/ 书寒公子系列）：江南书生平叙温润、句尾轻扬悬收留白；<br>
4. <b>调式四 · 唐调沉雄古文赋调</b>（t16《陋室铭》/ t18《前赤壁赋》/ t19《岳阳楼记》）：句前快诵铺排、句末重锤大跨度沉落压韵。
</div>
{"".join(html_groups)}
<script>
async function downloadWav(url, filename, btn) {{
  const orig = btn.textContent;
  btn.textContent = '下载中...';
  btn.disabled = true;
  try {{
    const sep = url.includes('?') ? '&' : '?';
    const resp = await fetch(url + sep + 'download=1');
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const blob = await resp.blob();
    const objUrl = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = objUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(objUrl), 5000);
  }} catch (e) {{
    alert('下载失败：' + e.message);
  }} finally {{
    btn.textContent = orig;
    btn.disabled = false;
  }}
}}
</script>
</body></html>"""
    out_html = WEB_DIST / "sing_compare.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"\n✅ V7 非近体四大调式试听台已刷新 -> {out_html}")


if __name__ == "__main__":
    main()
