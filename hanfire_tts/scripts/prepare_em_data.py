#!/usr/bin/env python3
"""提取近古新素材与中古《周南》24kHz 音频，并用本地 Whisper GPU 仅做粗切分与时间戳提取（后续用原文+大模型校对）。"""
import json, subprocess
from pathlib import Path
import whisper

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
EM_RAW = ROOT / "hanfire_tts" / "data_em_raw"
MC_RAW = ROOT / "hanfire_tts" / "data_mc_raw"
EM_CLIPS = EM_RAW / "wav24k"
EM_CLIPS.mkdir(parents=True, exist_ok=True)

EM_VIDEOS = [
    ("xyj", EM_RAW / "xiyouji_vocals_24k.wav"),
    ("yyl", ASSETS / "近古音/岳阳楼记，但用明朝官话读出来【明中期官话朗读，大体采用《洪武正韵》俗音】_720P.mp4"),
    ("ths", ASSETS / "近古音/明末官话试读 之 【桃花扇．试一齣．先声】 选段 (20)_720P.mp4"),
    ("dyc", ASSETS / "近古音/明末官话拟音试读《钓鱼城行》西儒耳目资音系  重录_720P.mp4"),
    ("wbcs", ASSETS / "近古音/明末官话试读 之 【唐诗三百首·五言古诗·杜甫·赠卫八处士】 (21)_720P.mp4"),
    ("yjk", ASSETS / "近古音/明代初年官话音朗读：《咏荆轲》【东晋】陶渊明.作   【依据《韵略易通》《洪武正韵译训》订定发音】_720P.mp4"),
    ("tag_new", ASSETS / "近古音/明末汉语试读 桃花庵歌 唐伯虎_720P.mp4"),
    ("chysx", ASSETS / "近古音/明朝官话--沧海一声笑.mp4"),
    ("gj_ming", ASSETS / "近古音/明朝读书音（洪武正韵）朗诵《诗经·周南·关雎》【近古汉语】_720P.mp4"),
    ("qya", ASSETS / "近古音/1600s的正宗大明南京官话——《西儒耳目资》音系读辛弃疾词《青玉案·元夕》_720P.mp4"),
    ("jw", ASSETS / "近古音/明末官话读顾炎武《精卫》（西儒耳目资音系）_720P.mp4"),
]
MC_VIDEOS = [
    ("zn_mc", ASSETS / "中古音/詩·國風·周南 中古漢語切韻音系朗讀_480P.mp4"),
]

model = whisper.load_model("small", device="cuda")

def process_list(pairs, out_dir: Path, out_json: Path, era: str):
    records = []
    for tag, src in pairs:
        wav_p = src if src.suffix == ".wav" else (out_dir / f"{tag}.wav")
        if src.suffix != ".wav":
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-ac", "1", "-ar", "24000", str(wav_p)], check=True)
        res = model.transcribe(str(wav_p), language="zh", task="transcribe", fp16=True, verbose=False,
                               initial_prompt="以下是中国古典诗词散文文言文朗读，请使用标准汉字输出，不要输出拼音。")
        segs = [{"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip()}
                for s in res["segments"] if s["end"] - s["start"] >= 0.4]
        print(f"  [{era}:{tag}] {wav_p.name} -> {len(segs)} segs")
        records.append({"tag": tag, "era": era, "wav": str(wav_p), "segments": segs})
    out_json.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")

process_list(EM_VIDEOS, EM_CLIPS, EM_RAW / "em_raw_segments.json", "em")
process_list(MC_VIDEOS, MC_RAW, MC_RAW / "mc_zn_raw_segments.json", "mc")
print("✅ 粗切分完成")
