#!/usr/bin/env python3
"""对齐《一口气听完中古切韵音全常用音节》：OCR 行序列 + 鼠标指针锚点 → 音节片段标签 (v2)。

- 帧时间：f_i 位于 (i-1)*10 s（ffmpeg fps=1/10），补帧 g_i 位于 (i-1)*10-5 s
- 合并：按时间排序的所有帧做“最长后缀-前缀重叠”拼接成全局行序列
- 锚点：帧内鼠标指针所指行 ↔ 该时刻正在读/紧随其后的音频片段
- 区间分配：相邻锚点间片段数 ds 与行数 dg 相等直接一一对应；ds > dg 时剔除最短的 (ds-dg) 个片段（噪声/切碎）；ds < dg 跳过
"""
import json, re, unicodedata
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parents[1] / "data_mc_raw"
frames = []
for fn, kind in (("ocr_frames10s.json", "f"), ("ocr_frames_gap.json", "g")):
    p = D / fn
    if not p.exists():
        continue
    for fr in json.load(open(p)):
        i = int(re.search(r"(\d+)", fr["file"]).group(1))
        fr["t"] = (i - 1) * 10.0 - (5.0 if kind == "g" else 0.0)
        frames.append(fr)
frames.sort(key=lambda x: x["t"])
segs = json.load(open(D / "segments_v1.json"))


def norm_ipa(s):
    return unicodedata.normalize("NFC", s.strip().replace(" ", ""))


def key(l):
    # 以字头（前两个大字）为键：比 IPA 更不易被 OCR 读出细微差异
    return l["chars"][:2]


G, anchors, cur_header = [], [], None
n_noov = 0
for fr in frames:
    lines = [l for l in fr["lines"] if l["chars"] and l["ipa"]]
    if fr.get("header"):
        cur_header = fr["header"]
    keys = [key(l) for l in lines]

    def feq(a, b):
        return a == b or a[:1] == b[:1] or (len(a) > 1 and len(b) > 1 and a[1:2] == b[1:2])

    best = 0
    for k in range(min(len(G), len(keys)), 0, -1):
        tailk = [g["key"] for g in G[-k:]]
        mism = sum(not feq(a, b) for a, b in zip(tailk, keys[:k]))
        if mism == 0 or (k >= 5 and mism <= 1):
            best = k
            break
    if best == 0 and G:
        n_noov += 1
    gidx, start = [], len(G) - best
    for j, l in enumerate(lines):
        if j < best:
            gidx.append(start + j)
        else:
            G.append({"key": keys[j], "ipa": l["ipa"], "chars": l["chars"], "header": cur_header})
            gidx.append(len(G) - 1)
    c = fr.get("cursor_line")
    if c is not None and 0 <= c < len(gidx):
        anchors.append((fr["t"], gidx[c]))

print(f"帧 {len(frames)} | 无重叠拼接 {n_noov} | 全局行数 {len(G)} | 锚点 {len(anchors)} | 音频片段 {len(segs)}")


def seg_at(t):
    for i, (a, b) in enumerate(segs):
        if b >= t:
            return i
    return len(segs) - 1


A = sorted({(seg_at(t), g) for t, g in anchors})
mono = []
for s, g in A:
    if not mono or (s > mono[-1][0] and g > mono[-1][1]):
        mono.append((s, g))
labels, ok, fixed, bad = {}, 0, 0, 0
for (s0, g0), (s1, g1) in zip(mono, mono[1:]):
    ds, dg = s1 - s0, g1 - g0
    idx = list(range(s0, s1))
    if ds > dg and ds - dg <= 2:
        durs = sorted(idx, key=lambda i: segs[i][1] - segs[i][0])
        drop = set(durs[: ds - dg])
        idx = [i for i in idx if i not in drop]
        fixed += 1
    elif ds != dg:
        bad += 1
        continue
    else:
        ok += 1
    for k, s in enumerate(idx):
        labels[s] = g0 + k
print(f"锚点区间: 一致 {ok} | 剔噪后一致 {fixed} | 放弃 {bad} | 已标注片段 {len(labels)}")

out = []
for s, g in sorted(labels.items()):
    L = G[g]
    if " " in L["ipa"].strip():
        continue
    out.append({"seg": s, "start": segs[s][0], "end": segs[s][1], "ipa": L["ipa"], "chars": L["chars"], "header": L["header"], "g": g})
json.dump(out, open(D / "xiaoqieyun_labels_v3.json", "w"), ensure_ascii=False, indent=0)
json.dump(G, open(D / "xiaoqieyun_rows_v3.json", "w"), ensure_ascii=False, indent=0)
print("单音节标签:", len(out), "| 覆盖不同行:", len({o["g"] for o in out}))
