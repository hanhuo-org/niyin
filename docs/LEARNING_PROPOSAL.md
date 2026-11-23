# Learning Proposal: 固化汉火古音 TTS 防“按下葫芦浮起瓢”规则与改音技能

## 1. 分类与设计理由 (Classification & Rationale)

根据本次交互中定位到的跨层耦合退化根因与 `agy-customizations` 规范（当前工作区尚无 `AGENTS.md` 或 `.agents/`），提议拆分为 **“常驻红线规则 (Rule) + 按需操作手册 (Skill)”** 两部分：

| 类型 | 目标路径 | 为什么用该类型 | 预估大小 |
| :--- | :--- | :--- | :--- |
| **Workspace Rule** | `/mnt/s790/project/tts/AGENTS.md` | 承载**每次修改都必须无条件遵守的硬性约束**（底座冻结、参考音零变速、零有声前导词、Vocos 甜区增益、测试集隔离、交付前必跑 `audit_regression.py`）。独立 `AGENTS.md` 不含 frontmatter，对工作区及子智能体常驻生效。 | ~2.6 KB（远低于 24 KB 上限） |
| **Workspace Skill** | `/mnt/s790/project/tts/.agents/skills/hanfire-pronunciation-tuning/SKILL.md` | 承载**多步操作命令与参数速查表**（如何通过 `overrides_*.json`、句内 `[字=规格]`、`calibrate_pronunciation.py` 修改单字读音与押韵，以及如何渲染和排查声学异常）。采用渐进式披露，平时不占上下文。 | ~2.8 KB |

---

## 2. 拟新增文件一：`/mnt/s790/project/tts/AGENTS.md`

```markdown
# 汉火古音 TTS (Hanfire Historical Chinese TTS) 工程红线守则

本工作区（特别是 `hanfire_tts/`）是一个“音韵编译器 → 因子化嵌入 → ZipVoice 流匹配 → Vocos 声码器”四层强耦合系统。为防止“按下葫芦浮起瓢”（修好一处听感却破坏另一处），任何代码修改、训练或音频渲染必须严格遵守以下硬性约束：

## 1. 模型训练与数据红线
- **非棚录干声严禁解冻主干**：凡经 Demucs 人声分离、视频网站二次压缩或含背景残响的语料，**严禁解冻 `fm_decoder`（含 LoRA）与 `text_encoder`**（否则必将分离残余相位伪影学成 `-50dB` 机械音底噪，并破坏句首注意力）。此类语料仅可用于训练词元嵌入或做音韵对齐。
- **单字语料仅训嵌入**：单音节孤立录音（如切韵全音节表）不含句首停顿词元 `"."`，绝对不可用于微调 `text_encoder` 或 `fm_decoder`。
- **测试集严格隔离**：中古音评测基准诗《静夜思》《春晓》《江雪》严禁以任何形式进入训练集。
- **合规与评测禁令**：严禁抓取或使用 `qieyun-tts.com`（含 Turnstile 及 `ai-train=no` 声明）数据训练；严禁用现代普通话 ASR（如 Whisper）评判上古/中古/近古拟音准确率。

## 2. 推理采样与声码器红线（防电音、防破音、防吞字、防抢音）
- **参考音零变速**：**严禁**对参考音频使用 `librosa.effects.time_stretch` 或 `ffmpeg atempo`（会产生 STFT/WSOLA 相位涂抹“电音感”并被克隆）。参考音必须使用 `rate=1.0` 原生未拉伸 24kHz 录音室干声，且参考文本词元必须与参考音内容 100% 一致。
- **字速严丝合缝匹配**：目标 `frames_per_100w` 必须与所选参考音的真实字速匹配（偏差 ≤ ±15%）。严禁用快语速参考音强配大 `fp100` 帧预算，否则流匹配会在句中硬塞空白停顿导致语速忽快忽慢。
- **零有声前导词**：**严禁**在目标句前拼接发音汉字（如 `"吟"`）做牺牲前导词（其鼻音尾音会残留为句首 `"en"` 杂音）。必须使用 `sample_clean_zero_carrier`（纯静音短逗号衔接 + 首字 `+44w` 帧预算补偿 + 温和边缘修剪 `pre_ms=35`）。
- **Vocos 甜区增益与零内部硬削波**：
  - 参考音送入 `PureVocosFbank` 前须衰减至甜区（标准 `RMS≈0.10` 干声乘 `0.45`），保证 Vocos 内部 `exp(mag)` 处于 `40~75` 线性区，避免触碰 `100` 幅度截断；
  - `PureVocosDecoder.decode_mel` 内部**禁止** `.clamp(-0.96, 0.96)` 硬削波，统一在外部做无损峰值保护归一化（`target_rms=0.055, max_peak<=0.72`）与 `guidance_scale=1.0`。

## 3. 音韵编译拓扑红线
- **零声母开口单字封顶**：近古/普通话流形下的零声母开口或鼻尾字（如 `安/庵`）必须保持原生单词元（如 `('an1', 62)`，权重 ≤ 65w），**严禁前置 `OC_ʔ`**（会变为舌尖塞音 `t-` 读成“贪”），**严禁叠加句尾延音或句中元音间逗号**（会裂变成“安安”或断词）。
- **禁插冗余双辅音头**：对原生已含 `-i-/-u-` 介音的韵母词元（如 `ian/ia/iu/ua/uo`），直接采用双词元 `(声母, 30) + (韵母, 70)`，禁止额外插入 `y0/w0` 挤成 3~4 帧双声母跳变（会引发高频爆音毛刺）；仅在韵母词元本身不含介音且会被普通话声韵拉塌时（如中古精组三四等尖音 `絕/雪`、见组三等魚虞韵 `舉/去`）才使用 `y0` 桥接。
- **切韵特殊 IPA 符号防退化与纯细音锚定**：解析 `qieyun-js` 的 `mid_tang` 拟音前必须先将 `ɨ/ʉ/ɯ/ɜ/œ/ɒ` 归一化为标准元音（否则会全部退化为央元音 `ə`，导致 `笠→了`、`月→乐`、`舉→葛`、`疑→内`）；且校准入声高前元音（如緝韵 `-ip`“笠”）须用纯细音词元（`in4+ie4`）而非与舌尖元音 `[ɿ/ʐ̩]` 混用的 `i4`。

## 4. 交付前强制回归门禁
任何编译器、嵌入、采样器或渲染脚本改动后，在向用户交付音频前**必须运行并通过全维度自动门禁**：
```bash
/mnt/s790/project/hanzi_train/.venv/bin/python hanfire_tts/scripts/audit_regression.py
```
6 大门禁（G1 句首无抢音、G2 分句数与时长极差、G3 零削波与低跳变 `max|dw|`、G4 底噪纯度、G5 押韵对比度、G6 关键音位共振峰）全部 `PASS` 后方可交付。
```

---

## 3. 拟新增文件二：`/mnt/s790/project/tts/.agents/skills/hanfire-pronunciation-tuning/SKILL.md`

```markdown
---
name: hanfire-pronunciation-tuning
description: >-
  Workflow and runbook for modifying individual character pronunciations, fixing
  polyphones or rhymes, calibrating embeddings without retraining the backbone,
  and rendering/verifying historical Chinese (OC/MC/EM) audio in hanfire_tts.
  Activate whenever the user reports a mispronounced character, broken rhyme,
  pacing glitch, or asks to synthesize new classical poems.
---

# Hanfire 古音单字精修与纯净渲染手册

修改个别字读音或修复押韵时，按“成本从低到高、绝不破坏全局”的顺序依次选用 L1 → L2，最后必跑回归门禁。

## Level 1：零重训即时改音（首选，0 秒生效）

### 方式 A：修改三代持久覆写字典（全局生效）
直接编辑 JSON 配置文件（编译器初始化时自动加载）：
- 中古音 (`mc`)：`hanfire_tts/configs/overrides_mc.json`
- 近古音 (`em`)：`hanfire_tts/configs/overrides_em.json`
- 上古音 (`oc`)：`hanfire_tts/configs/overrides_oc.json`

支持的规格写法：
1. **切韵音韵地位或调类**（仅 MC）：`"處": "昌開三魚去"` 或 `"行": "去"`
2. **同字借音**（三代通用）：`"滅": "同絕"`
3. **底层原子词元+权重直写**（三代通用）：`"花": "Q_曉:30+ua1:70"` 或 `"仙": "s0:30+ian1:70"`
4. **IPA+调标覆写**（OC/EM）：`"種": "tʂuŋh"`（尾 `h` 去声、尾 `ʔ` 上声）

### 方式 B：句内按位置标注（解决同句多音字）
在输入诗句中直接对特定位置的字加 `[字=规格]`，不影响同句其他同形字：
- 示例：`"[處=昌開三魚去]處聞啼鳥"`、`"桃花仙人[種=tʂuŋh]桃樹"`、`"萬徑人蹤[滅=同絕]"`

---

## Level 2：外科手术级嵌入流形校准（0.5～5 秒，冻结主干）

当某个韵类词元（如 `Q_麻合二_舒_平陰`）整体音色偏离预期时，使用 `calibrate_pronunciation.py` **仅更新该词元嵌入行**（100% 冻结 `text_encoder`、`fm_decoder` 与其余词元）：

```bash
# 1) 凸组合流形插值（如强化麻合二 -ua 介音）
/mnt/s790/project/hanzi_train/.venv/bin/python hanfire_tts/scripts/calibrate_pronunciation.py \
  --ckpt hanfire_tts/v10_outputs/best_fp32_v10_studio.pt \
  --blend "Q_麻合二_舒_平陰=ua1:0.75"

# 2) 同韵部硬共享锁定（强制同韵字共享同一韵核向量）
/mnt/s790/project/hanzi_train/.venv/bin/python hanfire_tts/scripts/calibrate_pronunciation.py \
  --ckpt hanfire_tts/v10_outputs/best_fp32_v10_studio.pt \
  --lock-rhyme "Q_仙三A_入_入陽:Q_仙合三_入_入陽,Q_仙合三_入_入陰"

# 3) 用单字干净干声做 35 步掩码精修（带 L2 流形锚定防漂移）
/mnt/s790/project/hanzi_train/.venv/bin/python hanfire_tts/scripts/calibrate_pronunciation.py \
  --ckpt hanfire_tts/v10_outputs/best_fp32_v10_studio.pt \
  --target-tok "Q_麻合二_舒_平陰" --clause "花" --wav /path/to/clean_ref.wav --steps 35
```

---

## 渲染与强制回归验证流程

1. **运行纯净自然渲染脚本**（已内置 `sample_clean_zero_carrier`、0.45 甜区增益、75Hz–9.5kHz 滤波与无损峰值保护）：
   ```bash
   HF_HUB_OFFLINE=1 /mnt/s790/project/hanzi_train/.venv/bin/python hanfire_tts/scripts/render_v10_studio.py
   ```
2. **运行 6 大自动回归门禁**（必须 `全部通过 ✅`）：
   ```bash
   /mnt/s790/project/hanzi_train/.venv/bin/python hanfire_tts/scripts/audit_regression.py
   ```
3. **单独审计某首新诗的押韵对比度**：
   ```bash
   /mnt/s790/project/hanzi_train/.venv/bin/python hanfire_tts/scripts/audit_rhyme.py <wav_path> --rhyme 0,1,3 --labels 句1,句2,句3,句4
   ```
```
