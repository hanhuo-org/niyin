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
   若修改了采样器、初值噪声、编译器或嵌入，还须运行 **8 种子统计回归门禁**（检验抗随机波动下限与零 G2 分句断裂）：
   ```bash
   HF_HUB_OFFLINE=1 /mnt/s790/project/hanzi_train/.venv/bin/python hanfire_tts/scripts/audit_regression.py --multiseed 8
   ```
3. **运行三代全音系与 15 首代表诗词全检**（大版本更新时运行）：
   ```bash
   /mnt/s790/project/hanzi_train/.venv/bin/python hanfire_tts/scripts/overnight_full_audit.py
   /mnt/s790/project/hanzi_train/.venv/bin/python hanfire_tts/scripts/audit_all_three_era_phonemes.py
   ```
4. **单独审计某首新诗的押韵对比度**：
   ```bash
   /mnt/s790/project/hanzi_train/.venv/bin/python hanfire_tts/scripts/audit_rhyme.py <wav_path> --rhyme 0,1,3 --labels 句1,句2,句3,句4
   ```
