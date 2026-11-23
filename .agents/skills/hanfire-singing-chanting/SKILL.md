---
name: hanfire-singing-chanting
description: >-
  Workflow, speaker routing, acoustic tuning laws, and verification runbook for
  the Hanfire Historical Chinese Chanting & Singing Engine (hanfire_sing/,
  DiffSinger 44.1kHz female & male models). Activate whenever synthesizing,
  tuning, debugging, or extending Shijing, Guwen/Fu, Tang poetry chanting,
  male/female DiffSinger training, or YuE dual-track ancient singing.
---

# 汉火古音唱吟 (`hanfire_sing/`) 工程操作手册

## 1. 现役双声部棚录底座与音区甜区表

两个底座 **100% 共享 `opencpop_clean.txt` 词表**与 **`pc_nsf_hifigan_44.1k_hop512_128bin_2025.02` 声码器**，所有上层旋律/韵律引擎可零改动一键切换男女声部：

| 底座目录 (`hanfire_sing/checkpoints/`) | 训练语料 | 说话人 (`--spk`) | 原生音域 / 换声警戒线 | 最佳适配调式与气质 |
|---|---|---|---|---|
| `hanfire_sing_base` (`ckpt=40000`) | OpenCpop 5.2h 44.1kHz 女声 | `opencpop_base` | `172~345Hz`（避开 `300~311Hz` 换声区气声带） | 全品类纯净女声基准 |
| `hanfire_sing_male` (`ckpt=40000`) | M4Singer 6 男声 10.35h 44.1kHz | `m4_bass1` | `78~131Hz`（增广 `62~175Hz`，胸声 `<=185Hz`） | **首选学者男低中音**：《蒹葭》《采薇》《陋室铭》《岳阳楼记》、深沉版《无衣》/《短歌行》 |
| `hanfire_sing_male` (`ckpt=40000`) | 同上 | `m4_tenor7` | `117~247Hz`（胸声 `<=260Hz`） | **首选清朗抒情男声（男版 OpenCpop）**：《关雎》《桃夭》《兰亭集序》《前赤壁赋》 |
| `hanfire_sing_male` (`ckpt=40000`) | 同上 | `m4_tenor6` | `110~247Hz`（胸声 `<=275Hz`） | **首选昂扬男高音**：秦风《无衣》豪放铿锵调 |
| `hanfire_sing_male` (`ckpt=40000`) | 同上 | `m4_tenor5` / `m4_bass3` / `m4_tenor2` | `92~311Hz` 备选音色 | 温润男中 / 磁性男低 / 清越男高备选 |

> **关键禁忌**：推理统一设 `--gender=0.0`、`--steps 24`、`--seed 42`（A/B 复现）；严禁对女声降 `--gender=-0.5` 冒充男声；严禁复用旧实验 `hanfire_tang_local`（仅 6.15min + Whisper 错标导致辅音毁损）。

---

## 2. 核心引擎与一键渲染命令

- **音素编译器与多音字/中古防 OOD 覆写**：`hanfire_sing/hanfire_sing_phonemes.py`
  - 单字现代音覆写：`_ZH_POLYPHONE_OVERRIDE`（如 `俟→si`, `蕃→fan`）
  - 上下文多音词覆写：`_ZH_CONTEXT_POLYPHONES`（如 `会稽→kuai ji`, `属客→zhu ke`, `冯虚→ping xu`, `扁舟→pian zhou`, `少长→shao zhang`, `汤汤→shang shang`）
  - 中古/近古合法 OpenCpop 音素守卫：严禁拼出 `y iao / w ui / w uo / w uang / y iu` 等 OOD 组合
- **《诗经》三大调式引擎（V9.6，已验收）**：`hanfire_sing/shijing_v9_engine.py`（`shijing_jianjia` / `shijing_guanju` / `shijing_wuyi`）
- **古文/辞赋三大调式引擎（V16，已验收）**：`hanfire_sing/guwen_v14_engine.py`（`guwen_lanting` 第一组《兰亭集序》 / `guwen_loushi` 第二组《陋室铭》 / `fu_chibi` 第三组《前赤壁赋》+《岳阳楼记》泛化）
  - V16 核心律则：逐字对照真人母带（`t09`/`t16`/`t18`/`t19`/`t10`）真实音高轨；对仗 `3+3`/`4+4` 并句连气吟诵（句间仅极短自然换气不硬断）；上声低兜（如 `浩浩`）、倒数第二字去声高昂亮峰（如 `际`/`万`）、末字阳平/平声雄浑归宫（如 `涯`/`千`）。

### 一键刷新男女双声部对比台（在仓库根目录 `niyin/` 下执行）
```bash
# 1) 刷新《诗经》三大调式双声部 -> http://127.0.0.1:8080/sing_compare.html
/mnt/s790/project/hanzi_train/.venv/bin/python hanfire_sing/scripts/render_shijing_lab.py

# 2) 刷新古文/辞赋三大调式双声部 -> http://127.0.0.1:8080/guwen_compare.html
/mnt/s790/project/hanzi_train/.venv/bin/python hanfire_sing/scripts/render_guwen_lab.py

# 3) 启动本地 Web 工作台与对比试听台（8080 HTTP / 8444 HTTPS）
/mnt/s790/project/hanzi_train/.venv/bin/python hanfire_tts/web_wasm/serve_web.py
```

---

## 3. 声学疑难排查速查表（诊断 → 根治）

1. **某字声母糊成半元音（如 `女→yu`、`归→wei`、`淑→如`）**：
   - 先查是否落入该声部第一换声点（女声 `300~311Hz`、男低 `>185Hz`、男中 `>260Hz`）→ 调整 `key_hz / male_key_hz` 退回实音区；
   - 查清擦音是否给足 `88~115ms`；
   - 长元音后的鼻音 `n/m` / 塞音 `b/p/d/t/g/k` 是否给足 `102~108ms` 并在高斯平滑**之后**加了 Hanning 基频微谷（`-1.45 / -1.75st`）。
2. **句中某字被“突然强调”或停顿怪异**：
   - 立即排查是否误插了词内 `SP`（哪怕 15ms），全部删除；
   - 查母带该字原位是否偶发 `+1.4st` 音高突拱或相邻字过渡深坑（如 `4.32s / 4.75s`），改切纯净稳段或参照同诗其他章的真实五声落位。
3. **长句高音突刺 / 爆音**：
   - 查 `librosa.pyin` 是否在母带某帧发生八度跳变毛刺（如 `+12~+14st`），必须在 `min_st..max_st` 内严密限幅并剔除毛刺句段。
4. **泛化新古文/辞赋不自然**：
   - 先从 `hanfire_sing/data_tang_processed/{tracks,clauses}/` 中找同体裁母带（如 `t19`/`t10`/`t18`）做逐字 F0 与时长对比分析，归纳四声平仄+对仗句法规律，绝不写死针对某首诗的特判。
5. **验证抗随机波动（防单种子幸存者偏差）**：
   - 改动引擎底层后，务必用 `seed = 101, 202, 303` 对关键难点句做 3 种子盲跑，检查辅音成阻 RMS 微谷与前后元音响度差，并通过 `read_file` 直接实听全部音频后再向用户交付。

---

## 4. 新增男/女棚录底座或云端 Colab 训练规范
- 参考脚本：`hanfire_sing/scripts/colab_train_male_acoustic.py` 与 `hanfire_sing/scripts/pull_male_ckpt.sh`；
- 从 HF 拉取大量小文件（如 M4Singer `TextGrid + wav`）务必用 **32 线程 `requests.Session` 直连 HTTP 断点续传**（实测 `20~40 files/s`，比默认 `snapshot_download` 快 10 倍，省去 50+ 分钟 A100 空烧）；
- 拉回权重到本地后须同步修补 `config.yaml` 的 `binary_data_dir` 与 `vocoder_ckpt` 本地路径、复制 `opencpop_clean.txt` 为 `dictionary.txt`/`dictionary-zh.txt`，通过 `torch.load` 校验后立即 `colab stop` 停机。
