# 汉火拟音 (Hanfire Niyin — v1.0)

> **三代历史汉语拟音 TTS（上古 / 中古 / 近古）× 浏览器端 WebGPU/WASM 离线推理 × 唐调《诗经》/古文/辞赋男女双声部 44.1kHz 唱吟引擎**（线上体验：[yin.hanhuo.org](https://yin.hanhuo.org/)）。

- **子系统一 · 三代古音拟音 TTS（`hanfire_tts/`，V10 Studio）**
  - **三代全贯通**：覆盖上古汉火音（`oc`）、中古切韵盛中唐音（`mc`，38 声母 + 133 韵类因子化嵌入）、近古《中原音韵》（`em` / `em_zyyy`），`51,735` 字三代联合词表零 OOD、100% 一字一音零吞字。
  - **跨端同轨确定性引擎**：`ZipVoice` 流匹配 + `Vocos 24kHz` 声码器 + `W8A8 QAT` 真 `INT8` 量化（`470 MB → 124 MB`），Python 与浏览器端统一采用 `Mulberry32 + Box-Muller` 球壳初值与 `F5-TTS Sway(s=-0.8)` 调度，支持 **WebGPU 加速** 与 **多线程 SIMD WASM** 无缝回退。
- **子系统二 · 汉火古音唱吟（`hanfire_sing/`，《诗经》V9.6 + 古文/辞赋 V16）**
  - **真人活腔残差 × 中古四声依字行腔**：零词内 `SP`、零动态响度压缩、零人工 EQ 染色（`x_pure_studio=True`）；完整实现《诗经》三大调式（《关雎》《蒹葭》《无衣》）与古文/辞赋三大调式（《兰亭集序》《陋室铭》《前赤壁赋》及跨篇排比连气泛化如《爱莲说》《诫子书》《岳阳楼记》）。
  - **44.1kHz 男女纯棚录双底座**：女声 `hanfire_sing_base`（OpenCpop 5.2h, 40k steps）+ 男声 `hanfire_sing_male`（M4Singer 6 男声 10.35h, 40k steps），100% 共享 `opencpop_clean.txt` 词表与 `PC-NSF-HiFiGAN 44.1kHz 2025.02` 声码器。

---

## 一、模型与素材托管位置

| 资源类别 | 托管地址 | 说明 |
| :--- | :--- | :--- |
| **源代码** | [GitHub `hanhuo-org/niyin`](https://github.com/hanhuo-org/niyin) | 本仓库（亦作为 `hanhuo-org/hanhuo-dadian` 子模块 `niyin/`） |
| **训练好模型（Release）** | [GitHub Release `v1.0`](https://github.com/hanhuo-org/niyin/releases/tag/v1.0) | 含 Web 运行包、TTS FP32/INT8/ONNX、唱吟男女双底座、44.1kHz 声码器、活腔母带与 `SHA256SUMS.txt` |
| **模型镜像（Hugging Face）** | [`hanhuo-org/hanhuo-niyin-models`](https://huggingface.co/hanhuo-org/hanhuo-niyin-models) | 同款训练好权重 + 完整 FP32 ONNX 与 V5 基础续训权重 |
| **训练素材与母带（HF）** | [`hanhuo-org/hanhuo-niyin-dataset`](https://huggingface.co/datasets/hanhuo-org/hanhuo-niyin-dataset) | 上古/中古/近古原始音韵素材、TTS 精切训练集、唐调唱吟母带及离线试听音频 |

---

## 二、快速启动（2 步即用）

### 1. 浏览器离线三代古音 TTS 工作台（约 168 MB）

无需 GPU 或 Cloudflare，纯 Python 标准库即可启动支持多线程 SIMD WASM / WebGPU 的本地离线服务：

```bash
git clone https://github.com/hanhuo-org/niyin.git
cd niyin

# 1. 从 GitHub Release 拉取并校验 Web 运行时模型包 (~168 MB)
python scripts/download_weights.py
# 若使用 Hugging Face 镜像源：
# python scripts/download_weights.py --source hf

# 2. 启动本地离线 Web 服务（自动附加 COOP/COEP 响应头）
python hanfire_tts/web_wasm/serve_web.py --port 8080
# 浏览器打开 http://127.0.0.1:8080/
```

### 2. Python 本地推理与唐调古音唱吟（男女双声部）

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 下载全部预训练权重（Web + TTS FP32/INT8 + 唱吟男女双底座 + 44.1kHz 声码器 + 唐调母带）
python scripts/download_weights.py --all

# A. 运行三代古音 TTS 全维度 6 大门禁回归评测
python hanfire_tts/scripts/audit_regression.py

# B. 渲染《诗经》三大调式男女双声部对比音频 -> hanfire_tts/web_wasm/dist_cf/sing_compare.html
python hanfire_sing/scripts/render_shijing_lab.py

# C. 渲染古文/辞赋三大调式男女双声部对比音频 -> hanfire_tts/web_wasm/dist_cf/guwen_compare.html
python hanfire_sing/scripts/render_guwen_lab.py
```

---

## 三、按需拉取训练素材与复现训练

```bash
# 按需从 Hugging Face (hanhuo-org/hanhuo-niyin-dataset) 拉取训练素材：
python scripts/download_dataset.py --raw-assets  # ~472 MB：上古/中古/近古原始素材与音节表
python scripts/download_dataset.py --tts-train   # ~650 MB：三代 TTS 训练集、24kHz 干声与小切韵/西游记对齐
python scripts/download_dataset.py --sing-tang   # ~695 MB：唐调唱吟母带与微调语料
python scripts/download_dataset.py --showcase    # ~121 MB：离线试听对比台音频
# 或一步拉取全部：
# python scripts/download_dataset.py --all
```

常用训练与导出入口：
- **三代 TTS V10 训练 & W8A8 QAT 量化**：`hanfire_tts/scripts/train_v10_a100.py`、`hanfire_tts/scripts/train_v10_studio_qat.py`
- **单字读音/韵类嵌入无损校准**：`hanfire_tts/scripts/calibrate_pronunciation.py`
- **浏览器 ONNX 导出与 16MB 分片打包**：`hanfire_tts/scripts/export_web_onnx.py`、`hanfire_tts/scripts/pack_web_int8_chunks.py`
- **唱吟 44.1kHz DiffSinger 男女底座训练**：`hanfire_sing/scripts/colab_train_acoustic.py`、`hanfire_sing/scripts/colab_train_male_acoustic.py`

> 开发与调音前请务必阅读工程红线守则：[AGENTS.md](AGENTS.md)。

---

## 四、目录结构

```text
niyin/
├── README.md
├── AGENTS.md                        # 古音 TTS 与唱吟工程红线守则
├── pyproject.toml / requirements.txt
├── docs/                            # 总体架构、唱吟经验沉淀、三代审计与部署文档
├── scripts/
│   ├── download_weights.py          # 从 GitHub Release / HF 拉取并校验全部训练好模型
│   ├── download_dataset.py          # 从 HF (hanhuo-org/hanhuo-niyin-dataset) 按需拉取训练素材
│   └── publish_release.sh           # 一键通过 gh release create 发布权重包
├── hanfire_tts/                     # 三代历史汉语拟音 TTS + 浏览器 WebGPU/WASM 离线工作台
│   ├── configs/                     # 三代读音覆写表 (overrides_oc/mc/em.json)
│   ├── hanfire_core/                # 音韵编译器、切韵因子化槽、ZipVoice 流形、W8A8 QAT
│   ├── scripts/                     # 训练、校准、导出与 6 大门禁回归审计脚本
│   ├── tools/qieyun/                # qieyun-js 切韵音韵地位导表工具
│   ├── vendor/ZipVoice/             # 轻量 ZipVoice 源码（含 fp32 softplus 与无 tensorboard 补丁）
│   └── web_wasm/                    # 浏览器端 WebGPU/WASM 推理与 Cloudflare Workers / 本地服务
└── hanfire_sing/                    # 古音唱吟引擎（《诗经》V9.6 + 古文/辞赋 V16）
    ├── hanfire_sing_phonemes.py     # 去声调三代唱吟音素编译器（0 OOD 对齐 OpenCpop）
    ├── shijing_v9_engine.py         # 《诗经》三大调式 V9.6 活腔引擎
    ├── guwen_v9_engine.py           # 古文/辞赋篇章规划与中古四声平仄分析
    ├── guwen_v14_engine.py          # 古文/辞赋三大调式 V16（兰亭/陋室/赤壁×岳阳楼排比连气律）
    ├── tang_melody_prior.py         # 唐调近体诗旋律先验
    ├── scripts/                     # 语料预处理、Colab 双声部训练、实验室渲染脚本
    └── DiffSinger/                  # 轻量 OpenVPI DiffSinger 源码 + 兼容补丁与 opencpop_clean.txt 词表
```

---

## 五、开源协议与致谢
- 基础架构鸣谢：[k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice)、[charactr-platform/vocos](https://github.com/charactr-platform/vocos)、[openvpi/DiffSinger](https://github.com/openvpi/DiffSinger)、[nk2028/qieyun-js](https://github.com/nk2028/qieyun-js)、[OpenCpop](https://wenet.org.cn/opencpop/)、[M4Singer](https://github.com/M4Singer/M4Singer)。
