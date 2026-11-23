# 汉火古音 TTS (`hanfire_tts`) —— 架构说明与操作手册 (V7 三代贯通 + INT8 QAT + 浏览器 WebGPU/WASM)

> 完整总体设计文档：[HANFIRE_ANCIENT_TTS_PLAN.md](../HANFIRE_ANCIENT_TTS_PLAN.md)

## 1. 核心目录与模块地图

```text
hanfire_tts/
├── hanfire_core/
│   ├── phonology.py             # 单核 4 槽位拆解器 [Onset | Medial | Nucleus | Coda]
│   ├── zipvoice_zh_manifold.py  # 核心底座：ZipVoice 中文流形手术 (86古音槽) + ThreeEraHanfireCompiler (oc/mc/em/em_zyyy)
│   ├── zipvoice_ancient.py      # PureVocosFbank (24kHz 100-Mel) 与声学底座工具
│   ├── qat_int8.py              # ★ W8A8 INT8 量化感知训练 (STE) + 真 torch.int8 打包推理层 (491MB -> 128MB)
│   ├── referee.py               # 物理声学裁判 (F0 / LPC共振峰Bark距离 / 频谱质心 / 颤音调制指数)
│   ├── lexicon_builder.py       # 汉火上古音 13,076 字单核词表构建 + 《诗经》自动叶韵
│   └── async_checkpoint.py      # 异步非阻塞 Checkpoint 管理器
├── scripts/
│   ├── prepare_v5_colab_bundle.py    # 构建三代联合词表 (51,735条 0 OOV) 与全频带 24kHz 精切训练集 (632条)
│   ├── colab_train_v5_qat.py         # ★ 三代球面精调 + W8A8 INT8 QAT + FP32/INT8 男女双音色四模式合成
│   └── export_web_onnx.py            # ★ 导出 WebGPU (FP32 零控制流) + WASM (QAT-INT8) 单文件 ONNX 与前端词表包
├── web_wasm/                         # ★ 纯浏览器端 WebGPU + WASM 离线推理工作台
│   ├── index.html                    # 古风交互工作台 (支持四模式古音、自定义干声上传、逐字 IPA 流形检视、24kHz WAV 下载)
│   ├── hanfire_worker.mjs            # Web Worker 调度器：优先探测 WebGPU，失败/无 GPU 自动无缝回退多核 WASM SIMD
│   ├── hanfire_dsp.mjs               # 纯 JS Radix-2 1024-pt FFT/ISTFT、24kHz Vocos Mel 提取器与一字一拍帧分配器
│   ├── serve_web.py                  # 内置 COOP/COEP (`SharedArrayBuffer` 多线程 WASM) 本地 HTTP 服务
│   ├── models/                       # WebGPU FP32 套件 + WASM QAT-INT8 轻量套件 (145.7 MB 总计)
│   └── assets/                       # 51,735 字预编译词表 (`lexicon_3era.json.gz` 0.61MB) + 双音色锚 (`voice_prompts.json`)
├── v5_bundle/                        # 三代联合词表、训练集与 24kHz 纯净男女参考锚 (128Hz 男 / 252Hz 女)
└── v5_outputs/                       # 训练产出的最优权重 (best_fp32.pt, best_int8.pt) 与 40 首展示音频
```

## 2. 浏览器 `WebGPU + WASM` 纯前端推理快速启动

无需任何后端推理接口，启动带 `COOP/COEP` 头的静态文件服务即可激活多核 `SharedArrayBuffer` WASM 与 `WebGPU`：

```bash
# 1. 启动本地 Web 服务 (默认监听 http://127.0.0.1:8080/)
python3 /mnt/s790/project/tts/hanfire_tts/web_wasm/serve_web.py --port 8080

# 2. (可选) 若重新训练了权重，一键重新导出 ONNX 双套件并执行 PyTorch 数值对齐门禁测试：
/mnt/s790/project/hanzi_train/.venv/bin/python \
  /mnt/s790/project/tts/hanfire_tts/scripts/export_web_onnx.py
```

### 浏览器端双栖架构要点
1. **运行时选型 (`onnxruntime-web`)**：
   - **WebGPU 优先**：检测到浏览器支持 `navigator.gpu` 时，自动加载零控制流算子（剔除 `If`/`Loop`/`Atan`/`Complex64`，静态化 `WebGPUStaticRelPosEncoding`）的原生图 (`text_encoder.onnx` + `fm_decoder.onnx` + `vocos_spec.onnx`)，与 PyTorch 波形余弦相似度达 **`1.000000`**。
   - **WASM CPU 无缝回退**：当浏览器未开启 WebGPU、无独显或编译受限时，Worker 自动降级至 `SharedArrayBuffer` 多线程 `WASM SIMD` 后端，并切换加载由 Stage-2 QAT 权重导出的每通道真 `INT8` 套件 (`text_encoder_int8.onnx` `6.1MB` + `fm_decoder_int8.onnx` `124.7MB` + `vocos_spec_int8.onnx` `14.9MB`，总计仅 **`145.7 MB`**)。
2. **绕过 `Complex64` / `ISTFT` 算子壁垒**：
   - 将 `Vocos` 拆分为纯实数 `vocos_spec.onnx` (`mel -> spec_real, spec_imag`) + 前端 `hanfire_dsp.mjs` 中的零依赖 Radix-2 `1024` 点 `istftVocos1024` 重叠相加重建，彻底规避 WebGPU / WASM 对复数张量的不支持。

## 3. 本机 GPU 训练与真 `torch.int8` 权重加载

```bash
# 本机 GPU 全量三代精调 + W8A8 QAT 训练
/mnt/s790/project/hanzi_train/.venv/bin/python -u \
  /mnt/s790/project/tts/hanfire_tts/scripts/colab_train_v5_qat.py \
  --bundle-dir /mnt/s790/project/tts/hanfire_tts/v5_bundle \
  --out-dir /mnt/s790/project/tts/hanfire_tts/v5_outputs
```

```python
from zipvoice.models.zipvoice import ZipVoice
from hanfire_core.qat_int8 import load_int8_checkpoint

int8_model, t2i = load_int8_checkpoint("v5_outputs/best_int8.pt", ZipVoice, device="cuda")
```
