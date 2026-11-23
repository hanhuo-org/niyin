#!/usr/bin/env bash
# ============================================================================
# 汉火古音 TTS —— 免每次授权与秒级还原的 Colab GPU 远程控制助手 (colab_ctl.sh)
# 借鉴 colab-gpu-studio 思路：环境/底座权重/数据包全部离线固化，GPU 开机 10 秒开训！
#
# 用法：
#   bash scripts/colab_ctl.sh build-cache       # 0. 本地一次性打包离线 wheels + HF 底座缓存 (~530MB)
#   bash scripts/colab_ctl.sh init [A100|L4]    # 1. 创建 GPU 会话 + 上传并离线还原环境/权重/数据包（零外网安装）
#   bash scripts/colab_ctl.sh train             # 2. 运行训练（支持 TRAIN_SCRIPT / TRAIN_ARGS 自定义）
#   bash scripts/colab_ctl.sh status            # 3. 查看 Colab 会话状态与最新训练日志
#   bash scripts/colab_ctl.sh pull              # 4. 拉回权重与音频到本地
#   bash scripts/colab_ctl.sh stop              # 5. 立即关闭 GPU 会话（零闲置计费）
# ============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PIP="${PIP_BIN:-pip}"
CACHE_TAR="/tmp/hanfire_colab_env_cache.tar.gz"
SESSION="${COLAB_SESSION:-hanfire-v10}"
GPU_TYPE="${2:-A100}"
BUNDLE_NAME="${BUNDLE_NAME:-v10_bundle}"
BUNDLE_TAR="/tmp/hanfire_${BUNDLE_NAME}.tar.gz"

# 直连 Google Colab 控制面（不走本地 HTTP 代理），使用 ADC 免交互鉴权
COLAB_CMD=(env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY colab --auth=adc)

cmd="${1:-help}"

case "$cmd" in
  build-cache)
    echo "[Cache 1/2] 离线下载轻量纯 Python 依赖 wheels (不覆盖 Colab 原生 PyTorch/CUDA)..."
    STAGE="/tmp/hanfire_env_stage"
    rm -rf "$STAGE" && mkdir -p "$STAGE/wheels" "$STAGE/hf_hub"
    "$VENV_PIP" download --only-binary=:all: --no-deps -d "$STAGE/wheels" \
      vocos einops pypinyin soundfile pyyaml huggingface_hub opencc-python-reimplemented
    echo "[Cache 2/2] 打包 HuggingFace 底座权重 (ZipVoice + Vocos 24kHz)..."
    HF_HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
    for repo in models--k2-fsa--ZipVoice models--charactr--vocos-mel-24khz; do
      if [ -d "$HF_HUB/$repo" ]; then
        cp -rL "$HF_HUB/$repo" "$STAGE/hf_hub/$repo"
      fi
    done
    tar czf "$CACHE_TAR" -C "$STAGE" wheels hf_hub
    rm -rf "$STAGE"
    ls -lh "$CACHE_TAR"
    echo "✅ 离线环境固化包已生成: $CACHE_TAR"
    ;;

  init)
    if [ ! -f "$CACHE_TAR" ]; then
      bash "$0" build-cache
    fi
    echo "[1/4] 创建 Colab GPU 会话: session=${SESSION}, gpu=${GPU_TYPE} ..."
    "${COLAB_CMD[@]}" new -s "$SESSION" --gpu "$GPU_TYPE" --high-mem \
      || "${COLAB_CMD[@]}" new -s "$SESSION" --gpu "$GPU_TYPE" \
      || "${COLAB_CMD[@]}" new -s "$SESSION" --gpu L4

    echo "[2/4] 打包最新代码与数据包 (${BUNDLE_NAME})..."
    tar czf /tmp/hanfire_code.tar.gz -C "$PROJECT_DIR" hanfire_core scripts vendor tools/qieyun/mc_qieyun_unified.tsv
    tar czf "$BUNDLE_TAR" -C "$PROJECT_DIR" "$BUNDLE_NAME"

    echo "[3/4] 直传固化环境包 + 代码 + 数据包至 Colab VM (零 PyPI / 零 HF 下载 / 零 Drive 弹窗)..."
    "${COLAB_CMD[@]}" upload -s "$SESSION" "$CACHE_TAR" /content/hanfire_colab_env_cache.tar.gz
    "${COLAB_CMD[@]}" upload -s "$SESSION" /tmp/hanfire_code.tar.gz /content/hanfire_code.tar.gz
    "${COLAB_CMD[@]}" upload -s "$SESSION" "$BUNDLE_TAR" /content/hanfire_bundle.tar.gz

    echo "[4/4] 离线秒级还原环境与 HuggingFace 缓存..."
    cat << 'PYEOF' | "${COLAB_CMD[@]}" exec -s "$SESSION" --timeout 180
import subprocess, sys, os
subprocess.run(
    "mkdir -p /content/hanfire_tts /content/env_cache /root/.cache/huggingface/hub && "
    "tar xzf /content/hanfire_colab_env_cache.tar.gz -C /content/env_cache && "
    "cp -rn /content/env_cache/hf_hub/* /root/.cache/huggingface/hub/ && "
    "tar xzf /content/hanfire_code.tar.gz -C /content/hanfire_tts && "
    "tar xzf /content/hanfire_bundle.tar.gz -C /content/hanfire_tts",
    shell=True, check=True,
)
subprocess.run(
    f"{sys.executable} -m pip install -q --no-index --no-deps --find-links=/content/env_cache/wheels "
    "vocos einops pypinyin soundfile opencc-python-reimplemented",
    shell=True, check=True,
)
print("✅ 离线环境与底座权重还原完成 (HF_HUB_OFFLINE=1 就绪)")
PYEOF
    ;;

  train)
    STAGE_A_EP="${STAGE_A_EPOCHS:-15}"
    STAGE_B_EP="${STAGE_B_EPOCHS:-20}"
    BATCH_SZ="${BATCH_SIZE:-32}"
    echo "[Train] 启动 V10 A100 后台训练 (StageA=${STAGE_A_EP}, StageB=${STAGE_B_EP}, bs=${BATCH_SZ})..."
    cat << PYEOF | "${COLAB_CMD[@]}" exec -s "$SESSION" --timeout 60
import subprocess, sys, os
cmd = (
    f"nohup env HF_HUB_OFFLINE=1 {sys.executable} -u /content/hanfire_tts/scripts/train_v10_a100.py "
    "--bundle-dir /content/hanfire_tts/${BUNDLE_NAME} "
    "--out-dir /content/v10_outputs "
    "--stage-a-epochs ${STAGE_A_EP} "
    "--stage-b-epochs ${STAGE_B_EP} "
    "--batch-size ${BATCH_SZ} "
    "> /content/train.log 2>&1 &"
)
subprocess.run(cmd, shell=True, check=True)
print("🚀 后台训练已启动！日志: /content/train.log")
PYEOF
    ;;

  status)
    "${COLAB_CMD[@]}" status
    cat << 'PYEOF' | "${COLAB_CMD[@]}" exec -s "$SESSION" --timeout 60
from pathlib import Path
for p_str in ("/content/train.log", "/content/v5_train.log"):
    log_p = Path(p_str)
    if log_p.exists():
        lines = log_p.read_text(encoding="utf-8", errors="replace").splitlines()
        print(f"=== {p_str} 最近 25 行 ===")
        print("\n".join(lines[-25:]))
PYEOF
    ;;

  pull)
    OUT_LOCAL="$PROJECT_DIR/v10_outputs"
    mkdir -p "$OUT_LOCAL"
    "${COLAB_CMD[@]}" download -s "$SESSION" /content/v10_outputs/best_fp32_v10.pt "$OUT_LOCAL/best_fp32_v10.pt"
    echo "✅ 权重已拉回: $OUT_LOCAL/best_fp32_v10.pt"
    ;;

  stop)
    echo "[Stop] 正在安全停止 Colab 会话 ${SESSION} ..."
    "${COLAB_CMD[@]}" stop -s "$SESSION"
    ;;

  *)
    echo "用法: bash scripts/colab_ctl.sh {build-cache|init|train|status|pull|stop} [GPU_TYPE]"
    ;;
esac
