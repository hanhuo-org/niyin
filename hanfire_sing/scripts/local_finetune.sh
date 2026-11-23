#!/usr/bin/env bash
# ============================================================================
# 汉火古音唱吟 · 本地 RTX 2080 Ti (22GB) 一键微调启动脚本
# 前提：云端 Colab 底座已通过 pull_colab_ckpt.sh 拉回至 hanfire_sing/checkpoints/hanfire_sing_base/
# 用法：
#   bash hanfire_sing/scripts/local_finetune.sh           # 全自动备料+二进制化+本地微调 4000 步 (~15分钟)
#   bash hanfire_sing/scripts/local_finetune.sh prep-only # 仅做本地 44.1kHz 对齐与二进制化（不启动训练）
# ============================================================================
set -euo pipefail

VENV_PY="${PYTHON_BIN:-python3}"
SING_DIR="${NIYIN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}/hanfire_sing"
DS_DIR="${SING_DIR}/DiffSinger"
CFG="${DS_DIR}/configs/hanfire_local_finetune.yaml"

echo "[1/3] 准备本地 tier1_studio_clean 唐调 44.1kHz 微调语料与配置..."
"$VENV_PY" "${SING_DIR}/scripts/prepare_local_finetune.py"

echo "[2/3] 在本地执行 DiffSinger binarize.py (44.1kHz 128-mel + Parselmouth F0)..."
(cd "$DS_DIR" && "$VENV_PY" scripts/binarize.py --config "$CFG")

if [ "${1:-}" = "prep-only" ]; then
  echo "✅ 本地微调语料与二进制特征已备好！待云端底座权重拉回后直接运行本脚本即可开训。"
  exit 0
fi

# 检查云端底座权重是否存在
BASE_CKPT=$(ls -1v "${SING_DIR}/checkpoints/hanfire_sing_base"/model_ckpt_steps_*.ckpt 2>/dev/null | tail -n 1 || true)
if [ -z "$BASE_CKPT" ]; then
  echo "❌ 未在 ${SING_DIR}/checkpoints/hanfire_sing_base/ 找到云端底座权重，请先运行:"
  echo "   bash hanfire_sing/scripts/pull_colab_ckpt.sh pull"
  exit 1
fi

echo "[3/3] 基于云端底座 $(basename "$BASE_CKPT") 启动本地 RTX 2080 Ti FP16 微调 (4000 步)..."
# 将最新底座路径刷新进配置
"$VENV_PY" - "$CFG" "$BASE_CKPT" <<'PY'
import sys, yaml
p, ckpt = sys.argv[1], sys.argv[2]
c = yaml.safe_load(open(p))
c["finetune_ckpt_path"] = ckpt
open(p, "w").write(yaml.safe_dump(c, sort_keys=False, allow_unicode=True))
PY

cd "$DS_DIR"
exec "$VENV_PY" -u scripts/train.py \
  --config "$CFG" \
  --exp_name hanfire_tang_local \
  --reset
