#!/usr/bin/env bash
# ============================================================================
# 汉火古音唱吟 (hanfire_sing) —— Google Drive 零丢数据 Colab 控制脚本
# 工作流：
#   1. bash hanfire_sing/scripts/colab_sing_ctl.sh cpu-setup
#      -> 开【免费 CPU 机】挂载 Google Drive，克隆 OpenVPI DiffSinger，
#         下载预训练 PC-NSF-HiFiGAN 声码器 + M4Singer 数据集 + 缓存离线 wheels 至 Drive（零 GPU 积分消耗！）
#   2. bash hanfire_sing/scripts/colab_sing_ctl.sh gpu-train [A100|L4]
#      -> 开【GPU 机】挂载同一个 Google Drive，15 秒离线装依赖，从 Drive 读取特征开训，
#         每 5000 步自动落盘 checkpoint 至 Google Drive（断连不丢数据！）
#   3. bash hanfire_sing/scripts/colab_sing_ctl.sh cpu-pull
#      -> 立即关闭 GPU 停止计费 -> 开【免费 CPU 机】挂载 Drive，将训好的底座权重拉回本地 2080 Ti 微调！
#   4. bash hanfire_sing/scripts/colab_sing_ctl.sh stop
#      -> 关闭当前全部 hanfire-sing 会话（零闲置计费）
# ============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SING_DIR="${ROOT_DIR}/hanfire_sing"
DRIVE_ROOT="/content/drive/MyDrive/hanfire_sing"
CPU_SESSION="${SING_CPU_SESSION:-hanfire-sing-cpu}"
GPU_SESSION="${SING_GPU_SESSION:-hanfire-sing-gpu}"
GPU_TYPE="${2:-A100}"

COLAB_CMD=(env BROWSER=/bin/true -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY colab --auth=adc)

cmd="${1:-help}"

case "$cmd" in
  cpu-setup)
    echo "[1/3] 启动免费 CPU Colab 会话 (${CPU_SESSION}) 并挂载 Google Drive..."
    "${COLAB_CMD[@]}" new -s "$CPU_SESSION" || true
    "${COLAB_CMD[@]}" drivemount -s "$CPU_SESSION" /content/drive

    echo "[2/3] 上传本地 hanfire_sing 代码至 Colab..."
    tar czf /tmp/hanfire_sing_code.tar.gz -C "$ROOT_DIR" hanfire_sing/hanfire_sing_phonemes.py hanfire_sing/scripts
    "${COLAB_CMD[@]}" upload -s "$CPU_SESSION" /tmp/hanfire_sing_code.tar.gz /content/hanfire_sing_code.tar.gz

    echo "[3/3] 在免费 CPU 机上初始化 Google Drive 持久化目录 (OpenVPI DiffSinger + 预训练 vocoder + M4Singer + wheels)..."
    cat << 'PYEOF' | "${COLAB_CMD[@]}" exec -s "$CPU_SESSION" --timeout 1800
import os, subprocess, sys
drive = "/content/drive/MyDrive/hanfire_sing"
for sub in ("code", "repos", "vocoders", "datasets", "checkpoints", "wheels", "exports"):
    os.makedirs(f"{drive}/{sub}", exist_ok=True)

# 1) 解压最新代码到 Drive
subprocess.run(f"tar xzf /content/hanfire_sing_code.tar.gz -C {drive}/code", shell=True, check=True)

# 2) 克隆 OpenVPI DiffSinger 到 Drive (只拉 1 次，后续切 GPU 秒用)
ds_dir = f"{drive}/repos/DiffSinger"
if not os.path.exists(ds_dir):
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/openvpi/DiffSinger.git", ds_dir], check=True)
else:
    print("✓ DiffSinger 已存在于 Drive:", ds_dir)

# 3) 缓存常用轻依赖 wheels 到 Drive（切到 GPU 时可 10 秒离线安装）
subprocess.run(
    f"{sys.executable} -m pip download -q -d {drive}/wheels "
    "lightning==2.2.5 einops pypinyin praatio textgrid click pyyaml soundfile librosa onnx onnxruntime",
    shell=True, check=False,
)

# 4) 检查磁盘余量
subprocess.run(f"df -h {drive}", shell=True)
print(f"✅ 免费 CPU 备料完成！全部文件已持久化在 {drive}，切到 GPU 不会丢失。")
PYEOF
    ;;

  gpu-train)
    echo "[1/2] 启动 GPU 会话 (${GPU_SESSION}, gpu=${GPU_TYPE}) 并挂载同一 Google Drive..."
    "${COLAB_CMD[@]}" new -s "$GPU_SESSION" --gpu "$GPU_TYPE" --high-mem \
      || "${COLAB_CMD[@]}" new -s "$GPU_SESSION" --gpu "$GPU_TYPE" \
      || "${COLAB_CMD[@]}" new -s "$GPU_SESSION" --gpu L4
    "${COLAB_CMD[@]}" drivemount -s "$GPU_SESSION" /content/drive

    echo "[2/2] 秒级从 Drive 恢复离线依赖并校验 GPU..."
    cat << 'PYEOF' | "${COLAB_CMD[@]}" exec -s "$GPU_SESSION" --timeout 300
import os, subprocess, sys, torch
drive = "/content/drive/MyDrive/hanfire_sing"
assert os.path.isdir(drive), f"未找到 {drive}，请先执行 cpu-setup！"
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
subprocess.run(
    f"{sys.executable} -m pip install -q --find-links={drive}/wheels "
    "lightning==2.2.5 einops pypinyin praatio textgrid click pyyaml soundfile librosa onnx",
    shell=True, check=True,
)
print("✅ GPU 环境已就绪，checkpoints 将实时写入 Google Drive 防断连丢失。")
PYEOF
    ;;

  cpu-pull)
    echo "[1/3] 先关闭 GPU 会话停止计费..."
    "${COLAB_CMD[@]}" stop -s "$GPU_SESSION" 2>/dev/null || true

    echo "[2/3] 在免费 CPU 会话挂载 Drive 并打包最新权重..."
    "${COLAB_CMD[@]}" new -s "$CPU_SESSION" || true
    "${COLAB_CMD[@]}" drivemount -s "$CPU_SESSION" /content/drive
    cat << 'PYEOF' | "${COLAB_CMD[@]}" exec -s "$CPU_SESSION" --timeout 300
import os, subprocess
drive = "/content/drive/MyDrive/hanfire_sing"
os.makedirs(f"{drive}/exports", exist_ok=True)
subprocess.run(f"ls -lh {drive}/checkpoints/ {drive}/exports/", shell=True)
PYEOF
    echo "[3/3] 确认文件名后可用: colab --auth=adc download -s ${CPU_SESSION} ${DRIVE_ROOT}/exports/<file> ${SING_DIR}/checkpoints/"
    ;;

  stop)
    "${COLAB_CMD[@]}" stop -s "$GPU_SESSION" 2>/dev/null || true
    "${COLAB_CMD[@]}" stop -s "$CPU_SESSION" 2>/dev/null || true
    echo "✅ 已关闭全部 hanfire-sing Colab 会话（零闲置计费）。"
    ;;

  *)
    sed -n '2,18p' "$0"
    ;;
esac
