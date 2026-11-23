#!/usr/bin/env bash
# 从 Colab GPU 实时查看男声底座 (hanfire_sing_male) 训练进度、热拉取权重到本地并自动完整性校验与关机停计费
# 用法：
#   bash hanfire_sing/scripts/pull_male_ckpt.sh status
#   bash hanfire_sing/scripts/pull_male_ckpt.sh pull
#   bash hanfire_sing/scripts/pull_male_ckpt.sh pull-and-stop
set -euo pipefail

SESSION="${SING_MALE_SESSION:-hanfire-sing-male}"
LOCAL_CKPT_DIR="${NIYIN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}/hanfire_sing/checkpoints/hanfire_sing_male"
COLAB=(env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY BROWSER=/bin/true colab --auth=adc)
mkdir -p "$LOCAL_CKPT_DIR"

cmd="${1:-status}"
case "$cmd" in
  status)
    "${COLAB[@]}" exec -s "$SESSION" --timeout 30 <<'PY'
import glob, os, subprocess
print("GPU:", subprocess.getoutput("nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader"))
ckpts = sorted(glob.glob("/content/work/DiffSinger/checkpoints/hanfire_sing_male/*.ckpt"))
print("Checkpoints:", [(os.path.basename(c), round(os.path.getsize(c)/1e6, 1)) for c in ckpts])
for logp in ("/content/work/train_male.log", "/content/work/prep_male.log"):
    if os.path.exists(logp):
        lines = open(logp, errors="ignore").read().splitlines()
        print(f"--- Tail {os.path.basename(logp)} ---")
        print("\n".join(lines[-20:]))
        break
PY
    ;;

  pull|pull-and-stop)
    LATEST=$("${COLAB[@]}" exec -s "$SESSION" --timeout 30 <<'PY'
import glob, os
ckpts = sorted(glob.glob("/content/work/DiffSinger/checkpoints/hanfire_sing_male/model_ckpt_steps_*.ckpt"),
               key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]))
print(ckpts[-1] if ckpts else "")
PY
    )
    LATEST=$(echo "$LATEST" | tr -d '\r' | tail -n 1)
    if [ -z "$LATEST" ]; then
      echo "❌ 尚未生成 model_ckpt_steps_*.ckpt，请先用 status 确认训练进度。"
      exit 1
    fi
    BN=$(basename "$LATEST")
    echo ">>> 拉取男声底座权重: $BN -> ${LOCAL_CKPT_DIR}/${BN}"
    "${COLAB[@]}" download -s "$SESSION" "$LATEST" "${LOCAL_CKPT_DIR}/${BN}"
    "${COLAB[@]}" download -s "$SESSION" "/content/work/DiffSinger/checkpoints/hanfire_sing_male/config.yaml" "${LOCAL_CKPT_DIR}/config.yaml" || true
    "${COLAB[@]}" download -s "$SESSION" "/content/work/binary_m4_male/dictionary.txt" "${LOCAL_CKPT_DIR}/dictionary.txt" || true
    "${COLAB[@]}" download -s "$SESSION" "/content/work/binary_m4_male/spk_map.json" "${LOCAL_CKPT_DIR}/spk_map.json" || true
    # 自动建软链到 DiffSinger/checkpoints/hanfire_sing_male
    ln -sfn "$LOCAL_CKPT_DIR" "${NIYIN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}/hanfire_sing/DiffSinger/checkpoints/hanfire_sing_male"

    ${PYTHON_BIN:-python3} - "${LOCAL_CKPT_DIR}" "${BN}" <<'PY'
import sys, torch
from pathlib import Path
d, bn = Path(sys.argv[1]), sys.argv[2]
p = d / bn
sz = p.stat().st_size
assert sz > 50 * 1024 * 1024, f"权重体积异常 ({sz} bytes)，禁止关机！"
ck = torch.load(p, map_location="cpu", weights_only=False)
sd = ck.get("state_dict", ck)
assert len(sd) > 50, f"state_dict 键数异常 ({len(sd)})，禁止关机！"
step = ck.get("global_step", "?")
print(f"✅ 男声底座权重完整性校验通过: {bn} ({sz/1e6:.1f} MB, {len(sd)} tensors, step={step})")
PY
    if [ "$cmd" = "pull-and-stop" ]; then
      echo ">>> 男声底座权重已安全落盘 SSD，关闭 Colab GPU 停止计费！"
      "${COLAB[@]}" stop -s "$SESSION"
    fi
    ;;

  *)
    sed -n '2,6p' "$0"
    ;;
esac
