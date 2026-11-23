#!/usr/bin/env bash
# 从 Colab L4 实时查看训练进度、拉取最新 checkpoint 到本地 RTX 2080 Ti，或收工关机停计费
# 用法：
#   bash hanfire_sing/scripts/pull_colab_ckpt.sh status        # 查看当前步数 / loss / 已有 ckpt
#   bash hanfire_sing/scripts/pull_colab_ckpt.sh pull          # 拉取最新 ckpt + config 到本地 hanfire_sing/checkpoints/
#   bash hanfire_sing/scripts/pull_colab_ckpt.sh pull-and-stop # 拉取最新 ckpt 后立即关闭云端 GPU 停止计费！
set -euo pipefail

SESSION="${SING_GPU_SESSION:-hanfire-sing-gpu}"
LOCAL_CKPT_DIR="${NIYIN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}/hanfire_sing/checkpoints/hanfire_sing_base"
COLAB=(env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY colab --auth=adc)
mkdir -p "$LOCAL_CKPT_DIR"

cmd="${1:-status}"
case "$cmd" in
  status)
    "${COLAB[@]}" exec -s "$SESSION" --timeout 30 <<'PY'
import glob, os, subprocess
print("GPU:", subprocess.getoutput("nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader"))
ckpts = sorted(glob.glob("/content/work/DiffSinger/checkpoints/hanfire_sing_base/*.ckpt"))
print("Checkpoints:", [(os.path.basename(c), round(os.path.getsize(c)/1e6, 1)) for c in ckpts])
if os.path.exists("/content/work/train_base.log"):
    lines = open("/content/work/train_base.log", errors="ignore").read().splitlines()
    print("--- Tail train_base.log ---")
    print("\n".join(lines[-20:]))
PY
    ;;

  pull-file|pull-file-and-stop)
    # 纯 HTTP 拉取指定 ckpt（绝不占用正在前台训练的 Jupyter 内核！）
    BN="${2:-}"
    VAL_LOSS="${3:-}"
    if [ -z "$BN" ]; then
      echo "用法: $0 pull-file <model_ckpt_steps_XXXX.ckpt> [val_loss]"
      exit 1
    fi
    REMOTE="/content/work/DiffSinger/checkpoints/hanfire_sing_base/${BN}"
    echo ">>> HTTP 热拉取权重: ${BN} (val_loss=${VAL_LOSS:-N/A}) -> ${LOCAL_CKPT_DIR}/${BN}"
    "${COLAB[@]}" download -s "$SESSION" "$REMOTE" "${LOCAL_CKPT_DIR}/${BN}"
    "${COLAB[@]}" download -s "$SESSION" "/content/work/DiffSinger/checkpoints/hanfire_sing_base/config.yaml" "${LOCAL_CKPT_DIR}/config.yaml" || true
    ${PYTHON_BIN:-python3} - "${LOCAL_CKPT_DIR}" "${BN}" "${VAL_LOSS}" <<'PY'
import json, sys, torch
from pathlib import Path
d, bn, val_s = Path(sys.argv[1]), sys.argv[2], sys.argv[3].strip()
p = d / bn
sz = p.stat().st_size
assert sz > 50 * 1024 * 1024, f"权重体积异常 ({sz} bytes)，禁止关机！"
ck = torch.load(p, map_location="cpu", weights_only=False)
sd = ck.get("state_dict", ck)
assert len(sd) > 50, f"state_dict 键数异常 ({len(sd)})，禁止关机！"
step = ck.get("global_step", "?")
val_loss = float(val_s) if val_s else float("inf")
print(f"✅ 本地权重完整性校验通过: {bn} ({sz/1e6:.1f} MB, {len(sd)} tensors, step={step}, val_loss={val_s or 'N/A'})")
meta_p = d / "ckpt_keep_meta.json"
meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
prev_best_loss = float(meta.get("best_val_loss", float("inf")))
prev_best_file = meta.get("best_file", "")
if (not prev_best_file) or (not (d / prev_best_file).exists()) or (val_loss <= prev_best_loss):
    meta["best_file"] = bn
    meta["best_val_loss"] = val_loss if val_loss < float("inf") else prev_best_loss
    meta["best_step"] = step
meta["latest_file"] = bn
meta["latest_val_loss"] = val_loss if val_loss < float("inf") else None
meta["latest_step"] = step
meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
keep = {meta["best_file"], meta["latest_file"]}
for f in sorted(d.glob("model_ckpt_steps_*.ckpt")):
    if f.name not in keep:
        f.unlink()
        print(f"🧹 已清理旧中间权重: {f.name}")
print(f"📦 本地保留 ({len(keep)} 份): best={meta['best_file']} (val_loss={meta.get('best_val_loss')}) | latest={meta['latest_file']}")
PY
    if [ "$cmd" = "pull-file-and-stop" ]; then
      echo ">>> 权重已通过本地完整性校验并安全落盘 SSD，现在关闭 Colab GPU 会话停止计费！"
      "${COLAB[@]}" stop -s "$SESSION"
    fi
    ;;

  pull|pull-and-stop)
    LATEST=$("${COLAB[@]}" exec -s "$SESSION" --timeout 30 <<'PY'
import glob, os
ckpts = sorted(glob.glob("/content/work/DiffSinger/checkpoints/hanfire_sing_base/model_ckpt_steps_*.ckpt"),
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
    echo ">>> 拉取最新底座权重: $BN -> ${LOCAL_CKPT_DIR}/${BN}"
    "${COLAB[@]}" download -s "$SESSION" "$LATEST" "${LOCAL_CKPT_DIR}/${BN}"
    "${COLAB[@]}" download -s "$SESSION" "/content/work/DiffSinger/checkpoints/hanfire_sing_base/config.yaml" "${LOCAL_CKPT_DIR}/config.yaml" || true
    "${COLAB[@]}" download -s "$SESSION" "/content/work/binary_opencpop/dictionary.txt" "${LOCAL_CKPT_DIR}/dictionary.txt" || true
    "${COLAB[@]}" download -s "$SESSION" "/content/work/binary_opencpop/spk_map.json" "${LOCAL_CKPT_DIR}/spk_map.json" || true
    # 同步读取云端当前 val_loss（用于自动评判并保留「最好 + 最新」两份权重！）
    VAL_LOSS=$("${COLAB[@]}" exec -s "$SESSION" --timeout 30 <<'PY'
import os, re
v = ""
if os.path.exists("/content/work/train_base.log"):
    ms = re.findall(r"val_loss=([0-9.]+)", open("/content/work/train_base.log", errors="ignore").read())
    if ms:
        v = ms[-1]
print(v)
PY
    )
    VAL_LOSS=$(echo "$VAL_LOSS" | tr -d '\r' | tail -n 1)
    ls -lh "${LOCAL_CKPT_DIR}"
    # 强校验 + 双保留策略：永远保留「val_loss 最低的 best」和「最新的 latest」两份，清理其余旧检查点！
    ${PYTHON_BIN:-python3} - "${LOCAL_CKPT_DIR}" "${BN}" "${VAL_LOSS}" <<'PY'
import json, os, sys, torch
from pathlib import Path

d = Path(sys.argv[1])
bn = sys.argv[2]
val_s = sys.argv[3].strip()
p = d / bn

sz = p.stat().st_size
assert sz > 50 * 1024 * 1024, f"权重体积异常 ({sz} bytes)，禁止关机！"
ck = torch.load(p, map_location="cpu", weights_only=False)
sd = ck.get("state_dict", ck)
assert len(sd) > 50, f"state_dict 键数异常 ({len(sd)})，禁止关机！"
step = ck.get("global_step", "?")
val_loss = float(val_s) if val_s else float("inf")
print(f"✅ 本地权重完整性校验通过: {bn} ({sz/1e6:.1f} MB, {len(sd)} tensors, step={step}, val_loss={val_s or 'N/A'})")

meta_p = d / "ckpt_keep_meta.json"
meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
prev_best_loss = float(meta.get("best_val_loss", float("inf")))
prev_best_file = meta.get("best_file", "")

if (not prev_best_file) or (not (d / prev_best_file).exists()) or (val_loss <= prev_best_loss):
    meta["best_file"] = bn
    meta["best_val_loss"] = val_loss if val_loss < float("inf") else prev_best_loss
    meta["best_step"] = step
meta["latest_file"] = bn
meta["latest_val_loss"] = val_loss if val_loss < float("inf") else None
meta["latest_step"] = step
meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

keep = {meta["best_file"], meta["latest_file"]}
for f in sorted(d.glob("model_ckpt_steps_*.ckpt")):
    if f.name not in keep:
        f.unlink()
        print(f"🧹 已清理旧中间权重: {f.name}")
print(f"📦 当前本地保留 ({len(keep)} 份): best={meta['best_file']} (val_loss={meta.get('best_val_loss')}) | latest={meta['latest_file']}")
PY
    if [ "$cmd" = "pull-and-stop" ]; then
      echo ">>> 权重已通过本地完整性校验并安全落盘 SSD，现在关闭 Colab GPU 会话停止计费！"
      "${COLAB[@]}" stop -s "$SESSION"
    fi
    ;;

  *)
    sed -n '2,6p' "$0"
    ;;
esac
