#!/usr/bin/env bash
# 一键将 release_artifacts/ 中的汉火拟音 (Hanfire Niyin v1.0) 训练好模型发布到 GitHub Release
# 用法: bash scripts/publish_release.sh [repo] [tag]
set -euo pipefail

REPO="${1:-hanhuo-org/niyin}"
TAG="${2:-v1.0}"
DIR="$(cd "$(dirname "$0")/.." && pwd)/release_artifacts"

REQUIRED=(
  "niyin-tts-v10-web-models.tar.gz"
  "niyin_tts_v10_studio_fp32.pt"
  "niyin_tts_v10_studio_int8.pt"
  "niyin_tts_v10_onnx_int8.tar.gz"
  "niyin_tts_lexicon_and_refs.tar.gz"
  "niyin_sing_female_base_40k.tar.gz"
  "niyin_sing_male_m4_40k.tar.gz"
  "niyin_sing_vocoder_pc_nsf_hifigan_44k.tar.gz"
  "niyin_sing_tang_bodies.tar.gz"
  "SHA256SUMS.txt"
)

ARGS=()
for f in "${REQUIRED[@]}"; do
  if [ ! -f "$DIR/$f" ]; then
    echo "缺少发布文件: $DIR/$f" >&2
    exit 1
  fi
  ARGS+=("$DIR/$f")
done

echo "Publishing GitHub Release $TAG to $REPO ..."
gh release create "$TAG" "${ARGS[@]}" \
  --repo "$REPO" \
  --title "Hanfire Niyin v1.0 (Three-Era Ancient TTS V10 + Tang-Diao Chanting V16)" \
  --notes "Hanfire Niyin (汉火拟音) v1.0 Release:
- **Three-Era Historical Chinese TTS (V10 Studio)**: Old Chinese (上古), Middle Chinese (中古切韵), Early Mandarin (近古中原音韵) ZipVoice flow-matching checkpoints (FP32 + true QAT INT8) & browser-ready WebGPU/WASM INT8 chunked kit (168 MB).
- **Historical Chinese Chanting & Singing (V16)**: 44.1kHz Female (OpenCpop 40k) & Multi-Speaker Male (M4Singer 6-speaker 40k) DiffSinger acoustic bases + PC-NSF-HiFiGAN 2025.02 vocoder + Tang-diao live-F0 reference bodies covering Shijing V9.6 and Classical Prose/Fu V16.
- **Hugging Face Mirrors**: Models \`https://huggingface.co/hanhuo-org/hanhuo-niyin-models\` | Training Corpus \`https://huggingface.co/datasets/hanhuo-org/hanhuo-niyin-dataset\`"
