#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Qwen3-VL-32B 当前不建议使用此仓库内的 verl SFT 入口。"
echo "原因: 截至 verl v0.7.0，仓库内 GRPO 路径已按多模态数据格式整理，SFT 仍缺少稳定的 Qwen3-VL 专用接线。"
echo "建议: 使用 bash training/run_grpo.sh"
echo "参考: training/README_qwen3vl32b.md"
exit 1
