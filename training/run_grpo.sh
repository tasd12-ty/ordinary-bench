#!/bin/bash
# Qwen3-VL-32B + verl GRPO 启动脚本
# 用法:
#   bash training/run_grpo.sh --gpus 8
#   bash training/run_grpo.sh --model /path/to/Qwen3-VL-32B --config-name grpo_qwen3vl32b_lora
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"
require_python

MODEL="${MODEL_PATH:-Qwen/Qwen3-VL-32B}"
N_GPUS="${N_GPUS:-8}"
CONFIG_NAME="grpo_qwen3vl32b_lora"
TRAIN_FILE="${TRAIN_FILE:-$PROJECT_DIR/VLM-test/verl_data_qwen3vl32b/train.parquet}"
VAL_FILE="${VAL_FILE:-$PROJECT_DIR/VLM-test/verl_data_qwen3vl32b/test.parquet}"
REWARD_PATH="${REWARD_PATH:-$PROJECT_DIR/VLM-test/verl_reward.py}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --gpus) N_GPUS="$2"; shift 2 ;;
        --config-name) CONFIG_NAME="$2"; shift 2 ;;
        --train-file) TRAIN_FILE="$2"; shift 2 ;;
        --val-file) VAL_FILE="$2"; shift 2 ;;
        --reward-path) REWARD_PATH="$2"; shift 2 ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

CONFIG_FILE="$SCRIPT_DIR/configs/${CONFIG_NAME}.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$TRAIN_FILE" ]; then
    echo "错误: RL 训练数据不存在: $TRAIN_FILE"
    echo "请先运行: bash training/prepare_data.sh --data-dir $PROJECT_DIR/data-gen/output"
    exit 1
fi

echo "=== Qwen3-VL-32B / verl GRPO ==="
echo "模型:      $MODEL"
echo "GPU:       $N_GPUS"
echo "配置:      $CONFIG_FILE"
echo "训练数据:  $TRAIN_FILE"
echo "验证数据:  $VAL_FILE"
echo "奖励函数:  $REWARD_PATH"
echo ""

cd "$PROJECT_DIR"

export TOKENIZERS_PARALLELISM=false
export MODEL_PATH="$MODEL"
export N_GPUS="$N_GPUS"
export TRAIN_FILE="$TRAIN_FILE"
export VAL_FILE="$VAL_FILE"
export REWARD_PATH="$REWARD_PATH"

"$PYTHON_BIN" -m verl.trainer.main_ppo \
    --config-path "$SCRIPT_DIR/configs" \
    --config-name "$CONFIG_NAME" \
    actor_rollout_ref.model.path="$MODEL" \
    trainer.n_gpus_per_node="$N_GPUS" \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    custom_reward_function.path="$REWARD_PATH"
