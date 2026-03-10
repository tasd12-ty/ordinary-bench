#!/bin/bash
# ORDINARY-BENCH GRPO 训练启动脚本
# 用法:
#   bash training/run_grpo.sh --model Qwen/Qwen3-VL-32B --mode lora --gpus 8
#   bash training/run_grpo.sh --model Qwen/Qwen3.5-VL-27B --mode full --gpus 8
set -e

# 默认值
MODEL="Qwen/Qwen3-VL-32B"
MODE="lora"
N_GPUS=8

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --mode)  MODE="$2"; shift 2 ;;
        --gpus)  N_GPUS="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$SCRIPT_DIR/configs/grpo_${MODE}.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    echo "可选 mode: lora, full"
    exit 1
fi

# 检查数据是否存在
if [ ! -f "$PROJECT_DIR/VLM-test/verl_data/train.parquet" ]; then
    echo "错误: RL 训练数据不存在，请先运行数据准备脚本"
    echo "  bash training/prepare_data.sh"
    exit 1
fi

echo "=== ORDINARY-BENCH GRPO 训练 ==="
echo "模型:    $MODEL"
echo "模式:    $MODE"
echo "GPU:     $N_GPUS"
echo "配置:    $CONFIG_FILE"
echo "奖励函数: verl_reward.py (软评分)"
echo ""

cd "$PROJECT_DIR"

export MODEL_PATH="$MODEL"
export N_GPUS="$N_GPUS"
export TRAIN_FILE="$PROJECT_DIR/VLM-test/verl_data/train.parquet"
export VAL_FILE="$PROJECT_DIR/VLM-test/verl_data/test.parquet"
export REWARD_PATH="$PROJECT_DIR/VLM-test/verl_reward.py"

python3 -m verl.trainer.main_ppo \
    --config-path "$SCRIPT_DIR/configs" \
    --config-name "grpo_${MODE}" \
    actor_rollout_ref.model.path="$MODEL" \
    trainer.n_gpus_per_node="$N_GPUS" \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    custom_reward_function.path="$REWARD_PATH"
