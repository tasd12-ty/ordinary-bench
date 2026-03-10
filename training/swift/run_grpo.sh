#!/bin/bash
# Qwen3.5-27B + ms-swift GRPO 训练
# 用法:
#   bash training/swift/run_grpo.sh
#   bash training/swift/run_grpo.sh --model Qwen/Qwen3.5-27B --gpus 8
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
N_GPUS="${N_GPUS:-8}"
TRAIN_FILE="${TRAIN_FILE:-$PROJECT_DIR/prepared_data/swift/grpo/train.jsonl}"
VAL_FILE="${VAL_FILE:-$PROJECT_DIR/prepared_data/swift/grpo/test.jsonl}"
PLUGIN_PATH="${PLUGIN_PATH:-$SCRIPT_DIR/plugins/ordinary_bench_reward.py}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --gpus) N_GPUS="$2"; shift 2 ;;
        --train-file) TRAIN_FILE="$2"; shift 2 ;;
        --val-file) VAL_FILE="$2"; shift 2 ;;
        --num-generations) NUM_GENERATIONS="$2"; shift 2 ;;
        --lora-rank) LORA_RANK="$2"; shift 2 ;;
        --full) LORA_RANK=0; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [ ! -f "$TRAIN_FILE" ]; then
    echo "错误: 训练数据不存在: $TRAIN_FILE"
    echo "请先运行: python training/swift/prepare_swift_data.py --data-dir data-gen/output"
    exit 1
fi

echo "=== Qwen3.5-27B / ms-swift GRPO ==="
echo "模型:        $MODEL"
echo "GPU:         $N_GPUS"
echo "LoRA:        rank=$LORA_RANK alpha=$LORA_ALPHA"
echo "generations: $NUM_GENERATIONS"
echo "训练数据:    $TRAIN_FILE"
echo "奖励插件:    $PLUGIN_PATH"
echo ""

# LoRA 或全参数
TUNER_ARGS=()
if [ "$LORA_RANK" -gt 0 ]; then
    TUNER_ARGS=(
        --tuner_type lora
        --lora_rank "$LORA_RANK"
        --lora_alpha "$LORA_ALPHA"
        --target_modules all-linear
    )
fi

NPROC_PER_NODE="$N_GPUS" swift rlhf \
    --rlhf_type grpo \
    --model "$MODEL" \
    "${TUNER_ARGS[@]}" \
    --external_plugins "$PLUGIN_PATH" \
    --reward_funcs ordinary_bench_accuracy ordinary_bench_format \
    --dataset "$TRAIN_FILE" \
    --val_dataset "$VAL_FILE" \
    --num_generations "$NUM_GENERATIONS" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-6 \
    --num_train_epochs 5 \
    --max_completion_length 128 \
    --max_prompt_length 1536 \
    --epsilon 0.2 \
    --temperature 0.7 \
    --top_p 0.95 \
    --max_grad_norm 0.5 \
    --logging_steps 5 \
    --save_strategy steps \
    --save_steps 50 \
    --use_vllm true \
    --vllm_mode colocate \
    --enable_thinking false \
    --bf16 true \
    --deepspeed zero2 \
    --gradient_checkpointing true \
    --output_dir "$PROJECT_DIR/output/swift_grpo_qwen35_27b"
