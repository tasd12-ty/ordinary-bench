#!/bin/bash
# Qwen3.5-27B + ms-swift SFT 训练
# 用法:
#   bash training/swift/run_sft.sh
#   bash training/swift/run_sft.sh --model Qwen/Qwen3.5-27B --gpus 8
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
N_GPUS="${N_GPUS:-8}"
TRAIN_FILE="${TRAIN_FILE:-$PROJECT_DIR/training/swift/data/sft_train.jsonl}"
VAL_FILE="${VAL_FILE:-$PROJECT_DIR/training/swift/data/sft_test.jsonl}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --gpus) N_GPUS="$2"; shift 2 ;;
        --train-file) TRAIN_FILE="$2"; shift 2 ;;
        --val-file) VAL_FILE="$2"; shift 2 ;;
        --lora-rank) LORA_RANK="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [ ! -f "$TRAIN_FILE" ]; then
    echo "错误: 训练数据不存在: $TRAIN_FILE"
    echo "请先运行: python training/swift/prepare_swift_data.py --mode sft --data-dir data-gen/output"
    exit 1
fi

echo "=== Qwen3.5-27B / ms-swift SFT ==="
echo "模型:      $MODEL"
echo "GPU:       $N_GPUS"
echo "LoRA:      rank=$LORA_RANK alpha=$LORA_ALPHA"
echo "训练数据:  $TRAIN_FILE"
echo ""

NPROC_PER_NODE="$N_GPUS" swift sft \
    --model "$MODEL" \
    --tuner_type lora \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --target_modules all-linear \
    --dataset "$TRAIN_FILE" \
    --val_dataset "$VAL_FILE" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --max_length 2048 \
    --warmup_ratio 0.05 \
    --save_strategy epoch \
    --eval_strategy epoch \
    --logging_steps 10 \
    --enable_thinking false \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --bf16 true \
    --deepspeed zero2 \
    --gradient_checkpointing true \
    --output_dir "$PROJECT_DIR/output/swift_sft_qwen35_27b"
