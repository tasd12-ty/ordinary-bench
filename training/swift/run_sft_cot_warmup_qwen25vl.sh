#!/bin/bash
# Qwen2.5-VL-7B CoT SFT warmup：用结构化 CoT trace 教会模型新的输出协议。
#
# 与 Qwen3.5-27B 版本 (run_sft_cot_warmup.sh) 的差异：
#   - 模型：7B 参数，无需 zero3
#   - enable_thinking=false：Qwen2.5-VL 无原生 thinking 模式
#   - 学习率较高（5e-5）、步数较多（300）：较小模型从零学习 <think> 格式
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

MODEL="${MODEL:-}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-}"
N_GPUS="${N_GPUS:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
TRAIN_FILE="${TRAIN_FILE:-$PROJECT_DIR/prepared_data/swift/cot_sft/cot_sft_train.jsonl}"
VAL_FILE="${VAL_FILE:-$PROJECT_DIR/prepared_data/swift/cot_sft/cot_sft_test.jsonl}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
MAX_STEPS="${MAX_STEPS:-300}"
SAVE_STEPS="${SAVE_STEPS:-50}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output/swift_cot_sft_warmup_qwen25vl}"
DEEPSPEED_STAGE="${DEEPSPEED_STAGE:-zero2}"
BARRIER_DIR="${BARRIER_DIR:-$PROJECT_DIR/output/.swift_barrier}"

resolve_reference() {
    local input_path="$1"
    if [ -z "$input_path" ]; then
        return 0
    fi
    if [ ! -e "$input_path" ]; then
        echo "$input_path"
        return 0
    fi
    if [ -d "$input_path" ] && [[ "$(basename "$input_path")" != checkpoint-* ]]; then
        local latest_checkpoint
        latest_checkpoint=$(find "$input_path" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)
        if [ -n "$latest_checkpoint" ]; then
            echo "$latest_checkpoint"
            return 0
        fi
    fi
    echo "$input_path"
}

wait_for_nodes() {
    local barrier_name="$1"
    if [ "$NNODES" -le 1 ]; then
        return 0
    fi
    mkdir -p "$BARRIER_DIR/$barrier_name"
    local token_file="$BARRIER_DIR/$barrier_name/node_${NODE_RANK}"
    : > "$token_file"
    echo "等待多节点 barrier: $barrier_name (${NODE_RANK}/${NNODES})"
    while true; do
        local ready_count
        ready_count=$(find "$BARRIER_DIR/$barrier_name" -maxdepth 1 -type f -name 'node_*' | wc -l | tr -d ' ')
        if [ "$ready_count" -ge "$NNODES" ]; then
            break
        fi
        sleep 2
    done
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --stage1-checkpoint) STAGE1_CHECKPOINT="$2"; shift 2 ;;
        --gpus) N_GPUS="$2"; shift 2 ;;
        --nnodes) NNODES="$2"; shift 2 ;;
        --node-rank) NODE_RANK="$2"; shift 2 ;;
        --master-addr) MASTER_ADDR="$2"; shift 2 ;;
        --master-port) MASTER_PORT="$2"; shift 2 ;;
        --train-file) TRAIN_FILE="$2"; shift 2 ;;
        --val-file) VAL_FILE="$2"; shift 2 ;;
        --epochs) NUM_EPOCHS="$2"; shift 2 ;;
        --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
        --max-length) MAX_LENGTH="$2"; shift 2 ;;
        --max-steps) MAX_STEPS="$2"; shift 2 ;;
        --save-steps) SAVE_STEPS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --barrier-dir) BARRIER_DIR="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL" ] && [ -n "$STAGE1_CHECKPOINT" ]; then
    MODEL="$STAGE1_CHECKPOINT"
fi
MODEL="$(resolve_reference "${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}")"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "错误: 训练数据不存在: $TRAIN_FILE"
    echo "请先运行: python training/swift/prepare_swift_data.py --mode cot-sft --data-dir data-gen/output"
    exit 1
fi

mkdir -p "$OUTPUT_DIR" "$BARRIER_DIR"
wait_for_nodes "cot_sft_warmup_qwen25vl"

echo "=== ms-swift CoT SFT warmup (Qwen2.5-VL) ==="
echo "模型:        $MODEL"
echo "GPU:         $N_GPUS"
echo "节点:        $NNODES (node_rank=$NODE_RANK)"
echo "训练数据:    $TRAIN_FILE"
echo "验证数据:    $VAL_FILE"
echo "epochs:      $NUM_EPOCHS"
echo "lr:          $LEARNING_RATE"
echo "max_length:  $MAX_LENGTH"
echo "max_steps:   $MAX_STEPS"
echo "save_steps:  $SAVE_STEPS"
echo "deepspeed:   $DEEPSPEED_STAGE"
echo "thinking:    false (Qwen2.5-VL 无原生 thinking)"
echo "输出目录:    $OUTPUT_DIR"
echo ""

env \
    NPROC_PER_NODE="$N_GPUS" \
    NNODES="$NNODES" \
    NODE_RANK="$NODE_RANK" \
    MASTER_ADDR="$MASTER_ADDR" \
    MASTER_PORT="$MASTER_PORT" \
    swift sft \
    --model "$MODEL" \
    --train_type full \
    --dataset "$TRAIN_FILE" \
    --val_dataset "$VAL_FILE" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --num_train_epochs "$NUM_EPOCHS" \
    --max_length "$MAX_LENGTH" \
    --max_steps "$MAX_STEPS" \
    --warmup_ratio 0.05 \
    --save_strategy steps \
    --save_steps "$SAVE_STEPS" \
    --eval_strategy steps \
    --eval_steps "$SAVE_STEPS" \
    --logging_steps 5 \
    --enable_thinking false \
    --bf16 true \
    --deepspeed "$DEEPSPEED_STAGE" \
    --gradient_checkpointing true \
    --output_dir "$OUTPUT_DIR"
