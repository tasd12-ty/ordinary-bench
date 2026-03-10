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
NUM_EPOCHS="${NUM_EPOCHS:-5}"
ACCURACY_WEIGHT="${ACCURACY_WEIGHT:-1.0}"
FORMAT_WEIGHT="${FORMAT_WEIGHT:-1.0}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output/swift_grpo_qwen35_27b}"
ADAPTERS="${ADAPTERS:-}"
REF_ADAPTERS="${REF_ADAPTERS:-}"
SFT_ADAPTER="${SFT_ADAPTER:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

resolve_checkpoint_path() {
    local input_path="$1"
    if [ -z "$input_path" ]; then
        return 0
    fi
    if [ ! -e "$input_path" ]; then
        echo "错误: 路径不存在: $input_path" >&2
        exit 1
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

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --gpus) N_GPUS="$2"; shift 2 ;;
        --train-file) TRAIN_FILE="$2"; shift 2 ;;
        --val-file) VAL_FILE="$2"; shift 2 ;;
        --num-generations) NUM_GENERATIONS="$2"; shift 2 ;;
        --lora-rank) LORA_RANK="$2"; shift 2 ;;
        --epochs) NUM_EPOCHS="$2"; shift 2 ;;
        --accuracy-weight) ACCURACY_WEIGHT="$2"; shift 2 ;;
        --format-weight) FORMAT_WEIGHT="$2"; shift 2 ;;
        --reward-weights) ACCURACY_WEIGHT="$2"; FORMAT_WEIGHT="$3"; shift 3 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --adapters) ADAPTERS="$2"; shift 2 ;;
        --ref-adapters) REF_ADAPTERS="$2"; shift 2 ;;
        --sft-adapter) SFT_ADAPTER="$2"; shift 2 ;;
        --resume-from-checkpoint) RESUME_FROM_CHECKPOINT="$2"; shift 2 ;;
        --full) LORA_RANK=0; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [ ! -f "$TRAIN_FILE" ]; then
    echo "错误: 训练数据不存在: $TRAIN_FILE"
    echo "请先运行: python training/swift/prepare_swift_data.py --data-dir data-gen/output"
    exit 1
fi

if [ -n "$SFT_ADAPTER" ]; then
    SFT_ADAPTER="$(resolve_checkpoint_path "$SFT_ADAPTER")"
    if [ -z "$ADAPTERS" ]; then
        ADAPTERS="$SFT_ADAPTER"
    fi
    if [ -z "$REF_ADAPTERS" ]; then
        REF_ADAPTERS="$SFT_ADAPTER"
    fi
fi

if [ -n "$ADAPTERS" ]; then
    ADAPTERS="$(resolve_checkpoint_path "$ADAPTERS")"
fi

if [ -n "$REF_ADAPTERS" ]; then
    REF_ADAPTERS="$(resolve_checkpoint_path "$REF_ADAPTERS")"
fi

if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    RESUME_FROM_CHECKPOINT="$(resolve_checkpoint_path "$RESUME_FROM_CHECKPOINT")"
fi

echo "=== Qwen3.5-27B / ms-swift GRPO ==="
echo "模型:        $MODEL"
echo "GPU:         $N_GPUS"
echo "LoRA:        rank=$LORA_RANK alpha=$LORA_ALPHA"
echo "generations: $NUM_GENERATIONS"
echo "epochs:      $NUM_EPOCHS"
echo "reward:      accuracy=$ACCURACY_WEIGHT format=$FORMAT_WEIGHT"
echo "训练数据:    $TRAIN_FILE"
echo "奖励插件:    $PLUGIN_PATH"
echo "输出目录:    $OUTPUT_DIR"
if [ -n "$ADAPTERS" ]; then
    echo "adapters:    $ADAPTERS"
fi
if [ -n "$REF_ADAPTERS" ]; then
    echo "ref_adapters:$REF_ADAPTERS"
fi
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    echo "resume_ckpt: $RESUME_FROM_CHECKPOINT"
fi
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

EXTRA_ARGS=()
if [ -n "$ADAPTERS" ]; then
    EXTRA_ARGS+=(--adapters "$ADAPTERS")
fi
if [ -n "$REF_ADAPTERS" ]; then
    EXTRA_ARGS+=(--ref_adapters "$REF_ADAPTERS")
fi
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    EXTRA_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

NPROC_PER_NODE="$N_GPUS" swift rlhf \
    --rlhf_type grpo \
    --model "$MODEL" \
    "${TUNER_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    --external_plugins "$PLUGIN_PATH" \
    --reward_funcs ordinary_bench_accuracy ordinary_bench_format \
    --reward_weights "$ACCURACY_WEIGHT" "$FORMAT_WEIGHT" \
    --dataset "$TRAIN_FILE" \
    --val_dataset "$VAL_FILE" \
    --num_generations "$NUM_GENERATIONS" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-6 \
    --num_train_epochs "$NUM_EPOCHS" \
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
    --output_dir "$OUTPUT_DIR"
