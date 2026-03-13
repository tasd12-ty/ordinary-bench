#!/bin/bash
# Qwen2.5-VL-7B CoT GRPO：从 CoT warmup checkpoint 继续做结构化推理奖励训练。
#
# 与 Qwen3.5-27B 版本 (run_grpo_cot.sh) 的核心差异：
#   - enable_thinking=false：Qwen2.5-VL 无原生 thinking 模式
#   - DeepSpeed zero2：7B 模型 8 卡无需 zero3
#   - format_weight=0.10：无原生 thinking 需加强格式约束
#   - beta=0.005：较小模型更易 collapse，需稍强 KL 约束
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

MODEL="${MODEL:-}"
COT_SFT_CHECKPOINT="${COT_SFT_CHECKPOINT:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
MULTI_VIEW="${MULTI_VIEW:-false}"
CHECK_MODEL="${CHECK_MODEL:-}"
MODEL_TYPE="${MODEL_TYPE:-}"
TEMPLATE="${TEMPLATE:-}"
N_GPUS="${N_GPUS:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
TRAIN_FILE="${TRAIN_FILE:-$PROJECT_DIR/prepared_data/swift/cot_grpo/train.jsonl}"
VAL_FILE="${VAL_FILE:-$PROJECT_DIR/prepared_data/swift/cot_grpo/test.jsonl}"
PLUGIN_PATH="${PLUGIN_PATH:-$SCRIPT_DIR/plugins/ordinary_bench_reward_cot.py}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-256}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
BETA="${BETA:-0.005}"
EPSILON="${EPSILON:-0.2}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
ACCURACY_WEIGHT="${ACCURACY_WEIGHT:-1.0}"
FORMAT_WEIGHT="${FORMAT_WEIGHT:-1.0}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.45}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output/swift_cot_grpo_qwen25vl}"
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
        --cot-sft-checkpoint) COT_SFT_CHECKPOINT="$2"; shift 2 ;;
        --resume-from-checkpoint) RESUME_FROM_CHECKPOINT="$2"; shift 2 ;;
        --check-model) CHECK_MODEL="$2"; shift 2 ;;
        --model-type) MODEL_TYPE="$2"; shift 2 ;;
        --template) TEMPLATE="$2"; shift 2 ;;
        --gpus) N_GPUS="$2"; shift 2 ;;
        --nnodes) NNODES="$2"; shift 2 ;;
        --node-rank) NODE_RANK="$2"; shift 2 ;;
        --master-addr) MASTER_ADDR="$2"; shift 2 ;;
        --master-port) MASTER_PORT="$2"; shift 2 ;;
        --train-file) TRAIN_FILE="$2"; shift 2 ;;
        --val-file) VAL_FILE="$2"; shift 2 ;;
        --epochs) NUM_EPOCHS="$2"; shift 2 ;;
        --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
        --num-generations) NUM_GENERATIONS="$2"; shift 2 ;;
        --max-completion-length) MAX_COMPLETION_LENGTH="$2"; shift 2 ;;
        --max-prompt-length) MAX_PROMPT_LENGTH="$2"; shift 2 ;;
        --accuracy-weight) ACCURACY_WEIGHT="$2"; shift 2 ;;
        --format-weight) FORMAT_WEIGHT="$2"; shift 2 ;;
        --reward-weights) ACCURACY_WEIGHT="$2"; FORMAT_WEIGHT="$3"; shift 3 ;;
        --multi-view) MULTI_VIEW=true; shift 1 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --barrier-dir) BARRIER_DIR="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# 多视角覆盖默认值（仅当用户未通过 --train-file 等显式指定时）
if [ "$MULTI_VIEW" = "true" ]; then
    [ "$TRAIN_FILE" = "$PROJECT_DIR/prepared_data/swift/cot_grpo/train.jsonl" ] && \
        TRAIN_FILE="$PROJECT_DIR/prepared_data/swift/cot_grpo_mv/train.jsonl"
    [ "$VAL_FILE" = "$PROJECT_DIR/prepared_data/swift/cot_grpo/test.jsonl" ] && \
        VAL_FILE="$PROJECT_DIR/prepared_data/swift/cot_grpo_mv/test.jsonl"
    [ "$MAX_PROMPT_LENGTH" = "2048" ] && MAX_PROMPT_LENGTH=4096
    [ "$VLLM_MAX_NUM_SEQS" = "32" ] && VLLM_MAX_NUM_SEQS=16
    [ "$VLLM_GPU_MEMORY_UTILIZATION" = "0.45" ] && VLLM_GPU_MEMORY_UTILIZATION=0.35
    [ "$GRAD_ACCUM_STEPS" = "8" ] && GRAD_ACCUM_STEPS=16
    [ "$OUTPUT_DIR" = "$PROJECT_DIR/output/swift_cot_grpo_qwen25vl" ] && \
        OUTPUT_DIR="$PROJECT_DIR/output/swift_cot_grpo_qwen25vl_mv"
fi

if [ -z "$MODEL" ] && [ -n "$COT_SFT_CHECKPOINT" ]; then
    MODEL="$COT_SFT_CHECKPOINT"
fi
MODEL="$(resolve_reference "${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}")"
RESUME_FROM_CHECKPOINT="$(resolve_reference "$RESUME_FROM_CHECKPOINT")"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "错误: 训练数据不存在: $TRAIN_FILE"
    echo "请先运行: python training/swift/prepare_swift_data.py --mode cot-grpo --data-dir data-gen/output"
    exit 1
fi

if [ ! -f "$PLUGIN_PATH" ]; then
    echo "错误: 奖励插件不存在: $PLUGIN_PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR" "$BARRIER_DIR"
wait_for_nodes "cot_grpo_qwen25vl"

echo "=== ms-swift CoT GRPO (Qwen2.5-VL) ==="
echo "模型:        $MODEL"
echo "GPU:         $N_GPUS"
echo "节点:        $NNODES (node_rank=$NODE_RANK)"
echo "训练数据:    $TRAIN_FILE"
echo "验证数据:    $VAL_FILE"
echo "奖励插件:    $PLUGIN_PATH"
echo "generations: $NUM_GENERATIONS"
echo "epochs:      $NUM_EPOCHS"
echo "reward:      accuracy=$ACCURACY_WEIGHT format=$FORMAT_WEIGHT"
echo "beta:        $BETA"
echo "multi_view:  $MULTI_VIEW"
echo "deepspeed:   $DEEPSPEED_STAGE"
echo "thinking:    false (Qwen2.5-VL 无原生 thinking)"
echo "输出目录:    $OUTPUT_DIR"
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    echo "resume_ckpt: $RESUME_FROM_CHECKPOINT"
fi
echo ""

EXTRA_ARGS=()
[ -n "$CHECK_MODEL" ] && EXTRA_ARGS+=(--check_model "$CHECK_MODEL")
[ -n "$MODEL_TYPE" ] && EXTRA_ARGS+=(--model_type "$MODEL_TYPE")
[ -n "$TEMPLATE" ] && EXTRA_ARGS+=(--template "$TEMPLATE")
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    EXTRA_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

export ROLLOUT_LOG_DIR="$OUTPUT_DIR"

env \
    NPROC_PER_NODE="$N_GPUS" \
    NNODES="$NNODES" \
    NODE_RANK="$NODE_RANK" \
    MASTER_ADDR="$MASTER_ADDR" \
    MASTER_PORT="$MASTER_PORT" \
    swift rlhf \
    --rlhf_type grpo \
    --model "$MODEL" \
    "${EXTRA_ARGS[@]}" \
    --external_plugins "$PLUGIN_PATH" \
    --reward_funcs ordinary_bench_cot_accuracy ordinary_bench_cot_format \
    --reward_weights "$ACCURACY_WEIGHT" "$FORMAT_WEIGHT" \
    --dataset "$TRAIN_FILE" \
    --val_dataset "$VAL_FILE" \
    --num_generations "$NUM_GENERATIONS" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --num_train_epochs "$NUM_EPOCHS" \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --beta "$BETA" \
    --epsilon "$EPSILON" \
    --warmup_ratio "$WARMUP_RATIO" \
    --logging_steps 5 \
    --save_strategy steps \
    --save_steps 50 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS" \
    --enable_thinking false \
    --bf16 true \
    --deepspeed "$DEEPSPEED_STAGE" \
    --gradient_checkpointing true \
    --output_dir "$OUTPUT_DIR"
