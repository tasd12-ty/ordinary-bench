#!/bin/bash
# 部署 LoRA 微调模型并运行 benchmark 评测
#
# 支持 vLLM 和 SGLang 两种后端，自动启动推理服务并评测。
#
# 用法:
#   # vLLM 部署 + 评测（默认）
#   bash training/swift/serve_and_eval.sh --adapter output/swift_sft_qwen35_27b/checkpoint-xxx
#
#   # SGLang 部署 + 评测
#   bash training/swift/serve_and_eval.sh --backend sglang --adapter output/swift_sft_qwen35_27b/checkpoint-xxx
#
#   # 只部署不评测（手动评测）
#   bash training/swift/serve_and_eval.sh --adapter output/swift_sft_qwen35_27b/checkpoint-xxx --serve-only
#
#   # 评测基座模型（不加 LoRA，用于对比）
#   bash training/swift/serve_and_eval.sh --no-lora
#
#   # 对比评测：先跑基座，再跑 LoRA
#   bash training/swift/serve_and_eval.sh --compare --adapter output/swift_sft_qwen35_27b/checkpoint-xxx
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# ---- 默认参数 ----
MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
BACKEND="${BACKEND:-vllm}"          # vllm | sglang
ADAPTER=""                          # LoRA adapter 路径
NO_LORA=false
SERVE_ONLY=false
COMPARE=false
PORT="${PORT:-8000}"
N_GPUS="${N_GPUS:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
CONCURRENCY="${CONCURRENCY:-16}"    # 评测并发数
EVAL_SPLIT="test"                   # test | train | all

# ---- 解析参数 ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --backend) BACKEND="$2"; shift 2 ;;
        --adapter) ADAPTER="$2"; shift 2 ;;
        --no-lora) NO_LORA=true; shift ;;
        --serve-only) SERVE_ONLY=true; shift ;;
        --compare) COMPARE=true; shift ;;
        --port) PORT="$2"; shift 2 ;;
        --gpus) N_GPUS="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --eval-split) EVAL_SPLIT="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ---- 自动查找最新 checkpoint ----
resolve_adapter() {
    local dir="$1"
    if [[ -f "$dir/adapter_config.json" ]]; then
        echo "$dir"
        return
    fi
    # 查找目录下最新的 checkpoint-* 子目录
    local latest
    latest=$(ls -d "$dir"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    if [[ -n "$latest" && -f "$latest/adapter_config.json" ]]; then
        echo "$latest"
        return
    fi
    echo "错误: 未找到有效的 LoRA adapter: $dir" >&2
    exit 1
}

if [[ -n "$ADAPTER" ]]; then
    ADAPTER=$(resolve_adapter "$ADAPTER")
    echo "LoRA adapter: $ADAPTER"
fi

if [[ -z "$ADAPTER" && "$NO_LORA" == false && "$COMPARE" == false ]]; then
    echo "错误: 请指定 --adapter <path> 或 --no-lora 或 --compare"
    exit 1
fi

# ---- 启动推理服务 ----
start_server() {
    local use_adapter="$1"
    local model_tag="$2"

    echo ""
    echo "=== 启动 $BACKEND 推理服务 ==="
    echo "模型:     $MODEL"
    echo "后端:     $BACKEND"
    echo "LoRA:     ${use_adapter:-无}"
    echo "GPU:      $N_GPUS"
    echo "端口:     $PORT"
    echo ""

    if [[ "$BACKEND" == "vllm" ]]; then
        local cmd=(
            python -m vllm.entrypoints.openai.api_server
            --model "$MODEL"
            --port "$PORT"
            --tensor-parallel-size "$N_GPUS"
            --max-model-len "$MAX_MODEL_LEN"
            --gpu-memory-utilization 0.9
            --trust-remote-code
            --disable-log-requests
        )
        if [[ -n "$use_adapter" ]]; then
            cmd+=(--enable-lora --lora-modules "lora=$use_adapter")
        fi

    elif [[ "$BACKEND" == "sglang" ]]; then
        local cmd=(
            python -m sglang.launch_server
            --model-path "$MODEL"
            --port "$PORT"
            --tp "$N_GPUS"
            --max-total-tokens "$MAX_MODEL_LEN"
            --trust-remote-code
            --disable-log-requests
        )
        if [[ -n "$use_adapter" ]]; then
            cmd+=(--lora-paths "lora=$use_adapter")
        fi
    else
        echo "错误: 不支持的后端 $BACKEND，请使用 vllm 或 sglang"
        exit 1
    fi

    echo "命令: ${cmd[*]}"
    "${cmd[@]}" &
    SERVER_PID=$!
    echo "服务 PID: $SERVER_PID"

    # 等待服务就绪
    echo "等待服务就绪..."
    for i in $(seq 1 120); do
        if curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
            echo "服务已就绪 (${i}s)"
            return 0
        fi
        sleep 1
    done
    echo "错误: 服务启动超时（120s）"
    kill $SERVER_PID 2>/dev/null
    exit 1
}

stop_server() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        echo "停止推理服务 (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null
        wait $SERVER_PID 2>/dev/null || true
        unset SERVER_PID
    fi
}
trap stop_server EXIT

# ---- 运行评测 ----
run_eval() {
    local model_name="$1"
    local eval_tag="$2"

    echo ""
    echo "=== 评测: $eval_tag ==="

    local split_flag=""
    case "$EVAL_SPLIT" in
        test)  split_flag="--test-only" ;;
        train) split_flag="--train-only" ;;
        all)   split_flag="" ;;
    esac

    cd "$PROJECT_DIR/VLM-test/API-test"

    VLM_BASE_URL="http://localhost:$PORT/v1" \
    VLM_MODEL="$model_name" \
    VLM_API_KEY="dummy" \
    VLM_CONCURRENCY="$CONCURRENCY" \
    VLM_MAX_TOKENS="$MAX_MODEL_LEN" \
        python run_batch.py $split_flag

    cd "$PROJECT_DIR"
    echo "评测完成: $eval_tag"
    echo "结果目录: VLM-test/output/results/$(echo "$model_name" | tr '/' '--')/"
}

# ---- 主流程 ----
if [[ "$COMPARE" == true ]]; then
    # 对比模式：先跑基座，再跑 LoRA
    if [[ -z "$ADAPTER" ]]; then
        echo "错误: --compare 模式需要 --adapter 参数"
        exit 1
    fi

    # 1. 基座模型评测
    start_server "" "base"
    run_eval "$MODEL" "基座模型 $MODEL"
    stop_server

    # 2. LoRA 模型评测
    start_server "$ADAPTER" "lora"
    run_eval "lora" "LoRA 微调模型"
    stop_server

    echo ""
    echo "=== 对比完成 ==="
    echo "基座结果: VLM-test/output/results/$(echo "$MODEL" | tr '/' '--')/"
    echo "LoRA结果: VLM-test/output/results/lora/"

elif [[ "$NO_LORA" == true ]]; then
    # 只评测基座模型
    start_server "" "base"
    if [[ "$SERVE_ONLY" == true ]]; then
        echo "服务已启动，按 Ctrl+C 停止"
        wait $SERVER_PID
    else
        run_eval "$MODEL" "基座模型 $MODEL"
    fi

else
    # 评测 LoRA 模型
    start_server "$ADAPTER" "lora"
    if [[ "$SERVE_ONLY" == true ]]; then
        echo "服务已启动，按 Ctrl+C 停止"
        wait $SERVER_PID
    else
        run_eval "lora" "LoRA 微调模型"
    fi
fi
