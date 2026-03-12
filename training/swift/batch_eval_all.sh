#!/bin/bash
# 8-GPU 并行评测 126 个 checkpoint（2 个模型族 × 62 ckpt + 2 基座）
#
# 每个 GPU 独占一个 worker，tp=1，从共享队列原子出队任务。
# 每个任务：启动 vLLM → single-view eval → multi-view eval → 杀服务 → 取下一个。
#
# 用法:
#   bash training/swift/batch_eval_all.sh \
#       --base-model-a /models/Qwen3-8B \
#       --base-model-b /models/Qwen3.5-4B \
#       --ckpt-dir-a /output/sft_qwen3_8b \
#       --ckpt-dir-b /output/sft_qwen35_4b \
#       --family-a qwen3-8b \
#       --family-b qwen35-4b \
#       --n-gpus 8 \
#       --skip-existing
#
#   # 只看任务列表
#   bash training/swift/batch_eval_all.sh ... --dry-run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# ---- 默认参数 ----
BASE_MODEL_A=""
BASE_MODEL_B=""
CKPT_DIR_A=""
CKPT_DIR_B=""
FAMILY_A=""
FAMILY_B=""
N_GPUS=8
SKIP_EXISTING=false
DRY_RUN=false
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
CONCURRENCY="${CONCURRENCY:-16}"
MAX_LORA_RANK="${MAX_LORA_RANK:-32}" # LoRA 最大秩（需 >= 训练时的 rank）
STARTUP_TIMEOUT=180          # vLLM 启动超时（秒）
BASE_PORT=8000               # worker 端口从 BASE_PORT + gpu_id 开始

# ---- 解析参数 ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --base-model-a)  BASE_MODEL_A="$2";  shift 2 ;;
        --base-model-b)  BASE_MODEL_B="$2";  shift 2 ;;
        --ckpt-dir-a)    CKPT_DIR_A="$2";    shift 2 ;;
        --ckpt-dir-b)    CKPT_DIR_B="$2";    shift 2 ;;
        --family-a)      FAMILY_A="$2";       shift 2 ;;
        --family-b)      FAMILY_B="$2";       shift 2 ;;
        --n-gpus)        N_GPUS="$2";         shift 2 ;;
        --skip-existing) SKIP_EXISTING=true;  shift ;;
        --dry-run)       DRY_RUN=true;        shift ;;
        --max-model-len) MAX_MODEL_LEN="$2";  shift 2 ;;
        --concurrency)   CONCURRENCY="$2";    shift 2 ;;
        --base-port)     BASE_PORT="$2";      shift 2 ;;
        --max-lora-rank) MAX_LORA_RANK="$2"; shift 2 ;;
        --timeout)       STARTUP_TIMEOUT="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ---- 参数验证 ----
if [[ -z "$BASE_MODEL_A" || -z "$FAMILY_A" ]]; then
    echo "错误: 至少需要 --base-model-a 和 --family-a"
    exit 1
fi

RESULTS_DIR="$PROJECT_DIR/VLM-test/output/results"

# ---- 工作目录 ----
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
WORK_DIR="$PROJECT_DIR/logs/batch_eval_$TIMESTAMP"
mkdir -p "$WORK_DIR"
QUEUE_FILE="$WORK_DIR/queue.txt"
LOCK_FILE="$WORK_DIR/queue.lock"
DONE_FILE="$WORK_DIR/done_count"
echo 0 > "$DONE_FILE"

# ---- 生成任务队列 ----
# 队列格式: model_name<TAB>base_model_path<TAB>adapter_path
# adapter_path 为空表示基座模型

should_run() {
    local model_name="$1"
    local dir_name="${model_name//\//--}"

    if [[ "$SKIP_EXISTING" == true ]]; then
        local sv_summary="$RESULTS_DIR/$dir_name/summary.json"
        local mv_summary="$RESULTS_DIR/${dir_name}_multi_view/summary.json"
        if [[ -f "$sv_summary" && -f "$mv_summary" ]]; then
            echo "  SKIP: $model_name (已完成)" >&2
            return 1
        fi
    fi
    return 0
}

add_family_jobs() {
    local family="$1"
    local base_model="$2"
    local ckpt_dir="$3"

    # 基座模型
    if should_run "$family/base"; then
        printf "%s\t%s\t\n" "$family/base" "$base_model" >> "$QUEUE_FILE"
    fi

    # 各 checkpoint
    if [[ -n "$ckpt_dir" && -d "$ckpt_dir" ]]; then
        for ckpt_path in $(ls -d "$ckpt_dir"/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
            local step
            step=$(basename "$ckpt_path" | sed 's/checkpoint-//')
            local name="$family/ckpt-$step"
            if should_run "$name"; then
                printf "%s\t%s\t%s\n" "$name" "$base_model" "$ckpt_path" >> "$QUEUE_FILE"
            fi
        done
    fi
}

> "$QUEUE_FILE"  # 清空队列

echo "=== 生成任务队列 ==="
add_family_jobs "$FAMILY_A" "$BASE_MODEL_A" "$CKPT_DIR_A"

if [[ -n "$BASE_MODEL_B" && -n "$FAMILY_B" ]]; then
    add_family_jobs "$FAMILY_B" "$BASE_MODEL_B" "${CKPT_DIR_B:-}"
fi

TOTAL_JOBS=$(wc -l < "$QUEUE_FILE" | tr -d ' ')
echo "总任务数: $TOTAL_JOBS"
echo "工作目录: $WORK_DIR"

if [[ "$TOTAL_JOBS" -eq 0 ]]; then
    echo "无待执行任务。"
    exit 0
fi

if [[ "$DRY_RUN" == true ]]; then
    echo ""
    echo "=== 任务列表 (dry-run) ==="
    awk -F'\t' '{printf "  %-30s  base=%s  adapter=%s\n", $1, $2, ($3 ? $3 : "(none)")}' "$QUEUE_FILE"
    exit 0
fi

# ---- 原子出队 ----
dequeue() {
    (
        flock -x 200
        if [[ ! -s "$QUEUE_FILE" ]]; then
            exit 0
        fi
        head -1 "$QUEUE_FILE"
        tail -n +2 "$QUEUE_FILE" > "${QUEUE_FILE}.tmp"
        mv "${QUEUE_FILE}.tmp" "$QUEUE_FILE"
    ) 200>"$LOCK_FILE"
}

# ---- 原子计数 ----
increment_done() {
    (
        flock -x 200
        local count
        count=$(cat "$DONE_FILE")
        count=$((count + 1))
        echo "$count" > "$DONE_FILE"
        echo "$count"
    ) 200>"${DONE_FILE}.lock"
}

# ---- 等待 vLLM 就绪 ----
wait_ready() {
    local port=$1 timeout=$2
    for _ in $(seq 1 "$timeout"); do
        if curl -s "http://localhost:$port/v1/models" > /dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

# ---- 运行评测 ----
run_eval() {
    local port=$1 model_name=$2 log=$3

    (
        cd "$PROJECT_DIR/VLM-test/API-test"

        echo "[eval] single-view: $model_name" >> "$log"
        VLM_BASE_URL="http://localhost:$port/v1" \
        VLM_MODEL="$model_name" \
        VLM_API_KEY="dummy" \
        VLM_CONCURRENCY="$CONCURRENCY" \
        VLM_MAX_TOKENS="$MAX_MODEL_LEN" \
            python run_batch.py --test-only 2>&1 | tee -a "$log"

        echo "[eval] multi-view: $model_name" >> "$log"
        VLM_BASE_URL="http://localhost:$port/v1" \
        VLM_MODEL="$model_name" \
        VLM_API_KEY="dummy" \
        VLM_CONCURRENCY="$CONCURRENCY" \
        VLM_MAX_TOKENS="$MAX_MODEL_LEN" \
            python run_multi_view.py --test-only 2>&1 | tee -a "$log"
    )
}

# ---- Worker ----
worker() {
    local gpu_id=$1
    local port=$((BASE_PORT + gpu_id))
    local log="$WORK_DIR/worker_gpu${gpu_id}.log"

    echo "[GPU $gpu_id] Worker 启动, port=$port" | tee -a "$log"

    while true; do
        local job
        job=$(dequeue)
        [[ -z "$job" ]] && break

        local model_name base_model adapter
        model_name=$(echo "$job" | cut -f1)
        base_model=$(echo "$job" | cut -f2)
        adapter=$(echo "$job" | cut -f3)

        echo "" | tee -a "$log"
        echo "[GPU $gpu_id] === 开始: $model_name ===" | tee -a "$log"

        # 检查端口占用
        if curl -s "http://localhost:$port/v1/models" > /dev/null 2>&1; then
            echo "[GPU $gpu_id] 警告: 端口 $port 已被占用，尝试清理..." | tee -a "$log"
            fuser -k "$port/tcp" 2>/dev/null || true
            sleep 2
        fi

        # 构建 vLLM 命令
        local cmd=(
            python -m vllm.entrypoints.openai.api_server
            --model "$base_model"
            --port "$port"
            --tensor-parallel-size 1
            --max-model-len "$MAX_MODEL_LEN"
            --gpu-memory-utilization 0.9
            --trust-remote-code
            --disable-log-requests
        )

        if [[ -n "$adapter" ]]; then
            cmd+=(--enable-lora --max-lora-rank "$MAX_LORA_RANK" --lora-modules "${model_name}=${adapter}")
        else
            cmd+=(--served-model-name "$model_name")
        fi

        echo "[GPU $gpu_id] vLLM: ${cmd[*]}" >> "$log"

        # 启动 vLLM（限定 GPU）
        CUDA_VISIBLE_DEVICES="$gpu_id" "${cmd[@]}" >> "$log" 2>&1 &
        local server_pid=$!

        # 等待就绪
        if ! wait_ready "$port" "$STARTUP_TIMEOUT"; then
            echo "[GPU $gpu_id] 错误: vLLM 启动超时 ($model_name)" | tee -a "$log"
            kill "$server_pid" 2>/dev/null; wait "$server_pid" 2>/dev/null || true
            increment_done > /dev/null
            continue
        fi
        echo "[GPU $gpu_id] vLLM 就绪 (PID=$server_pid)" | tee -a "$log"

        # 评测（失败不中断）
        run_eval "$port" "$model_name" "$log" || \
            echo "[GPU $gpu_id] 警告: 评测异常 ($model_name)" | tee -a "$log"

        # 停止 vLLM
        kill "$server_pid" 2>/dev/null
        wait "$server_pid" 2>/dev/null || true

        # 进度
        local done_count
        done_count=$(increment_done)
        echo "[GPU $gpu_id] 完成: $model_name [$done_count/$TOTAL_JOBS]" | tee -a "$log"
    done

    echo "[GPU $gpu_id] Worker 结束" | tee -a "$log"
}

# ---- 清理 ----
cleanup() {
    echo ""
    echo "收到中断信号，清理中..."
    # 杀掉所有子进程
    kill $(jobs -p) 2>/dev/null || true
    # 杀掉可能残留的 vLLM 进程
    for gpu_id in $(seq 0 $((N_GPUS - 1))); do
        local port=$((BASE_PORT + gpu_id))
        fuser -k "$port/tcp" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    echo "已清理完毕。"
}
trap cleanup INT TERM

# ---- 启动所有 worker ----
echo ""
echo "=== 启动 $N_GPUS 个 worker，共 $TOTAL_JOBS 个任务 ==="
echo "    GPU 端口范围: $BASE_PORT - $((BASE_PORT + N_GPUS - 1))"
echo "    日志目录: $WORK_DIR"
echo ""

for gpu_id in $(seq 0 $((N_GPUS - 1))); do
    worker "$gpu_id" &
done

# 等待所有 worker 完成
wait

FINAL_DONE=$(cat "$DONE_FILE")
echo ""
echo "=== 全部完成: $FINAL_DONE/$TOTAL_JOBS ==="
echo "结果目录: $RESULTS_DIR"
echo "日志目录: $WORK_DIR"
