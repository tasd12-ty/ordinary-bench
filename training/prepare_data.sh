#!/bin/bash
# Qwen3-VL-32B + verl 数据准备
# 用法:
#   bash training/prepare_data.sh --data-dir ./data-gen/output
#   bash training/prepare_data.sh --data-dir ./data-gen/output --multi-view
#   bash training/prepare_data.sh --data-dir ./data-gen/output --include-sft
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"
require_python

DATA_DIR="$PROJECT_DIR/data-gen/output"
OUTPUT_DIR="$PROJECT_DIR/VLM-test/verl_data_qwen3vl32b"
INCLUDE_SFT=0
MULTI_VIEW=0
N_VIEWS=4

while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --multi-view) MULTI_VIEW=1; shift ;;
        --n-views) N_VIEWS="$2"; shift 2 ;;
        --include-sft) INCLUDE_SFT=1; shift ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

echo "=== Qwen3-VL-32B 数据准备 ==="
echo "数据目录: $DATA_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "多视角:   $MULTI_VIEW (n_views=$N_VIEWS)"

cd "$PROJECT_DIR/VLM-test"

if [ ! -d "$DATA_DIR/scenes" ]; then
    echo "错误: 场景目录不存在: $DATA_DIR/scenes"
    exit 1
fi

SCENE_COUNT=$(find "$DATA_DIR/scenes" -maxdepth 1 -name '*.json' | wc -l | tr -d ' ')
echo "[检查] 场景数量: $SCENE_COUNT"
if [ "$SCENE_COUNT" -eq 0 ]; then
    echo "错误: 未找到场景数据，请先生成场景"
    exit 1
fi

Q_COUNT=$(find "$PROJECT_DIR/VLM-test/output/questions" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
# 检查问题缓存是否与当前 data-dir 的场景匹配（抽样验证前 3 个场景）
REGEN_QUESTIONS=0
if [ "$Q_COUNT" -ne "$SCENE_COUNT" ]; then
    REGEN_QUESTIONS=1
elif [ "$Q_COUNT" -gt 0 ]; then
    SAMPLE_SCENES=$(find "$DATA_DIR/scenes" -maxdepth 1 -name '*.json' | sort | head -3)
    MISMATCH=0
    for s in $SAMPLE_SCENES; do
        SID=$(basename "$s" .json)
        if [ ! -f "$PROJECT_DIR/VLM-test/output/questions/$SID.json" ]; then
            echo "[1/3] 场景 $SID 缺少对应问题文件"
            MISMATCH=1
            break
        fi
    done
    if [ "$MISMATCH" -eq 1 ]; then
        echo "[1/3] 问题缓存与当前场景不匹配，重新生成"
        REGEN_QUESTIONS=1
    fi
fi
if [ "$REGEN_QUESTIONS" -eq 1 ]; then
    echo "[1/3] 生成问题 ..."
    "$PYTHON_BIN" generate_questions.py --data "$DATA_DIR"
else
    echo "[1/3] 问题已存在 ($Q_COUNT 个)，跳过"
fi

if [ ! -f "$DATA_DIR/train_scenes.json" ] || [ ! -f "$DATA_DIR/test_scenes.json" ]; then
    echo "[2/3] 划分训练/测试集 ..."
    "$PYTHON_BIN" "$PROJECT_DIR/data-gen/split_train_test.py" --output-dir "$DATA_DIR"
else
    echo "[2/3] 训练/测试划分已存在，跳过"
fi

EXTRA_ARGS=()
if [ "$MULTI_VIEW" -eq 1 ]; then
    EXTRA_ARGS+=(--multi-view --n-views "$N_VIEWS")
fi

echo "[3/3] 生成 RL 训练数据 ..."
"$PYTHON_BIN" prepare_verl_data.py \
    --mode rl \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}"

if [ "$INCLUDE_SFT" -eq 1 ]; then
    echo "附加导出实验性 SFT 数据 ..."
    "$PYTHON_BIN" prepare_verl_data.py \
        --mode sft \
        --data-dir "$DATA_DIR" \
        --output-dir "${OUTPUT_DIR}_sft_experimental" \
        "${EXTRA_ARGS[@]}"
fi

echo ""
echo "=== 数据准备完成 ==="
echo "GRPO 数据: $OUTPUT_DIR"
if [ "$INCLUDE_SFT" -eq 1 ]; then
    echo "SFT 数据:  ${OUTPUT_DIR}_sft_experimental"
fi
