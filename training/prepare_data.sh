#!/bin/bash
# 数据准备一键脚本
# 用法: bash training/prepare_data.sh [--data-dir ../data-gen/output]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${1:-$PROJECT_DIR/data-gen/output}"

echo "=== 数据准备 ==="
echo "数据目录: $DATA_DIR"

cd "$PROJECT_DIR/VLM-test"

# 1. 检查场景数据是否存在
SCENE_COUNT=$(ls "$DATA_DIR/scenes/"*.json 2>/dev/null | wc -l)
echo "[检查] 场景数量: $SCENE_COUNT"
if [ "$SCENE_COUNT" -eq 0 ]; then
    echo "错误: 未找到场景数据，请先生成场景"
    echo "  cd data-gen && python3 generate.py --config config_expand_from20.toml --start-idx 20 --blender blender --gpu --workers 7"
    exit 1
fi

# 2. 生成问题（如果不存在）
Q_COUNT=$(ls "$PROJECT_DIR/VLM-test/output/questions/"*.json 2>/dev/null | wc -l)
if [ "$Q_COUNT" -ne "$SCENE_COUNT" ]; then
    echo "[2/5] 生成问题 ..."
    python3 generate_questions.py --data "$DATA_DIR"
else
    echo "[2/5] 问题已存在 ($Q_COUNT 个)，跳过"
fi

# 3. 划分训练/测试集
if [ ! -f "$DATA_DIR/train_scenes.json" ]; then
    echo "[3/5] 划分训练/测试集 ..."
    python3 "$PROJECT_DIR/data-gen/split_train_test.py" --output-dir "$DATA_DIR"
else
    echo "[3/5] 训练/测试划分已存在，跳过"
fi

# 4. 生成 RL 格式 parquet
echo "[4/5] 生成 RL 训练数据 (per-question) ..."
python3 prepare_verl_data.py --mode rl --data-dir "$DATA_DIR" --output-dir ./verl_data

# 5. 生成 SFT 格式 parquet
echo "[5/5] 生成 SFT 训练数据 (per-batch) ..."
python3 prepare_verl_data.py --mode sft --data-dir "$DATA_DIR" --output-dir ./verl_data_sft

echo ""
echo "=== 数据准备完成 ==="
echo "RL 数据:  VLM-test/verl_data/"
echo "SFT 数据: VLM-test/verl_data_sft/"
