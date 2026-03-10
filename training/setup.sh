#!/bin/bash
# ORDINARY-BENCH 训练环境安装脚本
# 用法: bash training/setup.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== ORDINARY-BENCH 训练环境安装 ==="
echo "项目目录: $PROJECT_DIR"

# 1. 克隆 verl
if [ ! -d "$PROJECT_DIR/verl" ]; then
    echo "[1/4] 克隆 verl ..."
    git clone https://github.com/volcengine/verl.git "$PROJECT_DIR/verl"
else
    echo "[1/4] verl 已存在，跳过克隆"
fi

# 2. 安装 Python 依赖
echo "[2/4] 安装 Python 依赖 ..."
pip install pandas pyarrow numpy

# 3. 安装 verl
echo "[3/4] 安装 verl ..."
cd "$PROJECT_DIR/verl"
pip install -e .

# 4. 安装 vLLM（GRPO rollout 需要）
echo "[4/4] 安装 vLLM ..."
pip install vllm

echo ""
echo "=== 安装完成 ==="
echo "下一步: 准备数据"
echo "  bash training/prepare_data.sh"
