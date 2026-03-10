#!/bin/bash
# ms-swift + Qwen3.5-27B 训练环境安装
# 用法: bash training/swift/setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "=== ms-swift / Qwen3.5-27B 环境安装 ==="

# 检查 Python 版本
python3 -c "import sys; assert sys.version_info >= (3, 10), 'Python >= 3.10 required'" 2>/dev/null || {
    echo "错误: 需要 Python >= 3.10"
    exit 1
}

# 安装 ms-swift
if ! python3 -c "import swift" 2>/dev/null; then
    echo "[1/3] 安装 ms-swift ..."
    pip install 'ms-swift[all]>=4.0'
else
    SWIFT_VER=$(python3 -c "import swift; print(swift.__version__)" 2>/dev/null || echo "unknown")
    echo "[1/3] ms-swift 已安装 (v$SWIFT_VER)"
fi

# 安装 transformers (需要支持 Qwen3.5)
echo "[2/3] 升级 transformers (Qwen3.5 需要最新版) ..."
pip install 'transformers>=5.3.0'

# 安装 vLLM (GRPO rollout 加速)
echo "[3/3] 安装 vLLM ..."
pip install vllm

echo ""
echo "=== 安装完成 ==="
echo "下一步:"
echo "  1. 准备数据: python training/swift/prepare_swift_data.py --data-dir data-gen/output"
echo "  2. SFT 训练: bash training/swift/run_sft.sh"
echo "  3. GRPO 训练: bash training/swift/run_grpo.sh"
