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

# 安装/升级 ms-swift (需要 >= 4.0)
SWIFT_VER=$(python3 -c "import swift; print(swift.__version__)" 2>/dev/null || echo "0.0.0")
SWIFT_OK=$(python3 -c "
from packaging.version import Version
print('yes' if Version('$SWIFT_VER') >= Version('4.0') else 'no')
" 2>/dev/null || echo "no")

if [ "$SWIFT_OK" = "yes" ]; then
    echo "[1/4] ms-swift v$SWIFT_VER OK (>= 4.0)"
else
    echo "[1/4] 安装 ms-swift >= 4.0 (当前: $SWIFT_VER) ..."
    pip install 'ms-swift[all]>=4.0'
fi

# 安装 transformers + 视觉依赖
echo "[2/5] 安装 transformers + 视觉依赖 ..."
pip install -U 'transformers>=5.2.0' 'qwen_vl_utils>=0.0.14' peft liger-kernel

# 安装 Qwen3.5 线性注意力内核 (必须从 git 源码安装)
echo "[3/5] 安装 Qwen3.5 线性注意力内核 ..."
pip install -U git+https://github.com/fla-org/flash-linear-attention
pip install -U git+https://github.com/Dao-AILab/causal-conv1d --no-build-isolation
pip install deepspeed

# 安装 vLLM (GRPO rollout 加速)
echo "[4/5] 安装 vLLM ..."
pip install -U 'vllm>=0.17.0'

# vLLM 可能覆盖 transformers 版本，重新确保
echo "[5/5] 确认 transformers 版本 ..."
pip install -U 'transformers>=5.2.0'

echo ""
echo "=== 安装完成 ==="
echo "下一步:"
echo "  1a. 准备 GRPO 数据: python training/swift/prepare_swift_data.py --data-dir data-gen/output"
echo "  1b. 准备 SFT 数据:  python training/swift/prepare_swift_data.py --mode sft --data-dir data-gen/output"
echo "  2. GRPO 训练: bash training/swift/run_grpo.sh"
echo "  3. SFT 训练:  bash training/swift/run_sft.sh"
