#!/bin/bash
# ms-swift + Qwen3.5-27B 训练环境安装
# 用法: bash training/swift/setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# 加载版本锁定
# shellcheck disable=SC1091
source "$SCRIPT_DIR/versions.env"

echo "=== ms-swift / Qwen3.5-27B 环境安装 ==="
echo "版本: swift=${SWIFT_VERSION} transformers=${TRANSFORMERS_VERSION} vllm=${VLLM_VERSION}"

# 检查 Python 版本
python3 -c "import sys; assert sys.version_info >= (3, 10), 'Python >= 3.10 required'" 2>/dev/null || {
    echo "错误: 需要 Python >= 3.10"
    exit 1
}

# 安装 ms-swift
SWIFT_VER=$(python3 -c "import swift; print(swift.__version__)" 2>/dev/null || echo "0.0.0")
if [ "$SWIFT_VER" = "$SWIFT_VERSION" ]; then
    echo "[1/5] ms-swift v$SWIFT_VER OK"
else
    echo "[1/5] 安装 ms-swift==${SWIFT_VERSION} (当前: $SWIFT_VER) ..."
    pip install "ms-swift[all]==${SWIFT_VERSION}"
fi

# 安装 transformers + 视觉依赖
echo "[2/5] 安装 transformers + 视觉依赖 ..."
pip install "transformers==${TRANSFORMERS_VERSION}" "qwen_vl_utils==${QWEN_VL_UTILS_VERSION}" \
            "peft==${PEFT_VERSION}" "liger-kernel==${LIGER_KERNEL_VERSION}"

# 安装 Qwen3.5 线性注意力内核 (必须从 git 源码安装)
echo "[3/5] 安装 Qwen3.5 线性注意力内核 ..."
pip install "git+https://github.com/fla-org/flash-linear-attention@${FLA_GIT_REF}"
pip install "git+https://github.com/Dao-AILab/causal-conv1d@${CAUSAL_CONV1D_GIT_REF}" --no-build-isolation
pip install "deepspeed==${DEEPSPEED_VERSION}"

# 安装 vLLM (GRPO rollout 加速)
echo "[4/5] 安装 vLLM ..."
pip install "vllm==${VLLM_VERSION}"

# vLLM 可能覆盖 transformers 版本，重新确保
echo "[5/5] 确认 transformers 版本 ..."
pip install "transformers==${TRANSFORMERS_VERSION}"

echo ""
echo "=== 安装完成 ==="
echo "下一步:"
echo "  1a. 准备 GRPO 数据: python training/swift/prepare_swift_data.py --data-dir data-gen/output"
echo "  1b. 准备 SFT 数据:  python training/swift/prepare_swift_data.py --mode sft --data-dir data-gen/output"
echo "  2. GRPO 训练: bash training/swift/run_grpo.sh"
echo "  3. SFT 训练:  bash training/swift/run_sft.sh"
