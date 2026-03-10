#!/bin/bash
# 使用 uv 创建可迁移的 Qwen3-VL-32B + verl 训练环境
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

require_uv

echo "=== ORDINARY-BENCH Qwen3-VL-32B / verl 环境安装 ==="
echo "项目目录: $PROJECT_DIR"
echo "虚拟环境: $VENV_DIR"
print_versions
echo ""

uv venv --python "${PYTHON_VERSION}" "$VENV_DIR"

echo "[1/5] 安装基础数据处理依赖 ..."
uv pip install --python "$PYTHON_BIN" \
    "numpy==${NUMPY_VERSION}" \
    "pandas==${PANDAS_VERSION}" \
    "pyarrow==${PYARROW_VERSION}" \
    "datasets==${DATASETS_VERSION}" \
    "accelerate==${ACCELERATE_VERSION}" \
    "ray[default]==${RAY_VERSION}" \
    "wandb==${WANDB_VERSION}" \
    "Pillow==${PILLOW_VERSION}" \
    "qwen-vl-utils==${QWEN_VL_UTILS_VERSION}"

echo "[2/5] 安装 PyTorch ..."
uv pip install --python "$PYTHON_BIN" \
    --index-url "${TORCH_INDEX_URL}" \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}"

echo "[3/5] 安装 vLLM / Transformers ..."
uv pip install --python "$PYTHON_BIN" \
    "transformers==${TRANSFORMERS_VERSION}" \
    "vllm==${VLLM_VERSION}"

if [ "${INSTALL_FLASH_ATTN}" = "1" ]; then
    echo "[4/5] 安装 flash-attn ..."
    uv pip install --python "$PYTHON_BIN" --no-build-isolation "flash-attn==${FLASH_ATTN_VERSION}"
else
    echo "[4/5] 跳过 flash-attn（如需安装，设置 INSTALL_FLASH_ATTN=1）"
fi

echo "[5/5] 安装 verl 与本项目 ..."
uv pip install --python "$PYTHON_BIN" "git+https://github.com/volcengine/verl.git@${VERL_GIT_REF}"
uv pip install --python "$PYTHON_BIN" \
    "ray[default]==${RAY_VERSION}" \
    "transformers==${TRANSFORMERS_VERSION}" \
    "vllm==${VLLM_VERSION}"
uv pip install --python "$PYTHON_BIN" -e "$PROJECT_DIR"

echo ""
echo "=== 安装完成 ==="
echo "激活方式:"
echo "  source \"$VENV_DIR/bin/activate\""
echo "下一步:"
echo "  bash training/prepare_data.sh --data-dir \"$PROJECT_DIR/data-gen/output\""
echo "  bash training/run_grpo.sh --gpus 8"
