#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"

if [ -f "$SCRIPT_DIR/versions.env" ]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/versions.env"
fi

require_uv() {
    if ! command -v uv >/dev/null 2>&1; then
        echo "错误: 未找到 uv，请先安装 uv"
        echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
}

require_python() {
    if [ ! -x "$PYTHON_BIN" ]; then
        echo "错误: 未找到训练虚拟环境: $PYTHON_BIN"
        echo "请先运行: bash training/setup_uv.sh"
        exit 1
    fi
}

print_versions() {
    echo "Python: ${PYTHON_VERSION:-unset}"
    echo "verl ref: ${VERL_GIT_REF:-unset}"
    echo "torch: ${TORCH_VERSION:-unset}"
    echo "vllm: ${VLLM_VERSION:-unset}"
    echo "transformers: ${TRANSFORMERS_VERSION:-unset}"
}
