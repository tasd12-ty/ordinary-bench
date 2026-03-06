"""
VLM 测试配置。

所有设置通过环境变量覆盖，切换模型只需修改：
  VLM_BASE_URL — API 地址（OpenRouter / 本地模型）
  VLM_MODEL    — 模型名称
  VLM_API_KEY  — API 密钥
"""

import os

CONFIG = {
    # API 连接
    "base_url": os.environ.get("VLM_BASE_URL", "https://openrouter.ai/api/v1"),
    "api_key": os.environ.get("VLM_API_KEY", ""),
    "model": os.environ.get("VLM_MODEL", "google/gemini-2.0-flash-001"),
    "provider": os.environ.get("VLM_PROVIDER", ""),  # OpenRouter 供应商（如 alibaba）

    # 并发与重试
    "max_concurrency": int(os.environ.get("VLM_CONCURRENCY", "4")),
    "timeout": int(os.environ.get("VLM_TIMEOUT", "120")),
    "max_retries": int(os.environ.get("VLM_MAX_RETRIES", "5")),
    "retry_base_delay": float(os.environ.get("VLM_RETRY_DELAY", "2.0")),

    # 生成参数
    "temperature": 0.0,
    "max_tokens": 4096,

    # 路径（相对于 API-test/ 目录）
    "questions_dir": "../output/questions",                    # batch 模式输入
    "images_dir": "../../data-gen/output/images/single_view",  # 场景图片
    "results_dir": "../output/results",                        # 结果输出
}
