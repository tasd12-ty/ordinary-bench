"""
OpenAI 兼容 API 客户端。

支持 OpenRouter、本地模型等所有兼容 OpenAI 格式的 API。
包含图片 base64 编码、消息构造、指数退避重试。
"""

import base64
import logging
import time

from openai import OpenAI, APIError, APIConnectionError, RateLimitError, APITimeoutError

logger = logging.getLogger(__name__)


def load_image_base64(image_path: str) -> str:
    """读取图片文件并返回 base64 编码字符串。"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_messages(system_prompt: str, user_text: str, image_b64: str) -> list:
    """构造 OpenAI vision 格式的消息列表（system + user with image）。"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{image_b64}",
            }},
            {"type": "text", "text": user_text},
        ]},
    ]


def call_vlm(
    client: OpenAI,
    messages: list,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    max_retries: int = 5,
    retry_base_delay: float = 2.0,
    timeout: int = 120,
    provider: str = "",
) -> str:
    """
    调用 VLM API，带指数退避重试。

    重试条件：RateLimitError、超时、服务器 5xx 错误。
    其他错误直接抛出。
    provider: OpenRouter 指定供应商（如 "alibaba"），为空则不指定。
    """
    # OpenRouter 供应商路由
    extra_body = {}
    if provider:
        extra_body["provider"] = {"order": [provider]}

    for attempt in range(max_retries):
        try:
            kwargs = dict(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            if extra_body:
                kwargs["extra_body"] = extra_body
            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content
            if content is None:
                logger.warning("VLM 返回空内容，视为可重试错误")
                raise APIError("VLM returned None content", response=None, body=None)
            return content
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            delay = min(retry_base_delay * (2 ** attempt), 60.0)
            logger.warning(f"重试 {attempt+1}/{max_retries}，{type(e).__name__}，等待 {delay:.1f}s")
            time.sleep(delay)
        except APIError as e:
            if getattr(e, 'status_code', None) and e.status_code >= 500:
                delay = min(retry_base_delay * (2 ** attempt), 60.0)
                logger.warning(f"重试 {attempt+1}/{max_retries}，服务器错误 {e.status_code}，等待 {delay:.1f}s")
                time.sleep(delay)
            else:
                raise
    raise RuntimeError(f"调用失败，已重试 {max_retries} 次")


def make_client(base_url: str, api_key: str) -> OpenAI:
    """创建 OpenAI 客户端实例。切换模型只需改 base_url 和 api_key。"""
    return OpenAI(base_url=base_url, api_key=api_key)
