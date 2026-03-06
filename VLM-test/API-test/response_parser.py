"""
Batch 模式 VLM 响应解析器。

处理 VLM 返回的文本，提取 JSON 数组格式的答案。
兼容常见问题：
  - markdown 代码围栏
  - 尾随逗号
  - 多段 JSON 数组拼接（模型分 QRR/TRR 输出两段 ][）
  - <think>...</think> 思考标签
  - 响应截断（JSON 不完整）
"""

import json
import re
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


def _strip_think_tags(text: str) -> str:
    """移除 <think>...</think> 思考标签，只保留最终输出。"""
    # 贪婪匹配最后一个 </think> 之后的内容
    m = re.search(r'</think>\s*(.*)', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # 如果有 <think> 但没闭合，取 <think> 之前的内容
    if '<think>' in text:
        return text.split('<think>')[0].strip()
    return text


def _merge_json_arrays(text: str) -> str:
    """合并多段 JSON 数组：][  →  ,（模型把 QRR 和 TRR 分两段输出）。"""
    return re.sub(r'\]\s*\[', ',', text)


def _fix_truncated_json(text: str) -> str:
    """修复截断的 JSON 数组：补全缺失的 ] 或 }]。"""
    text = text.rstrip()
    # 尝试逐步补全：}, ]
    for suffix in [']', '}]', '"}]', '"}]}]']:
        try:
            json.loads(text + suffix)
            return text + suffix
        except json.JSONDecodeError:
            continue
    return text


def extract_json(raw: str) -> Any:
    """
    从 VLM 响应中提取 JSON。

    按优先级依次尝试多种修复策略，尽量容错解析。
    """
    if not raw:
        raise ValueError("VLM 返回空响应")

    text = raw.strip()

    # 1. 移除思考标签
    text = _strip_think_tags(text)

    # 2. 剥离 markdown 代码围栏
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if m:
        text = m.group(1).strip()

    # 3. 合并多段 JSON 数组
    text = _merge_json_arrays(text)

    # 4. 修复尾随逗号
    text = re.sub(r',\s*([}\]])', r'\1', text)

    # 5. 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 6. 尝试修复截断
    try:
        fixed = _fix_truncated_json(text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # 7. 从文本中正则提取 JSON 片段
    for pattern in [r'(\[.*\])', r'(\{.*\})']:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            candidate = _merge_json_arrays(m.group(1))
            candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                # 尝试修复截断
                try:
                    return json.loads(_fix_truncated_json(candidate))
                except json.JSONDecodeError:
                    continue

    raise ValueError(f"无法从响应中解析 JSON: {text[:200]}...")


def parse_batch_response(raw: str, expected_qids: List[str]) -> Dict[str, Any]:
    """
    解析 batch 模式响应。

    返回 {qid: answer}，其中：
    - QRR 的 answer 是字符串 "<" / "~=" / ">"
    - TRR 的 answer 是整数 1-12
    - 缺失的 qid 对应 None
    """
    result = {qid: None for qid in expected_qids}

    try:
        data = extract_json(raw)
    except ValueError as e:
        logger.error(f"解析 batch 响应失败: {e}")
        return result

    if not isinstance(data, list):
        logger.error(f"期望 JSON 数组，实际得到 {type(data).__name__}")
        return result

    for item in data:
        if not isinstance(item, dict):
            continue
        qid = item.get("qid", "")
        answer = item.get("answer")
        if qid in result:
            result[qid] = answer

    return result
