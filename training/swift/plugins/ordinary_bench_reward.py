"""
ORDINARY-BENCH ms-swift 奖励函数插件。

适配 ms-swift 的 ORM (Outcome Reward Model) 接口，
复用 verl_reward.py 中的核心评分逻辑。

注册方式：
    swift rlhf --rlhf_type grpo \
        --external_plugins training/swift/plugins/ordinary_bench_reward.py \
        --reward_funcs ordinary_bench_accuracy ordinary_bench_format
"""

import json
import math
import re
from typing import List, Union


# ── 核心评分逻辑（与 verl_reward.py 保持一致） ──

def _normalize_qrr_answer(value: str) -> Union[str, None]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    mapping = {
        "<": "<", "lt": "<",
        ">": ">", "gt": ">",
        "~=": "~=", "=": "~=", "≈": "~=", "approx": "~=", "eq": "~=",
    }
    return mapping.get(normalized)


def _parse_answer(response: str, question_type: str) -> Union[str, int, None]:
    text = response.strip()
    if question_type == "qrr":
        for pattern in [r'"answer"\s*:\s*"([^"]+)"', r'\b([<>]|~=|=|≈|lt|gt|eq|approx)\b']:
            m = re.search(pattern, text)
            if m:
                return _normalize_qrr_answer(m.group(1))
        normalized = _normalize_qrr_answer(text)
        if normalized is not None:
            return normalized
        return None
    elif question_type == "trr":
        for pattern in [r'"answer"\s*:\s*(\d+)', r'\b(\d{1,2})\b']:
            m = re.search(pattern, text)
            if m:
                val = int(m.group(1))
                if 1 <= val <= 12:
                    return val
        return None
    return None


def _hour_to_quadrant(hour: int) -> int:
    if not 1 <= hour <= 12:
        raise ValueError(f"invalid hour: {hour}")
    return ((hour % 12) // 3) + 1


def _ratio_closeness(ratio: float) -> float:
    if ratio <= 0:
        return 0.0
    log_diff = abs(math.log(ratio))
    return math.exp(-3.0 * log_diff)


def _score_qrr_soft(predicted: str, gt_answer: str, ratio: float = None) -> float:
    predicted = _normalize_qrr_answer(predicted)
    gt_answer = _normalize_qrr_answer(gt_answer)
    if predicted is None or gt_answer is None:
        return 0.0
    if predicted == gt_answer:
        return 1.0
    if ratio is None:
        return 0.0
    closeness = _ratio_closeness(ratio)
    if gt_answer == "~=":
        return 0.5
    else:
        if predicted == "~=":
            return closeness * 0.8
        else:
            return closeness * 0.3 if closeness > 0.7 else 0.0


def _score_trr(pred_hour: int, gt_hour: int) -> float:
    if not 1 <= pred_hour <= 12:
        return 0.0
    if not 1 <= gt_hour <= 12:
        return 0.0
    if pred_hour == gt_hour:
        return 1.0
    if _hour_to_quadrant(pred_hour) == _hour_to_quadrant(gt_hour):
        return 0.5
    diff = abs(pred_hour - gt_hour)
    if diff <= 1 or diff >= 11:
        return 0.25
    return 0.0


def _score_single(response: str, gt: dict) -> float:
    q_type = gt["type"]
    gt_answer = gt["answer"]
    predicted = _parse_answer(response, q_type)
    if predicted is None:
        return 0.0
    if q_type == "qrr":
        ratio = gt.get("ratio")
        return _score_qrr_soft(str(predicted), str(gt_answer), ratio)
    elif q_type == "trr":
        try:
            return _score_trr(int(predicted), int(gt_answer))
        except (ValueError, TypeError):
            return 0.0
    return 0.0


# ── ms-swift ORM 接口 ──

try:
    from swift.plugin import orms
    _HAS_SWIFT = True
except ImportError:
    _HAS_SWIFT = False
    orms = {}


class OrdinaryBenchAccuracyORM:
    """准确率奖励：基于软评分计算 QRR/TRR 的得分。"""

    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        scores = []
        for completion, sol in zip(completions, solution):
            try:
                gt = json.loads(sol)
            except (json.JSONDecodeError, TypeError):
                scores.append(0.0)
                continue
            scores.append(_score_single(completion, gt))
        return scores


class OrdinaryBenchFormatORM:
    """格式奖励：检查模型输出是否为有效的答案格式。"""

    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        scores = []
        for completion, sol in zip(completions, solution):
            try:
                gt = json.loads(sol)
            except (json.JSONDecodeError, TypeError):
                scores.append(0.0)
                continue
            q_type = gt.get("type", "")
            parsed = _parse_answer(completion, q_type)
            scores.append(1.0 if parsed is not None else 0.0)
        return scores


# 注册到 ms-swift
orms["ordinary_bench_accuracy"] = OrdinaryBenchAccuracyORM
orms["ordinary_bench_format"] = OrdinaryBenchFormatORM
