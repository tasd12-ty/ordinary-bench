"""
ORDINARY-BENCH 的 verl 奖励函数（软评分版）。

QRR 软评分：当实际距离接近时，给部分分（基于 ratio）。
TRR 多级评分：精确=1.0, 象限=0.5, 相邻=0.25。

注册方法（在 verl 配置中使用）：
    custom_reward_function:
        path: verl_reward.py
        name: compute_score
"""

import json
import re
from typing import Union


def _normalize_qrr_answer(value: str) -> Union[str, None]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    mapping = {
        "<": "<",
        "lt": "<",
        ">": ">",
        "gt": ">",
        "~=": "~=",
        "=": "~=",
        "≈": "~=",
        "approx": "~=",
        "eq": "~=",
    }
    return mapping.get(normalized)


def _parse_answer(response: str, question_type: str) -> Union[str, int, None]:
    """从模型回复中解析答案。"""
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
    """时钟小时转象限: Q1=12/1/2, Q2=3/4/5, Q3=6/7/8, Q4=9/10/11。"""
    if not 1 <= hour <= 12:
        raise ValueError(f"invalid hour: {hour}")
    return ((hour % 12) // 3) + 1


def _ratio_closeness(ratio: float) -> float:
    """
    计算距离比值的接近程度。
    ratio = dist(pair1) / dist(pair2)
    返回 0.0 (差异很大) ~ 1.0 (完全相等)。
    """
    if ratio <= 0:
        return 0.0
    # |log(ratio)| 越小越接近，用指数衰减映射到 [0, 1]
    import math
    log_diff = abs(math.log(ratio))
    # log_diff=0 → closeness=1.0; log_diff=0.1 → ~0.9; log_diff=0.5 → ~0.6
    return math.exp(-3.0 * log_diff)


def _score_qrr_soft(predicted: str, gt_answer: str, ratio: float = None) -> float:
    """
    QRR 软评分。

    当 ratio 可用时（距离比值 dist1/dist2）：
    - 预测正确 → 1.0
    - 值接近时（ratio ≈ 1）:
        - 预测 "~=" → closeness（值越接近分越高）
        - 预测反向 → closeness * 0.3（少量部分分）
    - 值差异大时:
        - 预测错误 → 0.0

    无 ratio 时退化为硬评分。
    """
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
        # GT 是约等于，预测了方向：给部分分（方向可能也对）
        return 0.5
    else:
        # GT 有明确方向 (< 或 >)
        if predicted == "~=":
            # 预测约等于，值确实接近时给较高部分分
            return closeness * 0.8
        else:
            # 预测反向，值非常接近时给少量部分分
            return closeness * 0.3 if closeness > 0.7 else 0.0


def _score_trr(pred_hour: int, gt_hour: int) -> float:
    """TRR 多级评分。"""
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


def compute_score(data_source: str, solution_str: str, ground_truth: str, **kwargs) -> float:
    """
    verl 奖励函数入口。

    Args:
        data_source: 数据源标识（"ordinary-bench"）
        solution_str: 模型的输出文本
        ground_truth: GT 的 JSON 字符串

    Returns:
        float: 奖励分数 [0.0, 1.0]
    """
    try:
        gt = json.loads(ground_truth)
    except (json.JSONDecodeError, TypeError):
        return 0.0

    # 单问题模式 (RL per-question)
    if isinstance(gt, dict) and "type" in gt:
        return _score_single(solution_str, gt)

    # 批量模式 (SFT/batch RL)
    if isinstance(gt, list):
        return _score_batch(solution_str, gt)

    return 0.0


def _score_single(response: str, gt: dict) -> float:
    """评分单个问题（含软评分）。"""
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


def _score_batch(response: str, gt_list: list) -> float:
    """评分一批问题，返回平均分。"""
    try:
        text = response.strip()
        if "```" in text:
            match = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
            if match:
                text = match.group(1).strip()
        predictions = json.loads(text)
        if not isinstance(predictions, list):
            predictions = []
    except (json.JSONDecodeError, TypeError):
        predictions = []

    pred_map = {}
    for p in predictions:
        if isinstance(p, dict) and "qid" in p:
            pred_map[p["qid"]] = p.get("answer")

    if not gt_list:
        return 0.0

    total_score = 0.0
    for gt_item in gt_list:
        qid = gt_item["qid"]
        q_type = gt_item["type"]
        gt_answer = gt_item["answer"]

        pred = pred_map.get(qid)
        if pred is None:
            continue

        if q_type == "qrr":
            ratio = gt_item.get("ratio")
            total_score += _score_qrr_soft(pred, gt_answer, ratio)
        elif q_type == "trr":
            try:
                total_score += _score_trr(int(pred), int(gt_answer))
            except (ValueError, TypeError):
                pass

    return total_score / len(gt_list)
