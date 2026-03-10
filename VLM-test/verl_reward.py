"""
ORDINARY-BENCH 的 verl 奖励函数。

用于 GRPO/PPO 训练时评估模型输出的空间推理准确性。

注册方法（在 verl 配置中使用）：
    reward_model:
        reward_manager: naive
        custom_reward_function:
            path: verl_reward.py
            name: compute_score

或直接在代码中导入：
    from verl_reward import compute_score
"""

import json
import re
from typing import Union


def _parse_answer(response: str, question_type: str) -> Union[str, int, None]:
    """从模型回复中解析答案。"""
    text = response.strip()

    if question_type == "qrr":
        # 匹配 <, ~=, >
        for pattern in [r'"answer"\s*:\s*"([<>~=]+)"', r'\b([<>]|~=)\b']:
            m = re.search(pattern, text)
            if m:
                return m.group(1)
        # 直接匹配整个回复
        if text in ("<", ">", "~="):
            return text
        return None

    elif question_type == "trr":
        # 匹配整数 1-12
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
    return ((hour % 12) // 3) + 1


def compute_score(data_source: str, solution_str: str, ground_truth: str, **kwargs) -> float:
    """
    verl 奖励函数：评估模型对空间推理问题的回答。

    Args:
        data_source: 数据源标识（"ordinary-bench"）
        solution_str: 模型的输出文本
        ground_truth: GT 的 JSON 字符串

    Returns:
        float: 奖励分数
            QRR: 1.0（正确）或 0.0（错误）
            TRR: 1.0（hour精确）/ 0.5（象限正确）/ 0.25（相邻±1h）/ 0.0（错误）
    """
    try:
        gt = json.loads(ground_truth)
    except (json.JSONDecodeError, TypeError):
        return 0.0

    # 单问题模式 (RL)
    if isinstance(gt, dict) and "type" in gt:
        return _score_single(solution_str, gt)

    # 批量模式 (SFT/batch RL)
    if isinstance(gt, list):
        return _score_batch(solution_str, gt)

    return 0.0


def _score_single(response: str, gt: dict) -> float:
    """评分单个问题。"""
    q_type = gt["type"]
    gt_answer = gt["answer"]
    predicted = _parse_answer(response, q_type)

    if predicted is None:
        return 0.0

    if q_type == "qrr":
        return 1.0 if str(predicted) == str(gt_answer) else 0.0

    elif q_type == "trr":
        gt_hour = int(gt_answer)
        pred_hour = int(predicted)

        if pred_hour == gt_hour:
            return 1.0
        if _hour_to_quadrant(pred_hour) == _hour_to_quadrant(gt_hour):
            return 0.5
        diff = abs(pred_hour - gt_hour)
        if diff <= 1 or diff >= 11:
            return 0.25
        return 0.0

    return 0.0


def _score_batch(response: str, gt_list: list) -> float:
    """评分一批问题，返回平均分。"""
    # 解析模型的 JSON 数组回复
    try:
        # 尝试从回复中提取 JSON 数组
        text = response.strip()
        # 去掉 markdown 代码块
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
            if str(pred) == str(gt_answer):
                total_score += 1.0
        elif q_type == "trr":
            try:
                pred_hour = int(pred)
                gt_hour = int(gt_answer)
                if pred_hour == gt_hour:
                    total_score += 1.0
                elif _hour_to_quadrant(pred_hour) == _hour_to_quadrant(gt_hour):
                    total_score += 0.5
                else:
                    diff = abs(pred_hour - gt_hour)
                    if diff <= 1 or diff >= 11:
                        total_score += 0.25
            except (ValueError, TypeError):
                pass

    return total_score / len(gt_list)
