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
import logging
import re
from typing import Union

logger = logging.getLogger(__name__)

# 解析路径统计（用于监控训练过程中标签采用率）
_parse_path_counts: dict = {}
_parse_total = 0
_PARSE_LOG_INTERVAL = 100


def _record_parse_path(path: str):
    """记录解析路径并定期输出统计。"""
    global _parse_total
    _parse_path_counts[path] = _parse_path_counts.get(path, 0) + 1
    _parse_total += 1
    if _parse_total % _PARSE_LOG_INTERVAL == 0:
        tag_pct = _parse_path_counts.get("tag", 0) / _parse_total * 100
        logger.info(
            "parse_path_stats total=%d tag=%.1f%% %s",
            _parse_total, tag_pct, dict(_parse_path_counts),
        )


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


def _extract_from_answer_tag(text: str) -> Union[str, None]:
    """提取 <answer>...</answer> 标签内容，取最后一个匹配。"""
    matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None


def _parse_answer_with_path(response: str, question_type: str) -> tuple:
    """从模型回复中解析答案，返回 (answer, parse_path)。

    parse_path 取值: tag, json, keyword, symbol, fallback,
                      context_num, last_num, unparseable
    """
    text = response.strip()

    # === 优先级 0: <answer> 标签 ===
    tag_content = _extract_from_answer_tag(text)
    if tag_content is not None:
        if question_type == "qrr":
            normalized = _normalize_qrr_answer(tag_content)
            if normalized is not None:
                return normalized, "tag"
        elif question_type == "trr":
            m = re.search(r'\b(\d{1,2})\b', tag_content)
            if m:
                val = int(m.group(1))
                if 1 <= val <= 12:
                    return val, "tag"

    # === 以下为现有 regex 回退逻辑 ===
    if question_type == "qrr":
        # 1) JSON 格式: {"answer": "<"}
        m = re.search(r'"answer"\s*:\s*"([^"]+)"', text)
        if m:
            return _normalize_qrr_answer(m.group(1)), "json"

        # 2) 文本中的关键词 (lt/gt/eq/approx 需要 word boundary)
        m = re.search(r'\b(lt|gt|eq|approx|~=)\b', text, re.IGNORECASE)
        if m:
            return _normalize_qrr_answer(m.group(1)), "keyword"

        # 3) 独立的 <, >, ~=, ≈, = 符号 (不用 \b，用前后空白/边界)
        m = re.search(r'(?:^|[\s:=",({])([<>≈]|~=)(?:$|[\s,."\'}):\]])', text)
        if m:
            return _normalize_qrr_answer(m.group(1)), "symbol"

        # 4) 整个文本就是答案
        normalized = _normalize_qrr_answer(text)
        if normalized is not None:
            return normalized, "fallback"
        return None, "unparseable"

    elif question_type == "trr":
        # 1) JSON 格式
        m = re.search(r'"answer"\s*:\s*(\d+)', text)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 12:
                return val, "json"
        # 2) 带上下文的数字 (hour/position/at/answer/is + 数字)
        m = re.search(r'(?:hour|position|at|answer|is)\s*[:\s]\s*(\d{1,2})\b', text, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 12:
                return val, "context_num"
        # 3) 回退: 取最后一个 1-12 范围的数字
        matches = re.findall(r'\b(\d{1,2})\b', text)
        for num_str in reversed(matches):
            val = int(num_str)
            if 1 <= val <= 12:
                return val, "last_num"
        return None, "unparseable"

    return None, "unparseable"


def _parse_answer(response: str, question_type: str) -> Union[str, int, None]:
    """从模型回复中解析答案。"""
    answer, _ = _parse_answer_with_path(response, question_type)
    return answer


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
        # GT 是约等于，预测了方向：closeness 高时给多，低时给少
        return closeness * 0.5
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
    predicted, parse_path = _parse_answer_with_path(response, q_type)
    _record_parse_path(parse_path)

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
