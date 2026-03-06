"""
Batch 评分模块：比较 VLM 预测与 GT。

QRR：比较器精确匹配。
TRR：三粒度评分（hour 精确 / quadrant 象限 / adjacent ±1h）。
"""

from typing import Dict, List, Any
from dsl.comparators import Comparator
from dsl.predicates import hour_to_quadrant


# ── 单题评分 ──

def score_qrr(predicted: str, gt: str) -> bool:
    """QRR 精确匹配：比较器是否一致。"""
    try:
        return Comparator.from_string(predicted) == Comparator.from_string(gt)
    except ValueError:
        return False


def score_trr_hour(predicted: int, gt_hour: int) -> bool:
    """TRR hour 精确匹配。"""
    return predicted == gt_hour


def score_trr_quadrant(predicted: int, gt_quadrant: int) -> bool:
    """TRR 象限匹配：预测的 hour 是否落在 GT 所在象限。"""
    try:
        return hour_to_quadrant(int(predicted)) == gt_quadrant
    except (ValueError, TypeError):
        return False


def score_trr_adjacent(predicted: int, gt_hour: int) -> bool:
    """TRR 相邻匹配：预测 hour 与 GT hour 差 ≤ 1（12↔1 循环）。"""
    try:
        diff = abs(int(predicted) - gt_hour)
        return diff <= 1 or diff >= 11
    except (ValueError, TypeError):
        return False


# ── 场景评分 ──

def score_batch_scene(predictions: Dict[str, Any], questions: List[dict]) -> dict:
    """
    对单个场景的所有 batch 预测评分。

    返回 QRR/TRR 各项统计 + 逐题详情。
    """
    qrr_correct = qrr_total = 0
    trr_hour_correct = trr_quad_correct = trr_adj_correct = trr_total = 0
    missing = 0
    per_question = []

    for q in questions:
        qid = q["qid"]
        pred = predictions.get(qid)

        if pred is None:
            missing += 1
            per_question.append({"qid": qid, "type": q["type"], "predicted": None, "correct": False})
            if q["type"] == "qrr":
                qrr_total += 1
            else:
                trr_total += 1
            continue

        if q["type"] == "qrr":
            qrr_total += 1
            correct = score_qrr(str(pred), q["gt_comparator"])
            if correct:
                qrr_correct += 1
            per_question.append({
                "qid": qid, "type": "qrr",
                "predicted": str(pred), "gt": q["gt_comparator"],
                "correct": correct,
            })
        elif q["type"] == "trr":
            trr_total += 1
            try:
                pred_int = int(pred)
            except (ValueError, TypeError):
                pred_int = -1
            h = score_trr_hour(pred_int, q["gt_hour"])
            qu = score_trr_quadrant(pred_int, q["gt_quadrant"])
            a = score_trr_adjacent(pred_int, q["gt_hour"])
            if h:
                trr_hour_correct += 1
            if qu:
                trr_quad_correct += 1
            if a:
                trr_adj_correct += 1
            per_question.append({
                "qid": qid, "type": "trr",
                "predicted": pred_int, "gt_hour": q["gt_hour"],
                "hour_correct": h, "quadrant_correct": qu, "adjacent_correct": a,
            })

    return {
        "qrr_correct": qrr_correct, "qrr_total": qrr_total,
        "trr_hour_correct": trr_hour_correct,
        "trr_quadrant_correct": trr_quad_correct,
        "trr_adjacent_correct": trr_adj_correct,
        "trr_total": trr_total,
        "missing": missing,
        "per_question": per_question,
    }


# ── 结果聚合 ──

def aggregate_batch_results(scene_results: List[dict]) -> dict:
    """汇总所有场景的 batch 评分，按 split 分组统计。"""
    total = {"qrr_correct": 0, "qrr_total": 0,
             "trr_hour_correct": 0, "trr_quadrant_correct": 0,
             "trr_adjacent_correct": 0, "trr_total": 0, "missing": 0}
    by_split = {}

    for r in scene_results:
        scene_id = r["scene_id"]
        split = scene_id.rsplit("_", 1)[0]  # n04_000000 → n04
        s = r["scores"]

        for k in total:
            total[k] += s[k]

        if split not in by_split:
            by_split[split] = {k: 0 for k in total}
        for k in total:
            by_split[split][k] += s[k]

    def _acc(correct, total_count):
        return round(correct / total_count, 4) if total_count > 0 else 0.0

    summary = {
        "overall": {
            "qrr_accuracy": _acc(total["qrr_correct"], total["qrr_total"]),
            "trr_hour_accuracy": _acc(total["trr_hour_correct"], total["trr_total"]),
            "trr_quadrant_accuracy": _acc(total["trr_quadrant_correct"], total["trr_total"]),
            "trr_adjacent_accuracy": _acc(total["trr_adjacent_correct"], total["trr_total"]),
            **total,
        },
        "by_split": {},
    }
    for split, counts in sorted(by_split.items()):
        summary["by_split"][split] = {
            "qrr_accuracy": _acc(counts["qrr_correct"], counts["qrr_total"]),
            "trr_hour_accuracy": _acc(counts["trr_hour_correct"], counts["trr_total"]),
            "trr_quadrant_accuracy": _acc(counts["trr_quadrant_correct"], counts["trr_total"]),
            **counts,
        }

    return summary
