"""
ORDINARY-BENCH CoT ms-swift 奖励函数插件。

注册三个独立的 ORM:
  - ordinary_bench_cot_accuracy
  - ordinary_bench_cot_reasoning
  - ordinary_bench_cot_format
"""

import json
import logging
import math
import os
import random
import re
import threading
from datetime import datetime, timezone
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

# ── Rollout 日志：记录模型实际输出用于训练监控 ──
_ROLLOUT_LOG_INTERVAL = 50   # 每 50 次调用记录一批
_ROLLOUT_SAMPLE_SIZE = 4     # 每批最多采样 4 条
_rollout_call_count = 0
_rollout_lock = threading.Lock()
_rollout_log_dir = os.environ.get("ROLLOUT_LOG_DIR", "")


def _get_rollout_log_path() -> Optional[str]:
    if not _rollout_log_dir:
        return None
    return os.path.join(_rollout_log_dir, "rollout_log.jsonl")


def _maybe_log_rollout(
    completions: List[str],
    solution: List[str],
    accuracy_scores: List[float],
    kwargs: dict,
):
    """周期性采样 rollout 并写入 JSONL 日志。"""
    global _rollout_call_count
    log_path = _get_rollout_log_path()
    if log_path is None:
        return

    with _rollout_lock:
        _rollout_call_count += 1
        current_count = _rollout_call_count

    if current_count % _ROLLOUT_LOG_INTERVAL != 0:
        return

    n = len(completions)
    sample_indices = random.sample(range(n), min(_ROLLOUT_SAMPLE_SIZE, n))

    samples = []
    for idx in sample_indices:
        completion = completions[idx]
        sol_str = solution[idx]
        try:
            gt = json.loads(sol_str)
        except (json.JSONDecodeError, TypeError):
            continue

        think_dict, answer_tag, parse_info = _parse_think_block(completion)
        reasoning_score = _score_reasoning(completion, gt)
        format_score = _score_format(completion, gt)

        samples.append({
            "completion": completion[:500],  # 截断避免日志过大
            "solution_type": gt.get("type"),
            "gt_answer": gt.get("answer"),
            "scores": {
                "accuracy": accuracy_scores[idx],
                "reasoning": round(reasoning_score, 4),
                "format": round(format_score, 4),
            },
            "think_keys": parse_info.get("think_keys", []),
            "parse_path": _parse_answer_with_path(completion, gt.get("type", ""))[1],
            "completion_length": len(completion or ""),
        })

    if not samples:
        return

    record = {
        "call_count": current_count,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "samples": samples,
    }

    try:
        with open(log_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.warning("rollout log write failed: %s", e)

SUPPORTED_THINK_KEYS = {
    "task",
    "anchor",
    "facing",
    "target",
    "quadrant",
    "hour",
    "pair1",
    "pair2",
    "comparison",
}
TRR_REQUIRED_KEYS = {"task", "anchor", "facing", "target", "quadrant", "hour"}
QRR_REQUIRED_KEYS = {"task", "pair1", "pair2", "comparison"}

_parse_path_counts: Dict[str, int] = {}
_parse_total = 0
_think_stats = {
    "total": 0,
    "has_think": 0,
    "parsed_think": 0,
    "has_answer_tag": 0,
    "complete": 0,
}
_think_key_counts: Dict[str, int] = {}
_length_stats = {
    "total": 0,
    "chars": 0,
    "le_64": 0,
    "le_128": 0,
    "le_256": 0,
    "gt_256": 0,
}
_PARSE_LOG_INTERVAL = 100


def _record_parse_path(path: str):
    global _parse_total
    _parse_path_counts[path] = _parse_path_counts.get(path, 0) + 1
    _parse_total += 1
    if _parse_total % _PARSE_LOG_INTERVAL == 0:
        tag_pct = _parse_path_counts.get("tag", 0) / _parse_total * 100
        logger.info(
            "cot_parse_path_stats total=%d tag=%.1f%% %s",
            _parse_total,
            tag_pct,
            dict(_parse_path_counts),
        )


def _record_think_parse_stats(info: dict):
    _think_stats["total"] += 1
    if info.get("has_think_tag"):
        _think_stats["has_think"] += 1
    if info.get("think_parse_ok"):
        _think_stats["parsed_think"] += 1
    if info.get("has_answer_tag"):
        _think_stats["has_answer_tag"] += 1
    if info.get("complete"):
        _think_stats["complete"] += 1

    for key in info.get("think_keys", []):
        _think_key_counts[key] = _think_key_counts.get(key, 0) + 1

    total = _think_stats["total"]
    if total % _PARSE_LOG_INTERVAL == 0:
        key_rates = {
            key: round(_think_key_counts.get(key, 0) / total * 100, 1)
            for key in sorted(SUPPORTED_THINK_KEYS)
            if _think_key_counts.get(key, 0) > 0
        }
        logger.info(
            "cot_think_stats total=%d think=%.1f%% parsed=%.1f%% answer_tag=%.1f%% complete=%.1f%% keys=%s",
            total,
            _think_stats["has_think"] / total * 100,
            _think_stats["parsed_think"] / total * 100,
            _think_stats["has_answer_tag"] / total * 100,
            _think_stats["complete"] / total * 100,
            key_rates,
        )


def _record_length_stats(text: str):
    length = len(text or "")
    _length_stats["total"] += 1
    _length_stats["chars"] += length
    if length <= 64:
        _length_stats["le_64"] += 1
    elif length <= 128:
        _length_stats["le_128"] += 1
    elif length <= 256:
        _length_stats["le_256"] += 1
    else:
        _length_stats["gt_256"] += 1

    total = _length_stats["total"]
    if total % _PARSE_LOG_INTERVAL == 0:
        logger.info(
            "cot_length_stats total=%d avg_chars=%.1f buckets={<=64:%d,<=128:%d,<=256:%d,>256:%d}",
            total,
            _length_stats["chars"] / total,
            _length_stats["le_64"],
            _length_stats["le_128"],
            _length_stats["le_256"],
            _length_stats["gt_256"],
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
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None


def _parse_answer_with_path(response: str, question_type: str) -> tuple:
    text = (response or "").strip()

    tag_content = _extract_from_answer_tag(text)
    if tag_content is not None:
        if question_type == "qrr":
            normalized = _normalize_qrr_answer(tag_content)
            if normalized is not None:
                return normalized, "tag"
        elif question_type == "trr":
            match = re.search(r"\b(\d{1,2})\b", tag_content)
            if match:
                value = int(match.group(1))
                if 1 <= value <= 12:
                    return value, "tag"

    if question_type == "qrr":
        match = re.search(r'"answer"\s*:\s*"([^"]+)"', text)
        if match:
            return _normalize_qrr_answer(match.group(1)), "json"
        match = re.search(r"\b(lt|gt|eq|approx|~=)\b", text, re.IGNORECASE)
        if match:
            return _normalize_qrr_answer(match.group(1)), "keyword"
        match = re.search(r'(?:^|[\s:=",({])([<>≈]|~=)(?:$|[\s,."\'}):\]])', text)
        if match:
            return _normalize_qrr_answer(match.group(1)), "symbol"
        normalized = _normalize_qrr_answer(text)
        if normalized is not None:
            return normalized, "fallback"
        return None, "unparseable"

    if question_type == "trr":
        match = re.search(r'"answer"\s*:\s*(\d+)', text)
        if match:
            value = int(match.group(1))
            if 1 <= value <= 12:
                return value, "json"
        match = re.search(r"(?:hour|position|at|answer|is)\s*[:\s]\s*(\d{1,2})\b", text, re.IGNORECASE)
        if match:
            value = int(match.group(1))
            if 1 <= value <= 12:
                return value, "context_num"
        matches = re.findall(r"\b(\d{1,2})\b", text)
        for num_str in reversed(matches):
            value = int(num_str)
            if 1 <= value <= 12:
                return value, "last_num"
        return None, "unparseable"

    return None, "unparseable"


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
        return closeness * 0.5
    if predicted == "~=":
        return closeness * 0.8
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


def _extract_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    match = re.search(r"-?\d+", str(value))
    if not match:
        return None
    return int(match.group(0))


def _normalize_obj_id(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _parse_pair_value(value: Optional[str]) -> Optional[FrozenSet[str]]:
    if value is None:
        return None
    parts = [_normalize_obj_id(part) for part in str(value).split(",")]
    parts = [part for part in parts if part]
    if len(parts) != 2:
        return None
    return frozenset(parts)


def _required_think_keys(question_type: str) -> Set[str]:
    return TRR_REQUIRED_KEYS if question_type == "trr" else QRR_REQUIRED_KEYS


def _parse_think_block(text: str) -> Tuple[dict, Optional[str], dict]:
    response = (text or "").strip()
    think_matches = re.findall(r"<think>(.*?)</think>", response, re.DOTALL | re.IGNORECASE)
    answer_tag = _extract_from_answer_tag(re.sub(r"<think>.*?</think>", " ", response, flags=re.DOTALL | re.IGNORECASE))

    think_dict = {}
    parsed_lines = 0
    line_count = 0

    if think_matches:
        think_content = think_matches[-1]
        lines = [line.strip() for line in think_content.splitlines() if line.strip()]
        line_count = len(lines)
        for line in lines:
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if key in SUPPORTED_THINK_KEYS and value:
                think_dict[key] = value
                parsed_lines += 1

    parse_info = {
        "has_think_tag": bool(think_matches),
        "has_answer_tag": answer_tag is not None,
        "think_parse_ok": parsed_lines > 0,
        "think_keys": sorted(think_dict.keys()),
        "think_line_count": line_count,
        "think_parsed_lines": parsed_lines,
    }
    return think_dict, answer_tag, parse_info


def _score_accuracy(response: str, gt: dict) -> float:
    q_type = gt["type"]
    gt_answer = gt["answer"]
    predicted, parse_path = _parse_answer_with_path(response, q_type)
    _record_parse_path(parse_path)
    if predicted is None:
        return 0.0
    if q_type == "qrr":
        return _score_qrr_soft(str(predicted), str(gt_answer), gt.get("ratio"))
    if q_type == "trr":
        try:
            return _score_trr(int(predicted), int(gt_answer))
        except (ValueError, TypeError):
            return 0.0
    return 0.0


def _score_reasoning_trr(think_dict: dict, gt: dict) -> float:
    score = 0.0
    if think_dict.get("task", "").strip().lower() == "trr":
        score += 0.05
    if _normalize_obj_id(think_dict.get("anchor")) == gt.get("ref1"):
        score += 0.10
    if _normalize_obj_id(think_dict.get("facing")) == gt.get("ref2"):
        score += 0.10
    if _normalize_obj_id(think_dict.get("target")) == gt.get("target"):
        score += 0.10

    quadrant = _extract_int(think_dict.get("quadrant"))
    if quadrant == gt.get("gt_quadrant"):
        score += 0.35

    hour = _extract_int(think_dict.get("hour"))
    if hour is not None:
        score += 0.30 * _score_trr(hour, int(gt["answer"]))

    return score


def _score_reasoning_qrr(think_dict: dict, gt: dict) -> float:
    score = 0.0
    if think_dict.get("task", "").strip().lower() == "qrr":
        score += 0.05

    pair1 = _parse_pair_value(think_dict.get("pair1"))
    if pair1 is not None and pair1 == frozenset(gt.get("pair1", [])):
        score += 0.15

    pair2 = _parse_pair_value(think_dict.get("pair2"))
    if pair2 is not None and pair2 == frozenset(gt.get("pair2", [])):
        score += 0.15

    comparison = think_dict.get("comparison")
    score += 0.65 * _score_qrr_soft(comparison, gt["answer"], gt.get("ratio"))
    return score


def _score_reasoning(response: str, gt: dict) -> float:
    think_dict, _, info = _parse_think_block(response)
    required_keys = _required_think_keys(gt["type"])
    info["complete"] = bool(info["has_think_tag"] and required_keys.issubset(think_dict))
    _record_think_parse_stats(info)
    _record_length_stats(response)

    if not info["has_think_tag"]:
        return 0.0
    if gt["type"] == "trr":
        return _score_reasoning_trr(think_dict, gt)
    if gt["type"] == "qrr":
        return _score_reasoning_qrr(think_dict, gt)
    return 0.0


def _score_format(response: str, gt: dict) -> float:
    think_dict, _, info = _parse_think_block(response)
    required_keys = _required_think_keys(gt["type"])
    has_complete_think = info["has_think_tag"] and required_keys.issubset(think_dict)
    has_answer_tag = info["has_answer_tag"]
    parsed_answer, _ = _parse_answer_with_path(response, gt["type"])

    if has_complete_think and has_answer_tag:
        return 1.0
    if info["has_think_tag"] and has_answer_tag:
        return 0.5
    if has_answer_tag:
        return 0.2
    if parsed_answer is not None:
        return 0.1
    return 0.0


try:
    from swift.plugin import orms

    _HAS_SWIFT = True
except ImportError:
    _HAS_SWIFT = False
    orms = {}


class OrdinaryBenchCotAccuracyORM:
    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        scores = []
        for completion, sol in zip(completions, solution):
            try:
                gt = json.loads(sol)
            except (json.JSONDecodeError, TypeError):
                scores.append(0.0)
                continue
            scores.append(_score_accuracy(completion, gt))
        _maybe_log_rollout(completions, solution, scores, kwargs)
        return scores


class OrdinaryBenchCotReasoningORM:
    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        scores = []
        for completion, sol in zip(completions, solution):
            try:
                gt = json.loads(sol)
            except (json.JSONDecodeError, TypeError):
                scores.append(0.0)
                continue
            scores.append(_score_reasoning(completion, gt))
        return scores


class OrdinaryBenchCotFormatORM:
    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        scores = []
        for completion, sol in zip(completions, solution):
            try:
                gt = json.loads(sol)
            except (json.JSONDecodeError, TypeError):
                scores.append(0.0)
                continue
            scores.append(_score_format(completion, gt))
        return scores


orms["ordinary_bench_cot_accuracy"] = OrdinaryBenchCotAccuracyORM
orms["ordinary_bench_cot_reasoning"] = OrdinaryBenchCotReasoningORM
orms["ordinary_bench_cot_format"] = OrdinaryBenchCotFormatORM
