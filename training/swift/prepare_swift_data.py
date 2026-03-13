#!/usr/bin/env python3
"""
将 ORDINARY-BENCH 数据转换为 ms-swift 训练所需的 JSONL 格式。

ms-swift 数据格式:
  {"images": ["path/to/img.png"], "messages": [...], "solution": "{...json...}"}

用法:
    python training/swift/prepare_swift_data.py --data-dir data-gen/output
    python training/swift/prepare_swift_data.py --data-dir data-gen/output --multi-view
    python training/swift/prepare_swift_data.py --mode cot-sft --data-dir data-gen/output
    python training/swift/prepare_swift_data.py --mode cot-grpo --data-dir data-gen/output
"""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Callable, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a spatial reasoning assistant analyzing a 3D scene image.
You will receive a list of objects visible in the image and a spatial question.

Question types:
1. QRR (distance comparison): Compare the 3D distance between two pairs of objects.
   Answer with exactly one of: "<" (first pair closer), "~=" (approximately equal), ">" (first pair farther).
2. TRR (clock direction): Imagine standing at ref1, facing toward ref2 (12 o'clock direction).
   Answer with the clock hour (integer 1-12) where the target object appears.

Think step by step, then wrap your final answer in <answer> tags.
For QRR: <answer><</answer>, <answer>~=</answer>, or <answer>></answer>
For TRR: <answer>7</answer> (integer 1-12)"""

SYSTEM_PROMPT_COT = """\
You are a spatial reasoning assistant analyzing a 3D scene image.

For each question, first analyze the spatial relationships in a <think> block using structured key=value pairs, then give your final answer in an <answer> block.

For TRR (clock direction) questions, your think block should include:
task=trr
anchor=<the object you stand at>
facing=<the object defining 12 o'clock>
target=<the object to locate>
quadrant=<1-4, where 1=12-3, 2=3-6, 3=6-9, 4=9-12>
hour=<your predicted clock hour 1-12>

For QRR (distance comparison) questions:
task=qrr
pair1=<obj_a,obj_b>
pair2=<obj_c,obj_d>
comparison=<symbol from {<, ~=, >}>

Then wrap your final answer: <answer>X</answer>"""

SYSTEM_PROMPT_MV = """\
You are a spatial reasoning assistant analyzing multiple views of a 3D scene.
You will receive {n_views} images showing the same scene from different camera angles, \
a list of objects visible in the scene, and a spatial question.

Question types:
1. QRR (distance comparison): Compare the 3D distance between two pairs of objects.
   Answer with exactly one of: "<" (first pair closer), "~=" (approximately equal), ">" (first pair farther).
2. TRR (clock direction): Imagine standing at ref1, facing toward ref2 (12 o'clock direction).
   Answer with the clock hour (integer 1-12) where the target object appears.

Use information from all views to reason about the 3D spatial layout.
Think step by step, then wrap your final answer in <answer> tags.
For QRR: <answer><</answer>, <answer>~=</answer>, or <answer>></answer>
For TRR: <answer>7</answer> (integer 1-12)"""

SYSTEM_PROMPT_COT_MV = """\
You are a spatial reasoning assistant analyzing multiple views of a 3D scene.
You will receive {n_views} images showing the same scene from different camera angles.

For each question, first analyze the spatial relationships in a <think> block \
using structured key=value pairs, then give your final answer in an <answer> block.

For TRR (clock direction) questions, your think block should include:
task=trr
anchor=<the object you stand at>
facing=<the object defining 12 o'clock>
target=<the object to locate>
quadrant=<1-4, where 1=12-3, 2=3-6, 3=6-9, 4=9-12>
hour=<your predicted clock hour 1-12>

For QRR (distance comparison) questions:
task=qrr
pair1=<obj_a,obj_b>
pair2=<obj_c,obj_d>
comparison=<symbol from {{<, ~=, >}}>

Then wrap your final answer: <answer>X</answer>"""

_MV_PROMPT_MAP = {
    id(SYSTEM_PROMPT): SYSTEM_PROMPT_MV,
    id(SYSTEM_PROMPT_COT): SYSTEM_PROMPT_COT_MV,
}


def format_objects(objects: list) -> str:
    lines = ["Objects in the image:"]
    for obj in objects:
        lines.append(f"  - {obj['id']}: {obj['desc']}")
    return "\n".join(lines)


def format_single_question(q: dict) -> str:
    if q["type"] == "qrr":
        p1a, p1b = q["pair1"]
        p2a, p2b = q["pair2"]
        return (
            f"Compare the distance between {p1a} and {p1b} "
            f"vs the distance between {p2a} and {p2b}. "
            f"Answer: < / ~= / >"
        )
    if q["type"] == "trr":
        return (
            f"Standing at {q['ref1']}, facing {q['ref2']} "
            f"(12 o'clock), what clock hour (1-12) is {q['target']} at?"
        )
    return ""


def _compute_metric(obj_a: dict, obj_b: dict, metric: str) -> float:
    if metric in ("dist3D", "dist3d"):
        a = obj_a.get("3d_coords", [0, 0, 0])
        b = obj_b.get("3d_coords", [0, 0, 0])
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))
    if metric in ("dist2D", "dist2d"):
        a = obj_a.get("pixel_coords", [0, 0])[:2]
        b = obj_b.get("pixel_coords", [0, 0])[:2]
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))
    return 0.0


def _load_scene_objects(scene_id: str, data_dir: Path) -> dict:
    scene_file = data_dir / "scenes" / f"{scene_id}.json"
    if not scene_file.exists():
        return {}
    with open(scene_file) as f:
        scene = json.load(f)
    return {obj["id"]: obj for obj in scene.get("objects", [])}


def _compute_qrr_ratio(q: dict, scene_objects: dict):
    p1a, p1b = q["pair1"]
    p2a, p2b = q["pair2"]
    metric = q.get("metric", "dist3D")
    if p1a not in scene_objects or p1b not in scene_objects:
        return None
    if p2a not in scene_objects or p2b not in scene_objects:
        return None
    d1 = _compute_metric(scene_objects[p1a], scene_objects[p1b], metric)
    d2 = _compute_metric(scene_objects[p2a], scene_objects[p2b], metric)
    if d2 == 0:
        return float("inf") if d1 > 0 else 1.0
    return d1 / d2


_missing_image_count = 0


def get_image_paths(scene_id: str, data_dir: Path, multi_view: bool, n_views: int) -> list:
    global _missing_image_count
    if multi_view:
        paths = [
            data_dir / "images" / "multi_view" / scene_id / f"view_{i}.png"
            for i in range(n_views)
        ]
    else:
        paths = [data_dir / "images" / "single_view" / f"{scene_id}.png"]
    result = []
    for path in paths:
        if not path.exists():
            logger.warning("图片不存在: %s", path)
            _missing_image_count += 1
        result.append(str(path.resolve()))
    return result


def get_gt_answer(q: dict):
    if q["type"] == "qrr":
        return q["gt_comparator"]
    if q["type"] == "trr":
        return q["gt_hour"]
    return None


def _build_user_content(objects: list, q: dict, images: list[str]) -> str:
    obj_text = format_objects(objects)
    question_text = f"{obj_text}\n\nQuestion:\n{format_single_question(q)}"
    if not images:
        return question_text
    image_prefix = "\n".join("<image>" for _ in range(len(images)))
    return f"{image_prefix}\n{question_text}"


def _build_messages(system_prompt: str, user_content: str, assistant_content: str = None) -> list:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    if assistant_content is not None:
        messages.append({"role": "assistant", "content": assistant_content})
    return messages


def _build_solution(q: dict, scene_objects: dict, enriched: bool) -> dict:
    gt_data = {
        "type": q["type"],
        "qid": q["qid"],
        "answer": get_gt_answer(q),
    }

    if q["type"] == "qrr":
        ratio = _compute_qrr_ratio(q, scene_objects) if scene_objects else None
        if ratio is not None:
            gt_data["ratio"] = round(ratio, 6)
        if enriched:
            gt_data["pair1"] = list(q["pair1"])
            gt_data["pair2"] = list(q["pair2"])
    elif q["type"] == "trr" and enriched:
        gt_data["gt_quadrant"] = q["gt_quadrant"]
        gt_data["gt_angle_deg"] = q["gt_angle_deg"]
        gt_data["target"] = q["target"]
        gt_data["ref1"] = q["ref1"]
        gt_data["ref2"] = q["ref2"]

    return gt_data


def _build_cot_trace(q: dict) -> str:
    if q["type"] == "trr":
        lines = [
            "<think>",
            "task=trr",
            f"anchor={q['ref1']}",
            f"facing={q['ref2']}",
            f"target={q['target']}",
            f"quadrant={q['gt_quadrant']}",
            f"hour={q['gt_hour']}",
            "</think>",
            f"<answer>{q['gt_hour']}</answer>",
        ]
        return "\n".join(lines)

    if q["type"] == "qrr":
        lines = [
            "<think>",
            "task=qrr",
            f"pair1={','.join(q['pair1'])}",
            f"pair2={','.join(q['pair2'])}",
            f"comparison={q['gt_comparator']}",
            "</think>",
            f"<answer>{q['gt_comparator']}</answer>",
        ]
        return "\n".join(lines)

    raise ValueError(f"unsupported question type: {q['type']}")


def _build_dataset_samples(
    question_file: Path,
    data_dir: Path,
    multi_view: bool,
    n_views: int,
    system_prompt: str,
    include_solution: bool,
    enriched_solution: bool,
    assistant_builder: Optional[Callable[[dict], str]],
) -> list:
    if multi_view and id(system_prompt) in _MV_PROMPT_MAP:
        system_prompt = _MV_PROMPT_MAP[id(system_prompt)].format(n_views=n_views)

    with open(question_file) as f:
        qdata = json.load(f)

    scene_id = qdata["scene_id"]
    objects = qdata["objects"]
    images = get_image_paths(scene_id, data_dir, multi_view, n_views)
    scene_objects = _load_scene_objects(scene_id, data_dir) if include_solution else {}

    samples = []
    for batch in qdata["batches"]:
        for q in batch["questions"]:
            user_content = _build_user_content(objects, q, images)
            assistant_content = assistant_builder(q) if assistant_builder is not None else None
            sample = {
                "images": images,
                "messages": _build_messages(system_prompt, user_content, assistant_content),
            }
            if include_solution:
                sample["solution"] = json.dumps(
                    _build_solution(q, scene_objects, enriched_solution),
                    ensure_ascii=False,
                )
            samples.append(sample)

    return samples


def build_sft_samples(question_file: Path, data_dir: Path, multi_view: bool, n_views: int) -> list:
    return _build_dataset_samples(
        question_file=question_file,
        data_dir=data_dir,
        multi_view=multi_view,
        n_views=n_views,
        system_prompt=SYSTEM_PROMPT,
        include_solution=False,
        enriched_solution=False,
        assistant_builder=lambda q: f"<answer>{get_gt_answer(q)}</answer>",
    )


def build_cot_sft_samples(question_file: Path, data_dir: Path, multi_view: bool, n_views: int) -> list:
    return _build_dataset_samples(
        question_file=question_file,
        data_dir=data_dir,
        multi_view=multi_view,
        n_views=n_views,
        system_prompt=SYSTEM_PROMPT_COT,
        include_solution=False,
        enriched_solution=False,
        assistant_builder=_build_cot_trace,
    )


def build_samples(question_file: Path, data_dir: Path, multi_view: bool, n_views: int) -> list:
    return _build_dataset_samples(
        question_file=question_file,
        data_dir=data_dir,
        multi_view=multi_view,
        n_views=n_views,
        system_prompt=SYSTEM_PROMPT,
        include_solution=True,
        enriched_solution=False,
        assistant_builder=None,
    )


def build_cot_grpo_samples(question_file: Path, data_dir: Path, multi_view: bool, n_views: int) -> list:
    return _build_dataset_samples(
        question_file=question_file,
        data_dir=data_dir,
        multi_view=multi_view,
        n_views=n_views,
        system_prompt=SYSTEM_PROMPT_COT,
        include_solution=True,
        enriched_solution=True,
        assistant_builder=None,
    )


MODE_CONFIG = {
    "grpo": {
        "builder": build_samples,
        "default_output_dir": "prepared_data/swift/grpo",
        "prefix": "",
    },
    "sft": {
        "builder": build_sft_samples,
        "default_output_dir": "prepared_data/swift/sft",
        "prefix": "sft_",
    },
    "cot-sft": {
        "builder": build_cot_sft_samples,
        "default_output_dir": "prepared_data/swift/cot_sft",
        "prefix": "cot_sft_",
    },
    "cot-grpo": {
        "builder": build_cot_grpo_samples,
        "default_output_dir": "prepared_data/swift/cot_grpo",
        "prefix": "",
    },
}


def write_jsonl(samples: list, path: Path):
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    logger.info("写入: %s (%d 行)", path, len(samples))


def main():
    parser = argparse.ArgumentParser(description="生成 ms-swift 训练数据")
    parser.add_argument(
        "--mode",
        choices=["grpo", "sft", "cot-sft", "cot-grpo"],
        default="grpo",
        help="数据模式: grpo/sft/cot-sft/cot-grpo (默认: grpo)",
    )
    parser.add_argument("--data-dir", default="data-gen/output")
    parser.add_argument("--questions-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--multi-view", action="store_true")
    parser.add_argument("--n-views", type=int, default=4)
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="允许图片缺失（仅警告，不退出）",
    )
    args = parser.parse_args()

    mode_config = MODE_CONFIG[args.mode]
    data_dir = Path(args.data_dir).resolve()
    questions_dir = (
        Path(args.questions_dir).resolve()
        if args.questions_dir
        else Path("VLM-test/output/questions").resolve()
    )
    output_dir = Path(args.output_dir or mode_config["default_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = data_dir / "train_scenes.json"
    test_file = data_dir / "test_scenes.json"
    if not train_file.exists() or not test_file.exists():
        logger.error("未找到 train/test 划分，请先运行 split_train_test.py")
        return

    with open(train_file) as f:
        train_ids = {scene["scene_id"] for scene in json.load(f)}
    with open(test_file) as f:
        test_ids = {scene["scene_id"] for scene in json.load(f)}

    logger.info("训练集: %d 场景, 测试集: %d 场景", len(train_ids), len(test_ids))

    question_files = sorted(questions_dir.glob("*.json"))
    if not question_files:
        logger.error("未找到问题文件: %s", questions_dir)
        return

    build_fn = mode_config["builder"]
    logger.info("模式: %s", args.mode)

    all_ids = train_ids | test_ids
    q_scene_ids = {qf.stem for qf in question_files}

    missing = all_ids - q_scene_ids
    if missing:
        logger.error("以下场景缺少问题文件 (%d 个): %s", len(missing), sorted(missing)[:10])
        logger.error("请检查 --questions-dir 是否与 --data-dir 匹配，或重新生成问题")
        return

    extra = q_scene_ids - all_ids
    if extra:
        logger.warning("跳过 %d 个不在 train/test 中的问题文件", len(extra))

    train_samples, test_samples = [], []
    for qf in question_files:
        scene_id = qf.stem
        samples = build_fn(qf, data_dir, args.multi_view, args.n_views)
        if scene_id in train_ids:
            train_samples.extend(samples)
        elif scene_id in test_ids:
            test_samples.extend(samples)

    if _missing_image_count > 0:
        if args.allow_missing_images:
            logger.warning("共 %d 张图片缺失（已通过 --allow-missing-images 跳过）", _missing_image_count)
        else:
            logger.error("共 %d 张图片缺失，请检查图片目录。使用 --allow-missing-images 跳过此检查", _missing_image_count)
            return

    prefix = mode_config["prefix"]
    train_path = output_dir / f"{prefix}train.jsonl"
    test_path = output_dir / f"{prefix}test.jsonl"
    write_jsonl(train_samples, train_path)
    write_jsonl(test_samples, test_path)

    stats = {
        "mode": args.mode,
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "multi_view": args.multi_view,
    }
    if args.mode in {"grpo", "cot-grpo"}:
        stats["train_qrr"] = sum(1 for sample in train_samples if '"type": "qrr"' in sample.get("solution", ""))
        stats["train_trr"] = sum(1 for sample in train_samples if '"type": "trr"' in sample.get("solution", ""))

    with open(output_dir / f"{prefix}stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n=== ms-swift 数据准备完成 ({args.mode}) ===")
    print(f"训练集: {len(train_samples)} -> {train_path}")
    print(f"测试集: {len(test_samples)} -> {test_path}")


if __name__ == "__main__":
    main()
