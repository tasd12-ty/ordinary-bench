#!/usr/bin/env python3
"""
将 ORDINARY-BENCH 数据转换为 ms-swift GRPO 训练所需的 JSONL 格式。

ms-swift 数据格式:
  {"images": ["path/to/img.png"], "messages": [...], "solution": "{...json...}"}

用法:
    python training/swift/prepare_swift_data.py --data-dir data-gen/output
    python training/swift/prepare_swift_data.py --data-dir data-gen/output --multi-view
"""

import argparse
import json
import logging
import math
from pathlib import Path

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

Respond ONLY with the answer. For QRR: "<", "~=", or ">". For TRR: an integer 1-12."""


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
    elif q["type"] == "trr":
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
    elif metric in ("dist2D", "dist2d"):
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


def _compute_qrr_ratio(q: dict, scene_objects: dict) -> float:
    p1a, p1b = q["pair1"]
    p2a, p2b = q["pair2"]
    metric = q.get("metric", "dist3D")
    if p1a not in scene_objects or p1b not in scene_objects:
        return 1.0
    if p2a not in scene_objects or p2b not in scene_objects:
        return 1.0
    d1 = _compute_metric(scene_objects[p1a], scene_objects[p1b], metric)
    d2 = _compute_metric(scene_objects[p2a], scene_objects[p2b], metric)
    if d2 == 0:
        return float('inf') if d1 > 0 else 1.0
    return d1 / d2


def get_image_paths(scene_id: str, data_dir: Path, multi_view: bool, n_views: int) -> list:
    if multi_view:
        paths = [
            data_dir / "images" / "multi_view" / scene_id / f"view_{i}.png"
            for i in range(n_views)
        ]
    else:
        paths = [data_dir / "images" / "single_view" / f"{scene_id}.png"]
    result = []
    for p in paths:
        if not p.exists():
            logger.warning("图片不存在: %s", p)
        result.append(str(p.resolve()))
    return result


def get_gt_answer(q: dict):
    if q["type"] == "qrr":
        return q["gt_comparator"]
    elif q["type"] == "trr":
        return q["gt_hour"]
    return None


def build_sft_samples(question_file: Path, data_dir: Path, multi_view: bool, n_views: int) -> list:
    """SFT 模式：包含 assistant 回复，不含 solution 字段。"""
    with open(question_file) as f:
        qdata = json.load(f)

    scene_id = qdata["scene_id"]
    objects = qdata["objects"]
    images = get_image_paths(scene_id, data_dir, multi_view, n_views)
    obj_text = format_objects(objects)

    n_images = len(images)
    image_prefix = "\n".join("<image>" for _ in range(n_images))

    samples = []
    for batch in qdata["batches"]:
        for q in batch["questions"]:
            question_text = f"{obj_text}\n\nQuestion:\n{format_single_question(q)}"
            user_content = f"{image_prefix}\n{question_text}" if n_images > 0 else question_text
            gt = get_gt_answer(q)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": str(gt)},
            ]

            sample = {"images": images, "messages": messages}
            samples.append(sample)

    return samples


def build_samples(question_file: Path, data_dir: Path, multi_view: bool, n_views: int) -> list:
    with open(question_file) as f:
        qdata = json.load(f)

    scene_id = qdata["scene_id"]
    objects = qdata["objects"]
    images = get_image_paths(scene_id, data_dir, multi_view, n_views)
    obj_text = format_objects(objects)
    scene_objects = _load_scene_objects(scene_id, data_dir)

    n_images = len(images)
    image_prefix = "\n".join("<image>" for _ in range(n_images))

    samples = []
    for batch in qdata["batches"]:
        for q in batch["questions"]:
            question_text = f"{obj_text}\n\nQuestion:\n{format_single_question(q)}"
            user_content = f"{image_prefix}\n{question_text}" if n_images > 0 else question_text

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]

            gt_data = {
                "type": q["type"],
                "qid": q["qid"],
                "answer": get_gt_answer(q),
            }
            if q["type"] == "qrr" and scene_objects:
                gt_data["ratio"] = round(_compute_qrr_ratio(q, scene_objects), 6)

            sample = {
                "images": images,
                "messages": messages,
                "solution": json.dumps(gt_data, ensure_ascii=False),
            }
            samples.append(sample)

    return samples


def main():
    parser = argparse.ArgumentParser(description="生成 ms-swift 训练数据")
    parser.add_argument("--mode", choices=["grpo", "sft"], default="grpo",
                        help="数据模式: grpo=含solution字段, sft=含assistant回复 (默认: grpo)")
    parser.add_argument("--data-dir", default="data-gen/output")
    parser.add_argument("--questions-dir", default=None)
    parser.add_argument("--output-dir", default="training/swift/data")
    parser.add_argument("--multi-view", action="store_true")
    parser.add_argument("--n-views", type=int, default=4)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    questions_dir = Path(args.questions_dir).resolve() if args.questions_dir else Path("VLM-test/output/questions").resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = data_dir / "train_scenes.json"
    test_file = data_dir / "test_scenes.json"
    if not train_file.exists() or not test_file.exists():
        logger.error("未找到 train/test 划分，请先运行 split_train_test.py")
        return

    with open(train_file) as f:
        train_ids = {s["scene_id"] for s in json.load(f)}
    with open(test_file) as f:
        test_ids = {s["scene_id"] for s in json.load(f)}

    logger.info(f"训练集: {len(train_ids)} 场景, 测试集: {len(test_ids)} 场景")

    question_files = sorted(questions_dir.glob("*.json"))
    if not question_files:
        logger.error(f"未找到问题文件: {questions_dir}")
        return

    build_fn = build_sft_samples if args.mode == "sft" else build_samples
    logger.info(f"模式: {args.mode}")

    all_ids = train_ids | test_ids
    q_scene_ids = {qf.stem for qf in question_files}
    matched = q_scene_ids & all_ids
    if not matched:
        logger.error("问题文件与 train/test 场景无交集！请检查 --data-dir 和 --questions-dir 是否匹配")
        return
    unmatched = q_scene_ids - all_ids
    if unmatched:
        logger.warning("跳过 %d 个不在 train/test 中的问题文件: %s", len(unmatched), sorted(unmatched)[:5])

    train_samples, test_samples = [], []
    for qf in question_files:
        scene_id = qf.stem
        samples = build_fn(qf, data_dir, args.multi_view, args.n_views)
        if scene_id in train_ids:
            train_samples.extend(samples)
        elif scene_id in test_ids:
            test_samples.extend(samples)

    def write_jsonl(samples, path):
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        logger.info(f"写入: {path} ({len(samples)} 行)")

    prefix = "sft_" if args.mode == "sft" else ""
    write_jsonl(train_samples, output_dir / f"{prefix}train.jsonl")
    write_jsonl(test_samples, output_dir / f"{prefix}test.jsonl")

    stats = {
        "mode": args.mode,
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "multi_view": args.multi_view,
    }
    if args.mode == "grpo":
        stats["train_qrr"] = sum(1 for s in train_samples if '"type": "qrr"' in s.get("solution", ""))
        stats["train_trr"] = sum(1 for s in train_samples if '"type": "trr"' in s.get("solution", ""))
    with open(output_dir / f"{prefix}stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    train_path = output_dir / f"{prefix}train.jsonl"
    test_path = output_dir / f"{prefix}test.jsonl"
    print(f"\n=== ms-swift 数据准备完成 ({args.mode}) ===")
    print(f"训练集: {len(train_samples)} -> {train_path}")
    print(f"测试集: {len(test_samples)} -> {test_path}")


if __name__ == "__main__":
    main()
