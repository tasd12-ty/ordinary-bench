#!/usr/bin/env python3
"""
将 ORDINARY-BENCH 数据转换为 verl 可直接加载的 parquet 格式。

当前推荐路径:
  - rl:  每个问题一行，适用于 Qwen3-VL-32B + GRPO/PPO
  - sft: 每个 batch 一行，保留为实验性导出

关键约束:
  - RL prompt 输出为结构化 chat messages，而不是 JSON 字符串
  - images 输出为 [{"path": "..."}] 列表
  - 用户消息前自动补足 <image> 占位符，和图片数量一一对应
"""

import argparse
import json
import logging
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 提示模板 ──

SYSTEM_PROMPT_SINGLE = """\
You are a spatial reasoning assistant analyzing a 3D scene image.
You will receive a list of objects visible in the image and a spatial question.

Question types:
1. QRR (distance comparison): Compare the 3D distance between two pairs of objects.
   Answer with exactly one of: "<" (first pair closer), "~=" (approximately equal), ">" (first pair farther).
2. TRR (clock direction): Imagine standing at ref1, facing toward ref2 (12 o'clock direction).
   Answer with the clock hour (integer 1-12) where the target object appears.

Respond ONLY with the answer. For QRR: "<", "~=", or ">". For TRR: an integer 1-12."""

SYSTEM_PROMPT_BATCH = """\
You are a spatial reasoning assistant analyzing a 3D scene image.
You will receive a list of objects visible in the image and a set of spatial questions.

Question types:
1. QRR (distance comparison): Compare the 3D distance between two pairs of objects.
   Answer with exactly one of: "<" (first pair closer), "~=" (approximately equal), ">" (first pair farther).
2. TRR (clock direction): Imagine standing at ref1, facing toward ref2 (12 o'clock direction).
   Answer with the clock hour (integer 1-12) where the target object appears.

Respond ONLY with a JSON array. Each element must have "qid" and "answer".
For QRR: answer is a string "<", "~=", or ">".
For TRR: answer is an integer 1-12.

Example:
[{"qid": "qrr_0001", "answer": "<"}, {"qid": "trr_0001", "answer": 7}]"""

SYSTEM_PROMPT_MULTI_VIEW = """\
You are a spatial reasoning assistant analyzing a 3D scene from multiple viewpoints.
You will receive {n_views} images of the same scene taken from different camera angles,
followed by a list of objects visible in the scene and a spatial question.

Analyze ALL provided views to determine spatial relationships more accurately.

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


def format_batch_questions(questions: list) -> str:
    lines = ["Questions:"]
    for q in questions:
        if q["type"] == "qrr":
            p1a, p1b = q["pair1"]
            p2a, p2b = q["pair2"]
            lines.append(
                f"[{q['qid']}] Compare the distance between {p1a} and {p1b} "
                f"vs the distance between {p2a} and {p2b}. "
                f"Answer: < / ~= / >"
            )
        elif q["type"] == "trr":
            lines.append(
                f"[{q['qid']}] Standing at {q['ref1']}, facing {q['ref2']} "
                f"(12 o'clock), what clock hour (1-12) is {q['target']} at?"
            )
    return "\n".join(lines)


_missing_image_count = 0


def build_image_entries(scene_id: str, data_dir: Path, multi_view: bool, n_views: int = 4) -> list:
    """返回 verl 期望的图片字典列表。"""
    global _missing_image_count
    if multi_view:
        paths = [
            data_dir / "images" / "multi_view" / scene_id / f"view_{i}.png"
            for i in range(n_views)
        ]
    else:
        paths = [data_dir / "images" / "single_view" / f"{scene_id}.png"]

    images = []
    for path in paths:
        if not path.exists():
            logger.warning("图片不存在: %s", path)
            _missing_image_count += 1
        images.append({"path": str(path.resolve())})
    return images


def build_user_content(body: str, n_images: int) -> str:
    """在用户消息前补齐 verl 多模态 loader 所需的 <image> 占位符。"""
    if n_images <= 0:
        return body
    prefix = "\n".join("<image>" for _ in range(n_images))
    return f"{prefix}\n{body}"


def build_sft_prompt(system: str, user_text: str, n_images: int) -> str:
    """SFT 保留为字符串 prompt，便于后续按需要接入自定义 chat template。"""
    content = build_user_content(user_text, n_images)
    return (
        f"System:\n{system}\n\n"
        f"User:\n{content}\n\n"
        "Assistant:\n"
    )


def get_gt_answer(q: dict):
    """提取问题的 ground truth 答案。"""
    if q["type"] == "qrr":
        return q["gt_comparator"]
    elif q["type"] == "trr":
        return q["gt_hour"]
    return None


def _compute_metric(obj_a: dict, obj_b: dict, metric: str) -> float:
    """计算两个物体间的距离度量值。"""
    if metric in ("dist3D", "dist3d"):
        a = obj_a.get("3d_coords", [0, 0, 0])
        b = obj_b.get("3d_coords", [0, 0, 0])
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))
    elif metric in ("dist2D", "dist2d"):
        a = obj_a.get("pixel_coords", [0, 0])[:2]
        b = obj_b.get("pixel_coords", [0, 0])[:2]
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))
    elif metric in ("depthGap", "depthgap"):
        da = obj_a.get("pixel_coords", [0, 0, 0])[2] if len(obj_a.get("pixel_coords", [])) > 2 else 0
        db = obj_b.get("pixel_coords", [0, 0, 0])[2] if len(obj_b.get("pixel_coords", [])) > 2 else 0
        return abs(da - db)
    return 0.0


def _load_scene_objects(scene_id: str, data_dir: Path) -> dict:
    """加载场景 JSON，返回 {obj_id: obj_data} 映射。"""
    scene_file = data_dir / "scenes" / f"{scene_id}.json"
    if not scene_file.exists():
        return {}
    with open(scene_file) as f:
        scene = json.load(f)
    return {obj["id"]: obj for obj in scene.get("objects", [])}


def _compute_qrr_ratio(q: dict, scene_objects: dict):
    """计算 QRR 问题的距离比值 dist(pair1) / dist(pair2)。"""
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
        return float('inf') if d1 > 0 else 1.0
    return d1 / d2


def build_rl_samples(
    question_file: Path, data_dir: Path, multi_view: bool, n_views: int,
) -> list:
    """RL 模式：每个问题一行，包含 prompt + ground_truth（含距离比值用于软评分）。"""
    with open(question_file) as f:
        qdata = json.load(f)

    scene_id = qdata["scene_id"]
    objects = qdata["objects"]
    images = build_image_entries(scene_id, data_dir, multi_view, n_views)
    obj_text = format_objects(objects)
    split = scene_id.rsplit("_", 1)[0]

    # 加载场景 3D 坐标用于计算距离比值
    scene_objects = _load_scene_objects(scene_id, data_dir)

    if multi_view:
        system = SYSTEM_PROMPT_MULTI_VIEW.format(n_views=n_views)
    else:
        system = SYSTEM_PROMPT_SINGLE

    samples = []
    for batch in qdata["batches"]:
        for q in batch["questions"]:
            question_text = f"{obj_text}\n\nQuestion:\n{format_single_question(q)}"
            user_text = build_user_content(question_text, len(images))
            gt = get_gt_answer(q)

            prompt = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ]

            # 构造 ground truth，QRR 加入距离比值
            gt_data = {
                "type": q["type"],
                "qid": q["qid"],
                "answer": gt,
            }
            if q["type"] == "qrr" and scene_objects:
                ratio = _compute_qrr_ratio(q, scene_objects)
                if ratio is not None:
                    gt_data["ratio"] = round(ratio, 6)
            elif q["type"] == "trr":
                gt_data["angle_deg"] = q.get("gt_angle_deg", 0)

            sample = {
                "data_source": "ordinary-bench",
                "prompt": prompt,
                "ability": f"spatial_{q['type']}",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps(gt_data),
                },
                "extra_info": {
                    "scene_id": scene_id,
                    "split": split,
                    "n_objects": qdata["n_objects"],
                    "qid": q["qid"],
                    "question_type": q["type"],
                },
                "images": images,
            }
            samples.append(sample)

    return samples


def build_sft_samples(
    question_file: Path, data_dir: Path, multi_view: bool, n_views: int,
) -> list:
    """SFT 模式：每个 batch 一行，prompt + 正确 response。"""
    with open(question_file) as f:
        qdata = json.load(f)

    scene_id = qdata["scene_id"]
    objects = qdata["objects"]
    images = build_image_entries(scene_id, data_dir, multi_view, n_views)
    obj_text = format_objects(objects)
    split = scene_id.rsplit("_", 1)[0]

    # 加载场景 3D 坐标用于计算距离比值
    scene_objects = _load_scene_objects(scene_id, data_dir)

    samples = []
    for batch in qdata["batches"]:
        questions = batch["questions"]
        user_text = f"{obj_text}\n\n{format_batch_questions(questions)}"

        # 构造正确答案
        answers = []
        for q in questions:
            answers.append({"qid": q["qid"], "answer": get_gt_answer(q)})
        response = json.dumps(answers)

        prompt = build_sft_prompt(SYSTEM_PROMPT_BATCH, user_text, len(images))

        # 构造 ground truth，QRR 加入距离比值
        gt_list = []
        for q in questions:
            gt_item = {"qid": q["qid"], "type": q["type"], "answer": get_gt_answer(q)}
            if q["type"] == "qrr" and scene_objects:
                ratio = _compute_qrr_ratio(q, scene_objects)
                if ratio is not None:
                    gt_item["ratio"] = round(ratio, 6)
            gt_list.append(gt_item)

        sample = {
            "data_source": "ordinary-bench",
            "prompt": prompt,
            "ability": "spatial_reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps(gt_list),
            },
            "extra_info": {
                "scene_id": scene_id,
                "split": split,
                "n_objects": qdata["n_objects"],
                "batch_id": batch["batch_id"],
                "n_questions": batch["n_questions"],
            },
            "images": images,
            "response": response,
        }
        samples.append(sample)

    return samples


def main():
    parser = argparse.ArgumentParser(description="将 ORDINARY-BENCH 数据转换为 verl parquet 格式")
    parser.add_argument("--mode", choices=["rl", "sft"], default="rl",
                        help="数据模式: rl=每题一行, sft=每batch一行 (默认: rl)")
    parser.add_argument("--data-dir", default="../data-gen/output",
                        help="data-gen 输出目录")
    parser.add_argument("--questions-dir", default=None,
                        help="问题目录 (默认: ./output/questions)")
    parser.add_argument("--output-dir", default="./verl_data",
                        help="输出目录 (默认: ./verl_data)")
    parser.add_argument("--multi-view", action="store_true",
                        help="使用多视角图片")
    parser.add_argument("--n-views", type=int, default=4,
                        help="视角数量 (默认: 4)")
    parser.add_argument("--allow-missing-images", action="store_true",
                        help="允许图片缺失（仅警告，不退出）")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    questions_dir = Path(args.questions_dir) if args.questions_dir else Path("./output/questions")
    questions_dir = questions_dir.resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取 train/test 划分
    train_file = data_dir / "train_scenes.json"
    test_file = data_dir / "test_scenes.json"

    if not train_file.exists() or not test_file.exists():
        logger.error(f"未找到 train/test 划分文件，请先运行 split_train_test.py")
        logger.error(f"  需要: {train_file}")
        logger.error(f"  需要: {test_file}")
        return

    with open(train_file) as f:
        train_scenes = json.load(f)
    with open(test_file) as f:
        test_scenes = json.load(f)

    train_ids = {s["scene_id"] for s in train_scenes}
    test_ids = {s["scene_id"] for s in test_scenes}

    logger.info(f"模式: {args.mode}")
    logger.info(f"训练集: {len(train_ids)} 场景, 测试集: {len(test_ids)} 场景")
    logger.info(f"图片模式: {'multi_view' if args.multi_view else 'single_view'}")

    # 收集所有问题文件
    question_files = sorted(questions_dir.glob("*.json"))
    if not question_files:
        logger.error(f"未找到问题文件: {questions_dir}")
        return

    build_fn = build_rl_samples if args.mode == "rl" else build_sft_samples

    train_samples = []
    test_samples = []

    for qf in question_files:
        scene_id = qf.stem
        samples = build_fn(qf, data_dir, args.multi_view, args.n_views)

        if scene_id in train_ids:
            train_samples.extend(samples)
        elif scene_id in test_ids:
            test_samples.extend(samples)
        else:
            logger.warning(f"场景 {scene_id} 不在 train/test 划分中，跳过")

    logger.info(f"训练集样本: {len(train_samples)}")
    logger.info(f"测试集样本: {len(test_samples)}")

    if _missing_image_count > 0:
        if args.allow_missing_images:
            logger.warning("共 %d 张图片缺失（已通过 --allow-missing-images 跳过）", _missing_image_count)
        else:
            logger.error("共 %d 张图片缺失，请检查图片目录。使用 --allow-missing-images 跳过此检查", _missing_image_count)
            return

    def validate_samples(samples: list, mode: str) -> None:
        for idx, sample in enumerate(samples):
            if mode == "rl" and not isinstance(sample["prompt"], list):
                raise ValueError(f"RL prompt 必须是 list[dict]，第 {idx} 条不是")
            if mode == "sft" and not isinstance(sample["prompt"], str):
                raise ValueError(f"SFT prompt 必须是 string，第 {idx} 条不是")

            image_count = len(sample.get("images", []))
            if image_count == 0:
                continue

            if mode == "rl":
                user_messages = [m for m in sample["prompt"] if m["role"] == "user"]
                if not user_messages:
                    raise ValueError(f"第 {idx} 条缺少 user message")
                image_tokens = user_messages[0]["content"].count("<image>")
            else:
                image_tokens = sample["prompt"].count("<image>")

            if image_tokens != image_count:
                raise ValueError(
                    f"第 {idx} 条图片占位符数量不匹配: placeholders={image_tokens}, images={image_count}"
                )

    def write_parquet(samples: list, path: Path, mode: str) -> None:
        validate_samples(samples, mode)
        table = pa.Table.from_pylist(samples)
        pq.write_table(table, path)
        logger.info("写入: %s (%d 行)", path, table.num_rows)

    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"

    if train_samples:
        write_parquet(train_samples, train_path, args.mode)

    if test_samples:
        write_parquet(test_samples, test_path, args.mode)

    # 统计信息
    stats = {
        "mode": args.mode,
        "multi_view": args.multi_view,
        "n_views": args.n_views if args.multi_view else 1,
        "train_scenes": len(train_ids),
        "test_scenes": len(test_ids),
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "data_dir": str(data_dir),
        "questions_dir": str(questions_dir),
    }

    if train_samples:
        qrr_count = sum(1 for s in train_samples if "qrr" in s.get("ability", ""))
        trr_count = sum(1 for s in train_samples if "trr" in s.get("ability", ""))
        if args.mode == "rl":
            stats["train_qrr_samples"] = qrr_count
            stats["train_trr_samples"] = trr_count

    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n=== verl 数据准备完成 ({args.mode} 模式) ===")
    print(f"训练集: {len(train_samples)} 样本 -> {train_path}")
    print(f"测试集: {len(test_samples)} 样本 -> {test_path}")
    print(f"统计: {stats_path}")


if __name__ == "__main__":
    main()
