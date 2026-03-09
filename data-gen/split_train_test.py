#!/usr/bin/env python3
"""
将场景划分为训练集和测试集。

按场景 index 划分：每个 split (n04-n10) 内独立划分，
保证训练集和测试集的物体数量分布一致。

默认划分策略：
  - index 0-79 → train（80%）
  - index 80-99 → test（20%）

用法：
    python split_train_test.py --output-dir ./output
    python split_train_test.py --output-dir ./output --test-start 80
    python split_train_test.py --output-dir ./output --test-start 60 --test-end 80

输出：
    output/train_scenes.json  — 训练集场景 ID 列表
    output/test_scenes.json   — 测试集场景 ID 列表
    output/dataset_split.json — 划分统计信息
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="划分训练/测试数据集")
    parser.add_argument("--output-dir", "-o", default="./output",
                        help="data-gen 输出目录（包含 scenes/）")
    parser.add_argument("--test-start", type=int, default=80,
                        help="测试集起始 index（默认 80）")
    parser.add_argument("--test-end", type=int, default=None,
                        help="测试集结束 index（默认到最大）")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    scenes_dir = output_dir / "scenes"

    if not scenes_dir.exists():
        logger.error(f"scenes 目录不存在: {scenes_dir}")
        return

    # 收集所有场景
    scene_files = sorted(scenes_dir.glob("*.json"))
    if not scene_files:
        logger.error("未找到场景文件")
        return

    train_scenes = []
    test_scenes = []

    for sf in scene_files:
        scene_id = sf.stem  # e.g. n04_000023
        parts = scene_id.split("_")
        split_name = parts[0]  # e.g. n04
        index = int(parts[1])  # e.g. 23

        is_test = index >= args.test_start
        if args.test_end is not None:
            is_test = args.test_start <= index < args.test_end

        entry = {
            "scene_id": scene_id,
            "split": split_name,
            "index": index,
            "scene_path": f"scenes/{scene_id}.json",
            "single_view_image": f"images/single_view/{scene_id}.png",
        }

        if is_test:
            test_scenes.append(entry)
        else:
            train_scenes.append(entry)

    # 按 split 统计
    def count_by_split(scenes):
        counts = {}
        for s in scenes:
            sp = s["split"]
            counts[sp] = counts.get(sp, 0) + 1
        return counts

    train_by_split = count_by_split(train_scenes)
    test_by_split = count_by_split(test_scenes)

    # 保存
    train_file = output_dir / "train_scenes.json"
    test_file = output_dir / "test_scenes.json"
    split_info_file = output_dir / "dataset_split.json"

    with open(train_file, "w") as f:
        json.dump(train_scenes, f, indent=2)
    with open(test_file, "w") as f:
        json.dump(test_scenes, f, indent=2)

    split_info = {
        "total_scenes": len(train_scenes) + len(test_scenes),
        "train_scenes": len(train_scenes),
        "test_scenes": len(test_scenes),
        "test_index_range": f"{args.test_start}-{args.test_end or 'max'}",
        "train_by_split": train_by_split,
        "test_by_split": test_by_split,
    }
    with open(split_info_file, "w") as f:
        json.dump(split_info, f, indent=2)

    # 打印
    print(f"\n=== 数据集划分 ===")
    print(f"总场景数: {split_info['total_scenes']}")
    print(f"训练集: {len(train_scenes)} 场景")
    print(f"测试集: {len(test_scenes)} 场景")
    print(f"\n按 split 分布:")
    print(f"{'Split':>8} {'Train':>8} {'Test':>8}")
    print("-" * 28)
    for sp in sorted(set(list(train_by_split.keys()) + list(test_by_split.keys()))):
        print(f"{sp:>8} {train_by_split.get(sp, 0):>8} {test_by_split.get(sp, 0):>8}")
    print(f"\n输出: {train_file}, {test_file}")


if __name__ == "__main__":
    main()
