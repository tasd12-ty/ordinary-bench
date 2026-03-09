#!/usr/bin/env python3
"""
重建 split 索引文件和 dataset_info.json。

增量生成后，organize_split() 会覆盖 splits/*.json，仅包含新场景。
本脚本扫描 scenes/ 目录下所有场景 JSON，按 split 分组重建完整索引。

用法：
    python rebuild_splits.py                          # 默认 ./output
    python rebuild_splits.py --output-dir ./output    # 指定目录
    python rebuild_splits.py --n-views 4              # 指定视角数（默认 4）
"""

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def rebuild(output_dir: str, n_views: int):
    output = Path(output_dir)
    scenes_dir = output / "scenes"
    splits_dir = output / "splits"

    if not scenes_dir.exists():
        logger.error(f"scenes 目录不存在: {scenes_dir}")
        return

    splits_dir.mkdir(parents=True, exist_ok=True)

    # 扫描所有场景 JSON，按 split 分组
    split_entries = defaultdict(list)

    for scene_file in sorted(scenes_dir.glob("*.json")):
        with open(scene_file) as f:
            scene = json.load(f)

        scene_id = scene.get("scene_id", scene_file.stem)
        split_name = scene.get("split", scene_id.rsplit("_", 1)[0])
        n_objects = scene.get("n_objects", len(scene.get("objects", [])))

        entry = {
            "scene_id": scene_id,
            "single_view_image": f"images/single_view/{scene_id}.png",
            "multi_view_images": [
                f"images/multi_view/{scene_id}/view_{i}.png"
                for i in range(n_views)
            ],
            "scene_path": f"scenes/{scene_id}.json",
            "n_objects": n_objects,
            "split": split_name,
        }
        split_entries[split_name].append(entry)

    # 写入 split 索引文件
    total_scenes = 0
    for split_name in sorted(split_entries.keys()):
        entries = split_entries[split_name]
        split_file = splits_dir / f"{split_name}.json"
        with open(split_file, "w") as f:
            json.dump(entries, f, indent=2)
        total_scenes += len(entries)
        logger.info(f"  {split_name}: {len(entries)} 个场景 -> {split_file}")

    # 更新 dataset_info.json
    info = {
        "name": "ORDINARY-BENCH Dataset",
        "version": "1.0",
        "created": datetime.now().isoformat(),
        "config": {
            "n_views": n_views,
        },
        "splits": {},
        "statistics": {},
        "total_scenes": total_scenes,
        "total_images": 0,
    }

    total_images = 0
    for split_name in sorted(split_entries.keys()):
        entries = split_entries[split_name]
        n = len(entries)
        # 从 split 名称推断物体数量（如 n04 -> 4）
        obj_count = int(split_name[1:]) if split_name[1:].isdigit() else 0
        info["splits"][split_name] = {
            "n_scenes": n,
            "min_objects": obj_count,
            "max_objects": obj_count,
        }
        info["statistics"][split_name] = {
            "n_scenes": n,
            "n_single_view_images": n,
            "n_multi_view_images": n * n_views,
        }
        total_images += n + n * n_views

    info["total_images"] = total_images

    info_file = output / "dataset_info.json"
    with open(info_file, "w") as f:
        json.dump(info, f, indent=2)
    logger.info(f"\n总计: {total_scenes} 个场景, {total_images} 张图片")
    logger.info(f"已更新: {info_file}")


def main():
    parser = argparse.ArgumentParser(
        description="重建 split 索引文件和 dataset_info.json"
    )
    parser.add_argument(
        "--output-dir", "-o", default="./output",
        help="data-gen 输出目录（默认: ./output）",
    )
    parser.add_argument(
        "--n-views", type=int, default=4,
        help="每个场景的视角数（默认: 4）",
    )
    args = parser.parse_args()
    rebuild(args.output_dir, args.n_views)


if __name__ == "__main__":
    main()
