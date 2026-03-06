#!/usr/bin/env python3
"""
Generate VLM evaluation questions from data-gen scene JSONs.

Two output modes:
  1. questions/     - Batch mode: all QRR/TRR enumerated, split into batches
  2. extraction_tasks/ - Autonomous mode: objects + GT for later evaluation

Usage:
    python generate_questions.py --data ../data-gen/output
    python generate_questions.py --data ../data-gen/output --split n04 --batch-size 10
    python generate_questions.py --data ../data-gen/output --counts  # just show counts
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from extraction import parse_objects, object_description, extract_gt, load_scene
from question_bank import enumerate_qrr, enumerate_trr, make_batches, question_counts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def process_scene(scene_path: Path, batch_size: int, tau: float) -> dict:
    """Process one scene: extract GT, enumerate questions, build batches."""
    scene = load_scene(str(scene_path))
    scene_id = scene["scene_id"]
    objects = parse_objects(scene)

    # Build object descriptions for VLM prompts
    obj_list = []
    for obj_id in sorted(objects.keys()):
        obj = objects[obj_id]
        obj_list.append({
            "id": obj_id,
            "desc": object_description(obj),
        })

    # Enumerate all questions
    qrr_questions = enumerate_qrr(objects, tau=tau)
    trr_questions = enumerate_trr(objects, use_3d=True)

    all_questions = qrr_questions + trr_questions
    batches = make_batches(all_questions, batch_size)

    # Batch mode output
    batch_output = {
        "scene_id": scene_id,
        "image_path": f"images/single_view/{scene_id}.png",
        "objects": obj_list,
        "n_objects": len(objects),
        "total_qrr": len(qrr_questions),
        "total_trr": len(trr_questions),
        "total_questions": len(all_questions),
        "n_batches": len(batches),
        "tau": tau,
        "batches": batches,
    }

    # Extraction mode output
    gt = extract_gt(scene, tau=tau)
    extraction_output = {
        "scene_id": scene_id,
        "image_path": f"images/single_view/{scene_id}.png",
        "objects": obj_list,
        "n_objects": len(objects),
        "tau": tau,
        "ground_truth": gt,
    }

    return batch_output, extraction_output


def main():
    parser = argparse.ArgumentParser(
        description="Generate VLM evaluation questions from scene JSONs"
    )
    parser.add_argument(
        "--data", "-d", default=None,
        help="Path to data-gen output directory",
    )
    parser.add_argument(
        "--output", "-o", default="./output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--split", "-s", default=None,
        help="Only process scenes from this split (e.g. n04)",
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=20,
        help="Questions per batch (default: 20)",
    )
    parser.add_argument(
        "--tau", type=float, default=0.10,
        help="Tolerance parameter (default: 0.10)",
    )
    parser.add_argument(
        "--counts", action="store_true",
        help="Just print question count table and exit",
    )

    args = parser.parse_args()

    if args.counts:
        print(f"{'Objects':>8} {'QRR':>8} {'TRR':>8} {'Total':>8}")
        print("-" * 36)
        for n in range(4, 11):
            c = question_counts(n)
            print(f"{n:>8} {c['n_qrr']:>8} {c['n_trr']:>8} {c['total']:>8}")
        return

    if not args.data:
        logger.error("--data is required (path to data-gen output directory)")
        sys.exit(1)

    data_dir = Path(args.data)
    scenes_dir = data_dir / "scenes"
    if not scenes_dir.exists():
        logger.error(f"Scenes directory not found: {scenes_dir}")
        sys.exit(1)

    output_dir = Path(args.output)
    questions_dir = output_dir / "questions"
    extraction_dir = output_dir / "extraction_tasks"
    questions_dir.mkdir(parents=True, exist_ok=True)
    extraction_dir.mkdir(parents=True, exist_ok=True)

    # Find scene files
    scene_files = sorted(scenes_dir.glob("*.json"))
    if args.split:
        scene_files = [f for f in scene_files if f.stem.startswith(args.split)]

    if not scene_files:
        logger.error(f"No scene files found in {scenes_dir}")
        sys.exit(1)

    logger.info(f"Processing {len(scene_files)} scenes (batch_size={args.batch_size}, tau={args.tau})")

    total_qrr = 0
    total_trr = 0
    total_batches = 0

    for scene_path in scene_files:
        batch_out, extract_out = process_scene(
            scene_path, args.batch_size, args.tau
        )
        scene_id = batch_out["scene_id"]

        # Save batch mode
        with open(questions_dir / f"{scene_id}.json", 'w') as f:
            json.dump(batch_out, f, indent=2)

        # Save extraction mode
        with open(extraction_dir / f"{scene_id}.json", 'w') as f:
            json.dump(extract_out, f, indent=2)

        total_qrr += batch_out["total_qrr"]
        total_trr += batch_out["total_trr"]
        total_batches += batch_out["n_batches"]

        logger.info(
            f"  {scene_id}: {batch_out['n_objects']} objects, "
            f"{batch_out['total_qrr']} QRR + {batch_out['total_trr']} TRR = "
            f"{batch_out['total_questions']} questions, "
            f"{batch_out['n_batches']} batches"
        )

    # Summary
    summary = {
        "n_scenes": len(scene_files),
        "batch_size": args.batch_size,
        "tau": args.tau,
        "total_qrr": total_qrr,
        "total_trr": total_trr,
        "total_questions": total_qrr + total_trr,
        "total_batches": total_batches,
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone! {len(scene_files)} scenes processed")
    print(f"  QRR: {total_qrr}, TRR: {total_trr}, Total: {total_qrr + total_trr}")
    print(f"  Batches: {total_batches} (size={args.batch_size})")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
