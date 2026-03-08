#!/usr/bin/env python3
"""
多视角 Batch 提问模式。

与 run_batch.py 相同的评测流程，但发送多张视角图片给 VLM。
图片来源：data-gen/output/images/multi_view/{scene_id}/view_0.png ~ view_3.png

用法：
    python run_multi_view.py                           # 全部场景，4 视角
    python run_multi_view.py --n-views 2               # 只发前 2 张视角
    python run_multi_view.py --split n04               # 指定 split
    python run_multi_view.py --scene n04_000000        # 单场景
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CONFIG
from vlm_client import make_client, load_image_base64, build_multi_view_messages, call_vlm
from prompts import MULTI_VIEW_SYSTEM_PROMPT, format_batch_user_prompt, REACT_CORRECTION_PROMPT
from response_parser import parse_batch_response
from scoring import score_batch_scene, aggregate_batch_results

REACT_MAX_ROUNDS = 2
REACT_MISSING_THRESHOLD = 0.2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _model_dir_name(model: str) -> str:
    return model.replace("/", "--")


def _load_multi_view_images(scene_id: str, images_dir: str, n_views: int) -> list:
    """加载场景的多视角图片，返回 base64 列表。"""
    scene_dir = Path(images_dir) / scene_id
    images_b64 = []
    for i in range(n_views):
        img_path = scene_dir / f"view_{i}.png"
        if not img_path.exists():
            raise FileNotFoundError(f"视角图片不存在: {img_path}")
        images_b64.append(load_image_base64(str(img_path)))
    return images_b64


def process_scene(scene_id: str, config: dict, n_views: int) -> dict:
    """处理单个场景的所有 batch（多视角模式）。"""
    questions_path = Path(config["questions_dir"]) / f"{scene_id}.json"
    with open(questions_path) as f:
        scene_data = json.load(f)

    # 加载多视角图片
    images_b64 = _load_multi_view_images(
        scene_id, config["multi_view_images_dir"], n_views)

    client = make_client(config["base_url"], config["api_key"])

    system_prompt = MULTI_VIEW_SYSTEM_PROMPT.format(n_views=n_views)

    all_questions = []
    all_predictions = {}

    model_dir = _model_dir_name(config["model"]) + "_multi_view"
    raw_dir = Path(config["results_dir"]) / model_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for batch in scene_data["batches"]:
        batch_id = batch["batch_id"]
        questions = batch["questions"]
        all_questions.extend(questions)
        expected_qids = [q["qid"] for q in questions]

        user_prompt = format_batch_user_prompt(scene_data["objects"], questions)
        messages = build_multi_view_messages(system_prompt, user_prompt, images_b64)

        vlm_kwargs = dict(
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            max_retries=config["max_retries"],
            retry_base_delay=config["retry_base_delay"],
            timeout=config["timeout"],
            provider=config.get("provider", ""),
        )

        logger.info(f"  {scene_id} batch {batch_id}/{scene_data['n_batches']-1} "
                     f"({len(questions)} questions, {n_views} views)")
        t0 = time.time()
        raw_response = call_vlm(client, messages, config["model"], **vlm_kwargs)
        elapsed = time.time() - t0

        predictions = parse_batch_response(raw_response, expected_qids)
        n_missing = sum(1 for v in predictions.values() if v is None)

        # ReAct 纠正循环
        react_round = 0
        while (react_round < REACT_MAX_ROUNDS
               and n_missing > len(expected_qids) * REACT_MISSING_THRESHOLD):
            react_round += 1
            missing_qids = [qid for qid, v in predictions.items() if v is None]
            logger.info(f"  {scene_id} batch {batch_id} ReAct #{react_round}，"
                        f"缺失 {n_missing}/{len(expected_qids)}")

            correction_messages = messages + [
                {"role": "assistant", "content": raw_response or ""},
                {"role": "user", "content": REACT_CORRECTION_PROMPT.format(
                    missing_qids=", ".join(missing_qids[:20]),
                    n_missing=n_missing,
                    n_total=len(expected_qids),
                )},
            ]

            t1 = time.time()
            correction_response = call_vlm(
                client, correction_messages, config["model"], **vlm_kwargs)
            elapsed += time.time() - t1

            correction_preds = parse_batch_response(correction_response, missing_qids)
            for qid, val in correction_preds.items():
                if val is not None:
                    predictions[qid] = val

            raw_response = (raw_response or "") + f"\n\n--- ReAct #{react_round} ---\n" + (correction_response or "")
            n_missing = sum(1 for v in predictions.values() if v is None)

        raw_record = {
            "scene_id": scene_id, "batch_id": batch_id,
            "model": config["model"], "n_views": n_views,
            "timestamp": time.time(),
            "elapsed_seconds": round(elapsed, 2),
            "react_rounds": react_round,
            "raw_response": raw_response,
        }
        with open(raw_dir / f"{scene_id}_batch_{batch_id}.json", "w") as f:
            json.dump(raw_record, f, indent=2, ensure_ascii=False)

        all_predictions.update(predictions)

    scores = score_batch_scene(all_predictions, all_questions)

    return {
        "scene_id": scene_id,
        "model": config["model"],
        "n_views": n_views,
        "n_objects": scene_data["n_objects"],
        "n_batches": scene_data["n_batches"],
        "total_questions": len(all_questions),
        "scores": scores,
    }


def main():
    parser = argparse.ArgumentParser(description="多视角 Batch 提问模式")
    parser.add_argument("--split", default=None, help="只跑指定 split（如 n04）")
    parser.add_argument("--scene", default=None, help="只跑单个场景（如 n04_000000）")
    parser.add_argument("--n-views", type=int, default=4, choices=[1, 2, 3, 4],
                        help="发送的视角数量（默认 4）")
    args = parser.parse_args()

    config = CONFIG.copy()
    if not config["api_key"]:
        logger.error("请设置环境变量 VLM_API_KEY")
        sys.exit(1)

    questions_dir = Path(config["questions_dir"])
    if args.scene:
        scene_ids = [args.scene]
    else:
        files = sorted(questions_dir.glob("*.json"))
        scene_ids = [f.stem for f in files]
        if args.split:
            scene_ids = [s for s in scene_ids if s.startswith(args.split)]

    # 按文件大小降序，让 batch 多的场景先跑
    scene_ids.sort(key=lambda s: (questions_dir / f"{s}.json").stat().st_size, reverse=True)

    if not scene_ids:
        logger.error("未找到场景文件")
        sys.exit(1)

    logger.info(f"Multi-view 模式：{len(scene_ids)} 个场景，{args.n_views} 视角，模型 {config['model']}")

    results = []
    with ThreadPoolExecutor(max_workers=config["max_concurrency"]) as pool:
        futures = {
            pool.submit(process_scene, sid, config, args.n_views): sid
            for sid in scene_ids
        }
        for future in as_completed(futures):
            sid = futures[future]
            try:
                result = future.result()
                results.append(result)
                s = result["scores"]
                logger.info(
                    f"  {sid}: QRR {s['qrr_correct']}/{s['qrr_total']}, "
                    f"TRR hour {s['trr_hour_correct']}/{s['trr_total']}, "
                    f"missing {s['missing']}"
                )
            except Exception as e:
                logger.error(f"  {sid} 失败: {e}")

    model_dir = _model_dir_name(config["model"]) + "_multi_view"
    scenes_dir = Path(config["results_dir"]) / model_dir / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        with open(scenes_dir / f"{r['scene_id']}.json", "w") as f:
            json.dump(r, f, indent=2, ensure_ascii=False)

    summary = aggregate_batch_results(results)
    summary["model"] = config["model"]
    summary["n_views"] = args.n_views
    summary["n_scenes"] = len(results)

    summary_path = Path(config["results_dir"]) / model_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    o = summary["overall"]
    print(f"\n=== Multi-view 模式结果 ({len(results)} 场景, {args.n_views} 视角) ===")
    print(f"模型: {config['model']}")
    print(f"QRR 准确率: {o['qrr_accuracy']:.2%} ({o['qrr_correct']}/{o['qrr_total']})")
    print(f"TRR hour 准确率: {o['trr_hour_accuracy']:.2%} ({o['trr_hour_correct']}/{o['trr_total']})")
    print(f"TRR quadrant 准确率: {o['trr_quadrant_accuracy']:.2%} ({o['trr_quadrant_correct']}/{o['trr_total']})")
    print(f"缺失: {o['missing']}")


if __name__ == "__main__":
    main()
