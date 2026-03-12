#!/usr/bin/env python3
"""
方案1：Batch 提问模式。

从 output/questions/ 加载预生成的问题批次，
逐 batch 发送图片+问题给 VLM，解析响应并评分。

用法：
    python run_batch.py                        # 全部场景
    python run_batch.py --test-only            # 只跑测试集（140 场景）
    python run_batch.py --train-only           # 只跑训练集（560 场景）
    python run_batch.py --split n04            # 指定 split
    python run_batch.py --scene n04_000000     # 单场景
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# 将 VLM-test/ 加入 sys.path，使 dsl 模块可被 scoring/parser 等导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CONFIG
from vlm_client import make_client, load_image_base64, build_messages, call_vlm
from prompts import BATCH_SYSTEM_PROMPT, format_batch_user_prompt, REACT_CORRECTION_PROMPT
from response_parser import parse_batch_response
from scoring import score_batch_scene, aggregate_batch_results

# ReAct 最大纠正轮数
REACT_MAX_ROUNDS = 2
# 缺失率超过此阈值时触发纠正（如 0.2 = 20% 问题没解析到）
REACT_MISSING_THRESHOLD = 0.2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _model_dir_name(model: str) -> str:
    """将模型名转为安全的目录名（如 qwen/qwen3-vl → qwen--qwen3-vl）。"""
    return model.replace("/", "--")


def process_scene(scene_id: str, config: dict) -> dict:
    """处理单个场景的所有 batch，返回评分结果。"""
    questions_path = Path(config["questions_dir"]) / f"{scene_id}.json"
    with open(questions_path) as f:
        scene_data = json.load(f)

    # 加载图片
    image_path = Path(config["images_dir"]) / f"{scene_id}.png"
    image_b64 = load_image_base64(str(image_path))

    # 创建 API 客户端
    client = make_client(config["base_url"], config["api_key"])

    # 收集所有问题和预测
    all_questions = []
    all_predictions = {}

    # 按模型名建子目录，区分不同模型的结果
    model_dir = _model_dir_name(config["model"])
    raw_dir = Path(config["results_dir"]) / model_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for batch in scene_data["batches"]:
        batch_id = batch["batch_id"]
        questions = batch["questions"]
        all_questions.extend(questions)
        expected_qids = [q["qid"] for q in questions]

        # 构造初始 prompt
        user_prompt = format_batch_user_prompt(scene_data["objects"], questions)
        messages = build_messages(BATCH_SYSTEM_PROMPT, user_prompt, image_b64)

        # VLM 调用参数
        vlm_kwargs = dict(
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            max_retries=config["max_retries"],
            retry_base_delay=config["retry_base_delay"],
            timeout=config["timeout"],
            provider=config.get("provider", ""),
        )

        # === ReAct 循环：调用 → 解析 → 检查 → 纠正 ===
        logger.info(f"  {scene_id} batch {batch_id}/{scene_data['n_batches']-1} ({len(questions)} questions)")
        t0 = time.time()
        raw_response = call_vlm(client, messages, config["model"], **vlm_kwargs)
        elapsed = time.time() - t0

        predictions = parse_batch_response(raw_response, expected_qids)
        n_missing = sum(1 for v in predictions.values() if v is None)

        # 纠正循环：如果缺失过多，发送纠正 prompt 让模型重新输出
        react_round = 0
        while (react_round < REACT_MAX_ROUNDS
               and n_missing > len(expected_qids) * REACT_MISSING_THRESHOLD):
            react_round += 1
            missing_qids = [qid for qid, v in predictions.items() if v is None]
            logger.info(f"  {scene_id} batch {batch_id} ReAct 纠正 #{react_round}，"
                        f"缺失 {n_missing}/{len(expected_qids)}")

            # 追加 assistant 的原始回复 + user 的纠正请求
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

            # 用纠正响应补充缺失的预测
            correction_preds = parse_batch_response(correction_response, missing_qids)
            for qid, val in correction_preds.items():
                if val is not None:
                    predictions[qid] = val

            # 更新 raw_response 记录（追加纠正）
            raw_response = (raw_response or "") + f"\n\n--- ReAct #{react_round} ---\n" + (correction_response or "")
            n_missing = sum(1 for v in predictions.values() if v is None)

        # 保存原始响应（含纠正记录）
        raw_record = {
            "scene_id": scene_id, "batch_id": batch_id,
            "model": config["model"], "timestamp": time.time(),
            "elapsed_seconds": round(elapsed, 2),
            "react_rounds": react_round,
            "raw_response": raw_response,
        }
        with open(raw_dir / f"{scene_id}_batch_{batch_id}.json", "w") as f:
            json.dump(raw_record, f, indent=2, ensure_ascii=False)

        all_predictions.update(predictions)

    # 评分
    scores = score_batch_scene(all_predictions, all_questions)

    return {
        "scene_id": scene_id,
        "model": config["model"],
        "n_objects": scene_data["n_objects"],
        "n_batches": scene_data["n_batches"],
        "total_questions": len(all_questions),
        "scores": scores,
    }


def main():
    parser = argparse.ArgumentParser(description="方案1：Batch 提问模式")
    parser.add_argument("--split", default=None, help="只跑指定 split（如 n04）")
    parser.add_argument("--scene", default=None, help="只跑单个场景（如 n04_000000）")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--test-only", action="store_true",
                       help="只跑测试集场景（从 test_scenes.json 读取）")
    group.add_argument("--train-only", action="store_true",
                       help="只跑训练集场景（从 train_scenes.json 读取）")
    args = parser.parse_args()

    config = CONFIG.copy()
    if not config["api_key"]:
        logger.error("请设置环境变量 VLM_API_KEY")
        sys.exit(1)

    # 查找场景文件
    questions_dir = Path(config["questions_dir"])
    split_file_dir = Path(__file__).resolve().parent.parent.parent / "data-gen" / "output"
    if args.scene:
        scene_ids = [args.scene]
    else:
        files = sorted(questions_dir.glob("*.json"))
        scene_ids = [f.stem for f in files]
        if args.split:
            scene_ids = [s for s in scene_ids if s.startswith(args.split)]
        if args.test_only or args.train_only:
            split_name = "test_scenes.json" if args.test_only else "train_scenes.json"
            split_path = split_file_dir / split_name
            with open(split_path) as f:
                split_ids = {s["scene_id"] for s in json.load(f)}
            scene_ids = [s for s in scene_ids if s in split_ids]

    # 按文件大小降序，让 batch 多的场景先跑，提高并发利用率
    scene_ids.sort(key=lambda s: (questions_dir / f"{s}.json").stat().st_size, reverse=True)

    if not scene_ids:
        logger.error("未找到场景文件")
        sys.exit(1)

    logger.info(f"Batch 模式：{len(scene_ids)} 个场景，模型 {config['model']}")

    # 并发处理场景
    results = []
    with ThreadPoolExecutor(max_workers=config["max_concurrency"]) as pool:
        futures = {pool.submit(process_scene, sid, config): sid for sid in scene_ids}
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

    # 按模型名保存逐场景结果
    model_dir = _model_dir_name(config["model"])
    scenes_dir = Path(config["results_dir"]) / model_dir / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        with open(scenes_dir / f"{r['scene_id']}.json", "w") as f:
            json.dump(r, f, indent=2, ensure_ascii=False)

    # 汇总
    summary = aggregate_batch_results(results)
    summary["model"] = config["model"]
    summary["n_scenes"] = len(results)

    summary_path = Path(config["results_dir"]) / model_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 打印结果
    o = summary["overall"]
    print(f"\n=== Batch 模式结果 ({len(results)} 场景) ===")
    print(f"模型: {config['model']}")
    print(f"QRR 准确率: {o['qrr_accuracy']:.2%} ({o['qrr_correct']}/{o['qrr_total']})")
    print(f"TRR hour 准确率: {o['trr_hour_accuracy']:.2%} ({o['trr_hour_correct']}/{o['trr_total']})")
    print(f"TRR quadrant 准确率: {o['trr_quadrant_accuracy']:.2%} ({o['trr_quadrant_correct']}/{o['trr_total']})")
    print(f"缺失: {o['missing']}")


if __name__ == "__main__":
    main()
