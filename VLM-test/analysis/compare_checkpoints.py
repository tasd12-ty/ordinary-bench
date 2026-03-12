#!/usr/bin/env python3
"""
对比多个 checkpoint 的评测结果，生成 CSV + Markdown 表格。

扫描 results/ 目录，按模型族和 checkpoint 编号排序，
标记每个指标的最佳 checkpoint。

用法:
    cd VLM-test
    python analysis/compare_checkpoints.py
    python analysis/compare_checkpoints.py --results-dir output/results --output-dir output/analysis
    python analysis/compare_checkpoints.py --family qwen3-8b --family qwen35-4b
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_result_dir(dirname: str) -> Optional[Tuple[str, str, str]]:
    """解析结果目录名为 (family, ckpt_tag, eval_type)。

    Examples:
        qwen3-8b--base           → (qwen3-8b, base, single_view)
        qwen3-8b--ckpt-5         → (qwen3-8b, ckpt-5, single_view)
        qwen3-8b--ckpt-5_multi_view → (qwen3-8b, ckpt-5, multi_view)
    """
    eval_type = "single_view"
    name = dirname
    if name.endswith("_multi_view"):
        eval_type = "multi_view"
        name = name[: -len("_multi_view")]

    if "--" not in name:
        return None

    family, ckpt_tag = name.split("--", 1)
    return family, ckpt_tag, eval_type


def ckpt_sort_key(tag: str) -> Tuple[int, int]:
    """排序键：base 排第一，其余按数字升序。"""
    if tag == "base":
        return (0, 0)
    m = re.match(r"ckpt-(\d+)", tag)
    if m:
        return (1, int(m.group(1)))
    return (2, 0)


def load_summary(results_dir: Path, dirname: str) -> Optional[dict]:
    """加载某个结果目录的 summary.json。"""
    path = results_dir / dirname / "summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def scan_results(
    results_dir: Path, families: Optional[List[str]] = None
) -> Dict[str, Dict[str, Dict[str, dict]]]:
    """扫描 results 目录，按 family → ckpt_tag → eval_type 组织。

    Returns:
        {family: {ckpt_tag: {"single_view": summary, "multi_view": summary}}}
    """
    data: Dict[str, Dict[str, Dict[str, dict]]] = {}

    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        parsed = parse_result_dir(d.name)
        if parsed is None:
            continue

        family, ckpt_tag, eval_type = parsed
        if families and family not in families:
            continue

        data.setdefault(family, {}).setdefault(ckpt_tag, {})
        summary = load_summary(results_dir, d.name)
        if summary:
            data[family][ckpt_tag][eval_type] = summary

    return data


METRICS = [
    "sv_qrr",
    "sv_trr_hour",
    "sv_trr_quad",
    "mv_qrr",
    "mv_trr_hour",
    "mv_trr_quad",
    "composite",
]


def build_table(data: dict) -> List[dict]:
    """从扫描结果构建对比表行。"""
    rows = []
    for family in sorted(data.keys()):
        ckpts = data[family]
        for ckpt_tag in sorted(ckpts.keys(), key=ckpt_sort_key):
            sv = ckpts[ckpt_tag].get("single_view", {})
            mv = ckpts[ckpt_tag].get("multi_view", {})
            sv_o = sv.get("overall", {})
            mv_o = mv.get("overall", {})

            row = {
                "family": family,
                "checkpoint": ckpt_tag,
                "sv_qrr": sv_o.get("qrr_accuracy"),
                "sv_trr_hour": sv_o.get("trr_hour_accuracy"),
                "sv_trr_quad": sv_o.get("trr_quadrant_accuracy"),
                "mv_qrr": mv_o.get("qrr_accuracy"),
                "mv_trr_hour": mv_o.get("trr_hour_accuracy"),
                "mv_trr_quad": mv_o.get("trr_quadrant_accuracy"),
                "missing_pct": sv_o.get("missing_rate"),
                "n_scenes": sv.get("n_scenes") or mv.get("n_scenes"),
            }

            # 综合分: 0.5 × SV QRR + 0.5 × SV TRR-Quad
            if row["sv_qrr"] is not None and row["sv_trr_quad"] is not None:
                row["composite"] = 0.5 * row["sv_qrr"] + 0.5 * row["sv_trr_quad"]
            else:
                row["composite"] = None

            rows.append(row)

    return rows


def find_best(rows: List[dict], family: str, metric: str) -> Optional[float]:
    """找某 family 下某指标的最大值。"""
    values = [
        r[metric] for r in rows if r["family"] == family and r[metric] is not None
    ]
    return max(values) if values else None


def fmt(val: Optional[float], best: Optional[float], as_pct: bool = True) -> str:
    """格式化数值，最佳值加粗。"""
    if val is None:
        return "-"
    s = f"{val:.2%}" if as_pct else f"{val:.4f}"
    if best is not None and abs(val - best) < 1e-6:
        return f"**{s}**"
    return s


def output_markdown(rows: List[dict], output_path: Path) -> str:
    """生成 Markdown 对比表，返回内容并写入文件。"""
    families = sorted(set(r["family"] for r in rows))
    bests = {
        f: {m: find_best(rows, f, m) for m in METRICS} for f in families
    }

    lines = [
        "# Checkpoint Comparison",
        "",
        "| Family | Checkpoint | SV QRR | SV TRR-Hour | SV TRR-Quad "
        "| MV QRR | MV TRR-Hour | MV TRR-Quad | Composite | Missing% |",
        "|" + "---|" * 10,
    ]

    for row in rows:
        b = bests[row["family"]]
        cols = [
            row["family"],
            row["checkpoint"],
            fmt(row["sv_qrr"], b["sv_qrr"]),
            fmt(row["sv_trr_hour"], b["sv_trr_hour"]),
            fmt(row["sv_trr_quad"], b["sv_trr_quad"]),
            fmt(row["mv_qrr"], b["mv_qrr"]),
            fmt(row["mv_trr_hour"], b["mv_trr_hour"]),
            fmt(row["mv_trr_quad"], b["mv_trr_quad"]),
            fmt(row["composite"], b["composite"]),
            fmt(row["missing_pct"], None),
        ]
        lines.append("| " + " | ".join(cols) + " |")

    content = "\n".join(lines) + "\n"
    output_path.write_text(content)
    return content


def output_csv(rows: List[dict], output_path: Path):
    """输出 CSV（方便 Excel/pandas 分析）。"""
    fieldnames = [
        "family",
        "checkpoint",
        "sv_qrr",
        "sv_trr_hour",
        "sv_trr_quad",
        "mv_qrr",
        "mv_trr_hour",
        "mv_trr_quad",
        "composite",
        "missing_pct",
        "n_scenes",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main():
    parser = argparse.ArgumentParser(description="对比 checkpoint 评测结果")
    parser.add_argument(
        "--results-dir",
        default=None,
        help="结果目录（默认 VLM-test/output/results）",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录（默认 VLM-test/output/analysis）",
    )
    parser.add_argument(
        "--family",
        action="append",
        default=None,
        help="按模型族过滤（可多次指定）",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    results_dir = Path(args.results_dir) if args.results_dir else base / "output" / "results"
    output_dir = Path(args.output_dir) if args.output_dir else base / "output" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(f"结果目录不存在: {results_dir}")
        sys.exit(1)

    data = scan_results(results_dir, args.family)
    if not data:
        print("未找到结果。")
        sys.exit(1)

    rows = build_table(data)

    # 概览
    for family in sorted(data.keys()):
        n = len(data[family])
        print(f"{family}: {n} checkpoint(s)")

    # 输出
    md_path = output_dir / "checkpoint_comparison.md"
    csv_path = output_dir / "checkpoint_comparison.csv"

    md_content = output_markdown(rows, md_path)
    output_csv(rows, csv_path)
    print(f"\nMarkdown → {md_path}")
    print(f"CSV      → {csv_path}")

    # 打印 Markdown 到 stdout
    print("")
    print(md_content)

    # 每族最佳
    print("--- Best per family (composite) ---")
    for family in sorted(data.keys()):
        family_rows = [
            r for r in rows if r["family"] == family and r["composite"] is not None
        ]
        if family_rows:
            best = max(family_rows, key=lambda r: r["composite"])
            print(f"  {family}: {best['checkpoint']} (composite={best['composite']:.4f})")
        else:
            print(f"  {family}: no data")


if __name__ == "__main__":
    main()
