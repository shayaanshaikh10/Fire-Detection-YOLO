#!/usr/bin/env python3
"""Compare YOLO experiment runs and generate a metrics table + chart."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _to_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def _find_metric_column(columns: list[str], keyword: str) -> str:
    for column in columns:
        if keyword in column:
            return column
    raise ValueError(f"Could not find column containing '{keyword}'")


def summarize_run(run_dir: Path, label: str) -> dict[str, float | str | int]:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing results.csv in {run_dir}")

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty results.csv in {run_dir}")

    columns = list(rows[0].keys())
    map50_col = _find_metric_column(columns, "metrics/mAP50(B)")
    precision_col = _find_metric_column(columns, "metrics/precision")
    recall_col = _find_metric_column(columns, "metrics/recall")

    best_row = max(rows, key=lambda r: _to_float(r[map50_col]))
    train_time_s = sum(_to_float(r.get("time", "0")) for r in rows)

    return {
        "run": label,
        "best_map50": _to_float(best_row[map50_col]),
        "best_precision": _to_float(best_row[precision_col]),
        "best_recall": _to_float(best_row[recall_col]),
        "epochs": len(rows),
        "train_time_s": train_time_s,
    }


def _try_plot(rows: list[dict[str, float | str | int]], output_png: Path, warning_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError:
        warning_path.write_text(
            "matplotlib not installed: comparison chart was not generated.\n",
            encoding="utf-8",
        )
        return

    runs = [str(row["run"]) for row in rows]
    scores = [float(row["best_map50"]) for row in rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(runs, scores, color=["#1f77b4", "#ff7f0e"])
    ax.set_title("Best mAP@50 Comparison")
    ax.set_ylabel("mAP@50")
    for idx, value in enumerate(scores):
        ax.text(idx, value + 0.005, f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def compare_runs(baseline_dir: Path, experiment_dir: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        summarize_run(baseline_dir, baseline_dir.name),
        summarize_run(experiment_dir, experiment_dir.name),
    ]

    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run", "best_map50", "best_precision", "best_recall", "epochs", "train_time_s"],
        )
        writer.writeheader()
        writer.writerows(rows)

    _try_plot(rows, output_dir / "map50_comparison.png", output_dir / "plot_warning.txt")

    md_lines = [
        "# Baseline vs Experiment Comparison",
        "",
        "| Run | Best mAP@50 | Best Precision | Best Recall | Epochs | Train Time (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['run']} | {float(row['best_map50']):.4f} | {float(row['best_precision']):.4f} |"
            f" {float(row['best_recall']):.4f} | {int(row['epochs'])} | {float(row['train_time_s']):.2f} |"
        )
    report_path = output_dir / "comparison.md"
    report_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare YOLO training runs")
    parser.add_argument("--baseline", default="results/yolov8_baseline", help="Baseline run directory")
    parser.add_argument("--experiment", required=True, help="Experiment run directory")
    parser.add_argument(
        "--output-dir",
        default="results/analysis/comparison",
        help="Directory to store comparison artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compare_runs(Path(args.baseline), Path(args.experiment), Path(args.output_dir))


if __name__ == "__main__":
    main()
