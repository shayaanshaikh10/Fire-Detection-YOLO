#!/usr/bin/env python3
"""Analyze YOLO results.csv and generate report artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _to_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def _find_metric_column(columns: list[str], keyword: str) -> str:
    for column in columns:
        if keyword in column:
            return column
    raise ValueError(f"Could not find column containing '{keyword}' in results.csv")


def _try_plot(
    epochs: list[float],
    precision: list[float],
    recall: list[float],
    map50: list[float],
    map5095: list[float],
    output_png: Path,
    output_note: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError:
        output_note.write_text(
            "matplotlib not installed: metrics plot was not generated.\n",
            encoding="utf-8",
        )
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, precision, label="Precision")
    axes[0].plot(epochs, recall, label="Recall")
    axes[0].set_title("Precision / Recall by Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, map50, label="mAP@50")
    axes[1].plot(epochs, map5095, label="mAP@50-95")
    axes[1].set_title("mAP by Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def analyze_results(results_csv: Path, output_dir: Path) -> dict[str, float | int | str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    with results_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("results.csv is empty")

    columns = list(rows[0].keys())
    epoch_col = _find_metric_column(columns, "epoch")
    precision_col = _find_metric_column(columns, "metrics/precision")
    recall_col = _find_metric_column(columns, "metrics/recall")
    map50_col = _find_metric_column(columns, "metrics/mAP50(B)")
    map5095_col = _find_metric_column(columns, "metrics/mAP50-95")
    cls_loss_col = _find_metric_column(columns, "train/cls_loss")

    best_row = max(rows, key=lambda r: _to_float(r[map50_col]))
    summary: dict[str, Any] = {
        "epochs_recorded": int(_to_float(rows[-1][epoch_col]) + 1),
        "best_epoch": int(_to_float(best_row[epoch_col])),
        "best_precision": _to_float(best_row[precision_col]),
        "best_recall": _to_float(best_row[recall_col]),
        "best_map50": _to_float(best_row[map50_col]),
        "best_map50_95": _to_float(best_row[map5095_col]),
        "final_cls_loss": _to_float(rows[-1][cls_loss_col]),
    }

    epochs = [_to_float(r[epoch_col]) for r in rows]
    precision = [_to_float(r[precision_col]) for r in rows]
    recall = [_to_float(r[recall_col]) for r in rows]
    map50 = [_to_float(r[map50_col]) for r in rows]
    map5095 = [_to_float(r[map5095_col]) for r in rows]

    _try_plot(
        epochs,
        precision,
        recall,
        map50,
        map5095,
        output_dir / "metrics_trends.png",
        output_dir / "plot_warning.txt",
    )

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    summary_md = (
        "# Baseline Analysis\n\n"
        f"- Best epoch: **{summary['best_epoch']}**\n"
        f"- Best mAP@50: **{summary['best_map50']:.4f}**\n"
        f"- Best precision: **{summary['best_precision']:.4f}**\n"
        f"- Best recall: **{summary['best_recall']:.4f}**\n"
        f"- Best mAP@50-95: **{summary['best_map50_95']:.4f}**\n"
        f"- Final classification loss: **{summary['final_cls_loss']:.4f}**\n"
    )
    (output_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze YOLO results.csv")
    parser.add_argument(
        "--results-csv",
        default="results/yolov8_baseline/results.csv",
        help="Path to run results.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results/analysis/baseline",
        help="Directory to store summary and plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_results(Path(args.results_csv), Path(args.output_dir))


if __name__ == "__main__":
    main()
