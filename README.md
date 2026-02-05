# Fire Detection using YOLOv8

Research-oriented computer vision project for fire/smoke/no-fire detection with
YOLOv8, including baseline analysis and ablation-comparison utilities.

## Project Structure

```text
Fire-Detection-YOLO/
├── data/
├── notebooks/
│   └── yolo_fire_detection.ipynb
├── results/
│   ├── yolov8_baseline/
│   └── analysis/
├── scripts/
│   ├── train_yolo.py
│   ├── analyze_results.py
│   └── compare_runs.py
└── README.md
```

## Baseline Analysis (Day 4 - Part A)

Generate a compact analytical report from the baseline `results.csv`:

```bash
python scripts/analyze_results.py \
  --results-csv results/yolov8_baseline/results.csv \
  --output-dir results/analysis/baseline
```

Outputs:
- `summary.json` with best-epoch metrics.
- `summary.md` quick insights for reporting.
- `metrics_trends.png` precision/recall and mAP trends.

## Ablation Experiment (Day 4 - Part B)

Recommended ablation: model-size comparison (`yolov8n` vs `yolov8s`).

```bash
python scripts/train_yolo.py \
  --data data/data.yaml \
  --model yolov8s.pt \
  --epochs 50 \
  --imgsz 640 \
  --project results \
  --name yolov8s_ablation
```

Then compare baseline and ablation:

```bash
python scripts/compare_runs.py \
  --baseline results/yolov8_baseline \
  --experiment results/yolov8s_ablation \
  --output-dir results/analysis/comparison
```

Outputs:
- `comparison.csv` for quantitative tracking.
- `comparison.md` report-ready metric table.
- `map50_comparison.png` visual mAP@50 comparison.

## SOP-Ready Summary

This project develops a YOLOv8-based detector for `fire`, `smoke`, and
`no_fire` categories using a Roboflow dataset in YOLO format. A baseline
`yolov8n` model was trained and evaluated with strong detection quality
(mAP@50 around 0.87), followed by research-style ablations to study trade-offs
between model capacity and efficiency. The workflow emphasizes reproducibility,
clear metric interpretation, and experiment-driven iteration suitable for
real-world safety applications and research internship portfolios.
