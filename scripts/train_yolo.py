#!/usr/bin/env python3
"""Train YOLOv8 fire/smoke detector with configurable settings."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def train_yolo(
    data: str,
    model: str,
    epochs: int,
    imgsz: int,
    project: str,
    name: str,
    device: str,
    workers: int,
    seed: int,
    pretrained: bool,
) -> None:
    """Launch a YOLO training run."""
    project_path = Path(project)
    project_path.mkdir(parents=True, exist_ok=True)

    detector = YOLO(model)
    detector.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        project=str(project_path),
        name=name,
        device=device,
        workers=workers,
        seed=seed,
        pretrained=pretrained,
        exist_ok=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 fire/smoke detector")
    parser.add_argument("--data", default="data/data.yaml", help="Path to YOLO data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model checkpoint")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--project", default="results", help="Results root directory")
    parser.add_argument("--name", default="yolov8_experiment", help="Run directory name")
    parser.add_argument("--device", default="0", help="CUDA device index or cpu")
    parser.add_argument("--workers", type=int, default=8, help="Data loader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable pretrained backbone initialization",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_yolo(
        data=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        pretrained=not args.no_pretrained,
    )


if __name__ == "__main__":
    main()
