#!/usr/bin/env python3
"""
YOLO Fire Detection Training Script
Research implementation for fire detection using YOLOv5/v8 architectures
"""

import argparse
import yaml
import time
from pathlib import Path
from ultralytics import YOLO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load training configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_model(model_name, data_config, output_dir, epochs=50, imgsz=640):
    """
    Train YOLO model with specified configuration
    
    Args:
        model_name (str): Model architecture (yolov5s.pt, yolov8n.pt, etc.)
        data_config (str): Path to data.yaml configuration
        output_dir (str): Output directory for results
        epochs (int): Training epochs
        imgsz (int): Image size for training
    
    Returns:
        dict: Training results and metrics
    """
    logger.info(f"Starting training with {model_name}")
    
    # Load model
    model = YOLO(model_name)
    
    # Training configuration
    train_config = {
        "data": data_config,
        "epochs": epochs,
        "imgsz": imgsz,
        "patience": 10,
        "save": True,
        "project": output_dir,
        "name": f"{model_name.split('.')[0]}_train",
        "exist_ok": True
    }
    
    logger.info(f"Training configuration: {train_config}")
    
    # Start training
    start_time = time.time()
    results = model.train(**train_config)
    training_time = time.time() - start_time
    
    # Extract metrics
    final_metrics = {
        'model': model_name,
        'mAP50': results.results_dict.get('metrics/mAP50(B)', 0),
        'mAP50-95': results.results_dict.get('metrics/mAP50-95(B)', 0),
        'precision': results.results_dict.get('metrics/precision(B)', 0),
        'recall': results.results_dict.get('metrics/recall(B)', 0),
        'training_time': training_time,
        'save_dir': str(results.save_dir)
    }
    
    logger.info(f"Training completed in {training_time:.1f} seconds")
    logger.info(f"Final mAP50: {final_metrics['mAP50']:.4f}")
    
    return final_metrics, results

def main():
    parser = argparse.ArgumentParser(description='Train YOLO model for fire detection')
    parser.add_argument('--model', default='yolov8n.pt', 
                       help='Model architecture (yolov5s.pt, yolov8n.pt, etc.)')
    parser.add_argument('--data', default='data/fire_dataset/data.yaml',
                       help='Path to dataset configuration')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Training image size')
    parser.add_argument('--output', default='runs/detect',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Train model
    metrics, results = train_model(
        model_name=args.model,
        data_config=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        imgsz=args.imgsz
    )
    
    # Save metrics
    metrics_path = Path(args.output) / f"{args.model.split('.')[0]}_metrics.yaml"
    with open(metrics_path, 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False)
    
    logger.info(f"Training metrics saved to: {metrics_path}")
    return metrics

if __name__ == "__main__":
    main()