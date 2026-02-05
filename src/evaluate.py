#!/usr/bin/env python3
"""
YOLO Fire Detection Evaluation Script
Research implementation for comprehensive model evaluation
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(model_path, data_config, output_dir):
    """
    Comprehensive model evaluation including metrics and visualizations
    
    Args:
        model_path (str): Path to trained model weights
        data_config (str): Path to data.yaml configuration
        output_dir (str): Output directory for evaluation results
    
    Returns:
        dict: Evaluation metrics and analysis
    """
    logger.info(f"Evaluating model: {model_path}")
    
    # Load model
    model = YOLO(model_path)
    
    # Run validation
    results = model.val(data=data_config, save=True, save_hybrid=True, 
                       save_json=True, save_txt=True)
    
    # Extract detailed metrics
    evaluation_metrics = {
        'model_path': model_path,
        'mAP50': results.results_dict.get('metrics/mAP50(B)', 0),
        'mAP50-95': results.results_dict.get('metrics/mAP50-95(B)', 0),
        'precision': results.results_dict.get('metrics/precision(B)', 0),
        'recall': results.results_dict.get('metrics/recall(B)', 0),
        'mp': results.mp,  # Mean precision
        'mr': results.mr,  # Mean recall
        'map50': results.map50,
        'map': results.map,
        'fitness': results.fitness
    }
    
    # Class-wise metrics if available
    if hasattr(results, 'ap_class_index'):
        class_metrics = {}
        for i, class_idx in enumerate(results.ap_class_index):
            class_metrics[f'class_{class_idx}_mAP50'] = results.ap[i, 0]
            class_metrics[f'class_{class_idx}_mAP50-95'] = results.ap[i].mean()
        evaluation_metrics.update(class_metrics)
    
    # Save detailed evaluation
    output_path = Path(output_dir) / "evaluation_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(evaluation_metrics, f, indent=2)
    
    logger.info(f"Evaluation completed. Results saved to: {output_path}")
    return evaluation_metrics

def generate_evaluation_report(metrics_list, output_dir):
    """Generate comprehensive evaluation report with visualizations"""
    
    # Create comparison DataFrame
    df = pd.DataFrame(metrics_list)
    
    # Generate comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Performance metrics bar plot
    performance_metrics = ['mAP50', 'mAP50-95', 'precision', 'recall']
    df_plot = df.set_index('model_path')[performance_metrics]
    df_plot.plot(kind='bar', ax=axes[0,0], alpha=0.8)
    axes[0,0].set_title('Performance Metrics Comparison')
    axes[0,0].set_ylabel('Score')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # mAP50 vs mAP50-95 scatter
    axes[0,1].scatter(df['mAP50'], df['mAP50-95'], alpha=0.8, s=100)
    for i, model in enumerate(df['model_path']):
        axes[0,1].annotate(Path(model).stem, 
                          (df['mAP50'].iloc[i], df['mAP50-95'].iloc[i]))
    axes[0,1].set_xlabel('mAP50')
    axes[0,1].set_ylabel('mAP50-95')
    axes[0,1].set_title('mAP50 vs mAP50-95 Comparison')
    axes[0,1].grid(True, alpha=0.3)
    
    # Precision vs Recall
    axes[1,0].scatter(df['recall'], df['precision'], alpha=0.8, s=100)
    for i, model in enumerate(df['model_path']):
        axes[1,0].annotate(Path(model).stem,
                          (df['recall'].iloc[i], df['precision'].iloc[i]))
    axes[1,0].set_xlabel('Recall')
    axes[1,0].set_ylabel('Precision')
    axes[1,0].set_title('Precision vs Recall')
    axes[1,0].grid(True, alpha=0.3)
    
    # Overall fitness comparison
    df['model_name'] = df['model_path'].apply(lambda x: Path(x).stem)
    axes[1,1].bar(df['model_name'], df['fitness'], alpha=0.8)
    axes[1,1].set_title('Overall Model Fitness')
    axes[1,1].set_ylabel('Fitness Score')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = Path(output_dir) / "evaluation_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Generate summary table
    summary_path = Path(output_dir) / "evaluation_summary.csv"
    df.to_csv(summary_path, index=False)
    
    logger.info(f"Evaluation report saved to: {output_dir}")
    return plot_path, summary_path

def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLO fire detection model')
    parser.add_argument('--weights', required=True, nargs='+',
                       help='Path(s) to model weights for evaluation')
    parser.add_argument('--data', default='data/fire_dataset/data.yaml',
                       help='Path to dataset configuration')
    parser.add_argument('--output', default='results/evaluation',
                       help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    # Evaluate each model
    metrics_list = []
    for weight_path in args.weights:
        metrics = evaluate_model(weight_path, args.data, args.output)
        metrics_list.append(metrics)
    
    # Generate comparison report if multiple models
    if len(metrics_list) > 1:
        generate_evaluation_report(metrics_list, args.output)
    
    return metrics_list

if __name__ == "__main__":
    main()