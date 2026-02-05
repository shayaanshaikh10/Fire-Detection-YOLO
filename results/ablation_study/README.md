# Ablation Study Results

## Model Comparison Summary

| Model    | mAP50  | mAP50-95 | Precision | Recall | Parameters | Inference Time |
|----------|--------|----------|-----------|--------|------------|---------------|
| YOLOv8n  | 86.9%  | 49.2%    | 83.7%     | 80.9%  | 3.2M       | 12.3ms        |
| YOLOv5s  | 89.1%  | 50.8%    | 85.6%     | 82.9%  | 9.1M       | 16.8ms        |

## Key Findings

### Performance Analysis
- **Accuracy Improvement**: YOLOv5s achieves 2.5% higher mAP50 than YOLOv8n
- **Computational Cost**: YOLOv5s requires 2.9x more parameters for this accuracy gain
- **Efficiency Ratio**: YOLOv8n provides superior parameter efficiency (27.5 mAP50 per million parameters vs 9.8 for YOLOv5s)

### Real-Time Performance
- **YOLOv8n**: 12.3ms average inference (81 FPS) 
- **YOLOv5s**: 16.8ms average inference (59 FPS)
- **Both models**: Suitable for real-time deployment (>24 FPS)

### Recommendations
- **YOLOv8n**: Recommended for resource-constrained deployments requiring high efficiency
- **YOLOv5s**: Recommended when marginal accuracy improvements justify increased computational cost

## Experimental Configuration
- **Dataset**: 1,542 images (4 classes: fire, light, no-fire, smoke)
- **Training**: 50 epochs, 640×640 resolution
- **Hardware**: Standard GPU environment
- **Framework**: Ultralytics YOLOv5/v8 implementations