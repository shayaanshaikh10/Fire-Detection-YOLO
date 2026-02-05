#!/usr/bin/env python3
"""
YOLO Fire Detection Inference Script
Real-time inference and demo generation for research showcase
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_inference_on_images(model_path, input_dir, output_dir, conf_threshold=0.25):
    """
    Run inference on a directory of images
    
    Args:
        model_path (str): Path to trained model weights
        input_dir (str): Directory containing input images
        output_dir (str): Directory to save inference results
        conf_threshold (float): Confidence threshold for detections
    
    Returns:
        dict: Inference statistics
    """
    logger.info(f"Running inference with model: {model_path}")
    
    # Load model
    model = YOLO(model_path)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get image files
    input_path = Path(input_dir)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_path.glob(ext)))
        image_files.extend(list(input_path.glob(ext.upper())))
    
    logger.info(f"Found {len(image_files)} images for inference")
    
    # Statistics tracking
    stats = {
        'total_images': len(image_files),
        'total_detections': 0,
        'avg_inference_time': 0,
        'class_counts': {}
    }
    
    total_time = 0
    
    # Process each image
    for img_file in image_files:
        start_time = time.time()
        
        # Run inference
        results = model(str(img_file), conf=conf_threshold, save=False, verbose=False)
        
        inference_time = time.time() - start_time
        total_time += inference_time
        
        # Process results
        for r in results:
            # Count detections
            if r.boxes is not None:
                stats['total_detections'] += len(r.boxes)
                
                # Count by class
                for cls in r.boxes.cls:
                    class_name = model.names[int(cls)]
                    stats['class_counts'][class_name] = stats['class_counts'].get(class_name, 0) + 1
            
            # Save annotated image
            annotated_img = r.plot()
            output_file = output_path / f"detected_{img_file.name}"
            cv2.imwrite(str(output_file), annotated_img)
    
    # Calculate average inference time
    stats['avg_inference_time'] = total_time / len(image_files) if image_files else 0
    
    logger.info(f"Inference completed. Total detections: {stats['total_detections']}")
    logger.info(f"Average inference time: {stats['avg_inference_time']:.3f}s")
    
    return stats

def run_inference_on_video(model_path, video_path, output_path, conf_threshold=0.25):
    """
    Run inference on video file
    
    Args:
        model_path (str): Path to trained model weights
        video_path (str): Path to input video
        output_path (str): Path for output video
        conf_threshold (float): Confidence threshold for detections
    
    Returns:
        dict: Video inference statistics
    """
    logger.info(f"Running video inference on: {video_path}")
    
    # Load model
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Statistics
    stats = {
        'total_frames': total_frames,
        'processed_frames': 0,
        'total_detections': 0,
        'avg_fps': 0,
        'class_counts': {}
    }
    
    start_time = time.time()
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run inference
        results = model(frame, conf=conf_threshold, verbose=False)
        
        # Process results
        for r in results:
            if r.boxes is not None:
                stats['total_detections'] += len(r.boxes)
                
                # Count by class
                for cls in r.boxes.cls:
                    class_name = model.names[int(cls)]
                    stats['class_counts'][class_name] = stats['class_counts'].get(class_name, 0) + 1
            
            # Get annotated frame
            annotated_frame = r.plot()
            out.write(annotated_frame)
        
        frame_count += 1
        stats['processed_frames'] = frame_count
        
        if frame_count % 30 == 0:  # Progress update every 30 frames
            logger.info(f"Processed {frame_count}/{total_frames} frames")
    
    # Calculate statistics
    total_time = time.time() - start_time
    stats['avg_fps'] = frame_count / total_time if total_time > 0 else 0
    
    # Cleanup
    cap.release()
    out.release()
    
    logger.info(f"Video inference completed. Average FPS: {stats['avg_fps']:.1f}")
    return stats

def main():
    parser = argparse.ArgumentParser(description='Run YOLO fire detection inference')
    parser.add_argument('--weights', required=True,
                       help='Path to model weights')
    parser.add_argument('--source', required=True,
                       help='Path to image directory or video file')
    parser.add_argument('--output', default='results/inference',
                       help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--save-stats', action='store_true',
                       help='Save inference statistics to JSON')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Determine input type and run inference
    source_path = Path(args.source)
    
    if source_path.is_dir():
        # Image directory
        stats = run_inference_on_images(
            args.weights, args.source, args.output, args.conf
        )
        logger.info("Image inference completed")
        
    elif source_path.is_file() and source_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
        # Video file
        output_video = Path(args.output) / f"detected_{source_path.name}"
        stats = run_inference_on_video(
            args.weights, str(source_path), str(output_video), args.conf
        )
        logger.info("Video inference completed")
        
    else:
        logger.error(f"Invalid source: {args.source}")
        return
    
    # Save statistics if requested
    if args.save_stats:
        import json
        stats_path = Path(args.output) / "inference_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Statistics saved to: {stats_path}")
    
    # Print summary
    print(f"\nInference Summary:")
    print(f"Total detections: {stats['total_detections']}")
    print(f"Class distribution: {stats['class_counts']}")
    
    return stats

if __name__ == "__main__":
    main()