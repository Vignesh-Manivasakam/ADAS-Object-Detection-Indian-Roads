"""
ADAS Object Detection Inference Script
======================================

Run object detection on images, videos, or webcam feed using trained YOLOv8 model.

Usage:
    # Single image
    python inference.py --source image.jpg --weights weights/best.pt
    
    # Video file
    python inference.py --source video.mp4 --weights weights/best.pt
    
    # Webcam (real-time)
    python inference.py --source 0 --weights weights/best.pt
    
    # Directory of images
    python inference.py --source path/to/images/ --weights weights/best.pt
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2
import time
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='ADAS Object Detection Inference')
    
    parser.add_argument('--source', type=str, required=True,
                        help='Path to image/video/directory or webcam index (0 for default camera)')
    
    parser.add_argument('--weights', type=str, default='weights/best.pt',
                        help='Path to model weights (.pt file)')
    
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (0.0-1.0)')
    
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS IoU threshold (0.0-1.0)')
    
    parser.add_argument('--imgsz', type=int, default=1280,
                        help='Input image size (640, 960, or 1280)')
    
    parser.add_argument('--device', type=str, default='',
                        help='Device to run on (e.g., "0" or "cpu")')
    
    parser.add_argument('--save', action='store_true',
                        help='Save detection results')
    
    parser.add_argument('--save-txt', action='store_true',
                        help='Save results in YOLO label format')
    
    parser.add_argument('--save-conf', action='store_true',
                        help='Include confidence in saved labels')
    
    parser.add_argument('--output', type=str, default='runs/detect/inference',
                        help='Output directory for saved results')
    
    parser.add_argument('--view', action='store_true',
                        help='Display results in real-time')
    
    parser.add_argument('--fps', action='store_true',
                        help='Show FPS counter on display')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Print configuration
    print("="*70)
    print("ADAS OBJECT DETECTION - INFERENCE")
    print("="*70)
    print(f"Model:      {args.weights}")
    print(f"Source:     {args.source}")
    print(f"Conf:       {args.conf}")
    print(f"IoU:        {args.iou}")
    print(f"Image Size: {args.imgsz}")
    print(f"Device:     {args.device if args.device else 'auto-detect'}")
    print(f"Save:       {args.save}")
    print(f"Output:     {args.output}")
    print("="*70)
    
    # Verify model exists
    if not Path(args.weights).exists():
        print(f"\n❌ ERROR: Model file not found: {args.weights}")
        print("Please download or train a model first.")
        return
    
    # Load model
    print("\n📦 Loading model...")
    model = YOLO(args.weights)
    
    # Check device
    device = args.device if args.device else ('0' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Model loaded on: {device}")
    
    # Get class names
    class_names = model.names
    print(f"✅ Detecting {len(class_names)} classes: {', '.join(class_names.values())}")
    
    # Run inference
    print(f"\n🚀 Running inference on: {args.source}")
    print("-"*70)
    
    try:
        results = model.predict(
            source=args.source,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=device,
            save=args.save,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            project=args.output.split('/')[0] if '/' in args.output else 'runs',
            name=args.output.split('/')[-1] if '/' in args.output else 'detect',
            exist_ok=True,
            show=args.view,
            stream=True,  # Use generator for memory efficiency
        )
        
        # Process results
        frame_count = 0
        total_detections = 0
        start_time = time.time()
        
        for result in results:
            frame_count += 1
            num_detections = len(result.boxes)
            total_detections += num_detections
            
            # Calculate FPS
            if args.fps and frame_count % 10 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"\rFrame {frame_count} | FPS: {fps:.1f} | Detections: {num_detections}", end='')
        
        print(f"\n{'='*70}")
        print("✅ INFERENCE COMPLETE!")
        print(f"{'='*70}")
        print(f"Total frames processed: {frame_count}")
        print(f"Total detections:       {total_detections}")
        print(f"Average detections/frame: {total_detections/frame_count:.1f}")
        
        if args.save:
            print(f"\n📁 Results saved to: {args.output}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Inference interrupted by user.")
    
    except Exception as e:
        print(f"\n❌ ERROR during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
