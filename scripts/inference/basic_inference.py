"""
Basic YOLOv8 Inference Script
==============================

This is a simplified inference demo showing basic object detection.
For the complete ADAS collision avoidance system with:
- Multi-object tracking
- Time-to-Collision calculation
- Decision smoothing algorithm
- Adaptive safety behavior

Please contact for professional opportunities.

Usage:
    python scripts/inference/basic_inference.py --source video.mp4 --weights weights/best.pt
"""

import argparse
import cv2
import time
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='Basic YOLO Inference')
    parser.add_argument('--source', type=str, required=True, help='Video file or camera index (0)')
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='Model weights')
    parser.add_argument('--conf', type=float, default=0.30, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU threshold')
    parser.add_argument('--imgsz', type=int, default=640, help='Input size')
    parser.add_argument('--save', action='store_true', help='Save output video')
    parser.add_argument('--output', type=str, default='output.mp4', help='Output filename')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load model
    print(f"Loading model from {args.weights}...")
    model = YOLO(args.weights)
    
    # Open video
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video source {args.source}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n{'='*60}")
    print(f"Video: {width}x{height} @ {fps} FPS")
    print(f"Total frames: {total_frames}")
    print(f"{'='*60}\n")
    
    # Setup video writer if saving
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Saving output to: {args.output}\n")
    
    # Processing loop
    frame_count = 0
    start_time = time.time()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run inference
            results = model(frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)
            
            # Draw results
            annotated_frame = results[0].plot()
            
            # Add basic info overlay
            info_text = f"Frame: {frame_count}/{total_frames} | Objects: {len(results[0].boxes)}"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Calculate FPS
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            fps_text = f"FPS: {current_fps:.1f}"
            cv2.putText(annotated_frame, fps_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display
            cv2.imshow('YOLO Detection', annotated_frame)
            
            # Save if requested
            if writer is not None:
                writer.write(annotated_frame)
            
            # Progress update every 30 frames
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%) | FPS: {current_fps:.1f}")
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nInterrupted by user")
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        elapsed_total = time.time() - start_time
        avg_fps = frame_count / elapsed_total if elapsed_total > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"{'='*60}")
        print(f"Frames processed: {frame_count}/{total_frames}")
        print(f"Total time: {elapsed_total:.2f}s")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"{'='*60}\n")
        
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Output saved: {args.output}\n")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    print("\n" + "="*60)
    print("BASIC YOLO INFERENCE DEMO")
    print("="*60)
    print("This is a simplified version for demonstration.")
    print("Production ADAS system features:")
    print("  • Multi-object tracking with unique IDs")
    print("  • Distance & speed estimation")
    print("  • Time-to-Collision (TTC) calculation")
    print("  • 6-level decision system (STOP → CONTINUE)")
    print("  • Decision smoothing (80% false alert reduction)")
    print("  • Adaptive behavior (CITY/HIGHWAY modes)")
    print("="*60 + "\n")
    
    main()
