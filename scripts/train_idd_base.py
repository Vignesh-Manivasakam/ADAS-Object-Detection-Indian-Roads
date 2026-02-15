from ultralytics import YOLO
import torch
from datetime import datetime
import yaml
from pathlib import Path

def train_master_model():
    """
    TRAINING v3.0: The Master Model
    
    Combines all data engineering efforts into a single, high-performance training run.
    """
    
    # --- Verify Hardware ---
    if not torch.cuda.is_available():
        print("❌ FATAL ERROR: No GPU found. Training cannot proceed.")
        return
    
    print("="*70)
    print("🚀 ADAS MASTER MODEL TRAINING v3.0")
    print("="*70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print("\n✅ Pre-training Steps Complete: Master dataset and augmentations are ready.")
    print("\nSTRATEGY:")
    print("  - MODEL:      YOLOv8-small (yolov8s.pt) - High capacity for our large dataset.")
    print("  - DATA:       Master Dataset (IDD + IDD95k + Augmentations)")
    print("  - RESOLUTION: 640px - Critical for detecting small, distant objects.")
    print("  - GOAL:       Exceed 55% mAP50 by leveraging data diversity and model capacity.")
    print()
    
    # ============================================
    # 1. Load the Model
    # ============================================
    # We are using the Medium model for its higher capacity.
    model = YOLO('yolov8m.pt') 
    
    # ============================================
    # 2. Train the Model on the Master Dataset
    # ============================================
    results = model.train(
        # --- CRITICAL: Point to your new master YAML file ---
        data=r'C:\Users\\Documents\ADAS_Hackathon\datasets\MASTER_IDD_YOLO\master_data.yaml',
        
        # --- HARDWARE & PERFORMANCE ---
        imgsz=640,         # Use high resolution for small objects
        batch=32,            # Start with 8 for yolov8m @ 1280px. If no memory errors, you can try 10 or 12.
        workers=8,          # Utilize your powerful CPU for data loading
        device=0,           # Use the first GPU
        amp=True,           # Automatic mixed precision for faster training
        
        # --- TRAINING DURATION ---
        epochs=100,         # Train for longer on this large, diverse dataset
        patience=15,        # Stop if no improvement for 30 epochs
        
        # --- PROJECT SETUP ---
        project='runs/detect',
        name='MASTER_YOLOv8s_640px', # A descriptive name for our best run
        exist_ok=True,
        save=True,
        save_period=10,
        
        # --- AUGMENTATION (Online) ---
        # These are fast augmentations applied by the trainer in real-time
        mosaic=1.0,         # This is the most powerful YOLO augmentation
        mixup=0.1,
        copy_paste=0.2,     # Excellent for boosting our weak classes ('animal', 'bicycle')
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        
        # --- OPTIMIZER & LOSS ---
        optimizer='AdamW',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5, # Allow a longer warmup for the larger model and dataset
        box=7.5,
        cls=0.5, # This is the main knob for classification loss. Asymmetric loss is a more advanced topic.
        
        # --- LOGGING & VALIDATION ---
        val=True,
        plots=True,
        verbose=True,
    )
    
    print("\n" + "="*70)
    print("✅ MASTER MODEL TRAINING COMPLETE!")
    print("="*70)
    
    # The rest of your analysis code is good, let's just update the class list.
    
    # --- Final Validation and Analysis ---
    print("\n🧪 Validating the best performing checkpoint...")
    best_model_path = model.trainer.best
    best_model = YOLO(best_model_path)
    val_metrics = best_model.val()
    
    print(f"\n📊 Final Validation Metrics (from {Path(best_model_path).name}):")
    print(f"   mAP@50-95 (primary metric): {val_metrics.box.map:.4f}")
    print(f"   mAP@50 (secondary metric):  {val_metrics.box.map50:.4f}")
    print(f"   Precision:                  {val_metrics.box.mp:.4f}")
    print(f"   Recall:                     {val_metrics.box.mr:.4f}")
    
    # --- Correctly print per-class results using the final 11 classes ---
    print(f"\n⚠️ Per-Class Performance (mAP@50):")
    class_names = model.names
    for i, name in class_names.items():
        map50 = val_metrics.box.maps[i]
        print(f"   - {name:<15}: {map50:.4f}")
    
    print("\n" + "="*70)
    print("🎉 All steps complete! You are ready to analyze the results.")
    print("="*70)

if __name__ == '__main__':
    train_master_model()
    input("\nPress Enter to exit...")
