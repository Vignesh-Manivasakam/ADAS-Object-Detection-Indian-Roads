from ultralytics import YOLO
import torch

def train_finetune_ultra_high_res():
    """
    TRAINING v6.0: Ultra-High-Resolution Fine-Tuning (1280px)
    
    Final polish using maximum resolution for tiny object detection.
    """
    
    if not torch.cuda.is_available():
        print("❌ FATAL ERROR: No GPU found.")
        return
    
    print("="*70)
    print("🎯 ADAS HACKATHON - ULTIMATE RESOLUTION FINE-TUNING")
    print("="*70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("\nWARNING: This will push your GPU to its limits!")
    print("         Monitor GPU memory usage carefully.")
    print()
    
    # ============================================
    # LOAD YOUR BEST 960px MODEL
    # ============================================
    model = YOLO('C:/Users//Documents/ADAS_Hackathon/runs/detect/FINETUNE_YOLOv8s_960px_FINAL/weights/best.pt')
    
    results = model.train(
        # --- DATASET ---
        data=r'C:\Users\\Documents\ADAS_Hackathon\datasets\MASTER_IDD_YOLO\master_data.yaml',
        
        # ============================================
        # ### ULTRA-HIGH-RES PARAMETERS ###
        # ============================================
        imgsz=1280,         # Maximum reasonable resolution
        batch=12,           # REDUCED from 24 - Critical for memory!
        epochs=15,          # Short run - just for polish
        patience=5,         # Early stopping if no improvement
        
        # --- OPTIMIZER (VERY CONSERVATIVE) ---
        optimizer='AdamW',
        lr0=0.0005,         # HALVED from 960px run - very gentle updates
        lrf=0.0001,         # Very gradual decay
        momentum=0.937,
        weight_decay=0.0005,
        
        # --- LOSS WEIGHTS ---
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # ============================================
        
        workers=8,
        device=0,
        amp=True,           # Mixed precision - ESSENTIAL for memory
        
        # --- PROJECT ---
        project='runs/detect',
        name='FINETUNE_YOLOv8s_1280px_ULTRA',
        exist_ok=True,
        
        # --- AUGMENTATIONS (Reduced for stability) ---
        mosaic=0.5,         # Reduced from 1.0
        close_mosaic=5,     # Close early
        mixup=0.0,          # Disabled for final tuning
        copy_paste=0.0,     # Disabled
        degrees=5.0,        # Minimal rotation
        translate=0.1,      # Minimal translation
        scale=0.3,          # Reduced scaling
        
        # --- MEMORY OPTIMIZATIONS ---
        cache=False,        # Don't cache to RAM (use disk)
        rect=True,          # Rectangular training - saves memory
    )
    
    print("\n" + "="*70)
    print("✅ ULTRA-HIGH-RES FINE-TUNING COMPLETE!")
    print("="*70)
    
    # ============================================
    # VALIDATION
    # ============================================
    print("\n🧪 Final validation at 1280px...")
    best_model_path = results.save_dir / 'weights' / 'best.pt'
    best_model = YOLO(best_model_path)
    val_metrics = best_model.val()
    
    print(f"\n📊 FINAL ULTRA-HIGH-RES Metrics:")
    print(f"   mAP@50-95: {val_metrics.box.map:.4f}")
    print(f"   mAP@50:    {val_metrics.box.map50:.4f}")
    print(f"   Precision: {val_metrics.box.mp:.4f}")
    print(f"   Recall:    {val_metrics.box.mr:.4f}")
    
    # ============================================
    # PERFORMANCE COMPARISON
    # ============================================
    print("\n📈 RESOLUTION PROGRESSION:")
    print(f"   640px  → mAP@50-95: 0.2845")
    print(f"   960px  → mAP@50-95: 0.4056 (+42.6%)")
    print(f"   1280px → mAP@50-95: {val_metrics.box.map:.4f} ({((val_metrics.box.map/0.4056 - 1)*100):+.1f}%)")
    
    print("\n" + "="*70)
    print("🏆 TRAINING CAMPAIGN COMPLETE!")
    print("="*70)

if __name__ == '__main__':
    train_finetune_ultra_high_res()
    input("\nPress Enter to exit...")