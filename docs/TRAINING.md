# Training Guide

## Table of Contents
- [Prerequisites](#prerequisites)
- [Dataset Setup](#dataset-setup)
- [Training Stages](#training-stages)
- [Hyperparameters](#hyperparameters)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

**Minimum (for inference)**:
- CPU: 4 cores @ 2.5GHz
- RAM: 8GB
- GPU: Not required (CPU inference possible but slow)

**Recommended (for training)**:
- CPU: 8+ cores @ 3.0GHz
- RAM: 32GB
- GPU: NVIDIA GPU with 16GB+ VRAM (RTX 3090, RTX 4090, A5000, V100)
- Storage: 100GB SSD (for dataset + checkpoints)

**Tested Configuration**:
- GPU: NVIDIA RTX 3090 (24GB VRAM)
- CPU: AMD Ryzen 9 5900X
- RAM: 64GB DDR4
- Training Time: ~18 hours (all stages)

### Software Requirements

```bash
# Operating System
Ubuntu 20.04+ / Windows 10+ / macOS 12+

# CUDA Toolkit (for GPU training)
CUDA 11.8 or 12.1

# Python
Python 3.8+

# Package Manager
pip or conda
```

---

## Dataset Setup

### Option 1: Download IDD Dataset

```bash
# 1. Register at IDD website
Visit: https://idd.insaan.iiit.ac.in/

# 2. Download IDD Segmentation dataset
# (Object detection labels are derived from segmentation masks)

# 3. Convert to YOLO format (we provide converter script)
python scripts/convert_idd_to_yolo.py \
  --idd_root /path/to/IDD \
  --output_dir data/IDD_YOLO

# Expected structure:
data/IDD_YOLO/
├── images/
│   ├── train/      # ~15,000 images
│   └── val/        # ~3,000 images
└── labels/
    ├── train/
    └── val/
```

### Option 2: Use Pre-processed Dataset

```bash
# Download our pre-processed YOLO-format dataset
wget https://your-hosted-dataset.com/IDD_YOLO.zip
unzip IDD_YOLO.zip -d data/
```

### Dataset Statistics

| Split | Images | Annotations | Avg Objects/Image |
|-------|--------|-------------|-------------------|
| Train | 15,000 | 450,000+ | 30.2 |
| Val | 3,000 | 90,000+ | 30.1 |
| Test | 2,000 | 60,000+ | 30.0 |

### Class Distribution

```
Class Distribution (Training Set):
  car:           120,450 instances (26.8%)
  person:         98,230 instances (21.8%)
  motorcycle:     67,890 instances (15.1%)
  autorickshaw:   45,670 instances (10.1%)
  bus:            34,560 instances (7.7%)
  rider:          28,900 instances (6.4%)
  truck:          21,340 instances (4.7%)
  traffic_sign:   15,680 instances (3.5%)
  bicycle:         8,920 instances (2.0%)
  traffic_light:   6,740 instances (1.5%)
  vehicle:         1,230 instances (0.3%)
  animal:            250 instances (0.06%)
  caravan:            140 instances (0.03%)
```

---

## Training Stages

### Stage 1: Baseline Training @ 640px

**Purpose**: Establish baseline performance and learn general features.

```bash
python scripts/train_idd_base.py
```

**Configuration**:
```python
model: YOLOv8m (medium)
imgsz: 640
batch: 32
epochs: 100
patience: 15
optimizer: AdamW
lr0: 0.01
augmentations:
  - mosaic: 1.0
  - mixup: 0.1
  - copy_paste: 0.2
```

**Expected Results**:
- Training time: ~8 hours
- Final mAP@50-95: 0.28-0.30
- GPU memory: ~12GB

**Key Hyperparameters**:
- `batch=32`: Maximum for RTX 3090 @ 640px
- `mosaic=1.0`: Most effective augmentation for YOLO
- `copy_paste=0.2`: Helps with rare classes (animal, bicycle)

### Stage 2: Data Augmentation Generation

**Purpose**: Create 20,000 offline augmented images for weather/lighting robustness.

```bash
python scripts/run_augmentation.py
```

**Augmentation Pipeline**:
```python
Augmentations Applied:
  1. motion_blur (20%):    MotionBlur(blur_limit=(5, 11))
  2. gaussian_blur (15%):  GaussianBlur(blur_limit=(3, 7))
  3. rain (15%):           RandomRain(p=1.0)
  4. fog (15%):            RandomFog(p=1.0)
  5. night (10%):          RandomBrightness(-0.5, -0.3)
  6. dark (10%):           RandomBrightness(-0.8, -0.6)
  7. glare (10%):          RandomSunFlare(p=1.0)
  8. gamma (5%):           RandomGamma(50, 150)
```

**Output**:
```
data/MASTER_IDD_YOLO/
├── images/
│   ├── train/              # Original 15k
│   ├── train_augmented/    # NEW: 20k augmented
│   └── val/
└── labels/
    ├── train/
    ├── train_augmented/    # NEW: 20k augmented labels
    └── val/
```

**Important**: Update `master_data.yaml` to include augmented images:
```yaml
path: /path/to/data/MASTER_IDD_YOLO
train:
  - images/train
  - images/train_augmented  # Add this line
val: images/val
```

### Stage 3: Fine-Tuning @ 960px

**Purpose**: Improve small object detection (traffic lights, distant pedestrians).

```bash
python scripts/refine_tuning.py --imgsz 960 --epochs 50
```

**Configuration**:
```python
model: Load best.pt from Stage 1
imgsz: 960
batch: 16          # Reduced due to higher resolution
epochs: 50
patience: 10
lr0: 0.001         # Lower LR for fine-tuning
augmentations:
  - mosaic: 0.8    # Slightly reduced
  - mixup: 0.05    # Reduced
  - copy_paste: 0.1
```

**Expected Results**:
- Training time: ~6 hours
- Final mAP@50-95: 0.40-0.42
- GPU memory: ~22GB

### Stage 4 (Optional): Ultra-High-Res @ 1280px

**Purpose**: Final polish for maximum accuracy.

```bash
python scripts/refine_tuning.py --imgsz 1280 --epochs 15
```

**Configuration**:
```python
model: Load best.pt from Stage 3
imgsz: 1280
batch: 12          # Further reduced
epochs: 15         # Short run
patience: 5
lr0: 0.0005        # Very conservative
augmentations: Minimal (for stability)
```

**Expected Results**:
- Training time: ~3 hours
- Final mAP@50-95: 0.42-0.44
- GPU memory: ~24GB (requires 24GB+ GPU!)

---

## Hyperparameters

### Critical Hyperparameters Explained

#### Learning Rate Schedule
```python
lr0: 0.01           # Initial learning rate
lrf: 0.01           # Final learning rate (lr_final = lr0 * lrf)
warmup_epochs: 5    # Linear warmup for first N epochs
```

**Why these values?**
- `lr0=0.01`: Standard for AdamW optimizer with batch=32
- `lrf=0.01`: Decays to 1% of initial LR (10x reduction)
- `warmup_epochs=5`: Prevents early instability

#### Loss Weights
```python
box: 7.5            # Bounding box regression loss weight
cls: 0.5            # Classification loss weight
dfl: 1.5            # Distribution focal loss weight
```

**Tuning Strategy**:
- Increase `box` if bounding boxes are inaccurate
- Increase `cls` if classification errors are common
- Keep `dfl` at default unless you understand DFL deeply

#### Augmentation Probabilities
```python
mosaic: 1.0         # Always use mosaic (strongest augmentation)
mixup: 0.1          # 10% chance of mixing two images
copy_paste: 0.2     # 20% chance of pasting objects
degrees: 10.0       # Max rotation: ±10 degrees
translate: 0.1      # Max translation: ±10%
scale: 0.5          # Max scale: ±50%
fliplr: 0.5         # 50% horizontal flip
```

**Common Mistakes**:
- Setting `mosaic` too low → model doesn't learn small objects
- Setting `degrees` too high → unrealistic rotations
- Disabling `copy_paste` → rare classes don't improve

### Batch Size Selection

| GPU | VRAM | 640px | 960px | 1280px |
|-----|------|-------|-------|--------|
| RTX 3060 | 12GB | 16 | 8 | 4 |
| RTX 3070 | 8GB | 12 | 6 | 3 |
| RTX 3080 | 10GB | 20 | 10 | 6 |
| RTX 3090 | 24GB | 32 | 16 | 12 |
| RTX 4090 | 24GB | 40 | 20 | 14 |
| A100 | 40GB | 64 | 32 | 20 |

**Formula**: `batch_size ≈ VRAM_GB * k`, where `k = 2.0, 1.0, 0.5` for 640px, 960px, 1280px respectively.

---

## Troubleshooting

### Out of Memory (OOM) Errors

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```bash
# Option 1: Reduce batch size
python scripts/train_idd_base.py --batch 16  # Instead of 32

# Option 2: Use gradient accumulation
python scripts/train_idd_base.py --batch 8 --accumulate 4  # Effective batch=32

# Option 3: Enable mixed precision (AMP)
python scripts/train_idd_base.py --amp  # Already default in our scripts

# Option 4: Reduce image size
python scripts/train_idd_base.py --imgsz 512  # Instead of 640
```

### Training is Too Slow

**Issue**: Training takes >24 hours for Stage 1

**Solutions**:
```bash
# 1. Check GPU utilization
nvidia-smi -l 1  # Should be >90%

# 2. Increase workers
python scripts/train_idd_base.py --workers 8  # Max = CPU cores

# 3. Cache dataset to RAM
python scripts/train_idd_base.py --cache ram  # Requires 32GB+ RAM

# 4. Disable verbose logging
python scripts/train_idd_base.py --verbose False
```

### Model Not Converging

**Issue**: mAP stays <0.20 after 50+ epochs

**Diagnostics**:
```bash
# Check training curves
tensorboard --logdir runs/detect/train

# Common causes:
1. Learning rate too high → reduce lr0 to 0.001
2. Augmentations too aggressive → reduce mosaic to 0.5
3. Batch size too small → increase to 16+
4. Data quality issues → check labels manually
```

### Poor Performance on Rare Classes

**Issue**: Animal class has mAP < 0.10

**Solutions**:
```python
# 1. Increase copy-paste augmentation
copy_paste: 0.3  # From 0.2

# 2. Collect more rare class examples
# Use active learning or synthetic data

# 3. Adjust class weights (advanced)
# Edit ultralytics/yolo/utils/loss.py
```

### Overfitting

**Symptoms**:
- Training mAP = 0.60, Validation mAP = 0.30
- Loss curves diverge after epoch 30

**Solutions**:
```python
# 1. Stronger augmentations
mosaic: 1.0
mixup: 0.15
copy_paste: 0.3

# 2. Add weight decay
weight_decay: 0.001  # From 0.0005

# 3. Early stopping
patience: 10  # Stop earlier

# 4. Dropout (requires code modification)
dropout: 0.2  # Not natively supported in YOLOv8
```

---

## Advanced Training Techniques

### Learning Rate Finder

```python
from ultralytics import YOLO

model = YOLO('yolov8m.pt')
model.tune(
    data='data/master_data.yaml',
    epochs=10,
    iterations=300,
    optimizer='AdamW',
    plots=True
)
```

### Custom Loss Functions

```python
# Example: Add focal loss for classification
# Edit: ultralytics/yolo/utils/loss.py

class CustomLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        # Implement focal loss
        pass
```

### Multi-GPU Training

```bash
# Use DataParallel (not recommended for YOLO)
python scripts/train_idd_base.py --device 0,1

# Better: Use DistributedDataParallel
python -m torch.distributed.run --nproc_per_node=2 \
  scripts/train_idd_base.py --device 0,1
```

---

## Validation & Evaluation

### Run Validation

```bash
# Validate on test set
yolo val model=weights/best.pt data=data/master_data.yaml split=test

# Validate with specific IoU threshold
yolo val model=weights/best.pt data=data/master_data.yaml iou=0.6

# Save predictions for analysis
yolo val model=weights/best.pt data=data/master_data.yaml save_json=True
```

### Analyze Per-Class Performance

```python
from ultralytics import YOLO

model = YOLO('weights/best.pt')
metrics = model.val(data='data/master_data.yaml')

# Print per-class mAP
for i, class_name in model.names.items():
    print(f"{class_name}: {metrics.box.maps[i]:.4f}")
```

---

## Export & Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions on:
- INT8 Quantization
- ONNX export
- TensorRT optimization
- Mobile deployment (TFLite, CoreML)

---

## References

1. **YOLOv8 Documentation**: https://docs.ultralytics.com/
2. **IDD Dataset Paper**: Varma et al., WACV 2019
3. **Mosaic Augmentation**: Bochkovskiy et al., YOLOv4, 2020
4. **Copy-Paste Augmentation**: Ghiasi et al., CVPR 2021

---

## Support

For training-related issues:
- Open an issue on GitHub with `[TRAINING]` tag
- Include: GPU model, CUDA version, error logs, training command
