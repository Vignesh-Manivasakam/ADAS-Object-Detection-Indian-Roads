# 🚗 ADAS Object Detection & Real-Time Safety Decision System
### YOLOv8-based Multi-Class Detection + Production Inference Engine | 🏆 1st Prize

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00C9FF)](https://github.com/ultralytics/ultralytics)
[![ONNXRuntime](https://img.shields.io/badge/ONNXRuntime-005CED?logo=onnx)](https://onnxruntime.ai/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="assets/demo/detection_results.gif" alt="ADAS Detection Demo" width="800"/>
  <br>
  <em>Real-time object detection + safety decision-making optimized for challenging Indian road conditions</em>
</p>

---

## 🎯 Project Overview

This project implements a **full-stack, production-ready ADAS system** specifically built for **Indian road scenarios** — going beyond object detection to deliver real-time, prioritized driving decisions frame by frame.

**Two layers, built end-to-end:**
- **Layer 1 — Detection Model**: YOLOv8m trained on Indian road data, handling complex challenges unique to Indian traffic
- **Layer 2 — Inference Engine**: A real-time safety decision pipeline that wraps the model and translates detections into structured driving commands

### Key Achievements
- 🏆 **Won ADAS Hackathon** for Indian road scenario object detection
- 📊 **mAP@50-95: 0.420** | **mAP@50: 0.641** on IDD dataset
- 🎯 **Precision: 0.773** | **Recall: 0.569** | **F1-Score: 0.655**
- 🚀 **13 object classes** detected with real-time performance
- ⚡ **INT8 quantization** for edge deployment (3× size reduction, 2× speed boost)
- 🧠 **6-level safety decision hierarchy** with configurable decision smoothing
- 🔁 **Multi-object tracker** with persistent IDs and behavior classification

---

## 📊 Performance Metrics

<p align="center">
  <img src="assets/results/performance_dashboard.png" alt="Performance Dashboard" width="800"/>
</p>

### Model Performance Summary

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **mAP@50-95** | 0.420 | 0.400+ | ✅ Achieved |
| **mAP@50** | 0.641 | 0.550+ | ✅ Exceeded |
| **mAP@75** | 0.450 | 0.400+ | ✅ Achieved |
| **Precision** | 0.773 | 0.700+ | ✅ Exceeded |
| **Recall** | 0.569 | 0.500+ | ✅ Achieved |
| **F1-Score** | 0.655 | 0.600+ | ✅ Achieved |

### Class-Wise Performance (Top 10 Classes)

| Class | mAP@50-95 | Performance Level |
|-------|-----------|-------------------|
| Autorickshaw | 0.637 | 🟢 Excellent |
| Bus | 0.603 | 🟢 Excellent |
| Car | 0.591 | 🟢 Good |
| Motorcycle | 0.492 | 🟡 Good |
| Rider | 0.424 | 🟡 Moderate |
| Person | 0.370 | 🟡 Moderate |
| Traffic Sign | 0.325 | 🟠 Needs Improvement |
| Bicycle | 0.318 | 🟠 Needs Improvement |
| Traffic Light | 0.219 | 🔴 Challenging |
| Animal | 0.086 | 🔴 Rare Class |

<details>
<summary>📈 View Detailed Performance Charts</summary>

<p align="center">
  <img src="assets/results/precision_recall_curve.png" alt="Precision-Recall Curve" width="400"/>
  <img src="assets/results/radar_chart.png" alt="Performance Radar" width="400"/>
</p>

**Training Progression:**
- Baseline (640px): mAP@50-95 = 0.285
- Fine-tuned (960px): mAP@50-95 = 0.406 (+42.5%)
- Final (1280px): mAP@50-95 = 0.420 (+47.4%)

</details>

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     LAYER 1 — TRAINING PIPELINE                  │
│                                                                   │
│  IDD Dataset (13 classes) + IDD95k + Augmentations (~1,60,000)  │
│  → Progressive Training (640 → 960 → 1280px)                    │
│  → INT8 Quantization → FP32: 48.3MB | INT8: 15.7MB              │
└────────────────────────┬────────────────────────────────────────┘
                         │  YOLOv8 model (.pt / .onnx)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LAYER 2 — INFERENCE ENGINE                   │
│                                                                   │
│  Video Input                                                      │
│       ↓                                                           │
│  AdaptiveDetector (CITY / HIGHWAY / AUTO mode)                   │
│       ↓                                                           │
│  ObjectTracker (IoU-based, persistent IDs, class stabilization)  │
│       ↓                                                           │
│  Distance Estimation + TTC Calculation (per tracked object)      │
│       ↓                                                           │
│  Behavior Analysis (stopped / braking / accelerating / etc.)     │
│       ↓                                                           │
│  ADAS Decision Logic (6-level priority hierarchy)                │
│       ↓                                                           │
│  DecisionSmoother (consensus voting + safety-first override)     │
│       ↓                                                           │
│  Annotated Video Output + Session Report                         │
└─────────────────────────────────────────────────────────────────┘
```

### Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Detection Framework** | YOLOv8 (Ultralytics) | State-of-the-art object detection |
| **Model Backend** | PyTorch `.pt` / ONNX | GPU or edge/CPU deployment |
| **Object Tracking** | Custom IoU Tracker | Persistent multi-object tracking |
| **Vision Processing** | OpenCV | Frame capture, annotation, output |
| **Distance & TTC** | SciPy `linregress` | Monocular estimation + trend analysis |
| **Dataset** | IDD + IDD95k | 30k+ images of Indian roads |
| **Augmentation** | Albumentations + YOLO | Weather/lighting robustness |
| **Optimization** | INT8 Quantization | Edge device deployment |

---

## 🧠 Layer 2 — Real-Time ADAS Inference Engine

> *"The model detects. The engine decides."*

This is the layer that makes the system usable in a real ADAS context. Every frame goes through a full pipeline: detection → tracking → distance estimation → behavior analysis → decision → smoothing → display.

---

### 🚦 6-Level Safety Decision Hierarchy

Decisions are assigned a priority level and the highest-priority threat in frame drives the output:

| Priority | Decision | Trigger Condition |
|---|---|---|
| 🔴 **5** | **STOP** | Object <3m · Pedestrian in path <15m · Stopped vehicle in path |
| 🔴 **4** | **APPLY BRAKE** | Object in critical zone · TTC < 2s |
| 🟠 **3** | **READY TO BRAKE** | Preceding vehicle braking · TTC < 5s |
| 🟠 **3** | **FOLLOW CAUTIOUSLY** | Closing in warning zone · TTC < 5s |
| 🟡 **2** | **SLOW DOWN** | Object closing in caution zone · TTC < 10s |
| 🟢 **1** | **CONTINUE DRIVE** | No active threat detected in path |

Side-entry objects (entering from frame edges) receive reduced threat confidence and are excluded from STOP/APPLY BRAKE triggers until they've been tracked for 8+ frames in the path.

---
### Inference UI
<p align="center">
  <img src="assets/demo/inference UI.png" alt="Infernece UI" width="800"/>
</p>

### 🔁 Object Tracking & Class Stability

- **IoU-based multi-object tracker** — each detection is matched to existing tracks using bounding box overlap; unmatched detections spawn new tracks
- **Persistent track IDs** — objects retain their ID across frames (disappearance tolerance: 5 frames)
- **Class stabilization** — class label is not taken at face value. A majority vote over the last 20 frames is required (>75% consensus) before a class is committed to a track. This prevents misclassification flicker (e.g., `motorcycle` briefly detected as `bicycle`)
- **Lateral movement analysis** — tracks horizontal bbox movement per frame to detect lane changes; objects with high lateral speed are flagged and excluded from critical decisions

---

### 📐 Distance, Speed & TTC Estimation

**Monocular distance estimation** using class-specific real-world reference heights:

| Class | Reference Height |
|---|---|
| Car / Van | 1.6m |
| Person / Pedestrian | 1.7m |
| Bus | 3.0m |
| Motorcycle | 1.3m |
| Auto-rickshaw | 1.8m |

**Relative speed & TTC:**
- Linear regression (`scipy.stats.linregress`) fitted over a 10-frame distance history window
- Slope of the regression = closing rate (m/s) = relative speed
- TTC = current distance ÷ relative speed
- **TTC confidence score** derived from R² and history length — low-confidence TTC values are excluded from critical decision triggers
- Object speed estimated as: `ego_speed − relative_speed`

**Behavior classification per track:**

| Behavior | Detection Logic |
|---|---|
| `stopped` | Relative speed < 0.5 m/s AND object speed < 5 km/h |
| `braking` | Speed history shows delta < −8 km/h over recent frames |
| `accelerating` | Speed history shows delta > +8 km/h over recent frames |
| `lane_changing` | Lateral bbox movement exceeds 50px between frames |
| `moving` | Default — none of the above conditions met |

---

### 🧩 Decision Smoothing Engine

Raw per-frame decisions can fluctuate rapidly in dense traffic. The `DecisionSmoother` class adds a configurable buffer between raw detections and committed output decisions — reducing flicker while preserving genuine threat responsiveness.

**5 configurable presets:**

| Preset | Debounce Frames | Consensus Ratio | Use Case |
|---|---|---|---|
| `OFF` | 0 | — | Raw output, no filtering |
| `LIGHT` | 3 | 50% | Light traffic, fast response needed |
| `MEDIUM` | 5 | 60% | General city driving (default) |
| `STRONG` | 8 | 65% | Dense traffic, stability prioritized |
| `VERY_STRONG` | 12 | 70% | Validation / analysis mode |

**How it works:**
1. Each frame's raw decision is appended to a rolling history buffer
2. The most recent N frames (= debounce window) are majority-voted
3. If a decision reaches the consensus ratio, it is committed as output
4. If no consensus: the **highest-priority decision** in the window wins (safety-first fallback)
5. **Critical override**: For STOP or APPLY BRAKE, the debounce window is halved — emergency decisions propagate in half the normal time

**Side-by-side display mode** — raw and smoothed decisions rendered simultaneously for tuning and validation.

---

### 🚗 Adaptive Detection Modes

The system automatically selects the most appropriate detection configuration based on ego speed and object density:

| Mode | Conf Threshold | IoU Threshold | Auto-Selection Trigger |
|---|---|---|---|
| `CITY` | 0.30 | 0.40 | Object count > 8 OR speed < 40 km/h |
| `HIGHWAY` | 0.35 | 0.45 | Speed > 80 km/h AND object count < 3 |
| `AUTO` | 0.32 | 0.45 | Default adaptive mode |

Manual override available at runtime via keyboard.

---

### 📏 Speed-Adaptive Safety Zones

Distance thresholds are not fixed — they scale with the ego vehicle's current speed:

| Zone | Base Distance | Scaling |
|---|---|---|
| Critical | 5m | Clipped: 3m–10m |
| Warning | 15m | Clipped: 10m–30m |
| Caution | 30m | Clipped: 20m–50m |

Speed multiplier = `1 + ((current_speed − 40) / 100)`
Additional mode multipliers: Highway ×1.3, City ×0.8

---

### 🔺 Configurable Trapezoid Safety Zone

A trapezoidal ROI masks which detections are considered "in path" — filtering out objects in adjacent lanes that aren't a genuine threat.

- Adjustable in real time via keyboard (horizon, vanishing point X, bottom width, side angles)
- Partial overlap detection for large objects (trucks, buses) that straddle the lane boundary
- Configuration saved to `trapezoid_config.json` and auto-loaded on next run

---

### ⚡ Deployment & Output

| Feature | Detail |
|---|---|
| **Model backends** | PyTorch `.pt` (CUDA GPU) · ONNX (CPU / edge) — switchable at runtime |
| **Output resolutions** | FHD (1920×1080) · ULTRA (2048×1080) · QHD (2560×1440) |
| **Speed input** | Manual (keyboard ±5 km/h steps) or Auto (estimated from object density) |
| **Session recording** | Start/stop clip recording with bounding box overlay |
| **Session report** | Auto-saved `.txt` report on shutdown: FPS, detection counts, smoothing stats, safety config |
| **Performance rating** | FPS-based: Excellent (≥25) · Good (≥15) · Acceptable (≥10) · Needs Optimization (<10) |

---

## 🗂️ Dataset & Data Engineering

### Indian Driving Dataset (IDD)
- **Source**: IIT Hyderabad's benchmark dataset
- **Scope**: Urban and highway driving scenarios across India
- **Classes**: 13 object categories relevant to Indian roads

### Data Pipeline

```
Dataset Composition:
├── Base IDD Dataset:   ~45,000 images
├── IDD95k Extension:   ~95,000 images
└── Augmented Data:     ~20,000 synthetic images
    └── Total:          ~1,60,000 training images
```

### Augmentation Strategy

**Offline Augmentations** (Albumentations):
- 🌧️ `RandomRain` — simulates monsoon conditions
- 🌫️ `RandomFog` — low visibility scenarios
- 🌙 `RandomBrightnessContrast` — night/low-light (brightness: −0.8 to −0.3)
- 💡 `RandomSunFlare` — harsh glare conditions
- 📹 `MotionBlur` — camera shake simulation (blur: 5–11px)

**Online Augmentations** (YOLOv8 native):
- 🎲 `Mosaic=1.0` — combines 4 images → improves small object detection
- 🔀 `MixUp=0.1` — blends images → reduces overfitting
- 📋 `Copy-Paste=0.2` — inserts rare class objects → boosts animal/bicycle recall
- 🎨 `HSV Adjustments` — color and lighting diversity
- 🔄 `Geometric Transforms` — rotation (±10°), translation (±10%), scale (±50%)

---

## 🎓 Training Methodology

### 1. Progressive Training Strategy

| Stage | Resolution | Epochs | Purpose | mAP@50-95 |
|-------|-----------|--------|---------|-----------|
| **Stage 1** | 640px | 100 | Baseline training | 0.285 |
| **Stage 2** | 960px | 50 | Small object focus | 0.406 |
| **Stage 3** | 1280px | 15 | Final polish | **0.420** |

Higher resolutions increase VRAM exponentially — progressive training allows fast convergence at 640px then precision gains at higher resolutions without training from scratch.

### 2. Handling Class Imbalance

- **Copy-Paste Augmentation** (p=0.2) — inserts rare class instances into training scenes
- **Focal Loss** (implicit in YOLOv8) — down-weights easy negatives
- **Class Weight** — `cls` loss weight adjusted to 0.5 (from default 0.3)
- **Result**: Animal class improved from 0.04 → 0.086 mAP (+115%)

### 3. Addressing Poor Lighting Conditions

40% of Indian driving occurs in sub-optimal lighting. Augmentation pipeline covers night simulation, fog, rain, and glare. **Result**: mAP on night-time validation images improved by 28%.

### 4. Edge Optimization (INT8 Quantization)

```python
# Quantize Conv2D and Linear layers to INT8
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)
```

| | FP32 | INT8 |
|---|---|---|
| Model size | 48.3 MB | **15.7 MB** |
| Inference speed | 12.8 FPS | **24.5 FPS** |
| mAP drop | — | **<2%** |

---

## 🚀 Quick Start

### Prerequisites
```bash
python --version   # Python 3.8+
nvidia-smi         # CUDA GPU recommended
```

### Installation

```bash
git clone https://github.com/Vignesh-Manivasakam/ADAS-Object-Detection-Indian-Roads.git
cd ADAS-Object-Detection-Indian-Roads

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Training from Scratch

```bash
# Stage 1: Base training at 640px
python scripts/train_idd_base.py

# Stage 2: Fine-tune at 960px
python scripts/refine_tuning.py

# Generate augmented dataset
python scripts/run_augmentation.py
```

### Running the Inference Engine

```bash
# Configure paths in Final_Code.py before running:
#   PT_MODEL_PATH  = path to your .pt model
#   MODEL_PATH     = path to your .onnx model (optional)
#   VIDEO_PATH     = path to input video

python Final_Code.py
```

**Runtime keyboard controls:**

| Key | Action |
|---|---|
| `Space` | Pause / Resume |
| `↑` / `↓` | Increase / Decrease ego speed |
| `M` | Toggle speed mode (Manual / Auto) |
| `D` | Cycle decision smoothing preset |
| `0` / `9` | Increase / Decrease debounce frames |
| `B` | Toggle Raw vs Smoothed decision display |
| `C` / `H` / `A` | Force CITY / HIGHWAY / AUTO detection mode |
| `T` | Toggle trapezoid safety zone |
| `R` | Start / Stop session recording |
| `W` / `N` | Widen / Narrow trapezoid bottom |
| `O` | Save trapezoid config to JSON |
| `Q` / `ESC` | Quit |

### Standard Inference (Images / Video)

```bash
# Single image
python scripts/inference.py --source path/to/image.jpg --weights weights/best.pt

# Video file
python scripts/inference.py --source path/to/video.mp4 --weights weights/best.pt

# Webcam
python scripts/inference.py --source 0 --weights weights/best.pt
```

### Model Export

```bash
# ONNX (for TensorRT / edge deployment)
yolo export model=weights/best.pt format=onnx

# INT8 Quantization
python scripts/quantize_model.py --model weights/best.pt --data data/master_data.yaml

# TFLite / CoreML
yolo export model=weights/best.pt format=tflite
yolo export model=weights/best.pt format=coreml
```

---

## 📁 Project Structure

```
ADAS-Object-Detection-Indian-Roads/
│
├── README.md
├── requirements.txt
├── Final_Code.py                      # Full inference engine (main entry point)
│
├── scripts/
│   ├── train_idd_base.py              # Stage 1 training (640px)
│   ├── refine_tuning.py               # Stage 2 fine-tuning (960px)
│   ├── run_augmentation.py            # Offline augmentation pipeline
│   ├── quantize_model.py              # INT8 quantization
│   └── inference.py                   # Standard inference script
│
├── weights/
│   ├── best.pt                        # FP32 PyTorch model
│   └── best_int8.pt                   # INT8 quantized model
│
├── data/
│   ├── master_data.yaml               # Dataset config
│   └── README.md                      # Dataset setup guide
│
├── trapezoid_config.json              # Saved trapezoid ROI config
│
├── assets/
│   ├── demo/detection_results.gif
│   ├── results/
│   └── augmentation/
│
├── notebooks/
│   ├── exploratory_data_analysis.ipynb
│   ├── model_evaluation.ipynb
│   └── inference_demo.ipynb
│
└── docs/
    ├── TRAINING.md
    ├── DEPLOYMENT.md
    └── METRICS.md
```

---

## 📈 Results & Analysis

### Training Curves

<p align="center">
  <img src="assets/results/training_curves.png" alt="Training Curves" width="800"/>
</p>

**Key Observations:**
1. **Fast convergence** — model plateaus around epoch 60–70
2. **No overfitting** — validation loss tracks training loss closely
3. **Resolution boost** — significant mAP jump when moving to 1280px

### Failure Cases & Limitations

<details>
<summary>🔍 View Known Issues</summary>

**1. Occluded Objects**
- Problem: Partially hidden vehicles/pedestrians behind trees or poles
- Current performance: 35% recall for >70% occlusion
- Mitigation: Copy-paste augmentation with occlusion scenarios

**2. Extreme Lighting**
- Problem: Complete darkness or harsh noon glare
- Current performance: 15% mAP drop in extreme conditions
- Mitigation: Needs more diverse lighting augmentations

**3. Rare Classes (Animals)**
- Problem: Only ~250 training examples
- Current performance: mAP = 0.086 (too low for production)
- Mitigation: Synthetic data or active learning required

**4. Small, Distant Objects**
- Problem: Traffic lights >50m, pedestrians at crosswalks
- Current performance: 42% recall for objects <16px
- Mitigation: 1280px input helps but still challenging

</details>

---

## 🛠️ Advanced Usage

### Custom Dataset Training

```bash
# Prepare dataset in YOLO format, then create data.yaml:
cat > data/custom_data.yaml << EOF
path: /path/to/data
train: images/train
val: images/val
names:
  0: class_name_1
  1: class_name_2
EOF

python scripts/train_idd_base.py --data data/custom_data.yaml
```

### Hyperparameter Tuning

```bash
python scripts/train_idd_base.py --tune
```

---

## 📝 Citation

```bibtex
@misc{manivasakam2024adas,
  author = {Vignesh Manivasakam},
  title = {ADAS Object Detection for Indian Road Scenarios},
  year = {2024},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Vignesh-Manivasakam/ADAS-Object-Detection-Indian-Roads}}
}
```

**Dataset Citation (IDD):**
```bibtex
@inproceedings{varma2019idd,
  title={IDD: A Dataset for Exploring Problems of Autonomous Navigation in Unconstrained Environments},
  author={Varma, Girish and Subramanian, Anbumani and Namboodiri, Anoop and Chandraker, Manmohan and Jawahar, CV},
  booktitle={IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year={2019}
}
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

**Open areas for contribution:**
- [ ] Improve rare class detection (animal, bicycle)
- [ ] Add DeepSORT / ByteTrack for more robust multi-object tracking
- [ ] TensorRT optimization for <20ms inference on Jetson
- [ ] Transformer-based detection (DETR, DINO)
- [ ] Multi-camera fusion for 360° detection
- [ ] Expand decision logic with lane departure and blind-spot warnings

---

## 📄 License

Licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **IIT Hyderabad** for the Indian Driving Dataset (IDD)
- **Ultralytics** for the YOLOv8 framework
- **Albumentations** for the data augmentation library
- **ADAS Hackathon Organizers** for the platform and challenge

---

## 📧 Contact

**Vignesh Manivasakam**
- 📧 Email: vicky.manivasagam@gmail.com
- 💼 LinkedIn: [Vignesh Manivasakam](http://www.linkedin.com/in/vignesh-manivasakam-17b0a2128/)
- 🐙 GitHub: [@Vignesh-Manivasakam](https://github.com/Vignesh-Manivasakam)

---

<p align="center">
  <b>⭐ If you find this project useful, please consider giving it a star! ⭐</b>
</p>

<p align="center">
  <sub>Built with ❤️ for safer roads in India</sub>
</p>
