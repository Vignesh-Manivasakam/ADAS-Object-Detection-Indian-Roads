# Dataset Preparation Pipeline

## 📊 Overview

This document details the complete data engineering pipeline used to prepare the Indian Driving Dataset (IDD) for YOLO training. This pipeline was one of the most challenging and time-consuming parts of the project, involving:

- Merging two datasets with different annotation formats
- Cleaning inconsistent class labels
- Converting 160,000+ annotations to YOLO format
- Handling naming conflicts and edge cases

---

## 🎯 The Challenge

### Problem Statement

We had two excellent datasets:
1. **IDD Dataset** (~45K images): XML annotations (Pascal VOC format)
2. **IDD95k Dataset** (~95K images): JSON annotations (custom format)

But they had significant incompatibilities:

| Issue | IDD (XML) | IDD95k (JSON) | Impact |
|-------|-----------|---------------|--------|
| **Format** | Pascal VOC XML | Custom JSON | Different parsers needed |
| **Classes** | 16 classes | 19 classes | Name conflicts |
| **Class Names** | "traffic sign" | "trafficsign" | Mismatch |
| **Rare Classes** | trailer (18), train (60) | caravan (136) | Too few examples |
| **Ambiguous** | "vehicle fallback" (21K) | - | Mixed categories |

**Without proper cleaning, training would fail or produce poor results.**

---

## 🔧 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 1: ANALYSIS                              │
│  Scan all files → Count instances → Identify issues              │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 2: CLEANING                              │
│  Remove rare classes → Standardize names → Filter noise          │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 3: CONVERSION                            │
│  XML → YOLO    |    JSON → YOLO    |    Validate                │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 4: MERGING                               │
│  Combine datasets → Unique names → Split train/val/test         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📂 Scripts Overview

### 1. `check_classes_IDD.py` - Class Analysis

**Purpose**: Scan all XML files and analyze class distribution

```bash
python scripts/data_preparation/check_classes_IDD.py
```

**Output**: `detailed_class_analysis.txt`
```
Class: motorcycle
  Count: 103608
  Avg bbox size: 0.99% of image
  Sample files: [list of 5 example files]

Class: trailer
  Count: 18  ⚠️ TOO RARE!
  Avg bbox size: 9.16% of image
  ...
```

**Key Findings**:
- **trailer**: Only 18 instances → Remove
- **train**: Only 60 instances → Remove
- **caravan**: Only 136 instances → Remove
- **vehicle fallback**: 21K instances but ambiguous → Remove

---

### 2. `check_class_IDD95.py` - IDD95k Analysis

**Purpose**: Analyze JSON annotations for IDD95k dataset

```bash
python scripts/data_preparation/check_class_IDD95.py
```

**Discovers**:
- Class name differences: "trafficsign" vs "traffic sign"
- Additional classes not in IDD
- JSON structure variations

---

### 3. `check_dataset_IDD.py` - Integrity Validation

**Purpose**: Verify dataset integrity before conversion

```bash
python scripts/data_preparation/check_dataset_IDD.py
```

**Checks**:
- Missing image files
- Corrupted XML/JSON
- Bounding box validity (xmin < xmax, ymin < ymax)
- Image dimensions match annotations

---

### 4. `clean.py` - Data Cleaning

**Purpose**: Remove rare and ambiguous classes from raw annotations

```python
# Classes to remove
CLASSES_TO_REMOVE = {
    "trailer",          # 18 instances
    "train",            # 60 instances
    "caravan",          # 136 instances
    "ego vehicle",      # Camera car (irrelevant)
    "pole",             # Too generic
    "vehicle fallback"  # Ambiguous mix
}
```

```bash
python scripts/data_preparation/clean.py
```

**Process**:
1. Creates backup of original files (safety!)
2. Parses each XML/JSON file
3. Removes objects matching `CLASSES_TO_REMOVE`
4. Saves cleaned files

**Result**: 16 classes → **11 high-quality classes**

---

### 5. `convert_idd_to_yolo.py` - IDD Conversion

**Purpose**: Convert IDD XML annotations to YOLO format

```bash
python scripts/data_preparation/convert_idd_to_yolo.py
```

**Process**:

```python
def convert_xml_annotation(xml_file, output_label_file):
    # 1. Parse XML
    tree = ET.parse(xml_file)
    
    # 2. Extract image dimensions
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)
    
    # 3. Convert bounding boxes
    for obj in root.iter('object'):
        class_name = obj.find('name').text
        
        # Pascal VOC format: [xmin, ymin, xmax, ymax]
        xmin = float(xmlbox.find('xmin').text)
        ymin = float(xmlbox.find('ymin').text)
        xmax = float(xmlbox.find('xmax').text)
        ymax = float(xmlbox.find('ymax').text)
        
        # Convert to YOLO format: [x_center, y_center, width, height]
        # All normalized to [0, 1]
        x_center = (xmin + xmax) / 2.0 / width
        y_center = (ymin + ymax) / 2.0 / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        
        # Save: class_id x_center y_center width height
        label_file.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")
```

**Handles**:
- Unique filename generation (path collisions)
- Train/val/test splits
- Creates YAML configuration

---

### 6. `Convert_final.py` - Master Dataset Creation

**Purpose**: Merge IDD + IDD95k into single YOLO dataset

```bash
python scripts/data_preparation/Convert_final.py
```

**Key Features**:

#### A. Dual Format Parsing
```python
def convert_xml_annotation(xml_file):
    """Handles IDD XML (Pascal VOC)"""
    tree = ET.parse(xml_file)
    # ... XML parsing logic

def convert_json_annotation(json_file, image_size):
    """Handles IDD95k JSON (custom format)"""
    with open(json_file) as f:
        objects = json.load(f)
    # ... JSON parsing logic
```

#### B. Naming Collision Prevention
```python
# Problem: Both datasets have "scene1/000001.jpg"
# Solution: Add dataset prefix

# IDD image:
unique_name = "IDD_" + img_path.replace('/', '_')
# Result: IDD_frontFar_scene1_000001.jpg

# IDD95k image:
unique_name = f"IDD95k_train_{folder}_{filename}"
# Result: IDD95k_train_leftImg8bit_000001.jpg
```

#### C. Train/Val/Test Splitting
```python
for split in ['train', 'val', 'test']:
    # Process IDD split
    idd_images = read_split_file(f'IDD/{split}.txt')
    
    # Process IDD95k split
    idd95k_images = list((IDD95K_ROOT / split).rglob('*.jpg'))
    
    # Merge into master split
    all_images = idd_images + idd95k_images
```

**Output Structure**:
```
Master_IDD_YOLO/
├── images/
│   ├── train/     (126,000 images)
│   ├── val/       (18,000 images)
│   └── test/      (16,000 images)
├── labels/
│   ├── train/     (126,000 labels)
│   ├── val/       (18,000 labels)
│   └── test/      (16,000 labels)
└── master_data.yaml
```

---

### 7. `custom_dataset.py` - Advanced Augmentation

**Purpose**: Weather-specific augmentation pipeline using Albumentations

```python
# Rain simulation
A.RandomRain(
    slant_lower=-10,
    slant_upper=10,
    blur_value=3,
    p=0.25  # 25% of training images
)

# Fog simulation
A.RandomFog(
    fog_coef_lower=0.3,
    fog_coef_upper=0.7,
    p=0.20
)

# Night/low-light
A.RandomBrightnessContrast(
    brightness_limit=(-0.4, 0.1),
    p=0.35
)
```

---

### 8. `Final_Code.py` - Complete End-to-End Pipeline

**Purpose**: Run entire pipeline from raw data to training-ready dataset

```bash
python scripts/data_preparation/Final_Code.py
```

**Executes**:
1. Class analysis
2. Data cleaning
3. Format conversion
4. Dataset merging
5. Validation checks
6. YAML generation

**One command to rule them all!**

---

## 🎯 Usage Guide

### Option 1: Step-by-Step (Recommended for First Time)

```bash
# 1. Analyze classes
python scripts/data_preparation/check_classes_IDD.py
python scripts/data_preparation/check_class_IDD95.py

# 2. Review detailed_class_analysis.txt
# Decide which classes to remove

# 3. Clean data
python scripts/data_preparation/clean.py

# 4. Convert and merge
python scripts/data_preparation/Convert_final.py

# 5. Validate
python scripts/data_preparation/check_dataset_IDD.py
```

### Option 2: One-Shot (If Confident)

```bash
python scripts/data_preparation/Final_Code.py
```

---

## 📊 Before vs After

### Before Cleaning
```
Total Classes: 16
Total Instances: 528,000

Problems:
❌ trailer: 18 instances (too rare)
❌ train: 60 instances (too rare)
❌ caravan: 136 instances (too rare)
❌ vehicle fallback: 21,081 instances (ambiguous)
❌ ego vehicle: 850 instances (irrelevant)
```

### After Cleaning
```
Total Classes: 11
Total Instances: 486,000

Results:
✅ All classes have >3,000 instances
✅ Clear, unambiguous definitions
✅ Balanced distribution (relative to real-world)
✅ Training mAP improved by 3%
```

---

## 🚨 Common Issues & Solutions

### Issue 1: "File not found" during conversion
**Cause**: Image file doesn't match annotation filename
**Solution**: Check `train.txt` paths match actual files
```bash
# Debug
python -c "
with open('IDD_Detection/train.txt') as f:
    for line in f:
        img = f'IDD_Detection/JPEGImages/{line.strip()}.jpg'
        if not os.path.exists(img):
            print(f'Missing: {img}')
"
```

### Issue 2: Empty label files generated
**Cause**: All objects in image belong to removed classes
**Solution**: This is normal! Images with no valid objects are skipped

### Issue 3: "Permission denied" when cleaning
**Cause**: Files are read-only or in use
**Solution**: Close any programs using the files, run as administrator

### Issue 4: Duplicate filenames in merged dataset
**Cause**: Both IDD and IDD95k have same relative paths
**Solution**: Already handled by unique naming in `Convert_final.py`

---

## 📈 Performance Impact

| Metric | Without Cleaning | With Cleaning | Improvement |
|--------|------------------|---------------|-------------|
| mAP@50-95 | 0.387 | 0.420 | **+8.5%** |
| Training Time | 22 hours | 18 hours | **-18%** |
| Class Balance | Poor | Good | **Balanced** |
| Model Confusion | High | Low | **Clearer** |

**Key Insight**: Spending 2 days on data cleaning saved weeks of debugging poor model performance.

---

## 🎓 Lessons Learned

1. **Always analyze before cleaning**: We initially kept "vehicle fallback" thinking more data = better. Wrong! It confused the model.

2. **Rare classes need thresholds**: Classes with <100 instances are noise, not signal.

3. **Format conversion is error-prone**: Triple-check bounding box conversions with visual inspection.

4. **Backups are essential**: We corrupted 50 XML files during early testing. Backups saved us.

5. **Unique naming matters**: Filename collisions between datasets caused silent data loss initially.

---

## 🔗 Related Documentation

- [TRAINING.md](TRAINING.md) - Training guide using the prepared dataset
- [README.md](../README.md) - Main project overview
- [METRICS.md](METRICS.md) - Understanding model performance

---

## 📧 Questions?

If you encounter issues with the data preparation pipeline:
1. Check the console output for error messages
2. Verify file paths in the configuration section of each script
3. Run the validation script (`check_dataset_IDD.py`)
4. Open an issue on GitHub with [DATA PREP] tag

---

## 🎯 Summary

This pipeline transformed raw, inconsistent datasets into a clean, training-ready format:

**Input**: 2 datasets, 16-19 classes, 2 formats, inconsistent quality  
**Output**: 1 dataset, 11 classes, YOLO format, production-ready

**Time Investment**: 2 days of data engineering  
**ROI**: +8.5% mAP, -18% training time, ∞ debugging saved

**This is why data engineering is 80% of ML projects!** 🚀
