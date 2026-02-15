"""
COMPLETE IDD-YOLO DATASET INVESTIGATION SCRIPT
Performs all 10 critical checks on your converted dataset

Author: Hackathon ADAS Project
Date: Current Investigation
Purpose: Identify all data quality issues before further training
"""

import os
import glob
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# CONFIGURATION
# =====================================================

BASE_PATH = r"C:\Users\\Documents\Hackathon_ADAS\datasets\IDD_YOLO"

# Class mapping (adjust if yours is different)
CLASS_NAMES = {
    0: "person",
    1: "rider", 
    2: "car",
    3: "truck",
    4: "bus",
    5: "motorcycle",
    6: "bicycle",
    7: "autorickshaw",
    8: "animal",
    9: "traffic_light",
    10: "traffic_sign",
    11: "vehicle_fallback",
    12: "other"
}

# Safety-critical class IDs
SAFETY_CRITICAL_CLASSES = [0, 1, 5, 6]  # person, rider, motorcycle, bicycle

# Augmentation prefixes to check
AUGMENTATION_PREFIXES = ['blur_', 'rain_', 'fog_', 'night_']

# Output folder for reports
OUTPUT_DIR = os.path.join(BASE_PATH, "investigation_report")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("COMPREHENSIVE DATASET INVESTIGATION SCRIPT")
print("="*80)
print(f"Base Path: {BASE_PATH}")
print(f"Output Directory: {OUTPUT_DIR}")
print(f"Total Classes: {len(CLASS_NAMES)}")
print(f"Safety-Critical Classes: {[CLASS_NAMES[i] for i in SAFETY_CRITICAL_CLASSES]}")
print("="*80)

# =====================================================
# UTILITY FUNCTIONS
# =====================================================

def get_image_label_pairs(images_dir, labels_dir, split_name):
    """Get all image-label file pairs"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
    
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    
    return image_files, label_files

def get_filename_without_ext(filepath):
    """Get filename without extension"""
    return Path(filepath).stem

def read_yolo_label(label_path):
    """Read and parse YOLO format label file"""
    boxes = []
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    boxes.append({
                        'class_id': class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    })
    except Exception as e:
        return None, str(e)
    
    return boxes, None

def visualize_boxes_on_image(image_path, label_path, output_path):
    """Draw bounding boxes on image for visualization"""
    img = cv2.imread(image_path)
    if img is None:
        return False
    
    h, w = img.shape[:2]
    
    boxes, error = read_yolo_label(label_path)
    if error or boxes is None:
        return False
    
    for box in boxes:
        class_id = box['class_id']
        x_center = box['x_center'] * w
        y_center = box['y_center'] * h
        box_w = box['width'] * w
        box_h = box['height'] * h
        
        x1 = int(x_center - box_w/2)
        y1 = int(y_center - box_h/2)
        x2 = int(x_center + box_w/2)
        y2 = int(y_center + box_h/2)
        
        # Different colors for different class types
        if class_id in SAFETY_CRITICAL_CLASSES:
            color = (0, 0, 255)  # Red for safety-critical
        else:
            color = (0, 255, 0)  # Green for others
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        label = CLASS_NAMES.get(class_id, f"class_{class_id}")
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, color, 2)
    
    cv2.imwrite(output_path, img)
    return True

# =====================================================
# CHECK #1: DATASET STRUCTURE INTEGRITY
# =====================================================

def check_structure_integrity():
    """Verify image-label pairing and folder structure"""
    print("\n" + "="*80)
    print("CHECK #1: DATASET STRUCTURE INTEGRITY")
    print("="*80)
    
    results = {
        'train': {},
        'val': {}
    }
    
    for split in ['train', 'val']:
        print(f"\n--- Analyzing {split.upper()} split ---")
        
        images_dir = os.path.join(BASE_PATH, 'images', split)
        labels_dir = os.path.join(BASE_PATH, 'labels', split)
        
        image_files, label_files = get_image_label_pairs(images_dir, labels_dir, split)
        
        # Get base filenames
        image_basenames = {get_filename_without_ext(f) for f in image_files}
        label_basenames = {get_filename_without_ext(f) for f in label_files}
        
        # Find mismatches
        orphaned_images = image_basenames - label_basenames
        orphaned_labels = label_basenames - image_basenames
        matched = image_basenames & label_basenames
        
        results[split] = {
            'total_images': len(image_files),
            'total_labels': len(label_files),
            'matched_pairs': len(matched),
            'orphaned_images': len(orphaned_images),
            'orphaned_labels': len(orphaned_labels),
            'orphaned_image_list': list(orphaned_images)[:20],  # First 20 for report
            'orphaned_label_list': list(orphaned_labels)[:20]
        }
        
        print(f"✓ Total images found: {len(image_files)}")
        print(f"✓ Total labels found: {len(label_files)}")
        print(f"✓ Matched pairs: {len(matched)}")
        
        if orphaned_images:
            print(f"⚠ WARNING: {len(orphaned_images)} orphaned images (no labels)")
            print(f"  Examples: {list(orphaned_images)[:5]}")
        
        if orphaned_labels:
            print(f"⚠ WARNING: {len(orphaned_labels)} orphaned labels (no images)")
            print(f"  Examples: {list(orphaned_labels)[:5]}")
        
        # Check pairing percentage
        if len(image_files) > 0:
            pairing_rate = (len(matched) / len(image_files)) * 100
            results[split]['pairing_rate'] = pairing_rate
            
            if pairing_rate >= 98:
                print(f"✅ EXCELLENT: {pairing_rate:.2f}% pairing rate")
            elif pairing_rate >= 90:
                print(f"⚠ ACCEPTABLE: {pairing_rate:.2f}% pairing rate")
            else:
                print(f"❌ CRITICAL: Only {pairing_rate:.2f}% pairing rate!")
    
    return results

# =====================================================
# CHECK #2: LABEL FILE FORMAT CORRECTNESS
# =====================================================

def check_label_format():
    """Verify YOLO label format correctness"""
    print("\n" + "="*80)
    print("CHECK #2: LABEL FILE FORMAT CORRECTNESS")
    print("="*80)
    
    results = {
        'train': {},
        'val': {}
    }
    
    for split in ['train', 'val']:
        print(f"\n--- Analyzing {split.upper()} split ---")
        
        labels_dir = os.path.join(BASE_PATH, 'labels', split)
        label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
        
        format_errors = []
        coordinate_errors = []
        class_id_errors = []
        empty_files = []
        
        sample_size = min(len(label_files), 1000)  # Check first 1000 or all if less
        print(f"Checking {sample_size} label files for format issues...")
        
        for label_file in tqdm(label_files[:sample_size], desc="Validating"):
            # Check if empty
            if os.path.getsize(label_file) == 0:
                empty_files.append(label_file)
                continue
            
            boxes, error = read_yolo_label(label_file)
            
            if error:
                format_errors.append({'file': label_file, 'error': error})
                continue
            
            if boxes is None:
                format_errors.append({'file': label_file, 'error': 'Failed to parse'})
                continue
            
            # Validate each box
            for box in boxes:
                # Check class ID range
                if box['class_id'] < 0 or box['class_id'] >= len(CLASS_NAMES):
                    class_id_errors.append({
                        'file': label_file,
                        'class_id': box['class_id']
                    })
                
                # Check coordinate ranges
                if not (0.0 <= box['x_center'] <= 1.0):
                    coordinate_errors.append({
                        'file': label_file,
                        'field': 'x_center',
                        'value': box['x_center']
                    })
                
                if not (0.0 <= box['y_center'] <= 1.0):
                    coordinate_errors.append({
                        'file': label_file,
                        'field': 'y_center',
                        'value': box['y_center']
                    })
                
                if not (0.0 < box['width'] <= 1.0):
                    coordinate_errors.append({
                        'file': label_file,
                        'field': 'width',
                        'value': box['width']
                    })
                
                if not (0.0 < box['height'] <= 1.0):
                    coordinate_errors.append({
                        'file': label_file,
                        'field': 'height',
                        'value': box['height']
                    })
        
        results[split] = {
            'total_checked': sample_size,
            'format_errors': len(format_errors),
            'coordinate_errors': len(coordinate_errors),
            'class_id_errors': len(class_id_errors),
            'empty_files': len(empty_files),
            'format_error_details': format_errors[:10],
            'coordinate_error_details': coordinate_errors[:10],
            'class_id_error_details': class_id_errors[:10]
        }
        
        print(f"✓ Total files checked: {sample_size}")
        print(f"✓ Empty label files: {len(empty_files)}")
        print(f"✓ Format errors: {len(format_errors)}")
        print(f"✓ Coordinate errors: {len(coordinate_errors)}")
        print(f"✓ Class ID errors: {len(class_id_errors)}")
        
        error_rate = ((len(format_errors) + len(coordinate_errors) + len(class_id_errors)) / sample_size) * 100
        results[split]['error_rate'] = error_rate
        
        if error_rate == 0:
            print(f"✅ PERFECT: No format errors found!")
        elif error_rate < 1:
            print(f"✅ EXCELLENT: Only {error_rate:.2f}% error rate")
        elif error_rate < 5:
            print(f"⚠ ACCEPTABLE: {error_rate:.2f}% error rate")
        else:
            print(f"❌ CRITICAL: {error_rate:.2f}% error rate - major issues!")
    
    return results

# =====================================================
# CHECK #3: COMPLETE CLASS DISTRIBUTION ANALYSIS
# =====================================================

def check_class_distribution():
    """Analyze class distribution across entire dataset"""
    print("\n" + "="*80)
    print("CHECK #3: COMPLETE CLASS DISTRIBUTION ANALYSIS")
    print("="*80)
    
    results = {
        'train': {},
        'val': {},
        'combined': {}
    }
    
    for split in ['train', 'val']:
        print(f"\n--- Analyzing {split.upper()} split ---")
        
        labels_dir = os.path.join(BASE_PATH, 'labels', split)
        label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
        
        class_instance_counts = defaultdict(int)
        class_image_counts = defaultdict(int)
        total_instances = 0
        images_with_objects = 0
        empty_label_files = 0
        
        print(f"Scanning {len(label_files)} label files...")
        
        for label_file in tqdm(label_files, desc="Counting classes"):
            boxes, error = read_yolo_label(label_file)
            
            if error or boxes is None:
                continue
            
            if len(boxes) == 0:
                empty_label_files += 1
                continue
            
            images_with_objects += 1
            classes_in_image = set()
            
            for box in boxes:
                class_id = box['class_id']
                class_instance_counts[class_id] += 1
                classes_in_image.add(class_id)
                total_instances += 1
            
            for class_id in classes_in_image:
                class_image_counts[class_id] += 1
        
        # Calculate statistics
        class_stats = []
        for class_id in range(len(CLASS_NAMES)):
            instances = class_instance_counts.get(class_id, 0)
            images = class_image_counts.get(class_id, 0)
            instance_pct = (instances / total_instances * 100) if total_instances > 0 else 0
            image_pct = (images / len(label_files) * 100) if len(label_files) > 0 else 0
            
            class_stats.append({
                'class_id': class_id,
                'class_name': CLASS_NAMES[class_id],
                'instances': instances,
                'instance_percentage': instance_pct,
                'images_containing': images,
                'image_percentage': image_pct,
                'is_safety_critical': class_id in SAFETY_CRITICAL_CLASSES
            })
        
        # Sort by instance count
        class_stats.sort(key=lambda x: x['instances'], reverse=True)
        
        results[split] = {
            'total_label_files': len(label_files),
            'total_instances': total_instances,
            'images_with_objects': images_with_objects,
            'empty_label_files': empty_label_files,
            'class_statistics': class_stats
        }
        
        # Print summary
        print(f"\n📊 DISTRIBUTION SUMMARY:")
        print(f"  Total label files: {len(label_files)}")
        print(f"  Total object instances: {total_instances}")
        print(f"  Images with objects: {images_with_objects}")
        print(f"  Empty label files: {empty_label_files}")
        print(f"  Avg instances per image: {total_instances/images_with_objects:.2f}" if images_with_objects > 0 else "N/A")
        
        print(f"\n📋 PER-CLASS BREAKDOWN:")
        print(f"{'Class':<20} {'Instances':<12} {'%':<8} {'Images':<10} {'%':<8} {'Critical':<10}")
        print("-" * 78)
        
        for stat in class_stats:
            critical_mark = "⚠ YES" if stat['is_safety_critical'] else ""
            print(f"{stat['class_name']:<20} "
                  f"{stat['instances']:<12} "
                  f"{stat['instance_percentage']:<7.2f}% "
                  f"{stat['images_containing']:<10} "
                  f"{stat['image_percentage']:<7.2f}% "
                  f"{critical_mark:<10}")
        
        # Safety-critical analysis
        safety_critical_total = sum([s['instances'] for s in class_stats if s['is_safety_critical']])
        safety_critical_pct = (safety_critical_total / total_instances * 100) if total_instances > 0 else 0
        
        print(f"\n🚨 SAFETY-CRITICAL CLASSES TOTAL:")
        print(f"  Combined instances: {safety_critical_total} ({safety_critical_pct:.2f}%)")
        
        if safety_critical_pct >= 15:
            print(f"  ✅ EXCELLENT: Safety-critical classes well-represented")
        elif safety_critical_pct >= 10:
            print(f"  ⚠ ACCEPTABLE: Moderate representation")
        else:
            print(f"  ❌ CRITICAL: Severe under-representation!")
        
        results[split]['safety_critical_percentage'] = safety_critical_pct
    
    return results

# =====================================================
# CHECK #4: BOUNDING BOX QUALITY ANALYSIS (CONTINUED)
# =====================================================

def check_bbox_quality():
    """Verify bounding box quality through visualization and statistics"""
    print("\n" + "="*80)
    print("CHECK #4: BOUNDING BOX QUALITY ANALYSIS")
    print("="*80)
    
    results = {
        'train': {},
        'val': {}
    }
    
    # Create visualization folder
    viz_dir = os.path.join(OUTPUT_DIR, "bbox_visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    for split in ['train', 'val']:
        print(f"\n--- Analyzing {split.upper()} split ---")
        
        images_dir = os.path.join(BASE_PATH, 'images', split)
        labels_dir = os.path.join(BASE_PATH, 'labels', split)
        
        image_files, label_files = get_image_label_pairs(images_dir, labels_dir, split)
        
        # Sample random images for visualization
        sample_size = min(50, len(label_files))
        sample_indices = np.random.choice(len(label_files), sample_size, replace=False)
        
        tiny_boxes = []
        huge_boxes = []
        unusual_aspect_ratios = []
        box_sizes = []
        aspect_ratios = []
        
        print(f"Analyzing {sample_size} random samples...")
        
        for idx in tqdm(sample_indices, desc="Processing"):
            label_file = label_files[idx]
            label_basename = get_filename_without_ext(label_file)
            
            # Find corresponding image
            image_file = None
            for img_f in image_files:
                if get_filename_without_ext(img_f) == label_basename:
                    image_file = img_f
                    break
            
            if not image_file:
                continue
            
            boxes, error = read_yolo_label(label_file)
            if error or boxes is None:
                continue
            
            # Analyze boxes
            for box in boxes:
                width = box['width']
                height = box['height']
                area = width * height
                
                box_sizes.append(area)
                
                # Check for tiny boxes
                if area < 0.0001:  # Less than 0.01% of image
                    tiny_boxes.append({'file': label_file, 'area': area})
                
                # Check for huge boxes
                if area > 0.9:  # More than 90% of image
                    huge_boxes.append({'file': label_file, 'area': area})
                
                # Check aspect ratio
                if height > 0:
                    aspect_ratio = width / height
                    aspect_ratios.append(aspect_ratio)
                    
                    # Unusual aspect ratios
                    if aspect_ratio > 5 or aspect_ratio < 0.2:
                        unusual_aspect_ratios.append({
                            'file': label_file,
                            'aspect_ratio': aspect_ratio,
                            'class': CLASS_NAMES.get(box['class_id'], 'unknown')
                        })
            
            # Visualize first 10 samples
            if idx < 10:
                output_path = os.path.join(viz_dir, f"{split}_sample_{idx}.jpg")
                visualize_boxes_on_image(image_file, label_file, output_path)
        
        results[split] = {
            'samples_analyzed': sample_size,
            'tiny_boxes_count': len(tiny_boxes),
            'huge_boxes_count': len(huge_boxes),
            'unusual_aspect_ratios_count': len(unusual_aspect_ratios),
            'avg_box_area': np.mean(box_sizes) if box_sizes else 0,
            'median_box_area': np.median(box_sizes) if box_sizes else 0,
            'avg_aspect_ratio': np.mean(aspect_ratios) if aspect_ratios else 0,
            'tiny_box_examples': tiny_boxes[:10],
            'huge_box_examples': huge_boxes[:10],
            'unusual_aspect_examples': unusual_aspect_ratios[:10]
        }
        
        print(f"✓ Samples analyzed: {sample_size}")
        print(f"✓ Visualizations saved: {viz_dir}")
        print(f"✓ Tiny boxes found: {len(tiny_boxes)} (area < 0.01%)")
        print(f"✓ Huge boxes found: {len(huge_boxes)} (area > 90%)")
        print(f"✓ Unusual aspect ratios: {len(unusual_aspect_ratios)}")
        print(f"✓ Average box area: {np.mean(box_sizes)*100:.2f}%" if box_sizes else "N/A")
        print(f"✓ Average aspect ratio: {np.mean(aspect_ratios):.2f}" if aspect_ratios else "N/A")
        
        # Quality assessment
        issue_rate = ((len(tiny_boxes) + len(huge_boxes) + len(unusual_aspect_ratios)) / 
                      (sample_size * 10)) * 100  # Assuming ~10 boxes per image
        
        if issue_rate < 1:
            print(f"✅ EXCELLENT: Only {issue_rate:.2f}% problematic boxes")
        elif issue_rate < 5:
            print(f"⚠ ACCEPTABLE: {issue_rate:.2f}% problematic boxes")
        else:
            print(f"❌ CONCERNING: {issue_rate:.2f}% problematic boxes")
    
    return results

# =====================================================
# CHECK #5: CLASS LABEL MAPPING VERIFICATION
# =====================================================

def check_class_mapping():
    """Verify class ID to name mapping consistency"""
    print("\n" + "="*80)
    print("CHECK #5: CLASS LABEL MAPPING VERIFICATION")
    print("="*80)
    
    results = {}
    
    # Check if data.yaml exists
    data_yaml_path = os.path.join(BASE_PATH, "data.yaml")
    
    print(f"\n📋 DEFINED CLASS MAPPING:")
    for class_id, class_name in CLASS_NAMES.items():
        print(f"  {class_id}: {class_name}")
    
    # Scan all labels to find actual class IDs used
    all_class_ids = set()  # CHANGED: Use set from the start
    
    for split in ['train', 'val']:  # CHANGED: 'valid' to 'val' to match your folder
        labels_dir = os.path.join(BASE_PATH, 'labels', split)
        label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
        
        for label_file in tqdm(label_files[:1000], desc=f"Scanning {split}"):
            boxes, error = read_yolo_label(label_file)
            if error or boxes is None:
                continue
            
            for box in boxes:
                all_class_ids.add(box['class_id'])
    
    all_class_ids = sorted(list(all_class_ids))  # Convert to sorted list for display
    
    print(f"\n📊 ACTUAL CLASS IDs FOUND IN DATASET:")
    print(f"  {all_class_ids}")
    
    # Check for missing or extra classes
    defined_ids = set(CLASS_NAMES.keys())
    found_ids = set(all_class_ids)  # FIXED: Convert back to set for comparison
    missing_in_data = defined_ids - found_ids
    extra_in_data = found_ids - defined_ids
    
    results = {
        'defined_class_ids': list(defined_ids),
        'found_class_ids': all_class_ids,
        'missing_in_data': list(missing_in_data),
        'extra_in_data': list(extra_in_data),
        'mapping_consistent': len(extra_in_data) == 0
    }
    
    if missing_in_data:
        print(f"\n⚠ WARNING: Classes defined but not found in data:")
        for class_id in missing_in_data:
            print(f"  {class_id}: {CLASS_NAMES[class_id]}")
    
    if extra_in_data:
        print(f"\n❌ CRITICAL: Class IDs found in data but not defined:")
        for class_id in extra_in_data:
            print(f"  {class_id}: UNKNOWN")
    else:
        print(f"\n✅ PERFECT: All class IDs in data are properly defined")
    
    # Check data.yaml if exists
    if os.path.exists(data_yaml_path):
        print(f"\n✓ Found data.yaml at: {data_yaml_path}")
        results['data_yaml_exists'] = True
    else:
        print(f"\n⚠ WARNING: data.yaml not found at: {data_yaml_path}")
        results['data_yaml_exists'] = False
    
    return results

# =====================================================
# CHECK #6: AUGMENTED IMAGES VERIFICATION
# =====================================================

def check_augmented_images():
    """Verify augmented images quality and pairing"""
    print("\n" + "="*80)
    print("CHECK #6: AUGMENTED IMAGES VERIFICATION")
    print("="*80)
    
    results = {}
    
    # Only check training set (augmented images should only be in train)
    images_dir = os.path.join(BASE_PATH, 'images', 'train')
    labels_dir = os.path.join(BASE_PATH, 'labels', 'train')
    
    image_files, _ = get_image_label_pairs(images_dir, labels_dir, 'train')
    
    augmented_counts = {prefix: 0 for prefix in AUGMENTATION_PREFIXES}
    original_count = 0
    augmented_without_labels = []
    
    print(f"\nScanning {len(image_files)} training images...")
    
    for image_file in tqdm(image_files, desc="Checking augmentation"):
        basename = os.path.basename(image_file)
        filename_no_ext = get_filename_without_ext(image_file)
        
        # Check if augmented
        is_augmented = False
        for prefix in AUGMENTATION_PREFIXES:
            if basename.startswith(prefix):
                augmented_counts[prefix] += 1
                is_augmented = True
                break
        
        if not is_augmented:
            original_count += 1
        
        # Check if label exists
        label_file = os.path.join(labels_dir, filename_no_ext + '.txt')
        if not os.path.exists(label_file):
            if is_augmented:
                augmented_without_labels.append(basename)
    
    total_augmented = sum(augmented_counts.values())
    
    results = {
        'original_images': original_count,
        'total_augmented': total_augmented,
        'augmentation_breakdown': augmented_counts,
        'augmented_without_labels': len(augmented_without_labels),
        'augmented_without_labels_examples': augmented_without_labels[:10]
    }
    
    print(f"\n📊 AUGMENTATION SUMMARY:")
    print(f"  Original images: {original_count}")
    print(f"  Total augmented: {total_augmented}")
    print(f"  Total dataset size: {original_count + total_augmented}")
    
    print(f"\n📋 AUGMENTATION BREAKDOWN:")
    for prefix, count in augmented_counts.items():
        print(f"  {prefix:<10} {count:>6} images")
    
    if total_augmented > 0:
        augmentation_ratio = total_augmented / original_count if original_count > 0 else 0
        print(f"\n  Augmentation ratio: {augmentation_ratio:.2f}x")
        
        if augmentation_ratio >= 0.3:
            print(f"  ✅ GOOD: Substantial augmentation applied")
        else:
            print(f"  ⚠ LOW: Limited augmentation")
    else:
        print(f"\n  ❌ WARNING: No augmented images found!")
    
    if augmented_without_labels:
        print(f"\n❌ CRITICAL: {len(augmented_without_labels)} augmented images missing labels!")
        print(f"  Examples: {augmented_without_labels[:5]}")
    else:
        print(f"\n✅ PERFECT: All augmented images have corresponding labels")
    
    # Visual quality check on sample
    print(f"\nChecking augmentation visual quality (random samples)...")
    viz_dir = os.path.join(OUTPUT_DIR, "augmentation_samples")
    os.makedirs(viz_dir, exist_ok=True)
    
    for prefix in AUGMENTATION_PREFIXES:
        # Find images with this prefix
        prefix_images = [f for f in image_files if os.path.basename(f).startswith(prefix)]
        
        if prefix_images:
            # Save one sample
            sample_img = np.random.choice(prefix_images)
            img = cv2.imread(sample_img)
            if img is not None:
                output_path = os.path.join(viz_dir, f"sample_{prefix[:-1]}.jpg")
                cv2.imwrite(output_path, img)
    
    print(f"✓ Augmentation samples saved to: {viz_dir}")
    
    return results

# =====================================================
# CHECK #7: TRAIN/VALIDATION SPLIT INTEGRITY
# =====================================================

def check_train_val_split():
    """Verify train/validation split integrity"""
    print("\n" + "="*80)
    print("CHECK #7: TRAIN/VALIDATION SPLIT INTEGRITY")
    print("="*80)
    
    # Get all image basenames from both splits
    train_images_dir = os.path.join(BASE_PATH, 'images', 'train')
    val_images_dir = os.path.join(BASE_PATH, 'images', 'val')
    
    train_images, _ = get_image_label_pairs(train_images_dir, 
                                           os.path.join(BASE_PATH, 'labels', 'train'), 
                                           'train')
    val_images, _ = get_image_label_pairs(val_images_dir, 
                                         os.path.join(BASE_PATH, 'labels', 'val'), 
                                         'val')
    
    train_basenames = {get_filename_without_ext(f) for f in train_images}
    val_basenames = {get_filename_without_ext(f) for f in val_images}
    
    # Check for overlap
    overlap = train_basenames & val_basenames
    
    total_images = len(train_basenames) + len(val_basenames)
    train_percentage = (len(train_basenames) / total_images * 100) if total_images > 0 else 0
    val_percentage = (len(val_basenames) / total_images * 100) if total_images > 0 else 0
    
    results = {
        'train_images': len(train_basenames),
        'val_images': len(val_basenames),
        'total_images': total_images,
        'train_percentage': train_percentage,
        'val_percentage': val_percentage,
        'overlap_count': len(overlap),
        'overlap_examples': list(overlap)[:20]
    }
    
    print(f"\n📊 SPLIT STATISTICS:")
    print(f"  Training images: {len(train_basenames)} ({train_percentage:.1f}%)")
    print(f"  Validation images: {len(val_basenames)} ({val_percentage:.1f}%)")
    print(f"  Total unique images: {total_images}")
    
    # Check overlap
    if overlap:
        print(f"\n❌ CRITICAL: {len(overlap)} images appear in BOTH train and validation!")
        print(f"  This is DATA LEAKAGE - validation metrics are unreliable!")
        print(f"  Examples: {list(overlap)[:5]}")
    else:
        print(f"\n✅ PERFECT: No overlap between train and validation sets")
    
    # Check split ratio
    if 75 <= train_percentage <= 90:
        print(f"✅ OPTIMAL: Split ratio is appropriate")
    elif 70 <= train_percentage <= 95:
        print(f"⚠ ACCEPTABLE: Split ratio is acceptable")
    else:
        print(f"❌ SUBOPTIMAL: Unusual split ratio")
    
    # Check if augmented images in validation
    val_augmented = 0
    for val_img in val_images:
        basename = os.path.basename(val_img)
        for prefix in AUGMENTATION_PREFIXES:
            if basename.startswith(prefix):
                val_augmented += 1
                break
    
    results['val_augmented_count'] = val_augmented
    
    if val_augmented > 0:
        print(f"\n⚠ WARNING: {val_augmented} augmented images found in validation set!")
        print(f"  Validation should contain only original images")
    else:
        print(f"\n✅ GOOD: Validation contains only original images")
    
    return results

# =====================================================
# CHECK #8: ANNOTATION COMPLETENESS ANALYSIS
# =====================================================

def check_annotation_completeness():
    """Sample images to estimate annotation completeness"""
    print("\n" + "="*80)
    print("CHECK #8: ANNOTATION COMPLETENESS ANALYSIS")
    print("="*80)
    
    print("\n⚠ NOTE: This check requires MANUAL visual inspection")
    print("Generating sample images for manual review...")
    
    # Create samples folder
    samples_dir = os.path.join(OUTPUT_DIR, "annotation_completeness_samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    results = {}
    
    for split in ['train', 'val']:
        images_dir = os.path.join(BASE_PATH, 'images', split)
        labels_dir = os.path.join(BASE_PATH, 'labels', split)
        
        image_files, label_files = get_image_label_pairs(images_dir, labels_dir, split)
        
        # Sample 20 random images
        sample_size = min(20, len(image_files))
        sample_indices = np.random.choice(len(image_files), sample_size, replace=False)
        
        print(f"\nGenerating {sample_size} samples for {split} set...")
        
        for idx, sample_idx in enumerate(tqdm(sample_indices, desc="Creating samples")):
            image_file = image_files[sample_idx]
            basename = get_filename_without_ext(image_file)
            
            # Find label
            label_file = os.path.join(labels_dir, basename + '.txt')
            
            if os.path.exists(label_file):
                output_path = os.path.join(samples_dir, f"{split}_sample_{idx:02d}.jpg")
                visualize_boxes_on_image(image_file, label_file, output_path)
        
        results[split] = {
            'samples_generated': sample_size,
            'samples_location': samples_dir
        }
    
    print(f"\n✓ Sample images saved to: {samples_dir}")
    print(f"\n📋 MANUAL REVIEW INSTRUCTIONS:")
    print(f"  1. Open folder: {samples_dir}")
    print(f"  2. For each image, visually inspect:")
    print(f"     - Are all visible objects labeled?")
    print(f"     - Are small/distant objects labeled?")
    print(f"     - Are occluded objects labeled?")
    print(f"     - Are bounding boxes accurate?")
    print(f"  3. Estimate completeness rate manually")
    print(f"\n  Target: >90% completeness = GOOD")
    print(f"          80-90% = ACCEPTABLE")
    print(f"          <80% = NEEDS IMPROVEMENT")
    
    return results

# =====================================================
# CHECK #9: IMAGE QUALITY AND DIVERSITY ANALYSIS (CONTINUED)
# =====================================================

def check_image_diversity():
    """Analyze image quality and diversity"""
    print("\n" + "="*80)
    print("CHECK #9: IMAGE QUALITY AND DIVERSITY ANALYSIS")
    print("="*80)
    
    results = {
        'train': {},
        'valid': {}
    }
    
    for split in ['train', 'val']:
        print(f"\n--- Analyzing {split.upper()} split ---")
        
        images_dir = os.path.join(BASE_PATH, 'images', split)
        image_files, _ = get_image_label_pairs(images_dir, 
                                               os.path.join(BASE_PATH, 'labels', split), 
                                               split)
        
        # Sample images for analysis
        sample_size = min(200, len(image_files))
        sample_indices = np.random.choice(len(image_files), sample_size, replace=False)
        
        resolutions = []
        brightness_values = []
        
        print(f"Analyzing {sample_size} images for quality/diversity...")
        
        for idx in tqdm(sample_indices, desc="Processing"):
            img_path = image_files[idx]
            
            try:
                # Get resolution
                img = Image.open(img_path)
                width, height = img.size
                resolutions.append((width, height))
                
                # Get brightness (convert to grayscale and get mean)
                img_cv = cv2.imread(img_path)
                if img_cv is not None:
                    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                    brightness = np.mean(gray)
                    brightness_values.append(brightness)
                
            except Exception as e:
                continue
        
        # Analyze resolutions
        unique_resolutions = list(set(resolutions))
        resolution_counts = Counter(resolutions)
        most_common_res = resolution_counts.most_common(1)[0] if resolution_counts else ((0, 0), 0)
        
        # Analyze brightness
        brightness_mean = np.mean(brightness_values) if brightness_values else 0
        brightness_std = np.std(brightness_values) if brightness_values else 0
        
        # Categorize lighting
        very_dark = sum(1 for b in brightness_values if b < 50)
        dark = sum(1 for b in brightness_values if 50 <= b < 100)
        normal = sum(1 for b in brightness_values if 100 <= b < 180)
        bright = sum(1 for b in brightness_values if b >= 180)
        
        results[split] = {
            'samples_analyzed': sample_size,
            'unique_resolutions': len(unique_resolutions),
            'most_common_resolution': most_common_res[0],
            'resolution_uniformity': most_common_res[1] / sample_size if sample_size > 0 else 0,
            'avg_brightness': brightness_mean,
            'brightness_std': brightness_std,
            'lighting_distribution': {
                'very_dark': very_dark,
                'dark': dark,
                'normal': normal,
                'bright': bright
            }
        }
        
        print(f"\n📊 QUALITY METRICS:")
        print(f"  Samples analyzed: {sample_size}")
        print(f"  Unique resolutions: {len(unique_resolutions)}")
        print(f"  Most common: {most_common_res[0]} ({most_common_res[1]/sample_size*100:.1f}%)")
        
        if len(unique_resolutions) == 1:
            print(f"  ✅ PERFECT: All images have uniform resolution")
        elif len(unique_resolutions) <= 3:
            print(f"  ✅ GOOD: Very consistent resolutions")
        elif len(unique_resolutions) <= 10:
            print(f"  ⚠ ACCEPTABLE: Some resolution variation")
        else:
            print(f"  ❌ CONCERNING: High resolution diversity ({len(unique_resolutions)} different sizes)")
        
        print(f"\n💡 LIGHTING DIVERSITY:")
        print(f"  Average brightness: {brightness_mean:.1f}/255")
        print(f"  Brightness std dev: {brightness_std:.1f}")
        print(f"  Very dark (<50):   {very_dark:>4} ({very_dark/sample_size*100:>5.1f}%)")
        print(f"  Dark (50-100):     {dark:>4} ({dark/sample_size*100:>5.1f}%)")
        print(f"  Normal (100-180):  {normal:>4} ({normal/sample_size*100:>5.1f}%)")
        print(f"  Bright (>180):     {bright:>4} ({bright/sample_size*100:>5.1f}%)")
        
        # Diversity assessment
        if brightness_std > 40:
            print(f"  ✅ EXCELLENT: High lighting diversity")
        elif brightness_std > 25:
            print(f"  ✅ GOOD: Moderate lighting diversity")
        else:
            print(f"  ⚠ LIMITED: Low lighting diversity")
    
    return results

# =====================================================
# CHECK #10: SPATIAL PATTERN ANALYSIS
# =====================================================

def check_spatial_patterns():
    """Analyze spatial distribution patterns of each class"""
    print("\n" + "="*80)
    print("CHECK #10: SPATIAL PATTERN ANALYSIS (SAFETY-CRITICAL CLASSES)")
    print("="*80)
    
    results = {
        'train': {},
        'valid': {}
    }
    
    for split in ['train', 'val']:
        print(f"\n--- Analyzing {split.upper()} split ---")
        
        labels_dir = os.path.join(BASE_PATH, 'labels', split)
        label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
        
        # Collect spatial data for each class
        class_spatial_data = {class_id: {
            'x_centers': [],
            'y_centers': [],
            'widths': [],
            'heights': []
        } for class_id in CLASS_NAMES.keys()}
        
        print(f"Analyzing spatial patterns in {len(label_files)} label files...")
        
        for label_file in tqdm(label_files, desc="Processing"):
            boxes, error = read_yolo_label(label_file)
            if error or boxes is None:
                continue
            
            for box in boxes:
                class_id = box['class_id']
                if class_id in class_spatial_data:
                    class_spatial_data[class_id]['x_centers'].append(box['x_center'])
                    class_spatial_data[class_id]['y_centers'].append(box['y_center'])
                    class_spatial_data[class_id]['widths'].append(box['width'])
                    class_spatial_data[class_id]['heights'].append(box['height'])
        
        # Calculate statistics for each class
        class_stats = []
        
        for class_id in SAFETY_CRITICAL_CLASSES:
            data = class_spatial_data[class_id]
            
            if len(data['x_centers']) > 0:
                stats = {
                    'class_id': class_id,
                    'class_name': CLASS_NAMES[class_id],
                    'count': len(data['x_centers']),
                    'avg_x': np.mean(data['x_centers']),
                    'avg_y': np.mean(data['y_centers']),
                    'avg_width': np.mean(data['widths']),
                    'avg_height': np.mean(data['heights']),
                    'std_x': np.std(data['x_centers']),
                    'std_y': np.std(data['y_centers']),
                    'std_width': np.std(data['widths']),
                    'std_height': np.std(data['heights'])
                }
                class_stats.append(stats)
        
        results[split] = {
            'class_statistics': class_stats,
            'raw_data': class_spatial_data
        }
        
        # Print summary
        print(f"\n📍 SPATIAL DISTRIBUTION (Safety-Critical Classes Only):")
        print(f"{'Class':<15} {'Count':<8} {'Avg X':<8} {'Avg Y':<8} {'Avg W':<8} {'Avg H':<8}")
        print("-" * 63)
        
        for stat in class_stats:
            print(f"{stat['class_name']:<15} "
                  f"{stat['count']:<8} "
                  f"{stat['avg_x']:<8.3f} "
                  f"{stat['avg_y']:<8.3f} "
                  f"{stat['avg_width']:<8.3f} "
                  f"{stat['avg_height']:<8.3f}")
        
        print(f"\n📊 SPATIAL BIAS ASSESSMENT:")
        for stat in class_stats:
            # Check for extreme spatial bias
            x_bias = "CENTERED" if 0.3 < stat['avg_x'] < 0.7 else "BIASED"
            y_bias = "CENTERED" if 0.3 < stat['avg_y'] < 0.7 else "BIASED"
            
            # Check for size uniformity
            size_diversity = "DIVERSE" if stat['std_width'] > 0.03 else "UNIFORM"
            
            print(f"  {stat['class_name']}:")
            print(f"    Horizontal: {x_bias} (x={stat['avg_x']:.2f}±{stat['std_x']:.2f})")
            print(f"    Vertical: {y_bias} (y={stat['avg_y']:.2f}±{stat['std_y']:.2f})")
            print(f"    Size: {size_diversity} (w={stat['avg_width']:.3f}±{stat['std_width']:.3f})")
    
    return results

# =====================================================
# VISUALIZATION GENERATION
# =====================================================

def generate_visualizations(all_results):
    """Generate comprehensive visualization plots"""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATION PLOTS")
    print("="*80)
    
    viz_output_dir = os.path.join(OUTPUT_DIR, "visualizations")
    os.makedirs(viz_output_dir, exist_ok=True)
    
    # 1. Class Distribution Bar Chart
    if 'class_distribution' in all_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        train_stats = all_results['class_distribution']['train']['class_statistics']
        
        classes = [s['class_name'] for s in train_stats]
        instances = [s['instances'] for s in train_stats]
        percentages = [s['instance_percentage'] for s in train_stats]
        
        # Instance counts
        colors = ['red' if s['is_safety_critical'] else 'steelblue' for s in train_stats]
        ax1.bar(classes, instances, color=colors)
        ax1.set_title('Class Distribution (Instance Counts)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Number of Instances')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Percentages
        ax2.bar(classes, percentages, color=colors)
        ax2.set_title('Class Distribution (Percentages)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Percentage (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.axhline(y=10, color='orange', linestyle='--', label='10% threshold')
        ax2.axhline(y=5, color='red', linestyle='--', label='5% threshold')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Generated: class_distribution.png")
    
    # 2. Safety-Critical vs Others Pie Chart
    if 'class_distribution' in all_results:
        train_stats = all_results['class_distribution']['train']['class_statistics']
        
        safety_critical_total = sum([s['instances'] for s in train_stats if s['is_safety_critical']])
        others_total = sum([s['instances'] for s in train_stats if not s['is_safety_critical']])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sizes = [safety_critical_total, others_total]
        labels = [f"Safety-Critical\n({safety_critical_total:,} instances)", 
                  f"Other Classes\n({others_total:,} instances)"]
        colors = ['#ff6b6b', '#4ecdc4']
        explode = (0.1, 0)
        
        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax.set_title('Safety-Critical vs Other Classes Distribution', fontsize=16, fontweight='bold')
        
        plt.savefig(os.path.join(viz_output_dir, 'safety_critical_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Generated: safety_critical_distribution.png")
    
    # 3. Brightness Distribution Histogram
    if 'image_diversity' in all_results:
        # This would require storing actual brightness values, skip for now
        pass
    
    # 4. Spatial Heatmap for Safety-Critical Classes
    if 'spatial_patterns' in all_results:
        train_spatial = all_results['spatial_patterns']['train']['raw_data']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, class_id in enumerate(SAFETY_CRITICAL_CLASSES):
            data = train_spatial[class_id]
            
            if len(data['x_centers']) > 0:
                ax = axes[idx]
                
                # Create 2D histogram
                h, xedges, yedges = np.histogram2d(
                    data['x_centers'], 
                    data['y_centers'], 
                    bins=20, 
                    range=[[0, 1], [0, 1]]
                )
                
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                im = ax.imshow(h.T, origin='lower', extent=extent, cmap='hot', aspect='auto')
                ax.set_title(f'{CLASS_NAMES[class_id]} Spatial Distribution', fontweight='bold')
                ax.set_xlabel('X Position (normalized)')
                ax.set_ylabel('Y Position (normalized)')
                plt.colorbar(im, ax=ax, label='Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_output_dir, 'spatial_heatmaps.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Generated: spatial_heatmaps.png")
    
    print(f"\n✓ All visualizations saved to: {viz_output_dir}")

# =====================================================
# REPORT GENERATION
# =====================================================

def generate_comprehensive_report(all_results):
    """Generate detailed text report"""
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE REPORT")
    print("="*80)
    
    report_path = os.path.join(OUTPUT_DIR, "INVESTIGATION_REPORT.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("IDD-YOLO DATASET INVESTIGATION REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset Path: {BASE_PATH}\n")
        f.write("="*80 + "\n\n")
        
        # CHECK #1: Structure Integrity
        f.write("CHECK #1: DATASET STRUCTURE INTEGRITY\n")
        f.write("-" * 80 + "\n")
        if 'structure_integrity' in all_results:
            for split in ['train', 'val']:
                result = all_results['structure_integrity'][split]
                f.write(f"\n{split.upper()} Split:\n")
                f.write(f"  Total images: {result['total_images']}\n")
                f.write(f"  Total labels: {result['total_labels']}\n")
                f.write(f"  Matched pairs: {result['matched_pairs']}\n")
                f.write(f"  Pairing rate: {result['pairing_rate']:.2f}%\n")
                f.write(f"  Orphaned images: {result['orphaned_images']}\n")
                f.write(f"  Orphaned labels: {result['orphaned_labels']}\n")
                
                if result['pairing_rate'] >= 98:
                    f.write(f"  STATUS: ✅ EXCELLENT\n")
                elif result['pairing_rate'] >= 90:
                    f.write(f"  STATUS: ⚠ ACCEPTABLE\n")
                else:
                    f.write(f"  STATUS: ❌ CRITICAL ISSUE\n")
        f.write("\n\n")
        
        # CHECK #2: Label Format
        f.write("CHECK #2: LABEL FILE FORMAT CORRECTNESS\n")
        f.write("-" * 80 + "\n")
        if 'label_format' in all_results:
            for split in ['train', 'val']:
                result = all_results['label_format'][split]
                f.write(f"\n{split.upper()} Split:\n")
                f.write(f"  Files checked: {result['total_checked']}\n")
                f.write(f"  Format errors: {result['format_errors']}\n")
                f.write(f"  Coordinate errors: {result['coordinate_errors']}\n")
                f.write(f"  Class ID errors: {result['class_id_errors']}\n")
                f.write(f"  Empty files: {result['empty_files']}\n")
                f.write(f"  Error rate: {result['error_rate']:.2f}%\n")
                
                if result['error_rate'] < 1:
                    f.write(f"  STATUS: ✅ EXCELLENT\n")
                elif result['error_rate'] < 5:
                    f.write(f"  STATUS: ⚠ ACCEPTABLE\n")
                else:
                    f.write(f"  STATUS: ❌ CRITICAL ISSUE\n")
        f.write("\n\n")
        
        # CHECK #3: Class Distribution
        f.write("CHECK #3: CLASS DISTRIBUTION ANALYSIS\n")
        f.write("-" * 80 + "\n")
        if 'class_distribution' in all_results:
            result = all_results['class_distribution']['train']
            f.write(f"\nTRAINING Set:\n")
            f.write(f"  Total instances: {result['total_instances']}\n")
            f.write(f"  Images with objects: {result['images_with_objects']}\n")
            f.write(f"  Safety-critical %: {result['safety_critical_percentage']:.2f}%\n\n")
            
            f.write(f"  Per-Class Breakdown:\n")
            for stat in result['class_statistics']:
                f.write(f"    {stat['class_name']:<15} "
                       f"Instances: {stat['instances']:>6} ({stat['instance_percentage']:>5.2f}%) "
                       f"Images: {stat['images_containing']:>5} ({stat['image_percentage']:>5.2f}%)")
                if stat['is_safety_critical']:
                    f.write(" ⚠ CRITICAL")
                f.write("\n")
            
            if result['safety_critical_percentage'] >= 15:
                f.write(f"\n  STATUS: ✅ EXCELLENT - Safety-critical well represented\n")
            elif result['safety_critical_percentage'] >= 10:
                f.write(f"\n  STATUS: ⚠ ACCEPTABLE - Moderate representation\n")
            else:
                f.write(f"\n  STATUS: ❌ CRITICAL - Severe under-representation\n")
        f.write("\n\n")
        
        # CHECK #6: Augmentation
        f.write("CHECK #6: AUGMENTED IMAGES VERIFICATION\n")
        f.write("-" * 80 + "\n")
        if 'augmented_images' in all_results:
            result = all_results['augmented_images']
            f.write(f"  Original images: {result['original_images']}\n")
            f.write(f"  Total augmented: {result['total_augmented']}\n")
            f.write(f"  Augmentation breakdown:\n")
            for prefix, count in result['augmentation_breakdown'].items():
                f.write(f"    {prefix}: {count}\n")
            f.write(f"  Augmented without labels: {result['augmented_without_labels']}\n")
            
            if result['total_augmented'] > 0 and result['augmented_without_labels'] == 0:
                f.write(f"  STATUS: ✅ EXCELLENT\n")
            elif result['augmented_without_labels'] > 0:
                f.write(f"  STATUS: ❌ CRITICAL - Missing labels for augmented images\n")
            else:
                f.write(f"  STATUS: ⚠ WARNING - No augmented images found\n")
        f.write("\n\n")
        
        # CHECK #7: Train/Val Split
        f.write("CHECK #7: TRAIN/VALIDATION SPLIT INTEGRITY\n")
        f.write("-" * 80 + "\n")
        if 'train_val_split' in all_results:
            result = all_results['train_val_split']
            f.write(f"  Training images: {result['train_images']} ({result['train_percentage']:.1f}%)\n")
            f.write(f"  Validation images: {result['val_images']} ({result['val_percentage']:.1f}%)\n")
            f.write(f"  Overlap (data leakage): {result['overlap_count']}\n")
# =====================================================
# REPORT GENERATION (COMPLETE)
# =====================================================

def generate_comprehensive_report(all_results):
    """Generate detailed text report"""
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE REPORT")
    print("="*80)
    
    report_path = os.path.join(OUTPUT_DIR, "INVESTIGATION_REPORT.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("IDD-YOLO DATASET INVESTIGATION REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset Path: {BASE_PATH}\n")
        f.write("="*80 + "\n\n")
        
        # CHECK #1: Structure Integrity
        f.write("CHECK #1: DATASET STRUCTURE INTEGRITY\n")
        f.write("-" * 80 + "\n")
        if 'structure_integrity' in all_results:
            for split in ['train', 'val']:
                result = all_results['structure_integrity'][split]
                f.write(f"\n{split.upper()} Split:\n")
                f.write(f"  Total images: {result['total_images']}\n")
                f.write(f"  Total labels: {result['total_labels']}\n")
                f.write(f"  Matched pairs: {result['matched_pairs']}\n")
                f.write(f"  Pairing rate: {result['pairing_rate']:.2f}%\n")
                f.write(f"  Orphaned images: {result['orphaned_images']}\n")
                f.write(f"  Orphaned labels: {result['orphaned_labels']}\n")
                
                if result['orphaned_images'] > 0:
                    f.write(f"  Orphaned image examples: {result['orphaned_image_list'][:5]}\n")
                if result['orphaned_labels'] > 0:
                    f.write(f"  Orphaned label examples: {result['orphaned_label_list'][:5]}\n")
                
                if result['pairing_rate'] >= 98:
                    f.write(f"  STATUS: ✅ EXCELLENT\n")
                elif result['pairing_rate'] >= 90:
                    f.write(f"  STATUS: ⚠ ACCEPTABLE\n")
                else:
                    f.write(f"  STATUS: ❌ CRITICAL ISSUE\n")
        f.write("\n\n")
        
        # CHECK #2: Label Format
        f.write("CHECK #2: LABEL FILE FORMAT CORRECTNESS\n")
        f.write("-" * 80 + "\n")
        if 'label_format' in all_results:
            for split in ['train', 'val']:
                result = all_results['label_format'][split]
                f.write(f"\n{split.upper()} Split:\n")
                f.write(f"  Files checked: {result['total_checked']}\n")
                f.write(f"  Format errors: {result['format_errors']}\n")
                f.write(f"  Coordinate errors: {result['coordinate_errors']}\n")
                f.write(f"  Class ID errors: {result['class_id_errors']}\n")
                f.write(f"  Empty files: {result['empty_files']}\n")
                f.write(f"  Error rate: {result['error_rate']:.2f}%\n")
                
                if result['format_errors'] > 0:
                    f.write(f"\n  Format error examples:\n")
                    for err in result['format_error_details'][:5]:
                        f.write(f"    {os.path.basename(err['file'])}: {err['error']}\n")
                
                if result['coordinate_errors'] > 0:
                    f.write(f"\n  Coordinate error examples:\n")
                    for err in result['coordinate_error_details'][:5]:
                        f.write(f"    {os.path.basename(err['file'])}: {err['field']}={err['value']}\n")
                
                if result['class_id_errors'] > 0:
                    f.write(f"\n  Class ID error examples:\n")
                    for err in result['class_id_error_details'][:5]:
                        f.write(f"    {os.path.basename(err['file'])}: class_id={err['class_id']}\n")
                
                if result['error_rate'] < 1:
                    f.write(f"\n  STATUS: ✅ EXCELLENT\n")
                elif result['error_rate'] < 5:
                    f.write(f"\n  STATUS: ⚠ ACCEPTABLE\n")
                else:
                    f.write(f"\n  STATUS: ❌ CRITICAL ISSUE\n")
        f.write("\n\n")
        
        # CHECK #3: Class Distribution
        f.write("CHECK #3: CLASS DISTRIBUTION ANALYSIS\n")
        f.write("-" * 80 + "\n")
        if 'class_distribution' in all_results:
            result = all_results['class_distribution']['train']
            f.write(f"\nTRAINING Set:\n")
            f.write(f"  Total instances: {result['total_instances']}\n")
            f.write(f"  Images with objects: {result['images_with_objects']}\n")
            f.write(f"  Empty label files: {result['empty_label_files']}\n")
            f.write(f"  Safety-critical %: {result['safety_critical_percentage']:.2f}%\n\n")
            
            f.write(f"  Per-Class Breakdown:\n")
            f.write(f"  {'Class':<20} {'Instances':<12} {'%':<8} {'Images':<10} {'%':<8} {'Critical'}\n")
            f.write(f"  {'-'*78}\n")
            for stat in result['class_statistics']:
                critical_mark = "⚠ YES" if stat['is_safety_critical'] else ""
                f.write(f"  {stat['class_name']:<20} "
                       f"{stat['instances']:<12} "
                       f"{stat['instance_percentage']:<7.2f}% "
                       f"{stat['images_containing']:<10} "
                       f"{stat['image_percentage']:<7.2f}% "
                       f"{critical_mark}\n")
            
            f.write(f"\n  Safety-Critical Classes Summary:\n")
            for stat in result['class_statistics']:
                if stat['is_safety_critical']:
                    f.write(f"    {stat['class_name']}: {stat['instances']} instances "
                           f"({stat['instance_percentage']:.2f}%)\n")
            
            if result['safety_critical_percentage'] >= 15:
                f.write(f"\n  STATUS: ✅ EXCELLENT - Safety-critical well represented\n")
            elif result['safety_critical_percentage'] >= 10:
                f.write(f"\n  STATUS: ⚠ ACCEPTABLE - Moderate representation\n")
            else:
                f.write(f"\n  STATUS: ❌ CRITICAL - Severe under-representation\n")
            
            # Validation set
            result_val = all_results['class_distribution']['val']
            f.write(f"\nVALIDATION Set:\n")
            f.write(f"  Total instances: {result_val['total_instances']}\n")
            f.write(f"  Images with objects: {result_val['images_with_objects']}\n")
            f.write(f"  Safety-critical %: {result_val['safety_critical_percentage']:.2f}%\n")
        f.write("\n\n")
        
        # CHECK #4: Bounding Box Quality
        f.write("CHECK #4: BOUNDING BOX QUALITY ANALYSIS\n")
        f.write("-" * 80 + "\n")
        if 'bbox_quality' in all_results:
            for split in ['train', 'val']:
                result = all_results['bbox_quality'][split]
                f.write(f"\n{split.upper()} Split:\n")
                f.write(f"  Samples analyzed: {result['samples_analyzed']}\n")
                f.write(f"  Tiny boxes (<0.01% area): {result['tiny_boxes_count']}\n")
                f.write(f"  Huge boxes (>90% area): {result['huge_boxes_count']}\n")
                f.write(f"  Unusual aspect ratios: {result['unusual_aspect_ratios_count']}\n")
                f.write(f"  Average box area: {result['avg_box_area']*100:.3f}%\n")
                f.write(f"  Median box area: {result['median_box_area']*100:.3f}%\n")
                f.write(f"  Average aspect ratio: {result['avg_aspect_ratio']:.2f}\n")
                
                if result['tiny_boxes_count'] > 0:
                    f.write(f"\n  Tiny box examples:\n")
                    for ex in result['tiny_box_examples'][:3]:
                        f.write(f"    {os.path.basename(ex['file'])}: area={ex['area']:.6f}\n")
                
                if result['huge_boxes_count'] > 0:
                    f.write(f"\n  Huge box examples:\n")
                    for ex in result['huge_box_examples'][:3]:
                        f.write(f"    {os.path.basename(ex['file'])}: area={ex['area']:.3f}\n")
                
                if result['unusual_aspect_ratios_count'] > 0:
                    f.write(f"\n  Unusual aspect ratio examples:\n")
                    for ex in result['unusual_aspect_examples'][:3]:
                        f.write(f"    {os.path.basename(ex['file'])}: {ex['class']} "
                               f"ratio={ex['aspect_ratio']:.2f}\n")
                
                issue_count = (result['tiny_boxes_count'] + result['huge_boxes_count'] + 
                              result['unusual_aspect_ratios_count'])
                if issue_count == 0:
                    f.write(f"\n  STATUS: ✅ EXCELLENT - No problematic boxes found\n")
                elif issue_count < result['samples_analyzed'] * 0.05:
                    f.write(f"\n  STATUS: ✅ GOOD - Minimal issues\n")
                else:
                    f.write(f"\n  STATUS: ⚠ REVIEW NEEDED - Multiple issues detected\n")
        f.write("\n\n")
        
        # CHECK #5: Class Mapping
        f.write("CHECK #5: CLASS LABEL MAPPING VERIFICATION\n")
        f.write("-" * 80 + "\n")
        if 'class_mapping' in all_results:
            result = all_results['class_mapping']
            f.write(f"\n  Defined class IDs: {result['defined_class_ids']}\n")
            f.write(f"  Found in dataset: {result['found_class_ids']}\n")
            f.write(f"  Missing classes: {result['missing_in_data']}\n")
            f.write(f"  Extra/undefined classes: {result['extra_in_data']}\n")
            f.write(f"  data.yaml exists: {result['data_yaml_exists']}\n")
            
            if result['mapping_consistent']:
                f.write(f"\n  STATUS: ✅ EXCELLENT - Mapping is consistent\n")
            else:
                f.write(f"\n  STATUS: ❌ CRITICAL - Undefined class IDs found in dataset\n")
        f.write("\n\n")
        
        # CHECK #6: Augmentation
        f.write("CHECK #6: AUGMENTED IMAGES VERIFICATION\n")
        f.write("-" * 80 + "\n")
        if 'augmented_images' in all_results:
            result = all_results['augmented_images']
            f.write(f"  Original images: {result['original_images']}\n")
            f.write(f"  Total augmented: {result['total_augmented']}\n")
            f.write(f"  Total dataset size: {result['original_images'] + result['total_augmented']}\n")
            f.write(f"\n  Augmentation breakdown:\n")
            for prefix, count in result['augmentation_breakdown'].items():
                f.write(f"    {prefix:<12} {count:>6} images\n")
            f.write(f"\n  Augmented without labels: {result['augmented_without_labels']}\n")
            
            if result['augmented_without_labels'] > 0:
                f.write(f"  Examples:\n")
                for ex in result['augmented_without_labels_examples'][:5]:
                    f.write(f"    {ex}\n")
            
            aug_ratio = result['total_augmented'] / result['original_images'] if result['original_images'] > 0 else 0
            f.write(f"\n  Augmentation ratio: {aug_ratio:.2f}x\n")
            
            if result['total_augmented'] > 0 and result['augmented_without_labels'] == 0:
                f.write(f"\n  STATUS: ✅ EXCELLENT - Augmentation properly applied\n")
            elif result['augmented_without_labels'] > 0:
                f.write(f"\n  STATUS: ❌ CRITICAL - Missing labels for augmented images\n")
            else:
                f.write(f"\n  STATUS: ⚠ WARNING - No augmented images found\n")
        f.write("\n\n")
        
        # CHECK #7: Train/Val Split
        f.write("CHECK #7: TRAIN/VALIDATION SPLIT INTEGRITY\n")
        f.write("-" * 80 + "\n")
        if 'train_val_split' in all_results:
            result = all_results['train_val_split']
            f.write(f"  Training images: {result['train_images']} ({result['train_percentage']:.1f}%)\n")
            f.write(f"  Validation images: {result['val_images']} ({result['val_percentage']:.1f}%)\n")
            f.write(f"  Total images: {result['total_images']}\n")
            f.write(f"  Overlap (data leakage): {result['overlap_count']}\n")
            f.write(f"  Augmented in validation: {result['val_augmented_count']}\n")
            
            if result['overlap_count'] > 0:
                f.write(f"\n  ❌ DATA LEAKAGE DETECTED!\n")
                f.write(f"  Overlap examples:\n")
                for ex in result['overlap_examples'][:5]:
                    f.write(f"    {ex}\n")
            
            if result['val_augmented_count'] > 0:
                f.write(f"\n  ⚠ WARNING: Augmented images in validation set\n")
            
            if result['overlap_count'] == 0 and result['val_augmented_count'] == 0:
                if 75 <= result['train_percentage'] <= 90:
                    f.write(f"\n  STATUS: ✅ EXCELLENT - Clean split with optimal ratio\n")
                else:
                    f.write(f"\n  STATUS: ✅ GOOD - Clean split, ratio acceptable\n")
            else:
                f.write(f"\n  STATUS: ❌ CRITICAL - Data integrity issues\n")
        f.write("\n\n")
        
        # CHECK #8: Annotation Completeness
        f.write("CHECK #8: ANNOTATION COMPLETENESS ANALYSIS\n")
        f.write("-" * 80 + "\n")
        if 'annotation_completeness' in all_results:
            for split in ['train', 'val']:
                result = all_results['annotation_completeness'][split]
                f.write(f"\n{split.upper()} Split:\n")
                f.write(f"  Samples generated for manual review: {result['samples_generated']}\n")
                f.write(f"  Location: {result['samples_location']}\n")
            f.write(f"\n  NOTE: Manual visual inspection required\n")
            f.write(f"  Please review samples and estimate annotation completeness\n")
        f.write("\n\n")
        
        # CHECK #9: Image Diversity
        f.write("CHECK #9: IMAGE QUALITY AND DIVERSITY ANALYSIS\n")
        f.write("-" * 80 + "\n")
        if 'image_diversity' in all_results:
            for split in ['train', 'val']:
                result = all_results['image_diversity'][split]
                f.write(f"\n{split.upper()} Split:\n")
                f.write(f"  Samples analyzed: {result['samples_analyzed']}\n")
                f.write(f"  Unique resolutions: {result['unique_resolutions']}\n")
                f.write(f"  Most common: {result['most_common_resolution']} "
                       f"({result['resolution_uniformity']*100:.1f}%)\n")
                f.write(f"  Average brightness: {result['avg_brightness']:.1f}/255\n")
                f.write(f"  Brightness std dev: {result['brightness_std']:.1f}\n")
                
                f.write(f"\n  Lighting distribution:\n")
                lighting = result['lighting_distribution']
                total = sum(lighting.values())
                f.write(f"    Very dark (<50):  {lighting['very_dark']:>4} "
                       f"({lighting['very_dark']/total*100:>5.1f}%)\n")
                f.write(f"    Dark (50-100):    {lighting['dark']:>4} "
                       f"({lighting['dark']/total*100:>5.1f}%)\n")
                f.write(f"    Normal (100-180): {lighting['normal']:>4} "
                       f"({lighting['normal']/total*100:>5.1f}%)\n")
                f.write(f"    Bright (>180):    {lighting['bright']:>4} "
                       f"({lighting['bright']/total*100:>5.1f}%)\n")
                
                if result['unique_resolutions'] <= 3 and result['brightness_std'] > 25:
                    f.write(f"\n  STATUS: ✅ EXCELLENT - Uniform quality, diverse lighting\n")
                elif result['brightness_std'] > 25:
                    f.write(f"\n  STATUS: ✅ GOOD - Diverse lighting conditions\n")
                else:
                    f.write(f"\n  STATUS: ⚠ LIMITED - Low diversity\n")
        f.write("\n\n")
        
        # CHECK #10: Spatial Patterns
        f.write("CHECK #10: SPATIAL PATTERN ANALYSIS (SAFETY-CRITICAL CLASSES)\n")
        f.write("-" * 80 + "\n")
        if 'spatial_patterns' in all_results:
            result = all_results['spatial_patterns']['train']
            f.write(f"\nTRAINING Set Spatial Distribution:\n\n")
            f.write(f"  {'Class':<15} {'Count':<8} {'Avg X':<8} {'Avg Y':<8} "
                   f"{'Avg W':<8} {'Avg H':<8} {'Std X':<8} {'Std Y':<8}\n")
            f.write(f"  {'-'*95}\n")
            
            for stat in result['class_statistics']:
                f.write(f"  {stat['class_name']:<15} "
                       f"{stat['count']:<8} "
                       f"{stat['avg_x']:<8.3f} "
                       f"{stat['avg_y']:<8.3f} "
                       f"{stat['avg_width']:<8.3f} "
                       f"{stat['avg_height']:<8.3f} "
                       f"{stat['std_x']:<8.3f} "
                       f"{stat['std_y']:<8.3f}\n")
            
            f.write(f"\n  Spatial Bias Assessment:\n")
            for stat in result['class_statistics']:
                x_bias = "CENTERED" if 0.3 < stat['avg_x'] < 0.7 else "EDGE-BIASED"
                y_bias = "CENTERED" if 0.3 < stat['avg_y'] < 0.7 else "POSITION-BIASED"
                size_var = "DIVERSE" if stat['std_width'] > 0.03 else "UNIFORM"
                
                f.write(f"    {stat['class_name']}:\n")
                f.write(f"      Horizontal: {x_bias} (x={stat['avg_x']:.2f}±{stat['std_x']:.2f})\n")
                f.write(f"      Vertical: {y_bias} (y={stat['avg_y']:.2f}±{stat['std_y']:.2f})\n")
                f.write(f"      Size: {size_var} (w={stat['avg_width']:.3f}±{stat['std_width']:.3f})\n")
        f.write("\n\n")
        
        # SUMMARY AND RECOMMENDATIONS (CONTINUED)
        f.write("="*80 + "\n")
        f.write("SUMMARY AND RECOMMENDATIONS\n")
        f.write("="*80 + "\n\n")
        
        critical_issues = []
        warnings = []
        good_points = []
        
        # Analyze results and categorize
        if 'structure_integrity' in all_results:
            train_pairing = all_results['structure_integrity']['train']['pairing_rate']
            if train_pairing >= 98:
                good_points.append("✅ Excellent image-label pairing")
            elif train_pairing >= 90:
                warnings.append("⚠ Some orphaned images/labels exist")
            else:
                critical_issues.append("❌ CRITICAL: Significant image-label pairing issues")
        
        if 'label_format' in all_results:
            train_error = all_results['label_format']['train']['error_rate']
            if train_error < 1:
                good_points.append("✅ Label format is correct")
            elif train_error < 5:
                warnings.append("⚠ Some label format errors detected")
            else:
                critical_issues.append("❌ CRITICAL: High label format error rate")
        
        if 'class_distribution' in all_results:
            safety_pct = all_results['class_distribution']['train']['safety_critical_percentage']
            if safety_pct >= 15:
                good_points.append("✅ Safety-critical classes well represented")
            elif safety_pct >= 10:
                warnings.append("⚠ Safety-critical classes moderately represented")
            else:
                critical_issues.append("❌ CRITICAL: Safety-critical classes severely under-represented")
        
        if 'train_val_split' in all_results:
            overlap = all_results['train_val_split']['overlap_count']
            if overlap == 0:
                good_points.append("✅ No train/val data leakage")
            else:
                critical_issues.append("❌ CRITICAL: Train/validation data leakage detected")
        
        if 'augmented_images' in all_results:
            aug_total = all_results['augmented_images']['total_augmented']
            aug_missing = all_results['augmented_images']['augmented_without_labels']
            if aug_total > 0 and aug_missing == 0:
                good_points.append("✅ Augmentation properly implemented")
            elif aug_missing > 0:
                critical_issues.append("❌ CRITICAL: Augmented images missing labels")
            elif aug_total == 0:
                warnings.append("⚠ No data augmentation detected")
        
        if 'class_mapping' in all_results:
            if all_results['class_mapping']['mapping_consistent']:
                good_points.append("✅ Class mapping is consistent")
            else:
                critical_issues.append("❌ CRITICAL: Undefined class IDs in dataset")
        
        # Write summary sections
        if critical_issues:
            f.write("🚨 CRITICAL ISSUES (MUST FIX IMMEDIATELY):\n")
            for issue in critical_issues:
                f.write(f"  {issue}\n")
            f.write("\n")
        
        if warnings:
            f.write("⚠️  WARNINGS (SHOULD ADDRESS):\n")
            for warning in warnings:
                f.write(f"  {warning}\n")
            f.write("\n")
        
        if good_points:
            f.write("✅ POSITIVE FINDINGS:\n")
            for point in good_points:
                f.write(f"  {point}\n")
            f.write("\n")
        
        # Overall assessment
        f.write("-" * 80 + "\n")
        f.write("OVERALL DATASET QUALITY ASSESSMENT:\n")
        f.write("-" * 80 + "\n\n")
        
        if len(critical_issues) == 0 and len(warnings) <= 2:
            f.write("  RATING: ⭐⭐⭐⭐⭐ EXCELLENT\n")
            f.write("  Your dataset is high quality and ready for training.\n")
        elif len(critical_issues) == 0:
            f.write("  RATING: ⭐⭐⭐⭐ GOOD\n")
            f.write("  Your dataset is usable with minor improvements recommended.\n")
        elif len(critical_issues) <= 2:
            f.write("  RATING: ⭐⭐⭐ ACCEPTABLE\n")
            f.write("  Your dataset has issues that should be addressed before training.\n")
        else:
            f.write("  RATING: ⭐⭐ NEEDS IMPROVEMENT\n")
            f.write("  Your dataset has critical issues that must be fixed.\n")
        
        f.write("\n")
        
        # Recommendations
        f.write("-" * 80 + "\n")
        f.write("RECOMMENDED ACTIONS:\n")
        f.write("-" * 80 + "\n\n")
        
        action_number = 1
        
        # Critical actions
        if critical_issues:
            f.write("IMMEDIATE ACTIONS (Critical):\n\n")
            
            if any("pairing" in issue.lower() for issue in critical_issues):
                f.write(f"  {action_number}. FIX IMAGE-LABEL PAIRING:\n")
                f.write(f"     - Re-run conversion script to ensure all images have labels\n")
                f.write(f"     - Remove orphaned files\n")
                f.write(f"     - Verify pairing reaches 98%+\n\n")
                action_number += 1
            
            if any("format" in issue.lower() for issue in critical_issues):
                f.write(f"  {action_number}. FIX LABEL FORMAT ERRORS:\n")
                f.write(f"     - Review format error examples in this report\n")
                f.write(f"     - Fix conversion script bugs\n")
                f.write(f"     - Re-convert affected labels\n\n")
                action_number += 1
            
            if any("leakage" in issue.lower() for issue in critical_issues):
                f.write(f"  {action_number}. ELIMINATE DATA LEAKAGE:\n")
                f.write(f"     - Remove overlapping images from validation set\n")
                f.write(f"     - Ensure completely separate train/val splits\n")
                f.write(f"     - Re-validate your model after fixing\n\n")
                action_number += 1
            
            if any("under-represented" in issue.lower() for issue in critical_issues):
                f.write(f"  {action_number}. ADDRESS CLASS IMBALANCE:\n")
                f.write(f"     - Consider processing IDD-117K for more safety-critical examples\n")
                f.write(f"     - Apply class-specific augmentation to underrepresented classes\n")
                f.write(f"     - Increase asymmetric loss weighting (3-4x for critical classes)\n\n")
                action_number += 1
            
            if any("mapping" in issue.lower() for issue in critical_issues):
                f.write(f"  {action_number}. FIX CLASS MAPPING:\n")
                f.write(f"     - Verify data.yaml matches your class definitions\n")
                f.write(f"     - Ensure conversion script uses correct class IDs\n")
                f.write(f"     - Remove or remap undefined classes\n\n")
                action_number += 1
        
        # Warning-level actions
        if warnings:
            f.write("RECOMMENDED IMPROVEMENTS (Important):\n\n")
            
            if any("augmentation" in warning.lower() for warning in warnings):
                f.write(f"  {action_number}. IMPLEMENT DATA AUGMENTATION:\n")
                f.write(f"     - Apply weather augmentation (rain, fog, night)\n")
                f.write(f"     - Use geometric augmentation (rotation, scaling)\n")
                f.write(f"     - Target 0.3-0.5x augmentation ratio\n\n")
                action_number += 1
            
            if any("moderate" in warning.lower() for warning in warnings):
                f.write(f"  {action_number}. BOOST SAFETY-CRITICAL CLASS REPRESENTATION:\n")
                f.write(f"     - Apply targeted augmentation to pedestrian/rider images\n")
                f.write(f"     - Consider selective IDD-117K addition\n")
                f.write(f"     - Increase asymmetric loss weighting to 2.5-3x\n\n")
                action_number += 1
        
        # General improvements
        f.write("OPTIONAL ENHANCEMENTS:\n\n")
        
        f.write(f"  {action_number}. IMPLEMENT RESEARCH FEATURES:\n")
        f.write(f"     - Asymmetric loss function (if class imbalance exists)\n")
        f.write(f"     - SCIR metric for safety-aware evaluation\n")
        f.write(f"     - CCAT metric for deployment efficiency analysis\n")
        f.write(f"     - Adaptive inference system for demo\n\n")
        action_number += 1
        
        f.write(f"  {action_number}. IMPROVE MODEL PERFORMANCE:\n")
        f.write(f"     - Test-time augmentation for +1-2% mAP\n")
        f.write(f"     - Model ensemble (2-3 checkpoints)\n")
        f.write(f"     - Class-specific post-processing thresholds\n\n")
        action_number += 1
        
        # Decision on IDD-117K
        f.write("-" * 80 + "\n")
        f.write("IDD-117K DATASET DECISION:\n")
        f.write("-" * 80 + "\n\n")
        
        if 'class_distribution' in all_results:
            safety_pct = all_results['class_distribution']['train']['safety_critical_percentage']
            
            if safety_pct < 10:
                f.write("  RECOMMENDATION: ✅ YES, PROCESS IDD-117K\n\n")
                f.write("  Reasoning:\n")
                f.write(f"    - Safety-critical classes only {safety_pct:.1f}% of dataset\n")
                f.write(f"    - Severe under-representation hurts model performance\n")
                f.write(f"    - IDD-117K can provide additional pedestrian/rider examples\n\n")
                f.write("  Approach:\n")
                f.write("    - Process selectively (only images with pedestrians/riders)\n")
                f.write("    - Expected time: 4-6 hours\n")
                f.write("    - Expected gain: 3-5% mAP improvement\n")
            elif safety_pct < 15:
                f.write("  RECOMMENDATION: ⚠ MAYBE, SELECTIVE PROCESSING\n\n")
                f.write("  Reasoning:\n")
                f.write(f"    - Safety-critical classes at {safety_pct:.1f}% (moderate)\n")
                f.write(f"    - Could benefit from more examples but not critical\n")
                f.write(f"    - Consider if current model plateaus below 42% mAP\n\n")
                f.write("  Alternative:\n")
                f.write("    - First try targeted augmentation on existing data\n")
                f.write("    - Only process IDD-117K if model performance inadequate\n")
            else:
                f.write("  RECOMMENDATION: ❌ NO, SKIP IDD-117K\n\n")
                f.write("  Reasoning:\n")
                f.write(f"    - Safety-critical classes at {safety_pct:.1f}% (good)\n")
                f.write(f"    - Current dataset is well-balanced\n")
                f.write(f"    - Time better spent on research features\n\n")
                f.write("  Better Investment:\n")
                f.write("    - Implement asymmetric loss, SCIR, CCAT metrics\n")
                f.write("    - Build adaptive inference system\n")
                f.write("    - Focus on presentation and demo quality\n")
        
        f.write("\n")
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"✅ Report saved to: {report_path}")
    return report_path

# =====================================================
# JSON RESULTS EXPORT
# =====================================================

def export_results_json(all_results):
    """Export results to JSON for programmatic access"""
    json_path = os.path.join(OUTPUT_DIR, "investigation_results.json")
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(all_results)
    
    with open(json_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"✅ JSON results saved to: {json_path}")
    return json_path

# =====================================================
# MAIN EXECUTION
# =====================================================

def main():
    """Main execution function - runs all checks"""
    print("\n" + "="*80)
    print("STARTING COMPREHENSIVE DATASET INVESTIGATION")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    all_results = {}
    
    try:
        # CHECK #1: Structure Integrity
        all_results['structure_integrity'] = check_structure_integrity()
        
        # CHECK #2: Label Format
        all_results['label_format'] = check_label_format()
        
        # CHECK #3: Class Distribution
        all_results['class_distribution'] = check_class_distribution()
        
        # CHECK #4: Bounding Box Quality
        all_results['bbox_quality'] = check_bbox_quality()
        
        # CHECK #5: Class Mapping
        all_results['class_mapping'] = check_class_mapping()
        
        # CHECK #6: Augmented Images
        all_results['augmented_images'] = check_augmented_images()
        
        # CHECK #7: Train/Val Split
        all_results['train_val_split'] = check_train_val_split()
        
        # CHECK #8: Annotation Completeness
        all_results['annotation_completeness'] = check_annotation_completeness()
        
        # CHECK #9: Image Diversity
        all_results['image_diversity'] = check_image_diversity()
        
        # CHECK #10: Spatial Patterns
        all_results['spatial_patterns'] = check_spatial_patterns()
        
        # Generate visualizations
        generate_visualizations(all_results)
        
        # Generate comprehensive report
        report_path = generate_comprehensive_report(all_results)
        
        # Export to JSON
        json_path = export_results_json(all_results)
        
        print("\n" + "="*80)
        print("✅ INVESTIGATION COMPLETE!")
        print("="*80)
        print(f"\n📊 Results saved to: {OUTPUT_DIR}")
        print(f"📄 Text report: {report_path}")
        print(f"📊 JSON data: {json_path}")
        print(f"📈 Visualizations: {os.path.join(OUTPUT_DIR, 'visualizations')}")
        print(f"🖼️  Sample images: {os.path.join(OUTPUT_DIR, 'bbox_visualizations')}")
        print(f"\n🎯 NEXT STEPS:")
        print(f"   1. Review the text report: INVESTIGATION_REPORT.txt")
        print(f"   2. Check visualizations in the 'visualizations' folder")
        print(f"   3. Review sample bbox visualizations")
        print(f"   4. Read the SUMMARY AND RECOMMENDATIONS section")
        print(f"   5. Decide on IDD-117K based on recommendations")
        print("="*80)
        
        # Print quick summary to console
        print("\n" + "="*80)
        print("QUICK SUMMARY")
        print("="*80)
        
        if 'structure_integrity' in all_results:
            train = all_results['structure_integrity']['train']
            print(f"\n📦 Dataset Size:")
            print(f"   Training: {train['total_images']} images, {train['total_labels']} labels")
            print(f"   Pairing: {train['pairing_rate']:.1f}%")
        
        if 'class_distribution' in all_results:
            train = all_results['class_distribution']['train']
            print(f"\n📊 Class Distribution:")
            print(f"   Total instances: {train['total_instances']:,}")
            print(f"   Safety-critical: {train['safety_critical_percentage']:.1f}%")
            
            # Show top 5 classes
            top_classes = sorted(train['class_statistics'], 
                               key=lambda x: x['instances'], reverse=True)[:5]
            print(f"\n   Top 5 classes:")
            for stat in top_classes:
                critical = "⚠" if stat['is_safety_critical'] else " "
                print(f"   {critical} {stat['class_name']}: {stat['instances']:,} "
                      f"({stat['instance_percentage']:.1f}%)")
        
        if 'augmented_images' in all_results:
            aug = all_results['augmented_images']
            print(f"\n🎨 Augmentation:")
            print(f"   Original: {aug['original_images']}")
            print(f"   Augmented: {aug['total_augmented']}")
            total = aug['original_images'] + aug['total_augmented']
            print(f"   Total: {total}")
        
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n❌ ERROR during investigation: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nPartial results may be available in OUTPUT_DIR")
    
    return all_results

# =====================================================
# SCRIPT ENTRY POINT
# =====================================================

if __name__ == "__main__":
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  COMPREHENSIVE IDD-YOLO DATASET INVESTIGATION SCRIPT".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    print("\n")
    
    # Verify base path exists
    if not os.path.exists(BASE_PATH):
        print(f"❌ ERROR: Dataset path does not exist: {BASE_PATH}")
        print(f"Please update BASE_PATH variable in the script")
        exit(1)
    
    # Run investigation
    results = main()
    
    print("\n✅ Script execution completed successfully!\n")