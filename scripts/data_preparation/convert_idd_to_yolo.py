import os
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import shutil

# ============================================
# CONFIGURATION
# ============================================
IDD_ROOT = r"C:\Users\\Documents\Hackathon_ADAS\IDD117K_Detection\IDD_Detection"
OUTPUT_ROOT = r"C:\Users\\Documents\Hackathon_ADAS\datasets\IDD_YOLO"

# Verified classes
CLASSES = [
    'animal',
    'autorickshaw',
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'person',
    'rider',
    'traffic sign',
    'trailer',
    'train',
    'truck',
    'vehicle fallback',
]

class_to_idx = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}

print(f"📋 Using {len(CLASSES)} classes from your actual data\n")

def convert_box(size, box):
    """Convert Pascal VOC bbox to YOLO format"""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    
    x_center = (box[0] + box[2]) / 2.0
    y_center = (box[1] + box[3]) / 2.0
    width = box[2] - box[0]
    height = box[3] - box[1]
    
    x_center *= dw
    width *= dw
    y_center *= dh
    height *= dh
    
    return (x_center, y_center, width, height)

def convert_annotation(xml_file, output_label_file):
    """Convert XML to YOLO format"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        
        labels = []
        
        for obj in root.iter('object'):
            cls_name = obj.find('name').text
            
            if cls_name not in class_to_idx:
                continue
            
            cls_id = class_to_idx[cls_name]
            
            xmlbox = obj.find('bndbox')
            bbox = (
                float(xmlbox.find('xmin').text),
                float(xmlbox.find('ymin').text),
                float(xmlbox.find('xmax').text),
                float(xmlbox.find('ymax').text)
            )
            
            yolo_bbox = convert_box((w, h), bbox)
            labels.append(f"{cls_id} {' '.join([f'{x:.6f}' for x in yolo_bbox])}\n")
        
        if labels:
            with open(output_label_file, 'w') as f:
                f.writelines(labels)
            return True
        return False
    except Exception as e:
        return False

def process_split(split_name, split_file):
    """Process train/val/test split"""
    print(f"\n{'='*60}")
    print(f"Processing {split_name.upper()} split...")
    print(f"{'='*60}")
    
    with open(split_file, 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]
    
    images_output = Path(OUTPUT_ROOT) / 'images' / split_name
    labels_output = Path(OUTPUT_ROOT) / 'labels' / split_name
    
    success_count = 0
    failed_count = 0
    
    for img_path in tqdm(image_paths, desc=f"Converting {split_name}"):
        # Add .jpg extension
        img_file = Path(IDD_ROOT) / 'JPEGImages' / f"{img_path}.jpg"
        xml_file = Path(IDD_ROOT) / 'Annotations' / f"{img_path}.xml"
        
        # Check if files exist
        if not img_file.exists():
            failed_count += 1
            continue
        
        if not xml_file.exists():
            failed_count += 1
            continue
        
        # ============================================
        # FIX: Create unique filename using full path
        # ============================================
        # Replace path separators with underscores to make unique names
        # frontFar/scene1/001542_r → frontFar_scene1_001542_r.jpg
        unique_name = img_path.replace('/', '_').replace('\\', '_')
        
        output_img_path = images_output / f"{unique_name}.jpg"
        output_label_path = labels_output / f"{unique_name}.txt"
        
        try:
            # Copy image
            shutil.copy2(img_file, output_img_path)
            
            # Convert annotation
            if convert_annotation(xml_file, output_label_path):
                success_count += 1
            else:
                failed_count += 1
        except Exception as e:
            failed_count += 1
    
    print(f"\n✅ {split_name.upper()}: {success_count} images converted")
    print(f"❌ {split_name.upper()}: {failed_count} failed")
    
    return success_count

if __name__ == "__main__":
    print("🚀 IDD to YOLO Conversion - FIXED VERSION")
    print(f"Input: {IDD_ROOT}")
    print(f"Output: {OUTPUT_ROOT}\n")
    
    # Clear existing data to start fresh
    print("🧹 Cleaning old data...")
    for split in ['train', 'val', 'test']:
        img_dir = Path(OUTPUT_ROOT) / 'images' / split
        lbl_dir = Path(OUTPUT_ROOT) / 'labels' / split
        
        if img_dir.exists():
            for f in img_dir.glob('*'):
                f.unlink()
        if lbl_dir.exists():
            for f in lbl_dir.glob('*'):
                f.unlink()
        
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
    
    print("✅ Ready for conversion\n")
    
    total_images = 0
    
    splits = {
        'train': Path(IDD_ROOT) / 'train.txt',
        'val': Path(IDD_ROOT) / 'val.txt',
        'test': Path(IDD_ROOT) / 'test.txt'
    }
    
    for split_name, split_file in splits.items():
        if split_file.exists():
            count = process_split(split_name, split_file)
            total_images += count
    
    # Create YAML configuration
    yaml_content = f"""# IDD Detection Dataset - Verified Classes
path: {OUTPUT_ROOT}
train: images/train
val: images/val
test: images/test

# Number of classes
nc: {len(CLASSES)}

# Class names (verified from actual XML files)
names: {CLASSES}
"""
    
    yaml_path = Path(OUTPUT_ROOT) / 'idd.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n{'='*60}")
    print("🎉 CONVERSION COMPLETE!")
    print(f"{'='*60}")
    print(f"✅ Total images converted: {total_images}")
    print(f"📋 Classes: {len(CLASSES)}")
    print(f"📄 Config file: {yaml_path}")
    
    # Verify counts
    print(f"\n📊 Verification:")
    for split in ['train', 'val', 'test']:
        img_count = len(list((Path(OUTPUT_ROOT) / 'images' / split).glob('*.jpg')))
        lbl_count = len(list((Path(OUTPUT_ROOT) / 'labels' / split).glob('*.txt')))
        print(f"   {split.upper()}: {img_count} images, {lbl_count} labels")
    
    print(f"\n🚀 Next: Run training with python train_idd_base.py")
    
    input("\nPress Enter to exit...")