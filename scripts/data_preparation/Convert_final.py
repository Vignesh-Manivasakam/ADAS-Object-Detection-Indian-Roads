import os
import xml.etree.ElementTree as ET
import json ### NEW ###
from pathlib import Path
from tqdm import tqdm
import shutil
import cv2 ### NEW ###

# ============================================
# 1. CONFIGURATION
# ============================================

### MODIFIED ### - Paths for both datasets and a new master output
IDD_XML_ROOT = Path("C:/Users//Documents/Hackathon_ADAS/IDD117K_Detection/IDD_Detection")
IDD95K_JSON_ROOT = Path("C:/Users//Documents/Hackathon_ADAS/IDD117K_Detection/IDD_95kDetection")
MASTER_OUTPUT_ROOT = Path("C:/Users//Documents/Hackathon_ADAS/datasets/Master_IDD_YOLO")

### MODIFIED ### - Our final "Golden" 11-class list
FINAL_CLASSES = [
    'animal', 'autorickshaw', 'bicycle', 'bus', 'car', 'motorcycle', 
    'person', 'rider', 'traffic light', 'traffic sign', 'truck'
]

class_to_idx = {cls_name: idx for idx, cls_name in enumerate(FINAL_CLASSES)}

print(f"📋 Using {len(FINAL_CLASSES)} final classes for the master dataset.\n")

# --- Utility function (no changes needed) ---
def convert_box(size, box):
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

# ============================================
# 2. PARSER FUNCTIONS (one for XML, one for JSON)
# ============================================

def convert_xml_annotation(xml_file, output_label_file):
    """Converts a single XML annotation to YOLO format."""
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
            bbox = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
            yolo_bbox = convert_box((w, h), bbox)
            labels.append(f"{cls_id} {' '.join([f'{x:.6f}' for x in yolo_bbox])}\n")
        if labels:
            with open(output_label_file, 'w') as f:
                f.writelines(labels)
            return True
        return False
    except Exception:
        return False

### NEW ### - A dedicated function to parse the IDD95k JSON format
def convert_json_annotation(json_file, image_size, output_label_file):
    """Converts a single JSON annotation to YOLO format."""
    try:
        with open(json_file, 'r') as f:
            objects = json.load(f)
        
        w, h = image_size
        labels = []
        for obj in objects:
            cls_name = obj.get("name")
            if cls_name not in class_to_idx:
                continue
            cls_id = class_to_idx[cls_name]
            box = obj.get("bbox")
            bbox = (float(box['xmin']), float(box['ymin']), float(box['xmax']), float(box['ymax']))
            yolo_bbox = convert_box((w, h), bbox)
            labels.append(f"{cls_id} {' '.join([f'{x:.6f}' for x in yolo_bbox])}\n")
        
        if labels:
            with open(output_label_file, 'w') as f:
                f.writelines(labels)
            return True
        return False
    except Exception:
        return False

# ============================================
# 3. MAIN EXECUTION LOGIC
# ============================================

if __name__ == "__main__":
    print("🚀 Creating Master YOLO Dataset from IDD and IDD95k")
    print(f"Output will be saved to: {MASTER_OUTPUT_ROOT}\n")
    
    # --- Clean and set up output directories ---
    print("🧹 Cleaning old data...")
    if MASTER_OUTPUT_ROOT.exists():
        shutil.rmtree(MASTER_OUTPUT_ROOT)
    for split in ['train', 'val']:
        (MASTER_OUTPUT_ROOT / 'images' / split).mkdir(parents=True, exist_ok=True)
        (MASTER_OUTPUT_ROOT / 'labels' / split).mkdir(parents=True, exist_ok=True)
    print("✅ Output directories created.\n")

    # --- Process each split (train and val) ---
    for split_name in ['train', 'val']:
        print(f"\n{'='*60}")
        print(f"Processing MASTER {split_name.upper()} split...")
        print(f"{'='*60}")
        
        images_output_dir = MASTER_OUTPUT_ROOT / 'images' / split_name
        labels_output_dir = MASTER_OUTPUT_ROOT / 'labels' / split_name
        
        # --- Part A: Process the XML dataset ---
        print(f"--- Processing IDD (XML) source for {split_name} split ---")
        idd_split_file = IDD_XML_ROOT / f'{split_name}.txt'
        if idd_split_file.exists():
            with open(idd_split_file, 'r') as f:
                image_paths = [line.strip() for line in f.readlines()]
            
            for img_path_str in tqdm(image_paths, desc=f"Converting IDD {split_name}"):
                img_file = IDD_XML_ROOT / 'JPEGImages' / f"{img_path_str}.jpg"
                xml_file = IDD_XML_ROOT / 'Annotations' / f"{img_path_str}.xml"

                if not img_file.exists() or not xml_file.exists():
                    continue
                
                unique_name = "IDD_" + img_path_str.replace('/', '_').replace('\\', '_')
                output_img_path = images_output_dir / f"{unique_name}.jpg"
                output_label_path = labels_output_dir / f"{unique_name}.txt"
                
                shutil.copy2(img_file, output_img_path)
                convert_xml_annotation(xml_file, output_label_path)

        # --- Part B: Process the JSON dataset ---
        print(f"--- Processing IDD95k (JSON) source for {split_name} split ---")
        idd95k_split_dir = IDD95K_JSON_ROOT / split_name
        json_label_files = list((idd95k_split_dir / 'labelsJSON').rglob('*.json'))

        for json_file in tqdm(json_label_files, desc=f"Converting IDD95k {split_name}"):
            # Construct the corresponding image path
            relative_path = json_file.relative_to(idd95k_split_dir / 'labelsJSON')
            img_file = idd95k_split_dir / 'leftImg8bit' / relative_path.with_suffix('.jpg')

            if not img_file.exists():
                continue

            # We need to read the image to get its dimensions for JSON conversion
            image = cv2.imread(str(img_file))
            h, w, _ = image.shape
            
            unique_name = f"IDD95k_{split_name}_{relative_path.parent}_{relative_path.stem}"
            output_img_path = images_output_dir / f"{unique_name}.jpg"
            output_label_path = labels_output_dir / f"{unique_name}.txt"

            shutil.copy2(img_file, output_img_path)
            convert_json_annotation(json_file, (w, h), output_label_path)

    # --- Create the final YAML configuration file ---
    yaml_content = f"""# Master ADAS Dataset (IDD + IDD95k)
path: {str(MASTER_OUTPUT_ROOT.resolve())}
train: images/train
val: images/val

# Class configuration
nc: {len(FINAL_CLASSES)}
names: {FINAL_CLASSES}
"""
    yaml_path = MASTER_OUTPUT_ROOT / 'master_data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\n{'='*60}")
    print("🎉 MASTER DATASET CREATION COMPLETE!")
    print(f"{'='*60}")
    
    # --- Final Verification ---
    print(f"\n📊 Verification:")
    train_imgs = len(list((MASTER_OUTPUT_ROOT / 'images' / 'train').glob('*.jpg')))
    train_lbls = len(list((MASTER_OUTPUT_ROOT / 'labels' / 'train').glob('*.txt')))
    val_imgs = len(list((MASTER_OUTPUT_ROOT / 'images' / 'val').glob('*.jpg')))
    val_lbls = len(list((MASTER_OUTPUT_ROOT / 'labels' / 'val').glob('*.txt')))
    
    print(f"   Master TRAIN split: {train_imgs} images, {train_lbls} labels")
    print(f"   Master VAL split:   {val_imgs} images, {val_lbls} labels")
    print(f"📋 Final Classes: {len(FINAL_CLASSES)}")
    print(f"📄 Config file created: {yaml_path}")
    print(f"\n🚀 You are now ready to train on the master dataset!")
    
    input("\nPress Enter to exit...")

