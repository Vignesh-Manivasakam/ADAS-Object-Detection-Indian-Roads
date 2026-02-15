import albumentations as A
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import random
import sys # NEW: To allow us to exit the script

# ====================================================================
# 1. CONFIGURATION AND PATH VALIDATION
# ====================================================================

# Use forward slashes for better cross-platform compatibility and to avoid path errors
MASTER_DATASET_ROOT = Path("C:/Users//Documents/ADAS_Hackathon/datasets/MASTER_IDD_YOLO")
TOTAL_AUGMENTATIONS_TO_CREATE = 20000

# --- NEW: Pre-run Diagnostic Check ---
def validate_paths():
    print("--- Running Pre-flight Diagnostic Check ---")
    paths_to_check = {
        "MASTER_DATASET_ROOT": MASTER_DATASET_ROOT,
        "Images Directory": MASTER_DATASET_ROOT / "images",
        "Labels Directory": MASTER_DATASET_ROOT / "labels",
        "Training Images": MASTER_DATASET_ROOT / "images" / "train",
        "Training Labels": MASTER_DATASET_ROOT / "labels" / "train"
    }
    all_paths_ok = True
    for name, path in paths_to_check.items():
        if path.exists() and path.is_dir():
            print(f"✅ OK: Found directory '{name}' at: {path}")
        else:
            print(f"❌ FAILED: Could not find directory '{name}'. Looked for: {path}")
            all_paths_ok = False
            
    if not all_paths_ok:
        print("\nFATAL ERROR: One or more required directories were not found.")
        print("Please check the MASTER_DATASET_ROOT path and the folder structure inside it.")
        sys.exit() # Exit the script
    
    # Check if there are actual images in the folder
    image_files = list((MASTER_DATASET_ROOT / "images" / "train").glob("*.jpg"))
    if not image_files:
        print(f"❌ FAILED: The directory '{MASTER_DATASET_ROOT / 'images' / 'train'}' exists, but it contains 0 JPG images.")
        print("Please verify that your training images are in the correct folder.")
        sys.exit() # Exit the script
    
    print(f"✅ OK: Found {len(image_files)} images in the training directory.")
    print("--- Diagnostic Check Passed ---")

# Run the validation before anything else
validate_paths()

# ====================================================================
# 2. AUGMENTATION DEFINITIONS (No changes needed)
# ====================================================================
BBOX_PARAMS = A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.25)
AUGMENTATION_FUNCTIONS = {
    "motion_blur": A.Compose([A.MotionBlur(blur_limit=(5, 11), p=1.0)], bbox_params=BBOX_PARAMS),
    "gaussian_blur": A.Compose([A.GaussianBlur(blur_limit=(3, 7), p=1.0)], bbox_params=BBOX_PARAMS),
    "rain": A.Compose([A.RandomRain(p=1.0)], bbox_params=BBOX_PARAMS),
    "fog": A.Compose([A.RandomFog(p=1.0)], bbox_params=BBOX_PARAMS),
    "night": A.Compose([A.RandomBrightnessContrast(brightness_limit=(-0.5, -0.3), p=1.0)], bbox_params=BBOX_PARAMS),
    "dark": A.Compose([A.RandomBrightnessContrast(brightness_limit=(-0.8, -0.6), p=1.0)], bbox_params=BBOX_PARAMS),
    "glare": A.Compose([A.RandomSunFlare(p=1.0)], bbox_params=BBOX_PARAMS),
    "gamma": A.Compose([A.RandomGamma(gamma_limit=(50, 150), p=1.0)], bbox_params=BBOX_PARAMS),
    "mild_brightness": A.Compose([A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=1.0)], bbox_params=BBOX_PARAMS)
}

# ====================================================================
# 3. HELPER FUNCTIONS
# ====================================================================
def read_yolo_labels(label_path):
    if not label_path.exists(): return [], []
    bboxes, class_labels = [], []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                if not line.strip(): continue
                parts = line.split()
                if len(parts) >= 5:
                    class_labels.append(int(parts[0]))
                    bboxes.append(list(map(float, parts[1:5])))
    except Exception: pass
    return bboxes, class_labels

def write_yolo_labels(label_path, bboxes, class_labels):
    try:
        # Create parent directory for the label file if it doesn't exist
        label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_path, 'w') as f:
            for bbox, class_id in zip(bboxes, class_labels):
                f.write(f"{class_id} {' '.join(f'{x:.6f}' for x in bbox)}\n")
    except Exception as e:
        print(f"\n⚠️ Error writing {label_path}: {e}")

# ====================================================================
# 4. MAIN SCRIPT
# ====================================================================
def run_augmentation_pipeline():
    train_img_dir = MASTER_DATASET_ROOT / "images" / "train"
    train_label_dir = MASTER_DATASET_ROOT / "labels" / "train"
    
    aug_img_dir = MASTER_DATASET_ROOT / "images" / "train_augmented"
    aug_label_dir = MASTER_DATASET_ROOT / "labels" / "train_augmented"

    print("🧹 Setting up output directories...")
    # --- THE FIX ---
    # The 'parents=True' argument will create the entire directory tree
    # (e.g., 'Master_IDD_YOLO/images/train_augmented') if it doesn't exist.
    aug_img_dir.mkdir(parents=True, exist_ok=True)
    aug_label_dir.mkdir(parents=True, exist_ok=True)
    # ---------------
    print("✅ Ready to start augmentation.")

    all_image_paths = list(train_img_dir.glob("*.jpg"))
    if not all_image_paths:
        print(f"❌ FATAL ERROR: No images found in '{train_img_dir}'. Please check the MASTER_DATASET_ROOT path.")
        return

    print(f"✅ Found {len(all_image_paths)} source images to augment from.")
    
    created_count = 0
    
    with tqdm(total=TOTAL_AUGMENTATIONS_TO_CREATE, desc="Creating Augmented Images", unit="img") as pbar:
        while created_count < TOTAL_AUGMENTATIONS_TO_CREATE:
            try:
                source_image_path = random.choice(all_image_paths)
                label_path = train_label_dir / source_image_path.with_suffix('.txt').name
                if not label_path.exists():
                    continue
                
                aug_type, transform_function = random.choice(list(AUGMENTATION_FUNCTIONS.items()))
                
                image = cv2.imread(str(source_image_path))
                if image is None: continue
                
                bboxes, class_labels = read_yolo_labels(label_path)
                if not bboxes: continue

                augmented = transform_function(image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB), bboxes=bboxes, class_labels=class_labels)
                
                if not augmented['bboxes']: continue

                unique_filename = f"{aug_type}_{created_count}_{source_image_path.name}"
                output_image_path = aug_img_dir / unique_filename
                output_label_path = aug_label_dir / Path(unique_filename).with_suffix('.txt')
                
                cv2.imwrite(str(output_image_path), cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR))
                write_yolo_labels(output_label_path, augmented['bboxes'], augmented['class_labels'])
                
                created_count += 1
                pbar.update(1)
            except Exception as e:
                print(f"\n⚠️ Warning: Skipped one image due to an error: {e}")
                continue

    print("\n" + "="*60)
    print("✅ AUGMENTATION PROCESS COMPLETE!")
    print("="*60)
    print(f"Total new augmented images created: {created_count}")
    print(f"Augmented images saved in: '{aug_img_dir}'")
    print(f"Augmented labels saved in: '{aug_label_dir}'")
    print("\n🚀 Next Step: Remember to update your master_data.yaml file.")

if __name__ == "__main__":
    run_augmentation_pipeline()
    input("\nPress Enter to exit...")

