import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from pathlib import Path
import yaml

class WeatherAugmentedDataset:
    """
    Custom dataset with research-based weather augmentations:
    - Motion blur (rain, speed)
    - Rain simulation
    - Fog simulation
    - Low-light/night
    - Sun glare
    - Shadow effects
    """
    
    def __init__(self, img_dir, label_dir, mode='train'):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.mode = mode
        
        # Get all image paths
        self.image_paths = list(self.img_dir.glob('*.jpg'))
        
        # Define augmentation pipeline
        if mode == 'train':
            self.transform = self.get_train_transform()
        else:
            self.transform = self.get_val_transform()
    
    def get_train_transform(self):
        """
        Research-based augmentation for Indian road conditions
        """
        return A.Compose([
            # ============================================
            # 🌧️ RAIN SIMULATION
            # ============================================
            A.RandomRain(
                slant_lower=-10,
                slant_upper=10,
                drop_length=20,
                drop_width=1,
                drop_color=(200, 200, 200),
                blur_value=3,
                brightness_coefficient=0.9,
                rain_type='drizzle',
                p=0.25  # 25% of images
            ),
            
            # ============================================
            # 🌫️ FOG SIMULATION
            # ============================================
            A.RandomFog(
                fog_coef_lower=0.3,
                fog_coef_upper=0.7,
                alpha_coef=0.08,
                p=0.20  # 20% of images
            ),
            
            # ============================================
            # 💨 MOTION BLUR (Speed/Rain effect)
            # ============================================
            A.MotionBlur(
                blur_limit=(3, 7),  # Blur kernel size
                p=0.30  # 30% of images
            ),
            
            # OR use Gaussian blur for rain effect
            A.OneOf([
                A.MotionBlur(blur_limit=(3, 7)),
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MedianBlur(blur_limit=5),
            ], p=0.30),
            
            # ============================================
            # 🌙 NIGHT / LOW-LIGHT
            # ============================================
            A.RandomBrightnessContrast(
                brightness_limit=(-0.4, 0.1),  # Darker images
                contrast_limit=(-0.3, 0.2),
                p=0.35  # 35% of images
            ),
            
            # ============================================
            # ☀️ SUN GLARE / FLARE
            # ============================================
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),  # Top half of image
                angle_lower=0,
                angle_upper=1,
                num_flare_circles_lower=6,
                num_flare_circles_upper=10,
                src_radius=400,
                src_color=(255, 255, 255),
                p=0.15  # 15% of images
            ),
            
            # ============================================
            # 🌑 SHADOW (Trees, tunnels, buildings)
            # ============================================
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),  # Bottom half
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                p=0.25  # 25% of images
            ),
            
            # ============================================
            # 🎨 COLOR AUGMENTATION (Atmospheric effects)
            # ============================================
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
            
            # ============================================
            # 🔄 GEOMETRIC AUGMENTATION (Already in YOLO)
            # ============================================
            A.HorizontalFlip(p=0.5),
            
            # ============================================
            # 🎯 OCCLUSION SIMULATION (Dense traffic)
            # ============================================
            A.CoarseDropout(
                max_holes=8,
                max_height=50,
                max_width=50,
                min_holes=3,
                min_height=20,
                min_width=20,
                fill_value=0,
                p=0.20  # 20% of images
            ),
            
            # ============================================
            # 📷 CAMERA EFFECTS
            # ============================================
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1),
                A.GridDistortion(num_steps=5, distort_limit=0.1),
            ], p=0.15),
            
            # ============================================
            # 🌐 NOISE (Low-quality cameras)
            # ============================================
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50)),
                A.ISONoise(color_shift=(0.01, 0.05)),
            ], p=0.15),
            
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3  # Keep boxes with >30% visibility
        ))
    
    def get_val_transform(self):
        """No augmentation for validation"""
        return A.Compose([
            # Only basic normalization for validation
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))


# Export augmentation as YAML for YOLOv8
def create_augmented_yaml():
    """
    Create augmentation config that can be referenced
    """
    aug_config = {
        'rain': 0.25,
        'fog': 0.20,
        'motion_blur': 0.30,
        'night': 0.35,
        'glare': 0.15,
        'shadow': 0.25,
    }
    
    with open('augmentation_config.yaml', 'w') as f:
        yaml.dump(aug_config, f)
    
    print("✅ Augmentation config created: augmentation_config.yaml")

if __name__ == '__main__':
    create_augmented_yaml()
    print("Weather augmentation pipeline ready!")