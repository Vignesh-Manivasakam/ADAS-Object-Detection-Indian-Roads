import os
import json
from pathlib import Path
from collections import defaultdict
import datetime
from tqdm import tqdm
class IDD95kAnalyzer:
    """
    Analyzer for IDD95k dataset with JSON annotations
    Structure: train/val -> labelsJSON/leftImg8bit -> subfolders -> files
    """
    
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.train_path = self.base_path / 'train'
        self.val_path = self.base_path / 'val'
        
        # Data storage
        self.class_counts = defaultdict(int)  # Total instances per class
        self.image_counts = defaultdict(int)  # Images containing each class
        self.bbox_sizes = defaultdict(list)   # Bbox areas per class
        self.class_samples = defaultdict(list)  # Sample files per class
        
        self.total_images = 0
        self.total_annotations = 0
        self.files_with_annotations = 0
        self.files_without_annotations = 0
        
        # Split-wise tracking
        self.split_stats = {
            'train': {'images': 0, 'annotations': 0},
            'val': {'images': 0, 'annotations': 0}
        }
        
    def calculate_bbox_area(self, bbox):
        """Calculate bounding box area"""
        width = bbox['xmax'] - bbox['xmin']
        height = bbox['ymax'] - bbox['ymin']
        return width * height
    
    def parse_json_file(self, json_path, split_type):
        """Parse a single JSON annotation file"""
        try:
            with open(json_path, 'r') as f:
                annotations = json.load(f)
            
            if not annotations:
                self.files_without_annotations += 1
                return
            
            self.files_with_annotations += 1
            self.total_annotations += len(annotations)
            self.split_stats[split_type]['annotations'] += len(annotations)
            
            # Track classes in this image
            classes_in_image = set()
            
            for obj in annotations:
                class_name = obj['name']
                bbox = obj['bbox']
                
                # Count instances
                self.class_counts[class_name] += 1
                classes_in_image.add(class_name)
                
                # Calculate bbox area
                area = self.calculate_bbox_area(bbox)
                self.bbox_sizes[class_name].append(area)
                
                # Store sample (limit to 5 per class)
                if len(self.class_samples[class_name]) < 5:
                    self.class_samples[class_name].append(str(json_path))
            
            # Count images per class
            for cls in classes_in_image:
                self.image_counts[cls] += 1
                
        except Exception as e:
            print(f"Error parsing {json_path}: {e}")
    
    def scan_directory(self, labels_path, split_type):
        """Recursively scan labelsJSON directory with progress bar"""
        json_files = list(labels_path.rglob('*.json'))
        
        print(f"Found {len(json_files)} JSON files in {split_type}")
        
        # Progress bar
        for json_file in tqdm(json_files, desc=f"Processing {split_type}", unit="files"):
            self.parse_json_file(json_file, split_type)
            self.total_images += 1
            self.split_stats[split_type]['images'] += 1
    
    def analyze(self):
        """Main analysis function"""
        print("="*80)
        print("IDD95K DATASET ANALYZER")
        print("="*80)
        
        # Analyze train split
        train_labels = self.train_path / 'labelsJSON'
        if train_labels.exists():
            print(f"\n[TRAIN] Scanning: {train_labels}")
            self.scan_directory(train_labels, 'train')
        else:
            print(f"\n[WARNING] Train path not found: {train_labels}")
        
        # Analyze val split
        val_labels = self.val_path / 'labelsJSON'
        if val_labels.exists():
            print(f"\n[VAL] Scanning: {val_labels}")
            self.scan_directory(val_labels, 'val')
        else:
            print(f"\n[WARNING] Val path not found: {val_labels}")
        
        print(f"\n✅ Analysis Complete!")
        print(f"Total Images: {self.total_images}")
        print(f"Total Annotations: {self.total_annotations}")
        print(f"Unique Classes: {len(self.class_counts)}")
    
    def generate_reports(self, output_dir='IDD95k_Analysis_Reports'):
        """Generate all analysis reports"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Summary Report
        self._generate_summary_report(output_path, timestamp)
        
        # 2. Detailed Class Analysis
        self._generate_class_analysis(output_path, timestamp)
        
        # 3. Data Quality Report
        self._generate_quality_report(output_path, timestamp)
        
        # 4. Distribution JSON
        self._generate_distribution_json(output_path, timestamp)
        
        # 5. Sample Files List
        self._generate_sample_files(output_path, timestamp)
        
        # 6. Training Plan
        self._generate_training_plan(output_path, timestamp)
        
        print(f"\n📊 All reports saved to: {output_path.absolute()}")
    
    def _generate_summary_report(self, output_path, timestamp):
        """Generate summary report"""
        report_file = output_path / f'1_Summary_Report_{timestamp}.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("IDD95K DATASET ANALYSIS - SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset Path: {self.base_path}\n\n")
            
            # Overall Statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Images: {self.total_images:,}\n")
            f.write(f"Total Annotations: {self.total_annotations:,}\n")
            f.write(f"Unique Classes: {len(self.class_counts)}\n")
            f.write(f"Average Annotations per Image: {self.total_annotations/self.total_images:.2f}\n")
            f.write(f"Images with Annotations: {self.files_with_annotations:,}\n")
            f.write(f"Images without Annotations: {self.files_without_annotations:,}\n\n")
            
            # Split Statistics
            f.write("SPLIT STATISTICS\n")
            f.write("-"*80 + "\n")
            for split in ['train', 'val']:
                stats = self.split_stats[split]
                f.write(f"\n{split.upper()}:\n")
                f.write(f"  Images: {stats['images']:,}\n")
                f.write(f"  Annotations: {stats['annotations']:,}\n")
                if stats['images'] > 0:
                    f.write(f"  Avg Annotations/Image: {stats['annotations']/stats['images']:.2f}\n")
            
            # Class Distribution
            f.write("\n\nCLASS DISTRIBUTION (Top to Bottom by Instance Count)\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Class':<25} {'Instances':<12} {'Images':<12} {'% of Total':<12}\n")
            f.write("-"*80 + "\n")
            
            sorted_classes = sorted(self.class_counts.items(), key=lambda x: x[1], reverse=True)
            for class_name, count in sorted_classes:
                percentage = (count / self.total_annotations) * 100
                img_count = self.image_counts[class_name]
                f.write(f"{class_name:<25} {count:<12,} {img_count:<12,} {percentage:>10.2f}%\n")
        
        print(f"✅ Summary Report: {report_file.name}")

    def _generate_class_analysis(self, output_path, timestamp):
        """Generate detailed class analysis"""
        report_file = output_path / f'2_Detailed_Class_Analysis_{timestamp}.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("IDD95K DATASET - DETAILED CLASS ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            sorted_classes = sorted(self.class_counts.items(), key=lambda x: x[1], reverse=True)
            
            for idx, (class_name, count) in enumerate(sorted_classes, 1):
                f.write(f"\n{idx}. CLASS: {class_name}\n")
                f.write("-"*80 + "\n")
                f.write(f"Total Instances: {count:,}\n")
                f.write(f"Images Containing Class: {self.image_counts[class_name]:,}\n")
                f.write(f"Percentage of Total Annotations: {(count/self.total_annotations)*100:.2f}%\n")
                f.write(f"Avg Instances per Image (when present): {count/self.image_counts[class_name]:.2f}\n")
                
                # Bounding box statistics
                if class_name in self.bbox_sizes:
                    areas = self.bbox_sizes[class_name]
                    f.write(f"\nBounding Box Statistics:\n")
                    f.write(f"  Min Area: {min(areas):,} px²\n")
                    f.write(f"  Max Area: {max(areas):,} px²\n")
                    f.write(f"  Avg Area: {sum(areas)/len(areas):,.0f} px²\n")
                    f.write(f"  Median Area: {sorted(areas)[len(areas)//2]:,} px²\n")
                
                f.write("\n")
        
        print(f"✅ Detailed Class Analysis: {report_file.name}")

    def _generate_quality_report(self, output_path, timestamp):
        """Generate data quality report"""
        report_file = output_path / f'3_Data_Quality_Report_{timestamp}.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("IDD95K DATASET - DATA QUALITY REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Class Balance Analysis
            f.write("CLASS BALANCE ANALYSIS\n")
            f.write("-"*80 + "\n")
            
            sorted_classes = sorted(self.class_counts.items(), key=lambda x: x[1], reverse=True)
            max_count = sorted_classes[0][1]
            
            for class_name, count in sorted_classes:
                ratio = count / max_count
                percentage = (count / self.total_annotations) * 100
                
                if percentage < 1.0:
                    status = "⚠️ CRITICAL IMBALANCE"
                elif percentage < 5.0:
                    status = "⚠️ SEVERE IMBALANCE"
                elif percentage < 10.0:
                    status = "⚠️ MODERATE IMBALANCE"
                else:
                    status = "✅ ADEQUATE"
                
                f.write(f"{class_name:<25} | Ratio: 1:{max_count/count:>6.1f} | {percentage:>6.2f}% | {status}\n")
            
            # Coverage Analysis
            f.write(f"\n\nCOVERAGE ANALYSIS\n")
            f.write("-"*80 + "\n")
            f.write(f"Images with Annotations: {self.files_with_annotations:,} ({self.files_with_annotations/self.total_images*100:.1f}%)\n")
            f.write(f"Images without Annotations: {self.files_without_annotations:,} ({self.files_without_annotations/self.total_images*100:.1f}%)\n")
            
            # Recommendations
            f.write(f"\n\nRECOMMENDATIONS\n")
            f.write("-"*80 + "\n")
            
            critical_classes = [cls for cls, cnt in sorted_classes if (cnt/self.total_annotations)*100 < 1.0]
            if critical_classes:
                f.write(f"⚠️ CRITICAL: {len(critical_classes)} classes have <1% representation:\n")
                for cls in critical_classes:
                    f.write(f"   - {cls}\n")
                f.write(f"\n   Action: Consider aggressive augmentation or weighted loss\n")
            
            severe_classes = [cls for cls, cnt in sorted_classes if 1.0 <= (cnt/self.total_annotations)*100 < 5.0]
            if severe_classes:
                f.write(f"\n⚠️ SEVERE: {len(severe_classes)} classes have 1-5% representation:\n")
                for cls in severe_classes:
                    f.write(f"   - {cls}\n")
                f.write(f"\n   Action: Use class weights and targeted augmentation\n")
        
        print(f"✅ Data Quality Report: {report_file.name}")

    def _generate_distribution_json(self, output_path, timestamp):
        """Generate distribution data as JSON"""
        json_file = output_path / f'4_Distribution_Data_{timestamp}.json'
        
        data = {
            'metadata': {
                'analysis_date': datetime.datetime.now().isoformat(),
                'dataset_path': str(self.base_path),
                'total_images': self.total_images,
                'total_annotations': self.total_annotations,
                'unique_classes': len(self.class_counts)
            },
            'split_statistics': self.split_stats,
            'class_distribution': {
                cls: {
                    'instance_count': self.class_counts[cls],
                    'image_count': self.image_counts[cls],
                    'percentage': (self.class_counts[cls] / self.total_annotations) * 100,
                    'avg_bbox_area': sum(self.bbox_sizes[cls]) / len(self.bbox_sizes[cls]) if cls in self.bbox_sizes else 0
                }
                for cls in self.class_counts.keys()
            }
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Distribution JSON: {json_file.name}")

    def _generate_sample_files(self, output_path, timestamp):
        """Generate sample files list"""
        sample_file = output_path / f'5_Sample_Files_{timestamp}.txt'
        
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("IDD95K DATASET - SAMPLE FILES PER CLASS\n")
            f.write("="*80 + "\n\n")
            
            sorted_classes = sorted(self.class_counts.keys())
            
            for class_name in sorted_classes:
                f.write(f"\nCLASS: {class_name}\n")
                f.write("-"*80 + "\n")
                
                samples = self.class_samples[class_name]
                for sample in samples:
                    f.write(f"  {sample}\n")
                
                if not samples:
                    f.write("  (No samples available)\n")
        
        print(f"✅ Sample Files List: {sample_file.name}")

    def _generate_training_plan(self, output_path, timestamp):
        """Generate training recommendations"""
        plan_file = output_path / f'6_Training_Plan_{timestamp}.txt'
        
        with open(plan_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("IDD95K DATASET - TRAINING PLAN & RECOMMENDATIONS\n")
            f.write("="*80 + "\n\n")
            
            f.write("1. CLASS WEIGHTS RECOMMENDATION\n")
            f.write("-"*80 + "\n")
            
            sorted_classes = sorted(self.class_counts.items(), key=lambda x: x[1], reverse=True)
            max_count = sorted_classes[0][1]
            
            f.write("Suggested class weights (inverse frequency):\n\n")
            for class_name, count in sorted_classes:
                weight = max_count / count
                f.write(f"  {class_name:<25} : {weight:.3f}\n")
            
            f.write("\n\n2. AUGMENTATION STRATEGY\n")
            f.write("-"*80 + "\n")
            
            critical = [cls for cls, cnt in sorted_classes if (cnt/self.total_annotations)*100 < 1.0]
            severe = [cls for cls, cnt in sorted_classes if 1.0 <= (cnt/self.total_annotations)*100 < 5.0]
            
            if critical:
                f.write(f"\nCRITICAL Classes (Aggressive Augmentation):\n")
                for cls in critical:
                    f.write(f"  - {cls}: Apply 5-10x augmentation\n")
            
            if severe:
                f.write(f"\nSEVERE Classes (Moderate Augmentation):\n")
                for cls in severe:
                    f.write(f"  - {cls}: Apply 2-5x augmentation\n")
            
            f.write("\n\n3. TRAINING CONFIGURATION\n")
            f.write("-"*80 + "\n")
            f.write(f"Recommended batch size: 16-32 (based on GPU memory)\n")
            f.write(f"Training images: {self.split_stats['train']['images']:,}\n")
            f.write(f"Validation images: {self.split_stats['val']['images']:,}\n")
            f.write(f"Epochs recommended: 100-150\n")
            f.write(f"Use focal loss or class-balanced loss\n")
            f.write(f"Apply mosaic augmentation for small objects\n")
            
            f.write("\n\n4. DATASET STATISTICS SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Total classes: {len(self.class_counts)}\n")
            f.write(f"Most common class: {sorted_classes[0][0]} ({sorted_classes[0][1]:,} instances)\n")
            f.write(f"Least common class: {sorted_classes[-1][0]} ({sorted_classes[-1][1]:,} instances)\n")
            f.write(f"Class imbalance ratio: 1:{max_count/sorted_classes[-1][1]:.1f}\n")
        
        print(f"✅ Training Plan: {plan_file.name}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Set your dataset path
    DATASET_PATH = r"C:\Users\\Documents\Hackathon_ADAS\IDD117K_Detection\IDD_95kDetection"
    
    print("\n" + "="*80)
    print("STARTING IDD95K DATASET ANALYSIS")
    print("="*80 + "\n")
    
    # Initialize analyzer
    analyzer = IDD95kAnalyzer(DATASET_PATH)
    
    # Run analysis
    analyzer.analyze()
    
    # Generate reports
    analyzer.generate_reports()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE! ✅")
    print("="*80 + "\n")