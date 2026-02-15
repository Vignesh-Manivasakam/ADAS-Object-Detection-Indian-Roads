import os
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
import json
from pathlib import Path
import time

class DatasetAnalyzer:
    """Complete IDD Dataset Analyzer"""
    
    def __init__(self, annotations_dir):
        self.annotations_dir = annotations_dir
        self.class_info = defaultdict(lambda: {
            'count': 0,
            'all_files': [],
            'bbox_sizes': [],
            'bbox_areas': [],
            'aspect_ratios': [],
            'positions': []  # Store center positions
        })
        self.file_stats = {
            'total_files': 0,
            'processed_files': 0,
            'error_files': [],
            'empty_files': [],
            'multi_class_files': []
        }
        self.image_dimensions = []
        self.objects_per_image = []
    
    def _describe_position(self, x, y):
        """Describe position in human-readable format"""
        h_pos = "left" if x < 0.33 else "center" if x < 0.67 else "right"
        v_pos = "top" if y < 0.33 else "middle" if y < 0.67 else "bottom"
        return f"{v_pos}-{h_pos} (x={x:.2f}, y={y:.2f})"
    
    def analyze_single_file(self, xml_path):
        """Analyze a single XML annotation file"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Get image dimensions
            size = root.find('size')
            if size is not None:
                img_width = int(size.find('width').text)
                img_height = int(size.find('height').text)
                self.image_dimensions.append((img_width, img_height))
            else:
                img_width = img_height = 1920  # Default
            
            # Get filename
            filename_elem = root.find('filename')
            filename = filename_elem.text if filename_elem is not None else os.path.basename(xml_path)
            
            objects = root.findall('object')
            num_objects = len(objects)
            self.objects_per_image.append(num_objects)
            
            if num_objects == 0:
                self.file_stats['empty_files'].append(xml_path)
                return True
            
            classes_in_file = set()
            
            for obj in objects:
                name = obj.find('name')
                if name is None or not name.text:
                    continue
                
                class_name = name.text.strip()
                classes_in_file.add(class_name)
                
                # Update count
                self.class_info[class_name]['count'] += 1
                
                # Store file path
                rel_path = os.path.relpath(xml_path, self.annotations_dir)
                self.class_info[class_name]['all_files'].append(rel_path)
                
                # Get bounding box
                bbox = obj.find('bndbox')
                if bbox is not None:
                    try:
                        xmin = int(bbox.find('xmin').text)
                        xmax = int(bbox.find('xmax').text)
                        ymin = int(bbox.find('ymin').text)
                        ymax = int(bbox.find('ymax').text)
                        
                        # Calculate metrics
                        width = xmax - xmin
                        height = ymax - ymin
                        area = width * height
                        
                        # Normalized values
                        norm_width = width / img_width
                        norm_height = height / img_height
                        norm_area = area / (img_width * img_height)
                        
                        # Center position (normalized)
                        center_x = ((xmin + xmax) / 2) / img_width
                        center_y = ((ymin + ymax) / 2) / img_height
                        
                        # Aspect ratio
                        aspect_ratio = width / height if height > 0 else 0
                        
                        # Store all metrics
                        self.class_info[class_name]['bbox_sizes'].append((norm_width, norm_height))
                        self.class_info[class_name]['bbox_areas'].append(norm_area)
                        self.class_info[class_name]['aspect_ratios'].append(aspect_ratio)
                        self.class_info[class_name]['positions'].append((center_x, center_y))
                        
                    except Exception as e:
                        pass
            
            # Track multi-class images
            if len(classes_in_file) > 3:
                self.file_stats['multi_class_files'].append((xml_path, len(classes_in_file)))
            
            return True
            
        except Exception as e:
            self.file_stats['error_files'].append((xml_path, str(e)))
            return False
    
    def scan_all_files(self):
        """Scan all XML files in directory tree"""
        print(f"Starting comprehensive scan of: {self.annotations_dir}")
        print("=" * 100)
        
        start_time = time.time()
        
        # Collect all XML files first
        all_xml_files = []
        for root_dir, dirs, files in os.walk(self.annotations_dir):
            for file in files:
                if file.endswith('.xml'):
                    all_xml_files.append(os.path.join(root_dir, file))
        
        self.file_stats['total_files'] = len(all_xml_files)
        
        print(f"Found {len(all_xml_files)} XML files to process")
        print("Processing...\n")
        
        # Process each file
        for idx, xml_path in enumerate(all_xml_files, 1):
            if self.analyze_single_file(xml_path):
                self.file_stats['processed_files'] += 1
            
            # Progress indicator
            if idx % 500 == 0:
                elapsed = time.time() - start_time
                rate = idx / elapsed
                remaining = (len(all_xml_files) - idx) / rate
                print(f"Progress: {idx}/{len(all_xml_files)} ({idx/len(all_xml_files)*100:.1f}%) "
                      f"- Rate: {rate:.1f} files/sec - ETA: {remaining/60:.1f} min")
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ Scan complete in {elapsed_time/60:.2f} minutes")
        print(f"   Processed: {self.file_stats['processed_files']}/{self.file_stats['total_files']} files")
    
    def calculate_statistics(self):
        """Calculate statistical metrics for each class"""
        stats = {}
        
        for class_name, info in self.class_info.items():
            if info['count'] == 0:
                continue
            
            # Bbox size statistics
            if info['bbox_areas']:
                areas = info['bbox_areas']
                stats[class_name] = {
                    'count': info['count'],
                    'avg_area': sum(areas) / len(areas) * 100,  # As percentage
                    'min_area': min(areas) * 100,
                    'max_area': max(areas) * 100,
                    'small_objects': sum(1 for a in areas if a < 0.01),  # < 1% of image
                    'medium_objects': sum(1 for a in areas if 0.01 <= a < 0.1),
                    'large_objects': sum(1 for a in areas if a >= 0.1),
                }
                
                # Aspect ratio stats
                if info['aspect_ratios']:
                    ratios = info['aspect_ratios']
                    stats[class_name]['avg_aspect_ratio'] = sum(ratios) / len(ratios)
                
                # Position statistics (where objects typically appear)
                if info['positions']:
                    positions = info['positions']
                    avg_x = sum(p[0] for p in positions) / len(positions)
                    avg_y = sum(p[1] for p in positions) / len(positions)
                    stats[class_name]['typical_position'] = (avg_x, avg_y)
                
                # File distribution
                stats[class_name]['num_files'] = len(set(info['all_files']))
                stats[class_name]['avg_per_image'] = info['count'] / len(set(info['all_files']))
        
        return stats
    
    def generate_reports(self, output_dir='analysis_reports'):
        """Generate comprehensive analysis reports"""
        os.makedirs(output_dir, exist_ok=True)
        
        stats = self.calculate_statistics()
        
        # Report 1: Summary Statistics
        self._generate_summary_report(stats, output_dir)
        
        # Report 2: Detailed Class Analysis
        self._generate_detailed_class_report(stats, output_dir)
        
        # Report 3: Data Quality Report
        self._generate_quality_report(output_dir)
        
        # Report 4: Class Distribution Visualization Data
        self._generate_distribution_data(stats, output_dir)
        
        # Report 5: Sample Files per Class
        self._generate_sample_files_report(output_dir)
        
        # Report 6: Training Plan
        self._generate_training_plan(stats, output_dir)
        
        print(f"\n✅ All reports generated in: {output_dir}/")
    
    def _generate_summary_report(self, stats, output_dir):
        """Generate executive summary report"""
        filepath = os.path.join(output_dir, '01_SUMMARY_REPORT.txt')
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("IDD DATASET - EXECUTIVE SUMMARY\n")
            f.write("=" * 100 + "\n\n")
            
            # File statistics
            f.write("FILE STATISTICS:\n")
            f.write("-" * 100 + "\n")
            f.write(f"Total XML files found:        {self.file_stats['total_files']:,}\n")
            f.write(f"Successfully processed:       {self.file_stats['processed_files']:,}\n")
            f.write(f"Files with errors:            {len(self.file_stats['error_files']):,}\n")
            f.write(f"Empty annotation files:       {len(self.file_stats['empty_files']):,}\n")
            f.write(f"Complex scenes (>3 classes):  {len(self.file_stats['multi_class_files']):,}\n\n")
            
            # Image statistics
            if self.image_dimensions:
                f.write("IMAGE STATISTICS:\n")
                f.write("-" * 100 + "\n")
                widths = [w for w, h in self.image_dimensions]
                heights = [h for w, h in self.image_dimensions]
                f.write(f"Most common resolution:       {Counter(self.image_dimensions).most_common(1)[0][0]}\n")
                f.write(f"Width range:                  {min(widths)} - {max(widths)} pixels\n")
                f.write(f"Height range:                 {min(heights)} - {max(heights)} pixels\n\n")
            
            # Object statistics
            if self.objects_per_image:
                f.write("OBJECT STATISTICS:\n")
                f.write("-" * 100 + "\n")
                total_objects = sum(self.objects_per_image)
                avg_objects = total_objects / len(self.objects_per_image)
                f.write(f"Total objects annotated:      {total_objects:,}\n")
                f.write(f"Avg objects per image:        {avg_objects:.2f}\n")
                f.write(f"Max objects in single image:  {max(self.objects_per_image)}\n")
                f.write(f"Images with no objects:       {len(self.file_stats['empty_files']):,}\n\n")
            
            # Class statistics
            f.write("CLASS STATISTICS:\n")
            f.write("-" * 100 + "\n")
            f.write(f"Total unique classes:         {len(stats)}\n\n")
            
            # Top classes
            sorted_classes = sorted(stats.items(), key=lambda x: x[1]['count'], reverse=True)
            total_instances = sum(s['count'] for _, s in sorted_classes)
            
            f.write("CLASS DISTRIBUTION (Top to Bottom):\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Class Name':<25} {'Count':>12} {'%':>8} {'Avg Size':>12} {'Files':>10}\n")
            f.write("-" * 100 + "\n")
            
            for class_name, class_stats in sorted_classes:
                percentage = (class_stats['count'] / total_instances) * 100
                f.write(f"{class_name:<25} {class_stats['count']:>12,} {percentage:>7.2f}% "
                       f"{class_stats['avg_area']:>11.2f}% {class_stats['num_files']:>10,}\n")
            
            f.write("\n" + "=" * 100 + "\n")
            f.write("KEY INSIGHTS:\n")
            f.write("=" * 100 + "\n\n")
            
            # Class imbalance analysis
            top_class = sorted_classes[0]
            bottom_class = sorted_classes[-1]
            imbalance_ratio = top_class[1]['count'] / bottom_class[1]['count']
            
            f.write(f"1. CLASS IMBALANCE:\n")
            f.write(f"   Most common: '{top_class[0]}' with {top_class[1]['count']:,} instances\n")
            f.write(f"   Least common: '{bottom_class[0]}' with {bottom_class[1]['count']:,} instances\n")
            f.write(f"   Imbalance ratio: {imbalance_ratio:.1f}:1\n")
            f.write(f"   ⚠️  Consider using class weights or focal loss for training!\n\n")
            
            # Small object analysis
            small_obj_classes = [(name, s['small_objects']) for name, s in stats.items() 
                                if s['small_objects'] > s['count'] * 0.3]
            if small_obj_classes:
                f.write(f"2. SMALL OBJECT CHALLENGE:\n")
                f.write(f"   Classes with >30% small objects (<1% of image):\n")
                for cls, count in sorted(small_obj_classes, key=lambda x: x[1], reverse=True):
                    pct = (count / stats[cls]['count']) * 100
                    f.write(f"   - {cls}: {count:,} small objects ({pct:.1f}%)\n")
                f.write(f"   ⚠️  Consider multi-scale training and higher resolution inference!\n\n")
            
            # Position bias
            f.write(f"3. SPATIAL DISTRIBUTION:\n")
            for cls, s in sorted_classes[:5]:  # Top 5 classes
                if 'typical_position' in s:
                    x, y = s['typical_position']
                    location = self._describe_position(x, y)
                    f.write(f"   '{cls}' typically appears: {location}\n")
            f.write("\n")
        
        print(f"   ✓ Summary report: {filepath}")
    
    def _generate_detailed_class_report(self, stats, output_dir):
        """Generate detailed per-class analysis"""
        filepath = os.path.join(output_dir, '02_DETAILED_CLASS_ANALYSIS.txt')
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("DETAILED CLASS-BY-CLASS ANALYSIS\n")
            f.write("=" * 100 + "\n\n")
            
            sorted_classes = sorted(stats.items(), key=lambda x: x[1]['count'], reverse=True)
            
            for idx, (class_name, class_stats) in enumerate(sorted_classes, 1):
                f.write(f"\n{'='*100}\n")
                f.write(f"CLASS #{idx}: {class_name.upper()}\n")
                f.write(f"{'='*100}\n\n")
                
                # Basic stats
                f.write(f"Total instances:              {class_stats['count']:,}\n")
                f.write(f"Appears in files:             {class_stats['num_files']:,}\n")
                f.write(f"Avg instances per image:      {class_stats['avg_per_image']:.2f}\n\n")
                
                # Size statistics
                f.write(f"BOUNDING BOX SIZE ANALYSIS:\n")
                f.write(f"-" * 100 + "\n")
                f.write(f"Average size (% of image):    {class_stats['avg_area']:.3f}%\n")
                f.write(f"Smallest object:              {class_stats['min_area']:.3f}%\n")
                f.write(f"Largest object:               {class_stats['max_area']:.3f}%\n\n")
                
                # Object size distribution
                f.write(f"SIZE DISTRIBUTION:\n")
                total = class_stats['count']
                small_pct = (class_stats['small_objects'] / total) * 100
                medium_pct = (class_stats['medium_objects'] / total) * 100
                large_pct = (class_stats['large_objects'] / total) * 100
                
                f.write(f"  Small (<1% of image):       {class_stats['small_objects']:>8,} ({small_pct:>5.1f}%)\n")
                f.write(f"  Medium (1-10% of image):    {class_stats['medium_objects']:>8,} ({medium_pct:>5.1f}%)\n")
                f.write(f"  Large (>10% of image):      {class_stats['large_objects']:>8,} ({large_pct:>5.1f}%)\n\n")
                
                # Aspect ratio
                if 'avg_aspect_ratio' in class_stats:
                    f.write(f"Average aspect ratio:         {class_stats['avg_aspect_ratio']:.2f}\n")
                    if class_stats['avg_aspect_ratio'] > 2:
                        f.write(f"  → Typically wide objects\n")
                    elif class_stats['avg_aspect_ratio'] < 0.5:
                        f.write(f"  → Typically tall objects\n")
                    else:
                        f.write(f"  → Relatively square objects\n")
                    f.write("\n")
                
                # Spatial distribution
                if 'typical_position' in class_stats:
                    x, y = class_stats['typical_position']
                    f.write(f"Typical position in frame:    {self._describe_position(x, y)}\n\n")
                
                # Recommendations
                f.write(f"TRAINING RECOMMENDATIONS:\n")
                f.write(f"-" * 100 + "\n")
                
                if small_pct > 50:
                    f.write(f"⚠️  HIGH SMALL OBJECT RATIO ({small_pct:.1f}%)\n")
                    f.write(f"   → Use multi-scale training (scale=0.5)\n")
                    f.write(f"   → Increase input resolution (imgsz=1280)\n")
                    f.write(f"   → Consider ATSS or PAA anchor strategy\n\n")
                
                if class_stats['count'] < 1000:
                    f.write(f"⚠️  LOW SAMPLE COUNT ({class_stats['count']:,} instances)\n")
                    f.write(f"   → Apply strong augmentation for this class\n")
                    f.write(f"   → Use class weights in loss function\n")
                    f.write(f"   → Consider focal loss (gamma=2.0)\n\n")
                
                if class_stats['avg_per_image'] > 5:
                    f.write(f"ℹ️  HIGH DENSITY CLASS ({class_stats['avg_per_image']:.1f} per image)\n")
                    f.write(f"   → Important for NMS tuning (iou_threshold)\n")
                    f.write(f"   → May need higher confidence threshold\n\n")
        
        print(f"   ✓ Detailed class analysis: {filepath}")
    
    def _generate_quality_report(self, output_dir):
        """Generate data quality and issues report"""
        filepath = os.path.join(output_dir, '03_DATA_QUALITY_REPORT.txt')
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("DATA QUALITY ASSESSMENT\n")
            f.write("=" * 100 + "\n\n")
            
            # Error files
            f.write(f"FILES WITH PARSING ERRORS: {len(self.file_stats['error_files'])}\n")
            f.write("-" * 100 + "\n")
            if self.file_stats['error_files']:
                f.write("Sample errors (first 20):\n")
                for xml_path, error in self.file_stats['error_files'][:20]:
                    rel_path = os.path.relpath(xml_path, self.annotations_dir)
                    f.write(f"  {rel_path}\n")
                    f.write(f"    Error: {error}\n")
                f.write("\n")
            else:
                f.write("✅ No parsing errors found!\n\n")
            
            # Empty files
            f.write(f"EMPTY ANNOTATION FILES: {len(self.file_stats['empty_files'])}\n")
            f.write("-" * 100 + "\n")
            if self.file_stats['empty_files']:
                f.write("Files with no objects (first 50):\n")
                for xml_path in self.file_stats['empty_files'][:50]:
                    rel_path = os.path.relpath(xml_path, self.annotations_dir)
                    f.write(f"  {rel_path}\n")
                f.write(f"\n⚠️  Consider removing these from training or using as negative samples\n\n")
            else:
                f.write("✅ All files have annotations!\n\n")
            
            # Complex scenes
            f.write(f"COMPLEX SCENES (>3 CLASSES): {len(self.file_stats['multi_class_files'])}\n")
            f.write("-" * 100 + "\n")
            if self.file_stats['multi_class_files']:
                sorted_complex = sorted(self.file_stats['multi_class_files'], 
                                      key=lambda x: x[1], reverse=True)
                f.write("Most complex scenes (first 20):\n")
                for xml_path, num_classes in sorted_complex[:20]:
                    rel_path = os.path.relpath(xml_path, self.annotations_dir)
                    f.write(f"  {rel_path}: {num_classes} different classes\n")
                f.write(f"\nℹ️  These scenes are valuable for training robust models!\n\n")
            
            # Objects per image distribution
            if self.objects_per_image:
                f.write(f"OBJECTS PER IMAGE DISTRIBUTION:\n")
                f.write("-" * 100 + "\n")
                obj_counter = Counter(self.objects_per_image)
                f.write(f"Images with 0 objects:        {obj_counter[0]:,}\n")
                f.write(f"Images with 1-5 objects:      {sum(obj_counter[i] for i in range(1,6)):,}\n")
                f.write(f"Images with 6-10 objects:     {sum(obj_counter[i] for i in range(6,11)):,}\n")
                f.write(f"Images with 11-20 objects:    {sum(obj_counter[i] for i in range(11,21)):,}\n")
                f.write(f"Images with >20 objects:      {sum(obj_counter[i] for i in range(21,100)):,}\n")
                f.write("\n")
        
        print(f"   ✓ Quality report: {filepath}")
    
    def _generate_distribution_data(self, stats, output_dir):
        """Generate distribution data for visualization"""
        filepath = os.path.join(output_dir, '04_DISTRIBUTION_DATA.json')
        
        # Prepare data for easy plotting
        distribution_data = {
            'class_counts': {},
            'class_sizes': {},
            'class_positions': {},
            'size_distribution': {},
            'objects_per_image': self.objects_per_image[:10000]  # Sample for plotting
        }
        
        for class_name, class_stats in stats.items():
            distribution_data['class_counts'][class_name] = class_stats['count']
            distribution_data['class_sizes'][class_name] = {
                'avg': class_stats['avg_area'],
                'min': class_stats['min_area'],
                'max': class_stats['max_area']
            }
            if 'typical_position' in class_stats:
                distribution_data['class_positions'][class_name] = {
                    'x': class_stats['typical_position'][0],
                    'y': class_stats['typical_position'][1]
                }
            distribution_data['size_distribution'][class_name] = {
                'small': class_stats['small_objects'],
                'medium': class_stats['medium_objects'],
                'large': class_stats['large_objects']
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(distribution_data, f, indent=2)
        
        print(f"   ✓ Distribution data (JSON): {filepath}")
    
    def _generate_sample_files_report(self, output_dir):
        """Generate report with sample files for each class"""
        filepath = os.path.join(output_dir, '05_SAMPLE_FILES_PER_CLASS.txt')
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("SAMPLE FILES FOR EACH CLASS\n")
            f.write("=" * 100 + "\n\n")
            
            sorted_classes = sorted(self.class_info.items(), 
                                  key=lambda x: x[1]['count'], reverse=True)
            
            for class_name, info in sorted_classes:
                if info['count'] == 0:
                    continue
                
                f.write(f"\n{class_name.upper()}\n")
                f.write("-" * 100 + "\n")
                f.write(f"Total files containing this class: {len(set(info['all_files'])):,}\n")
                f.write(f"\nSample files (first 50):\n")
                
                # Get unique files
                unique_files = sorted(set(info['all_files']))[:50]
                for sample_file in unique_files:
                    f.write(f"  {sample_file}\n")
                
                f.write("\n")
        
        print(f"   ✓ Sample files report: {filepath}")
    
    def _generate_training_plan(self, stats, output_dir):
        """Generate comprehensive training plan"""
        plan_path = os.path.join(output_dir, '06_TRAINING_PLAN.txt')
        sorted_classes = sorted(stats.items(), key=lambda x: x[1]['count'], reverse=True)
        total_objects = sum(s['count'] for _, s in sorted_classes)
        
        with open(plan_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("COMPREHENSIVE TRAINING PLAN & RECOMMENDATIONS\n")
            f.write("=" * 100 + "\n\n")
            
            # Section 1: Class Imbalance Strategy
            f.write("1. CLASS IMBALANCE HANDLING\n")
            f.write("-" * 100 + "\n\n")
            
            # Calculate imbalance metrics
            max_count = sorted_classes[0][1]['count']
            min_count = sorted_classes[-1][1]['count']
            imbalance_ratio = max_count / min_count
            
            f.write(f"Imbalance Ratio: {imbalance_ratio:.1f}:1\n")
            f.write(f"Most common class: '{sorted_classes[0][0]}' ({max_count:,} instances)\n")
            f.write(f"Least common class: '{sorted_classes[-1][0]}' ({min_count:,} instances)\n\n")
            
            if imbalance_ratio > 10:
                f.write("⚠️  SEVERE CLASS IMBALANCE DETECTED!\n\n")
                f.write("RECOMMENDED STRATEGIES:\n\n")
                
                f.write("A. Use Asymmetric Loss Function:\n")
                f.write("   - Apply higher penalty for safety-critical classes (person, rider, bicycle)\n")
                f.write("   - Implement Focal Loss with gamma=2.0 for rare classes\n")
                f.write("   - Class weight formula: total_objects / (num_classes * class_count)\n\n")
                
                # Calculate suggested weights
                num_classes = len(sorted_classes)
                f.write("   Suggested Class Weights:\n")
                for idx, (class_name, class_stats) in enumerate(sorted_classes):
                    weight = total_objects / (num_classes * class_stats['count'])
                    f.write(f"   - {class_name}: {weight:.3f}\n")
                f.write("\n")
                
                f.write("B. Data Augmentation Strategy:\n")
                f.write("   - Apply stronger augmentation to rare classes\n")
                f.write("   - Use copy-paste augmentation for minority classes\n")
                f.write("   - Oversample images containing rare classes\n\n")
            
            # Section 2: Small Object Detection Strategy
            f.write("\n2. SMALL OBJECT DETECTION STRATEGY\n")
            f.write("-" * 100 + "\n\n")
            
            small_obj_classes = []
            for class_name, class_stats in stats.items():
                small_pct = (class_stats['small_objects'] / class_stats['count']) * 100
                if small_pct > 30:
                    small_obj_classes.append((class_name, small_pct, class_stats['small_objects']))
            
            if small_obj_classes:
                f.write(f"Classes with >30% small objects:\n")
                for cls, pct, count in sorted(small_obj_classes, key=lambda x: x[1], reverse=True):
                    f.write(f"  - {cls}: {pct:.1f}% ({count:,} small objects)\n")
                f.write("\n")
                
                f.write("RECOMMENDED STRATEGIES:\n\n")
                f.write("A. Training Configuration:\n")
                f.write("   - Use multi-scale training: scale=0.5 (images from 320px to 960px)\n")
                f.write("   - Higher input resolution: imgsz=1280 (instead of 640)\n")
                f.write("   - Mosaic augmentation: mosaic=1.0 (combines 4 images)\n")
                f.write("   - Copy-paste augmentation: copy_paste=0.3\n\n")
                
                f.write("B. Architecture Considerations:\n")
                f.write("   - Use YOLOv8m or YOLOv8l instead of YOLOv8n/s for better small object detection\n")
                f.write("   - Enable P2 layer if available (detects smaller features)\n")
                f.write("   - Consider SAHI (Slicing Aided Hyper Inference) for inference\n\n")
            
            # Section 3: Model Selection
            f.write("\n3. MODEL SELECTION RECOMMENDATION\n")
            f.write("-" * 100 + "\n\n")
            
            avg_objects = sum(self.objects_per_image) / len(self.objects_per_image) if self.objects_per_image else 0
            max_objects = max(self.objects_per_image) if self.objects_per_image else 0
            
            f.write(f"Dataset Complexity Metrics:\n")
            f.write(f"  - Average objects per image: {avg_objects:.2f}\n")
            f.write(f"  - Maximum objects per image: {max_objects}\n")
            f.write(f"  - Unique classes: {len(stats)}\n")
            f.write(f"  - Small object ratio: {sum(s['small_objects'] for s in stats.values()) / total_objects * 100:.1f}%\n\n")
            
            f.write("RECOMMENDED MODEL:\n")
            if avg_objects > 15 or len(small_obj_classes) > 5:
                f.write("  → YOLOv8m or YOLOv8l\n")
                f.write("    Reason: High scene complexity and many small objects require larger capacity\n")
            elif avg_objects > 8:
                f.write("  → YOLOv8s (Current choice is GOOD)\n")
                f.write("    Reason: Balanced performance for moderate complexity\n")
            else:
                f.write("  → YOLOv8n or YOLOv8s\n")
                f.write("    Reason: Simpler scenes allow smaller models\n")
            f.write("\n")
            
            # Section 4: Hyperparameter Recommendations
            f.write("\n4. HYPERPARAMETER RECOMMENDATIONS\n")
            f.write("-" * 100 + "\n\n")
            
            f.write("TRAINING HYPERPARAMETERS:\n\n")
            
            f.write("Basic Settings:\n")
            f.write("  epochs: 100-150 (monitor early stopping)\n")
            f.write("  batch: 16 (adjust based on GPU memory)\n")
            f.write("  imgsz: 1280 (for small objects) or 640 (standard)\n")
            f.write("  patience: 50 (early stopping patience)\n\n")
            
            f.write("Augmentation:\n")
            f.write("  hsv_h: 0.015 (hue augmentation)\n")
            f.write("  hsv_s: 0.7 (saturation)\n")
            f.write("  hsv_v: 0.4 (brightness)\n")
            f.write("  degrees: 0.0 (no rotation for road scenes)\n")
            f.write("  translate: 0.1 (slight translation)\n")
            f.write("  scale: 0.5 (multi-scale: 50% variation)\n")
            f.write("  fliplr: 0.5 (horizontal flip)\n")
            f.write("  flipud: 0.0 (no vertical flip for road scenes)\n")
            f.write("  mosaic: 1.0 (mosaic augmentation)\n")
            f.write("  mixup: 0.1 (mix two images)\n")
            f.write("  copy_paste: 0.3 (copy-paste augmentation)\n\n")
            
            f.write("Optimizer:\n")
            f.write("  optimizer: 'AdamW' or 'SGD'\n")
            f.write("  lr0: 0.01 (initial learning rate)\n")
            f.write("  lrf: 0.01 (final learning rate multiplier)\n")
            f.write("  momentum: 0.937 (for SGD)\n")
            f.write("  weight_decay: 0.0005\n\n")
            
            f.write("Loss Function:\n")
            f.write("  box: 7.5 (bbox loss weight)\n")
            f.write("  cls: 0.5 (classification loss weight)\n")
            f.write("  dfl: 1.5 (distribution focal loss weight)\n\n")
            
            # Section 5: Training Schedule
            f.write("\n5. TRAINING SCHEDULE (HACKATHON TIMELINE)\n")
            f.write("-" * 100 + "\n\n")
            
            f.write("PHASE 1: Baseline Training (24-48 hours)\n")
            f.write("  Objective: Establish baseline performance\n")
            f.write("  - Train YOLOv8s with default settings\n")
            f.write("  - 100 epochs, imgsz=640\n")
            f.write("  - Monitor mAP convergence\n")
            f.write("  - Expected mAP: 38-42%\n\n")
            
            f.write("PHASE 2: Asymmetric Loss Implementation (12-24 hours)\n")
            f.write("  Objective: Improve safety-critical class detection\n")
            f.write("  - Implement weighted loss for person, rider, bicycle\n")
            f.write("  - Apply focal loss for rare classes\n")
            f.write("  - Expected improvement: +2-3% mAP\n\n")
            
            f.write("PHASE 3: Small Object Optimization (24-48 hours)\n")
            f.write("  Objective: Boost small object detection\n")
            f.write("  - Increase resolution to imgsz=1280\n")
            f.write("  - Enable multi-scale training\n")
            f.write("  - Add copy-paste augmentation\n")
            f.write("  - Expected improvement: +1-2% mAP\n\n")
            
            f.write("PHASE 4: Post-Training Optimization (4-6 hours)\n")
            f.write("  Objective: Deployment optimization\n")
            f.write("  - INT8 quantization\n")
            f.write("  - Adaptive inference implementation\n")
            f.write("  - SCIR/CCAT metric calculation\n\n")
            
            # Section 6: Class-Specific Strategies
            f.write("\n6. CLASS-SPECIFIC TRAINING STRATEGIES\n")
            f.write("-" * 100 + "\n\n")
            
            for class_name, class_stats in sorted_classes:
                f.write(f"{class_name.upper()}:\n")
                
                count = class_stats['count']
                small_pct = (class_stats['small_objects'] / count) * 100
                avg_size = class_stats['avg_area']
                
                # Low count classes
                if count < 1000:
                    f.write(f"  ⚠️  Low sample count ({count:,})\n")
                    f.write(f"  → Apply class weight: {total_objects / (len(stats) * count):.2f}\n")
                    f.write(f"  → Use strong augmentation\n")
                
                # Small objects
                if small_pct > 50:
                    f.write(f"  ⚠️  Predominantly small objects ({small_pct:.1f}%)\n")
                    f.write(f"  → Train with higher resolution\n")
                    f.write(f"  → Use mosaic augmentation\n")
                
                # Large objects
                if avg_size > 15:
                    f.write(f"  ℹ️  Typically large objects ({avg_size:.1f}% of image)\n")
                    f.write(f"  → Should be easy to detect\n")
                    f.write(f"  → Can use as anchor for harder classes\n")
                
                # Spatial bias
                if 'typical_position' in class_stats:
                    x, y = class_stats['typical_position']
                    f.write(f"  ℹ️  Spatial pattern: {self._describe_position(x, y)}\n")
                
                f.write("\n")
            
            # Section 7: Validation Strategy
            f.write("\n7. VALIDATION & EVALUATION STRATEGY\n")
            f.write("-" * 100 + "\n\n")
            
            f.write("Metrics to Track:\n")
            f.write("  - Overall mAP@0.5\n")
            f.write("  - Overall mAP@0.5:0.95\n")
            f.write("  - Per-class mAP (focus on safety-critical classes)\n")
            f.write("  - Per-class Recall (especially for person, rider, bicycle)\n")
            f.write("  - SCIR (Safety Critical Inference Reliability)\n")
            f.write("  - CCAT (Compute Cost per Accurate Detection)\n\n")
            
            f.write("Safety-Critical Class Focus:\n")
            safety_classes = ['person', 'rider', 'bicycle', 'motorcycle']
            for cls in safety_classes:
                if cls in [c[0] for c in sorted_classes]:
                    f.write(f"  - {cls}: Target Recall >90%, mAP >45%\n")
            f.write("\n")
            
            f.write("Validation Best Practices:\n")
            f.write("  - Use stratified split (maintain class distribution)\n")
            f.write("  - Separate val set: 15-20% of total data\n")
            f.write("  - Include diverse scenarios: day/night, weather, density\n")
            f.write("  - Test on edge cases: heavy occlusion, small objects\n\n")
            
            # Section 8: Expected Timeline
            f.write("\n8. EXPECTED TIMELINE & MILESTONES\n")
            f.write("-" * 100 + "\n\n")
            
            total_images = self.file_stats['processed_files']
            epochs_per_day = 30  # Rough estimate
            
            f.write(f"Dataset Size: {total_images:,} images\n")
            f.write(f"Estimated training speed: ~{epochs_per_day} epochs/day (GPU dependent)\n\n")
            
            f.write("DAY 1-2: Baseline Training\n")
            f.write("  - YOLOv8s, 100 epochs\n")
            f.write("  - Milestone: Achieve 38-42% mAP\n\n")
            
            f.write("DAY 3: Implement Asymmetric Loss\n")
            f.write("  - Code custom loss function\n")
            f.write("  - Retrain 50 epochs\n")
            f.write("  - Milestone: Achieve 40-45% mAP\n\n")
            
            f.write("DAY 4: Small Object Optimization\n")
            f.write("  - Increase resolution\n")
            f.write("  - Enhanced augmentation\n")
            f.write("  - Milestone: Achieve 42-47% mAP\n\n")
            
            f.write("DAY 5: Post-Training Optimization\n")
            f.write("  - Quantization\n")
            f.write("  - Adaptive inference\n")
            f.write("  - Custom metrics\n")
            f.write("  - Milestone: Deployment-ready model\n\n")
            
            # Section 9: Risk Mitigation
            f.write("\n9. RISK MITIGATION STRATEGIES\n")
            f.write("-" * 100 + "\n\n")
            
            f.write("Risk: Model plateaus early\n")
            f.write("  → Implement learning rate warmup\n")
            f.write("  → Try cosine annealing scheduler\n")
            f.write("  → Increase model capacity (YOLOv8m)\n\n")
            
            f.write("Risk: Overfitting on majority classes\n")
            f.write("  → Apply stronger regularization (dropout, weight decay)\n")
            f.write("  → Use class weights\n")
            f.write("  → Monitor per-class validation metrics\n\n")
            
            f.write("Risk: Poor small object detection\n")
            f.write("  → Use SAHI (Slicing Aided Hyper Inference)\n")
            f.write("  → Increase input resolution to 1280 or 1536\n")
            f.write("  → Apply test-time augmentation\n\n")
            
            f.write("Risk: Time constraints\n")
            f.write("  → Use pretrained weights (don't train from scratch)\n")
            f.write("  → Focus on Phase 1-2 only\n")
            f.write("  → Parallelize experiments if multiple GPUs available\n\n")
            
            # Section 10: Success Criteria
            f.write("\n10. SUCCESS CRITERIA\n")
            f.write("-" * 100 + "\n\n")
            
            f.write("Minimum Acceptable Performance:\n")
            f.write("  - Overall mAP@0.5: >40%\n")
            f.write("  - Safety-critical class recall: >85%\n")
            f.write("  - Model size: <50MB (deployable)\n")
            f.write("  - Inference speed: <50ms on Jetson Nano\n\n")
            
            f.write("Target Performance:\n")
            f.write("  - Overall mAP@0.5: >45%\n")
            f.write("  - Safety-critical class recall: >90%\n")
            f.write("  - SCIR score: >0.85\n")
            f.write("  - Model size: <25MB (quantized)\n")
            f.write("  - Inference speed: <30ms on Jetson Nano\n\n")
            
            f.write("Exceptional Performance:\n")
            f.write("  - Overall mAP@0.5: >50%\n")
            f.write("  - Safety-critical class recall: >95%\n")
            f.write("  - SCIR score: >0.90\n")
            f.write("  - Real-time performance (30 FPS on edge device)\n\n")
            
            # Section 11: Recommended Training Command
            f.write("\n11. RECOMMENDED TRAINING COMMANDS\n")
            f.write("-" * 100 + "\n\n")
            
            f.write("PHASE 1 - Baseline Training:\n")
            f.write("-" * 50 + "\n")
            f.write("from ultralytics import YOLO\n\n")
            f.write("model = YOLO('yolov8s.pt')\n\n")
            f.write("results = model.train(\n")
            f.write("    data='data.yaml',\n")
            f.write("    epochs=100,\n")
            f.write("    imgsz=640,\n")
            f.write("    batch=16,\n")
            f.write("    patience=50,\n")
            f.write("    save=True,\n")
            f.write("    device=0,\n")
            f.write("    workers=8,\n")
            f.write("    project='runs/detect',\n")
            f.write("    name='baseline_yolov8s'\n")
            f.write(")\n\n")
            
            f.write("PHASE 2 - With Class Weights:\n")
            f.write("-" * 50 + "\n")
            f.write("# Calculate class weights based on analysis\n")
            f.write("class_weights = {\n")
            num_classes = len(sorted_classes)
            for idx, (class_name, class_stats) in enumerate(sorted_classes):
                weight = total_objects / (num_classes * class_stats['count'])
                f.write(f"    {idx}: {weight:.3f},  # {class_name}\n")
            f.write("}\n\n")
            f.write("# Apply in custom loss (requires modification)\n\n")
            
            f.write("PHASE 3 - High Resolution Training:\n")
            f.write("-" * 50 + "\n")
            f.write("model = YOLO('runs/detect/baseline_yolov8s/weights/best.pt')\n\n")
            f.write("results = model.train(\n")
            f.write("    data='data.yaml',\n")
            f.write("    epochs=100,\n")
            f.write("    imgsz=1280,  # Higher resolution\n")
            f.write("    batch=8,     # Reduce batch size\n")
            f.write("    scale=0.5,   # Multi-scale\n")
            f.write("    mosaic=1.0,\n")
            f.write("    copy_paste=0.3,\n")
            f.write("    mixup=0.1,\n")
            f.write("    patience=50,\n")
            f.write("    device=0,\n")
            f.write("    name='high_res_yolov8s'\n")
            f.write(")\n\n")
            
            f.write("PHASE 4 - Export & Quantization:\n")
            f.write("-" * 50 + "\n")
            f.write("model = YOLO('runs/detect/high_res_yolov8s/weights/best.pt')\n\n")
            f.write("# Export to ONNX with INT8 quantization\n")
            f.write("model.export(\n")
            f.write("    format='onnx',\n")
            f.write("    int8=True,\n")
            f.write("    data='data.yaml',\n")
            f.write("    imgsz=640,\n")
            f.write("    simplify=True\n")
            f.write(")\n\n")
            
            # Section 12: Key Insights Summary
            f.write("\n12. KEY DATASET INSIGHTS SUMMARY\n")
            f.write("-" * 100 + "\n\n")
            
            # Top 3 most common classes
            f.write(f"Most Common Classes (Top 3):\n")
            for i, (class_name, class_stats) in enumerate(sorted_classes[:3], 1):
                pct = (class_stats['count'] / total_objects) * 100
                f.write(f"  {i}. {class_name}: {class_stats['count']:,} instances ({pct:.1f}%)\n")
            f.write("\n")
            
            # Top 3 least common classes
            f.write(f"Least Common Classes (Bottom 3):\n")
            for i, (class_name, class_stats) in enumerate(sorted_classes[-3:][::-1], 1):
                pct = (class_stats['count'] / total_objects) * 100
                f.write(f"  {i}. {class_name}: {class_stats['count']:,} instances ({pct:.1f}%)\n")
            f.write("\n")
            
            # Small object challenge
            total_small = sum(s['small_objects'] for s in stats.values())
            small_pct = (total_small / total_objects) * 100
            f.write(f"Small Object Challenge:\n")
            f.write(f"  - {total_small:,} objects are small (<1% of image)\n")
            f.write(f"  - This is {small_pct:.1f}% of all objects\n")
            f.write(f"  - Priority: Use high-resolution training\n\n")
            
            # Class imbalance severity
            f.write(f"Class Imbalance Severity:\n")
            f.write(f"  - Imbalance ratio: {imbalance_ratio:.1f}:1\n")
            if imbalance_ratio > 100:
                f.write(f"  - Severity: EXTREME (require strong intervention)\n")
            elif imbalance_ratio > 50:
                f.write(f"  - Severity: SEVERE (require class weights + focal loss)\n")
            elif imbalance_ratio > 10:
                f.write(f"  - Severity: MODERATE (use class weights)\n")
            else:
                f.write(f"  - Severity: MILD (standard training acceptable)\n")
            f.write("\n")
            
            # Average scene complexity
            f.write(f"Scene Complexity:\n")
            f.write(f"  - Average objects per image: {avg_objects:.2f}\n")
            if avg_objects > 20:
                f.write(f"  - Complexity: VERY HIGH (use larger model)\n")
            elif avg_objects > 10:
                f.write(f"  - Complexity: HIGH (YOLOv8s/m recommended)\n")
            elif avg_objects > 5:
                f.write(f"  - Complexity: MODERATE (YOLOv8s optimal)\n")
            else:
                f.write(f"  - Complexity: LOW (YOLOv8n sufficient)\n")
            f.write("\n")
            
            # Final recommendations
            f.write("\n" + "=" * 100 + "\n")
            f.write("FINAL RECOMMENDATIONS FOR HACKATHON SUCCESS\n")
            f.write("=" * 100 + "\n\n")
            
            f.write("1. MUST DO (Critical):\n")
            f.write("   ✓ Start with YOLOv8s baseline training\n")
            f.write("   ✓ Implement class weights for imbalanced classes\n")
            f.write("   ✓ Use imgsz=1280 for small object detection\n")
            f.write("   ✓ Apply INT8 quantization for deployment\n")
            f.write("   ✓ Implement SCIR and CCAT custom metrics\n\n")
            
            f.write("2. SHOULD DO (High Impact):\n")
            f.write("   ✓ Implement asymmetric loss for safety-critical classes\n")
            f.write("   ✓ Use multi-scale training (scale=0.5)\n")
            f.write("   ✓ Enable mosaic and copy-paste augmentation\n")
            f.write("   ✓ Implement adaptive inference system\n\n")
            
            f.write("3. NICE TO HAVE (If Time Permits):\n")
            f.write("   ○ Test-time augmentation\n")
            f.write("   ○ Model ensemble\n")
            f.write("   ○ SAHI for inference\n")
            f.write("   ○ Try YOLOv8m for comparison\n\n")
            
            f.write("4. DON'T DO (Waste of Time):\n")
            f.write("   ✗ Architecture modifications\n")
            f.write("   ✗ Training from scratch (use pretrained)\n")
            f.write("   ✗ Extensive hyperparameter tuning\n")
            f.write("   ✗ Complex post-processing pipelines\n\n")
            
            f.write("=" * 100 + "\n")
            f.write("Good luck with your hackathon! 🚀\n")
            f.write("=" * 100 + "\n")
        
        print(f"   ✓ Training plan: {plan_path}")


# Main execution
if __name__ == "__main__":
    
    annotations_path = r"C:\Users\MQG1COB\Documents\Hackathon_ADAS\IDD117K_Detection\IDD_Detection\Annotations"
    
    print("=" * 100)
    print("IDD DATASET COMPREHENSIVE ANALYZER")
    print("=" * 100)
    print()
    
    # Create analyzer
    analyzer = DatasetAnalyzer(annotations_path)
    
    # Scan all files
    analyzer.scan_all_files()
    
    # Generate all reports
    analyzer.generate_reports(output_dir='analysis_reports')
    
    print("\n" + "=" * 100)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 100)
    print("\nGenerated Reports:")
    print("  1. 01_SUMMARY_REPORT.txt - Executive summary with key insights")
    print("  2. 02_DETAILED_CLASS_ANALYSIS.txt - Per-class deep dive")
    print("  3. 03_DATA_QUALITY_REPORT.txt - Issues and anomalies")
    print("  4. 04_DISTRIBUTION_DATA.json - Data for visualization")
    print("  5. 05_SAMPLE_FILES_PER_CLASS.txt - Sample file paths")
    print("  6. 06_TRAINING_PLAN.txt - Comprehensive training strategy")
    print("\n" + "=" * 100)
    print("Next Steps:")
    print("  1. Review 01_SUMMARY_REPORT.txt for overview")
    print("  2. Check 02_DETAILED_CLASS_ANALYSIS.txt for class-specific insights")
    print("  3. Follow 06_TRAINING_PLAN.txt for step-by-step guidance")
    print("  4. Start training with recommended hyperparameters!")
    print("=" * 100)