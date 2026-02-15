import torch
from ultralytics import YOLO
import torch.nn.utils.prune as prune
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class YOLOInt8Quantizer:
    """
    Simple INT8 Quantization for YOLO .pt models
    Focus: Size reduction, accuracy preservation, speed improvement
    """
    
    def __init__(self, model_path, data_yaml):
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.results = {}
        
    def create_int8_model(self):
        """
        Create INT8 quantized .pt model
        """
        print("="*70)
        print("🔪 CREATING INT8 QUANTIZED MODEL (.pt format)")
        print("="*70)
        
        # Load original model
        print("\n📦 Loading FP32 model...")
        model = YOLO(self.model_path)
        original_ckpt = torch.load(self.model_path, map_location='cpu')
        
        # Get original stats
        original_size = Path(self.model_path).stat().st_size / (1024**2)
        print(f"   Original size: {original_size:.2f} MB")
        
        # Validate original model
        print("\n📊 Validating original FP32 model...")
        fp32_metrics = model.val(data=self.data_yaml, batch=16, verbose=False, imgsz=640)
        fp32_map = fp32_metrics.box.map
        
        self.results['fp32'] = {
            'path': self.model_path,
            'size_mb': original_size,
            'map': fp32_map,
            'map50': fp32_metrics.box.map50,
            'precision': fp32_metrics.box.mp,
            'recall': fp32_metrics.box.mr
        }
        
        print(f"   FP32 mAP@50-95: {fp32_map:.4f}")
        
        # Apply INT8 quantization to model weights
        print("\n✂️ Applying INT8 quantization...")
        pytorch_model = model.model.cpu()
        pytorch_model.eval()
        
        # Quantize using PyTorch's dynamic quantization
        print("   Quantizing Conv2D and Linear layers...")
        quantized_model = torch.quantization.quantize_dynamic(
            pytorch_model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )
        
        # Save quantized model in .pt format
        output_dir = Path(self.model_path).parent
        int8_path = output_dir / 'best_int8.pt'
        
        print("\n💾 Saving INT8 model...")
        
        # Save in YOLO-compatible format
        quantized_ckpt = original_ckpt.copy()
        quantized_ckpt['model'] = quantized_model
        
        torch.save(quantized_ckpt, int8_path)
        
        int8_size = int8_path.stat().st_size / (1024**2)
        size_reduction = (1 - int8_size/original_size) * 100
        
        print(f"   ✅ INT8 model saved: {int8_path.name}")
        print(f"   Size: {int8_size:.2f} MB ({size_reduction:.1f}% reduction)")
        
        # Validate INT8 model
        print("\n📊 Validating INT8 model...")
        model_int8 = YOLO(str(int8_path))
        int8_metrics = model_int8.val(data=self.data_yaml, batch=16, verbose=False, imgsz=640)
        int8_map = int8_metrics.box.map
        
        accuracy_drop = (fp32_map - int8_map) * 100
        
        self.results['int8'] = {
            'path': str(int8_path),
            'size_mb': int8_size,
            'map': int8_map,
            'map50': int8_metrics.box.map50,
            'precision': int8_metrics.box.mp,
            'recall': int8_metrics.box.mr,
            'accuracy_drop': accuracy_drop
        }
        
        print(f"   INT8 mAP@50-95: {int8_map:.4f}")
        print(f"   Accuracy drop: {accuracy_drop:.2f}%")
        
        return str(int8_path)
    
    def benchmark_speed(self):
        """
        Benchmark inference speed for both models
        """
        print("\n" + "="*70)
        print("⚡ BENCHMARKING INFERENCE SPEED")
        print("="*70)
        
        dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        iterations = 100
        
        for model_type in ['fp32', 'int8']:
            print(f"\n🔍 Testing {model_type.upper()}...")
            
            model = YOLO(self.results[model_type]['path'])
            
            # Warmup
            for _ in range(10):
                model.predict(dummy_input, verbose=False, imgsz=640)
            
            # Benchmark
            times = []
            for i in range(iterations):
                if (i+1) % 20 == 0:
                    print(f"   Progress: {i+1}/{iterations}", end='\r')
                start = time.perf_counter()
                model.predict(dummy_input, verbose=False, imgsz=640)
                times.append(time.perf_counter() - start)
            
            avg_ms = np.mean(times) * 1000
            fps = 1000 / avg_ms
            
            self.results[model_type]['avg_ms'] = avg_ms
            self.results[model_type]['fps'] = fps
            
            print(f"\n   ✓ Avg: {avg_ms:.2f}ms | FPS: {fps:.1f}")
        
        # Calculate speedup
        speedup = self.results['fp32']['avg_ms'] / self.results['int8']['avg_ms']
        self.results['int8']['speedup'] = speedup
        
        print(f"\n🚀 Speedup: {speedup:.2f}x")
    
    def create_presentation_charts(self):
        """
        Create 3 essential charts for presentation
        """
        print("\n" + "="*70)
        print("📊 CREATING PRESENTATION CHARTS")
        print("="*70)
        
        plt.style.use('seaborn-v0_8-whitegrid')
        
        fig = plt.figure(figsize=(18, 6))
        
        # ============================================
        # Chart 1: Size Comparison
        # ============================================
        ax1 = plt.subplot(1, 3, 1)
        
        models = ['FP32', 'INT8']
        sizes = [self.results['fp32']['size_mb'], self.results['int8']['size_mb']]
        colors = ['#3498db', '#2ecc71']
        
        bars = ax1.bar(models, sizes, color=colors, edgecolor='black', linewidth=2, width=0.6)
        
        for i, (bar, size) in enumerate(zip(bars, sizes)):
            reduction = (1 - sizes[1]/sizes[0]) * 100 if i == 1 else 0
            label = f'{size:.1f} MB' if i == 0 else f'{size:.1f} MB\n↓ {reduction:.1f}%'
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    label, ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        ax1.set_ylabel('Model Size (MB)', fontsize=14, fontweight='bold')
        ax1.set_title('Model Size Comparison', fontsize=16, fontweight='bold', pad=15)
        ax1.set_ylim(0, max(sizes) * 1.25)
        ax1.grid(axis='y', alpha=0.3)
        
        # ============================================
        # Chart 2: Speed Comparison
        # ============================================
        ax2 = plt.subplot(1, 3, 2)
        
        fps_values = [self.results['fp32']['fps'], self.results['int8']['fps']]
        
        bars = ax2.bar(models, fps_values, color=colors, edgecolor='black', linewidth=2, width=0.6)
        
        for i, (bar, fps) in enumerate(zip(bars, fps_values)):
            speedup = fps_values[1] / fps_values[0] if i == 1 else 1.0
            label = f'{fps:.1f} FPS' if i == 0 else f'{fps:.1f} FPS\n↑ {speedup:.2f}x'
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    label, ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        ax2.set_ylabel('Inference Speed (FPS)', fontsize=14, fontweight='bold')
        ax2.set_title('Speed Comparison', fontsize=16, fontweight='bold', pad=15)
        ax2.set_ylim(0, max(fps_values) * 1.25)
        ax2.grid(axis='y', alpha=0.3)
        
        # ============================================
        # Chart 3: Accuracy Comparison
        # ============================================
        ax3 = plt.subplot(1, 3, 3)
        
        metrics = ['mAP@50-95', 'mAP@50', 'Precision', 'Recall']
        fp32_vals = [
            self.results['fp32']['map'],
            self.results['fp32']['map50'],
            self.results['fp32']['precision'],
            self.results['fp32']['recall']
        ]
        int8_vals = [
            self.results['int8']['map'],
            self.results['int8']['map50'],
            self.results['int8']['precision'],
            self.results['int8']['recall']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, fp32_vals, width, label='FP32', 
                       color='#3498db', edgecolor='black', linewidth=1.5)
        bars2 = ax3.bar(x + width/2, int8_vals, width, label='INT8', 
                       color='#2ecc71', edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', 
                        fontsize=10, fontweight='bold')
        
        ax3.set_ylabel('Score', fontsize=14, fontweight='bold')
        ax3.set_title('Accuracy Comparison', fontsize=16, fontweight='bold', pad=15)
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics, fontsize=11, rotation=15)
        ax3.set_ylim(0, 1.1)
        ax3.legend(fontsize=12)
        ax3.grid(axis='y', alpha=0.3)
        
        # Add accuracy drop annotation
        drop_text = f"Accuracy Drop: {self.results['int8']['accuracy_drop']:.2f}%"
        status = "✓ Excellent" if abs(self.results['int8']['accuracy_drop']) < 1 else "✓ Good" if abs(self.results['int8']['accuracy_drop']) < 2 else "⚠ Acceptable"
        ax3.text(0.5, 0.95, f"{drop_text}\n{status}", 
                transform=ax3.transAxes, ha='center', va='top',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        
        plt.suptitle('INT8 Quantization Results for YOLOv8', 
                    fontsize=20, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_path = Path(self.model_path).parent / 'int8_quantization_results.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Chart saved: {output_path}")
        plt.close()
    
    def generate_summary_report(self):
        """
        Generate text summary report
        """
        print("\n" + "="*70)
        print("📄 GENERATING SUMMARY REPORT")
        print("="*70)
        
        report = []
        report.append("="*70)
        report.append("INT8 QUANTIZATION SUMMARY REPORT")
        report.append("="*70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Model: {Path(self.model_path).name}")
        report.append("")
        
        # Results table
        report.append("QUANTIZATION RESULTS:")
        report.append("-"*70)
        report.append(f"{'Metric':<30} {'FP32':<20} {'INT8':<20} {'Change':<20}")
        report.append("-"*70)
        
        # Size
        size_change = (1 - self.results['int8']['size_mb']/self.results['fp32']['size_mb']) * 100
        report.append(f"{'Model Size (MB)':<30} {self.results['fp32']['size_mb']:<20.2f} "
                     f"{self.results['int8']['size_mb']:<20.2f} ↓ {size_change:.1f}%")
        
        # Speed
        speed_change = (self.results['int8']['fps'] / self.results['fp32']['fps'] - 1) * 100
        report.append(f"{'Inference Speed (FPS)':<30} {self.results['fp32']['fps']:<20.1f} "
                     f"{self.results['int8']['fps']:<20.1f} ↑ {speed_change:.1f}%")
        
                # Accuracy
        acc_change = self.results['int8']['accuracy_drop']
        report.append(f"{'mAP@50-95':<30} {self.results['fp32']['map']:<20.4f} "
                     f"{self.results['int8']['map']:<20.4f} {acc_change:+.2f}%")
        
        report.append(f"{'mAP@50':<30} {self.results['fp32']['map50']:<20.4f} "
                     f"{self.results['int8']['map50']:<20.4f}")
        
        report.append(f"{'Precision':<30} {self.results['fp32']['precision']:<20.4f} "
                     f"{self.results['int8']['precision']:<20.4f}")
        
        report.append(f"{'Recall':<30} {self.results['fp32']['recall']:<20.4f} "
                     f"{self.results['int8']['recall']:<20.4f}")
        
        report.append("")
        report.append("="*70)
        report.append("DEPLOYMENT RECOMMENDATION:")
        report.append("="*70)
        
        if abs(acc_change) < 1:
            report.append("✅ EXCELLENT - Ready for production deployment")
            report.append("   • Minimal accuracy loss (<1%)")
            report.append(f"   • {size_change:.1f}% size reduction")
            report.append(f"   • {speed_change:.1f}% speed improvement")
        elif abs(acc_change) < 2:
            report.append("✅ VERY GOOD - Recommended for deployment")
            report.append("   • Low accuracy loss (<2%)")
            report.append(f"   • {size_change:.1f}% size reduction")
            report.append(f"   • {speed_change:.1f}% speed improvement")
        elif abs(acc_change) < 5:
            report.append("⚠️ ACCEPTABLE - Deploy with caution")
            report.append("   • Moderate accuracy loss (2-5%)")
            report.append(f"   • {size_change:.1f}% size reduction")
            report.append(f"   • {speed_change:.1f}% speed improvement")
            report.append("   • Test thoroughly before production")
        else:
            report.append("❌ HIGH LOSS - Not recommended")
            report.append("   • High accuracy loss (>5%)")
            report.append("   • Consider alternative methods")
        
        report.append("")
        report.append("="*70)
        
        # Save report
        report_path = Path(self.model_path).parent / 'int8_quantization_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"✅ Report saved: {report_path}")
        
        # Print summary to console
        print("\n" + "="*70)
        print("📊 SUMMARY")
        print("="*70)
        print(f"Size Reduction:    {size_change:.1f}%")
        print(f"Speed Improvement: {speed_change:.1f}%")
        print(f"Accuracy Drop:     {acc_change:.2f}%")
        print("="*70)
    
    def run_complete_analysis(self):
        """
        Run complete INT8 quantization analysis
        """
        print("\n" + "="*70)
        print("🚀 STARTING INT8 QUANTIZATION ANALYSIS")
        print("="*70)
        
        start_time = time.time()
        
        # Step 1: Create INT8 model
        print("\n▶ Step 1/4: Creating INT8 quantized model...")
        self.create_int8_model()
        
        # Step 2: Benchmark speed
        print("\n▶ Step 2/4: Benchmarking inference speed...")
        self.benchmark_speed()
        
        # Step 3: Create charts
        print("\n▶ Step 3/4: Creating presentation charts...")
        self.create_presentation_charts()
        
        # Step 4: Generate report
        print("\n▶ Step 4/4: Generating summary report...")
        self.generate_summary_report()
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE!")
        print("="*70)
        print(f"Total time: {elapsed:.1f} seconds")
        print("\nGenerated files:")
        print(f"  1. {Path(self.results['int8']['path']).name} - INT8 model")
        print(f"  2. int8_quantization_results.png - Comparison charts")
        print(f"  3. int8_quantization_report.txt - Summary report")
        print(f"\nLocation: {Path(self.model_path).parent}")
        print("="*70)


# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == '__main__':
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║              INT8 QUANTIZATION FOR YOLO (.pt)                   ║
    ║                                                                  ║
    ║  This script will:                                              ║
    ║  ✓ Create INT8 quantized .pt model                             ║
    ║  ✓ Compare size reduction                                       ║
    ║  ✓ Benchmark inference speed                                    ║
    ║  ✓ Validate accuracy preservation                               ║
    ║  ✓ Generate presentation-ready charts                           ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    MODEL_PATH = r"C:\Users\\Documents\ADAS_Hackathon\runs\detect\FINETUNE_YOLOv8s_1280px_ULTRA\weights\model_best.pt"
    DATA_YAML = r"C:\Users\\Documents\ADAS_Hackathon\datasets\MASTER_IDD_YOLO\master_data.yaml"
    
    # Verify files exist
    if not Path(MODEL_PATH).exists():
        print(f"❌ ERROR: Model not found: {MODEL_PATH}")
        exit(1)
    
    if not Path(DATA_YAML).exists():
        print(f"❌ ERROR: Dataset YAML not found: {DATA_YAML}")
        exit(1)
    
    print(f"\n📦 Model: {Path(MODEL_PATH).name}")
    print(f"📊 Dataset: {Path(DATA_YAML).name}")
    print("\nStarting in 3 seconds...")
    time.sleep(3)
    
    # Run analysis
    quantizer = YOLOInt8Quantizer(MODEL_PATH, DATA_YAML)
    quantizer.run_complete_analysis()
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                        SUCCESS!                                 ║
    ║                                                                  ║
    ║  📊 Your INT8 quantized model is ready!                         ║
    ║                                                                  ║
    ║  Next steps:                                                    ║
    ║  1. Check int8_quantization_results.png for charts             ║
    ║  2. Read int8_quantization_report.txt for details              ║
    ║  3. Test best_int8.pt on your hardware                         ║
    ║  4. Deploy if accuracy drop is acceptable                       ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)