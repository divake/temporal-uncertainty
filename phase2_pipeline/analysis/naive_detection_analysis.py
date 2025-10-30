#!/usr/bin/env python3
"""
Naive Detection Analysis for MOT17-11-FRCNN
Analyzes raw YOLOv8 detections without uncertainty quantification
Compares with aleatoric uncertainty patterns
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
import torch
from ultralytics import YOLO

# Add parent directory to path
sys.path.append('/ssd_4TB/divake/temporal_uncertainty/phase2_pipeline')
sys.path.append('/ssd_4TB/divake/temporal_uncertainty')

from src.data.mot_loader import MOT17Sequence
from src.data.track_extractor import Track25Analyzer
from src.visualization.uncertainty_plots import UncertaintyVisualizer
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NaiveDetectionAnalyzer:
    """Analyze naive YOLO detections for Track 25"""

    def __init__(self, sequence_name: str = "MOT17-11-FRCNN", track_id: int = 25):
        """Initialize analyzer"""
        self.sequence_name = sequence_name
        self.track_id = track_id

        # Setup paths
        self.data_root = Path("/ssd_4TB/divake/temporal_uncertainty/data/MOT17/train")
        self.metadata_root = Path("/ssd_4TB/divake/temporal_uncertainty/metadata/raw_outputs")
        self.results_dir = Path("/ssd_4TB/divake/temporal_uncertainty/phase2_pipeline/analysis/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load sequence - pass full path to sequence directory
        seq_path = self.data_root / self.sequence_name
        self.sequence = MOT17Sequence(seq_path)

        # Load track analyzer
        self.track_analyzer = Track25Analyzer(str(self.metadata_root))

        # Load YOLO model
        self.model = YOLO('/ssd_4TB/divake/temporal_uncertainty/models/yolov8n.pt')
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')

        # Results storage
        self.detection_results = []
        self.analysis_results = {}

    def compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes [x, y, w, h]"""
        # Convert to [x1, y1, x2, y2]
        b1_x1, b1_y1 = box1[0], box1[1]
        b1_x2, b1_y2 = box1[0] + box1[2], box1[1] + box1[3]

        b2_x1, b2_y1 = box2[0], box2[1]
        b2_x2, b2_y2 = box2[0] + box2[2], box2[1] + box2[3]

        # Intersection
        inter_x1 = max(b1_x1, b2_x1)
        inter_y1 = max(b1_y1, b2_y1)
        inter_x2 = min(b1_x2, b2_x2)
        inter_y2 = min(b1_y2, b2_y2)

        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

        # Union
        b1_area = box1[2] * box1[3]
        b2_area = box2[2] * box2[3]
        union_area = b1_area + b2_area - inter_area

        return inter_area / (union_area + 1e-6)

    def run_naive_detection(self, start_frame: int = 1, end_frame: int = 900):
        """Run naive YOLO detection on sequence"""
        logger.info(f"Running naive detection on frames {start_frame}-{end_frame}")

        for frame_num in tqdm(range(start_frame, min(end_frame + 1, self.sequence.num_frames + 1)),
                             desc="Processing frames"):
            # Get frame
            frame_img = self.sequence.get_frame_by_number(frame_num)

            # Get ground truth for Track 25
            gt_bbox = self.sequence.get_track_bbox_for_frame(self.track_id, frame_num)

            # Run YOLO detection
            results = self.model(frame_img, verbose=False)

            # Process detections
            frame_result = {
                'frame': frame_num,
                'gt_bbox': gt_bbox.tolist() if gt_bbox is not None else None,
                'is_occluded': self.track_analyzer.extractor.is_track_occluded(self.track_id, frame_num),
                'classification': self.track_analyzer.classify_frame(frame_num),
                'detections': [],
                'matched_detection': None,
                'match_confidence': 0,
                'match_iou': 0,
                'num_detections': 0,
                'detection_found': False
            }

            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes

                # Convert to numpy arrays
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy()

                # Convert to xywh format
                xywh = np.zeros_like(xyxy)
                xywh[:, 0] = xyxy[:, 0]  # x
                xywh[:, 1] = xyxy[:, 1]  # y
                xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]  # w
                xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]  # h

                frame_result['num_detections'] = len(xywh)

                # Store all detections
                for i in range(len(xywh)):
                    frame_result['detections'].append({
                        'bbox': xywh[i].tolist(),
                        'confidence': float(conf[i]),
                        'class': int(cls[i])
                    })

                # Find best match for Track 25 if GT available
                if gt_bbox is not None:
                    ious = [self.compute_iou(gt_bbox, box) for box in xywh]
                    if len(ious) > 0 and max(ious) > 0.1:  # Very low threshold for analysis
                        best_idx = np.argmax(ious)
                        frame_result['matched_detection'] = xywh[best_idx].tolist()
                        frame_result['match_confidence'] = float(conf[best_idx])
                        frame_result['match_iou'] = float(ious[best_idx])
                        frame_result['detection_found'] = True

            self.detection_results.append(frame_result)

            # Clear GPU cache periodically
            if frame_num % 100 == 0:
                torch.cuda.empty_cache()

        logger.info(f"Processed {len(self.detection_results)} frames")

    def analyze_detection_patterns(self):
        """Analyze detection patterns across occlusion periods"""
        logger.info("Analyzing detection patterns...")

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.detection_results)

        # Group by classification
        classifications = ['occluded', 'clean', 'early_recovery', 'mid_recovery', 'late_recovery']

        analysis = {}
        for cls in classifications:
            cls_data = df[df['classification'] == cls]
            if len(cls_data) > 0:
                analysis[cls] = {
                    'num_frames': len(cls_data),
                    'detection_rate': cls_data['detection_found'].mean(),
                    'avg_confidence': cls_data['match_confidence'].mean(),
                    'avg_iou': cls_data['match_iou'].mean(),
                    'avg_num_detections': cls_data['num_detections'].mean(),
                    'confidence_std': cls_data['match_confidence'].std(),
                    'iou_std': cls_data['match_iou'].std()
                }

        self.analysis_results['by_classification'] = analysis

        # Analyze occlusion transitions
        occlusion_events = self.track_analyzer.get_occlusion_periods()
        transition_analysis = []

        for start, end in occlusion_events:
            # Analyze 10 frames before, during, and after
            before_start = max(1, start - 10)
            after_end = min(len(df), end + 10)

            before_data = df[(df['frame'] >= before_start) & (df['frame'] < start)]
            during_data = df[(df['frame'] >= start) & (df['frame'] <= end)]
            after_data = df[(df['frame'] > end) & (df['frame'] <= after_end)]

            transition = {
                'occlusion_start': start,
                'occlusion_end': end,
                'duration': end - start + 1,
                'before': {
                    'detection_rate': before_data['detection_found'].mean() if len(before_data) > 0 else 0,
                    'avg_confidence': before_data['match_confidence'].mean() if len(before_data) > 0 else 0
                },
                'during': {
                    'detection_rate': during_data['detection_found'].mean() if len(during_data) > 0 else 0,
                    'avg_confidence': during_data['match_confidence'].mean() if len(during_data) > 0 else 0
                },
                'after': {
                    'detection_rate': after_data['detection_found'].mean() if len(after_data) > 0 else 0,
                    'avg_confidence': after_data['match_confidence'].mean() if len(after_data) > 0 else 0
                }
            }
            transition_analysis.append(transition)

        self.analysis_results['occlusion_transitions'] = transition_analysis

        return analysis

    def save_results(self):
        """Save results to files"""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        # Save detection results as JSON
        json_path = self.results_dir / f"naive_detections_{self.sequence_name}_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(self.detection_results, f, indent=2)
        logger.info(f"Saved detection results to {json_path}")

        # Save as CSV for easy analysis
        df = pd.DataFrame(self.detection_results)
        csv_path = self.results_dir / f"naive_detections_{self.sequence_name}_{timestamp}.csv"

        # Flatten for CSV
        csv_data = []
        for _, row in df.iterrows():
            csv_row = {
                'frame': row['frame'],
                'is_occluded': row['is_occluded'],
                'classification': row['classification'],
                'detection_found': row['detection_found'],
                'match_confidence': row['match_confidence'],
                'match_iou': row['match_iou'],
                'num_detections': row['num_detections']
            }
            csv_data.append(csv_row)

        csv_df = pd.DataFrame(csv_data)
        csv_df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV to {csv_path}")

        # Save analysis results
        analysis_path = self.results_dir / f"naive_analysis_{self.sequence_name}_{timestamp}.json"
        with open(analysis_path, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        logger.info(f"Saved analysis to {analysis_path}")

        return json_path, csv_path, analysis_path

    def create_visualizations(self, save_dir: Path = None):
        """Create comprehensive visualizations"""
        if save_dir is None:
            save_dir = self.results_dir / "visualizations"
        save_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self.detection_results)

        # Create figure with multiple subplots
        fig, axes = plt.subplots(4, 1, figsize=(16, 12))

        # 1. Detection confidence over time
        ax1 = axes[0]
        frames = df['frame'].values
        confidences = df['match_confidence'].values

        # Mark occlusion periods
        occlusion_periods = self.track_analyzer.get_occlusion_periods()
        for start, end in occlusion_periods:
            ax1.axvspan(start, end, alpha=0.3, color='red', label='Occlusion' if start == occlusion_periods[0][0] else '')

        ax1.plot(frames, confidences, 'b-', alpha=0.7, label='Detection Confidence')
        ax1.set_ylabel('Confidence')
        ax1.set_title('Naive YOLO Detection Confidence Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Detection rate (binary: found/not found)
        ax2 = axes[1]
        detection_binary = df['detection_found'].astype(int).values

        for start, end in occlusion_periods:
            ax2.axvspan(start, end, alpha=0.3, color='red')

        ax2.plot(frames, detection_binary, 'g-', alpha=0.7, label='Detection Found')
        ax2.set_ylabel('Detection (0/1)')
        ax2.set_title('Track 25 Detection Success')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. IoU with ground truth
        ax3 = axes[2]
        ious = df['match_iou'].values

        for start, end in occlusion_periods:
            ax3.axvspan(start, end, alpha=0.3, color='red')

        ax3.plot(frames, ious, 'purple', alpha=0.7, label='IoU with GT')
        ax3.set_ylabel('IoU')
        ax3.set_title('Detection Quality (IoU with Ground Truth)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Number of total detections
        ax4 = axes[3]
        num_detections = df['num_detections'].values

        for start, end in occlusion_periods:
            ax4.axvspan(start, end, alpha=0.3, color='red')

        ax4.plot(frames, num_detections, 'orange', alpha=0.7, label='Total Detections')
        ax4.set_xlabel('Frame Number')
        ax4.set_ylabel('# Detections')
        ax4.set_title('Total YOLO Detections per Frame')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.suptitle(f'Naive YOLO Detection Analysis - {self.sequence_name} Track {self.track_id}',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()

        viz_path = save_dir / f"naive_detection_analysis.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved visualization to {viz_path}")

        # Create summary statistics plot
        self._create_summary_plot(save_dir)

        return viz_path

    def _create_summary_plot(self, save_dir: Path):
        """Create summary statistics visualization"""
        if 'by_classification' not in self.analysis_results:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        analysis = self.analysis_results['by_classification']
        classifications = list(analysis.keys())

        # Detection rates
        ax1 = axes[0, 0]
        detection_rates = [analysis[cls]['detection_rate'] for cls in classifications]
        colors = ['red' if 'occluded' in cls else 'green' if 'clean' in cls else 'blue'
                  for cls in classifications]
        ax1.bar(classifications, detection_rates, color=colors, alpha=0.7)
        ax1.set_ylabel('Detection Rate')
        ax1.set_title('Detection Rate by Period')
        ax1.set_xticklabels(classifications, rotation=45)

        # Average confidence
        ax2 = axes[0, 1]
        avg_conf = [analysis[cls]['avg_confidence'] for cls in classifications]
        ax2.bar(classifications, avg_conf, color=colors, alpha=0.7)
        ax2.set_ylabel('Average Confidence')
        ax2.set_title('Detection Confidence by Period')
        ax2.set_xticklabels(classifications, rotation=45)

        # Average IoU
        ax3 = axes[1, 0]
        avg_iou = [analysis[cls]['avg_iou'] for cls in classifications]
        ax3.bar(classifications, avg_iou, color=colors, alpha=0.7)
        ax3.set_ylabel('Average IoU')
        ax3.set_title('Detection Quality (IoU) by Period')
        ax3.set_xticklabels(classifications, rotation=45)

        # Number of frames
        ax4 = axes[1, 1]
        num_frames = [analysis[cls]['num_frames'] for cls in classifications]
        ax4.bar(classifications, num_frames, color=colors, alpha=0.7)
        ax4.set_ylabel('Number of Frames')
        ax4.set_title('Frame Distribution')
        ax4.set_xticklabels(classifications, rotation=45)

        plt.suptitle('Naive Detection Summary Statistics', fontsize=14, fontweight='bold')
        plt.tight_layout()

        summary_path = save_dir / "naive_detection_summary.png"
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved summary plot to {summary_path}")

    def print_summary(self):
        """Print analysis summary"""
        print("\n" + "="*80)
        print("NAIVE DETECTION ANALYSIS SUMMARY")
        print("="*80)

        if 'by_classification' in self.analysis_results:
            print("\nDetection Performance by Period:")
            print("-"*40)

            for cls, stats in self.analysis_results['by_classification'].items():
                print(f"\n{cls.upper()}:")
                print(f"  Frames: {stats['num_frames']}")
                print(f"  Detection Rate: {stats['detection_rate']:.2%}")
                print(f"  Avg Confidence: {stats['avg_confidence']:.3f}")
                print(f"  Avg IoU: {stats['avg_iou']:.3f}")
                print(f"  Avg # Detections: {stats['avg_num_detections']:.1f}")

        if 'occlusion_transitions' in self.analysis_results:
            print("\n\nOcclusion Impact Analysis:")
            print("-"*40)

            for i, trans in enumerate(self.analysis_results['occlusion_transitions'][:3]):  # Show first 3
                print(f"\nOcclusion Event {i+1} (Frames {trans['occlusion_start']}-{trans['occlusion_end']}):")
                print(f"  Before: Detection Rate = {trans['before']['detection_rate']:.2%}, "
                      f"Confidence = {trans['before']['avg_confidence']:.3f}")
                print(f"  During: Detection Rate = {trans['during']['detection_rate']:.2%}, "
                      f"Confidence = {trans['during']['avg_confidence']:.3f}")
                print(f"  After:  Detection Rate = {trans['after']['detection_rate']:.2%}, "
                      f"Confidence = {trans['after']['avg_confidence']:.3f}")

        print("\n" + "="*80)


def main():
    """Run naive detection analysis"""
    analyzer = NaiveDetectionAnalyzer()

    # Run detection on full sequence
    analyzer.run_naive_detection(start_frame=1, end_frame=900)

    # Analyze patterns
    analyzer.analyze_detection_patterns()

    # Save results
    json_path, csv_path, analysis_path = analyzer.save_results()

    # Create visualizations
    viz_path = analyzer.create_visualizations()

    # Print summary
    analyzer.print_summary()

    print(f"\nResults saved to:")
    print(f"  - Detections: {json_path}")
    print(f"  - CSV: {csv_path}")
    print(f"  - Analysis: {analysis_path}")
    print(f"  - Visualization: {viz_path}")


if __name__ == "__main__":
    main()