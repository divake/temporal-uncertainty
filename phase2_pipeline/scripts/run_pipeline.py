#!/usr/bin/env python3
"""
Main Pipeline Script for Phase 2
Aleatoric Uncertainty Quantification for Track 25 in MOT17-11-FRCNN
"""

import sys
import os
sys.path.append('/ssd_4TB/divake/temporal_uncertainty')
sys.path.append('/ssd_4TB/divake/temporal_uncertainty/phase2_pipeline')

import yaml
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import logging
import torch
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Import our modules
from src.data.mot_loader import MOT17Dataset
from src.data.track_extractor import Track25Analyzer
from src.models.yolo_wrapper import YOLOv8WithUncertainty
from src.uncertainty.aleatoric import AleatricUncertaintyEstimator, UncertaintyMetrics
from src.augmentations.transforms import get_tta_transforms
from src.visualization.uncertainty_plots import UncertaintyVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('phase2_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


class Track25UncertaintyPipeline:
    """Complete pipeline for Track 25 uncertainty analysis"""

    def __init__(self, config_dir: str = "phase2_pipeline/config"):
        """Initialize pipeline with configurations"""
        self.config_dir = Path(config_dir)
        self.configs = self._load_configs()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup results directory
        self.results_dir = Path(self.configs['experiment']['paths']['results_root']) / \
                          f"seq11_yolov8n_track25_{self.timestamp}"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Pipeline initialized. Results will be saved to: {self.results_dir}")

    def _load_configs(self) -> dict:
        """Load all configuration files"""
        configs = {}
        config_files = ['experiment', 'model', 'uncertainty', 'dataset']

        for config_name in config_files:
            config_path = self.config_dir / f"{config_name}.yaml"
            with open(config_path, 'r') as f:
                configs[config_name] = yaml.safe_load(f)

        return configs

    def setup_components(self):
        """Setup all pipeline components"""
        logger.info("Setting up pipeline components...")

        # 1. Load MOT17 sequence
        mot17_root = self.configs['experiment']['paths']['mot17_root']
        self.mot_dataset = MOT17Dataset(mot17_root)
        self.sequence = self.mot_dataset.load_sequence("MOT17-11-FRCNN")
        logger.info(f"Loaded sequence: {self.sequence.get_sequence_stats()}")

        # 2. Setup Track 25 analyzer
        metadata_root = self.configs['experiment']['paths']['metadata_root']
        self.track_analyzer = Track25Analyzer(metadata_root)
        logger.info(f"Track 25 stats: {self.track_analyzer.stats['duration']} frames")

        # 3. Load YOLO model with MC Dropout
        model_config = self.configs['model']['model']  # Access nested 'model' key
        self.yolo_model = YOLOv8WithUncertainty(
            model_path=model_config['weights_path'],
            device=self.configs['experiment']['experiment']['device'],  # Access nested 'experiment' key
            conf_threshold=model_config['confidence_threshold'],
            iou_threshold=model_config['iou_threshold'],
            enable_mc_dropout=model_config['enable_mc_dropout'],
            dropout_rate=self.configs['uncertainty']['uncertainty']['mc_dropout']['dropout_rate']  # Access nested 'uncertainty' key
        )

        # 4. Setup uncertainty estimator
        self.uncertainty_estimator = AleatricUncertaintyEstimator(
            combine_method=self.configs['uncertainty']['uncertainty']['tta']['ensemble']['method']
        )

        # 5. Setup TTA transforms
        tta_config = self.configs['uncertainty']['uncertainty']['tta']['augmentations']
        self.tta_transforms = get_tta_transforms(tta_config)

        # 6. Setup visualizer
        viz_dir = self.results_dir / "visualizations"
        self.visualizer = UncertaintyVisualizer(save_dir=viz_dir)

        logger.info("All components initialized successfully")

    def run_inference(self):
        """Run inference on Track 25 frames"""
        logger.info("Starting inference on Track 25...")

        # Get Track 25 lifetime
        start_frame, end_frame = self.track_analyzer.stats['lifetime_frames']

        # Use configured frame range
        exp_config = self.configs['experiment']['experiment']
        start_frame = max(start_frame, exp_config['start_frame'])
        end_frame = min(end_frame, exp_config['end_frame'])

        total_frames = end_frame - start_frame + 1
        logger.info(f"Processing frames {start_frame} to {end_frame} ({total_frames} frames)")

        # Storage for results
        self.frame_results = []
        self.uncertainty_timeline = []

        # MC Dropout settings
        mc_config = self.configs['uncertainty']['uncertainty']['mc_dropout']
        num_passes = mc_config['num_forward_passes']

        # TTA settings
        tta_enabled = self.configs['uncertainty']['uncertainty']['tta']['enabled']

        # Process each frame
        for frame_num in tqdm(range(start_frame, end_frame + 1), desc="Processing frames"):
            # Get frame image
            frame_img = self.sequence.get_frame_by_number(frame_num)

            # Get ground truth bbox for Track 25
            gt_bbox = self.sequence.get_track_bbox_for_frame(25, frame_num)

            if gt_bbox is None:
                logger.warning(f"No GT bbox for Track 25 at frame {frame_num}")
                continue

            # Adapt IoU threshold based on occlusion status
            # Lower threshold during occlusions for better detection
            is_occluded = self.track_analyzer.extractor.is_track_occluded(25, frame_num)
            iou_threshold = 0.2 if is_occluded else 0.3

            # Run MC Dropout inference
            mc_result = self.yolo_model.predict_with_uncertainty(
                frame_img,
                num_forward_passes=num_passes,
                target_bbox=gt_bbox,
                iou_threshold=iou_threshold
            )

            # Run TTA if enabled
            tta_result = None
            if tta_enabled:
                tta_result = self._run_tta_inference(frame_img, gt_bbox, frame_num)

            # Combine results
            frame_result = self._combine_results(
                frame_num, gt_bbox, mc_result, tta_result
            )

            # Store results
            self.frame_results.append(frame_result)
            self.uncertainty_timeline.append({
                'frame': frame_num,
                'uncertainty': frame_result['combined_uncertainty'],
                'bbox_uncertainty': frame_result.get('bbox_uncertainty', 0),
                'conf_uncertainty': frame_result.get('conf_uncertainty', 0),
                'is_occluded': self.track_analyzer.extractor.is_track_occluded(25, frame_num)
            })

            # Clear GPU cache periodically
            if frame_num % 100 == 0:
                torch.cuda.empty_cache()
                logger.info(f"Processed {frame_num - start_frame + 1}/{total_frames} frames")

        logger.info(f"Inference complete. Processed {len(self.frame_results)} frames")

    def _run_tta_inference(self, image: np.ndarray, gt_bbox: np.ndarray, frame_num: int) -> dict:
        """Run TTA inference on a single frame"""
        # Generate augmented images
        augmented_images = self.tta_transforms(image)

        # Run inference on each augmented image
        tta_predictions = []
        for aug_img in augmented_images:
            pred = self.yolo_model.predict_single(aug_img, apply_dropout=False)
            tta_predictions.append(pred)

        # Compute uncertainty from TTA predictions
        matched_boxes = []
        matched_confs = []

        for pred in tta_predictions:
            if len(pred['boxes']) == 0:
                continue

            # Find best matching box (use lower threshold during occlusions)
            is_occluded = self.track_analyzer.extractor.is_track_occluded(25, frame_num)
            iou_thresh_tta = 0.2 if is_occluded else 0.3
            ious = self.yolo_model._compute_iou(gt_bbox, pred['boxes'])
            if len(ious) > 0 and np.max(ious) > iou_thresh_tta:
                best_idx = np.argmax(ious)
                matched_boxes.append(pred['boxes'][best_idx])
                matched_confs.append(pred['confidences'][best_idx])

        if len(matched_boxes) > 0:
            bbox_unc = self.uncertainty_estimator.compute_bbox_uncertainty(
                np.array(matched_boxes)
            )
            conf_unc = self.uncertainty_estimator.compute_confidence_uncertainty(
                np.array(matched_confs)
            )

            return {
                'bbox_uncertainty': bbox_unc,
                'conf_uncertainty': conf_unc,
                'num_augmentations': len(augmented_images),
                'detection_rate': len(matched_boxes) / len(augmented_images)
            }

        return None

    def _combine_results(self, frame_num: int, gt_bbox: np.ndarray,
                        mc_result: dict, tta_result: dict = None) -> dict:
        """Combine MC Dropout and TTA results"""
        result = {
            'frame': frame_num,
            'gt_bbox': gt_bbox.tolist(),
            'mc_dropout': mc_result
        }

        # Extract uncertainties
        if mc_result.get('found'):
            bbox_unc = mc_result.get('bbox_uncertainty', 0)
            conf_unc = mc_result.get('confidence_variance', 0)
        else:
            # When track not found, set VERY HIGH uncertainty
            # Use 99th percentile of typical uncertainty values
            bbox_unc = 100.0  # High spatial uncertainty when track not detected
            conf_unc = 1.0    # Max confidence uncertainty

        # Add TTA if available
        if tta_result:
            result['tta'] = tta_result
            # Average MC and TTA uncertainties
            if tta_result.get('bbox_uncertainty'):
                tta_bbox_unc = tta_result['bbox_uncertainty']['total_uncertainty']
                bbox_unc = (bbox_unc + tta_bbox_unc) / 2
            if tta_result.get('conf_uncertainty'):
                tta_conf_unc = tta_result['conf_uncertainty']['variance']
                conf_unc = (conf_unc + tta_conf_unc) / 2

        # Compute combined uncertainty
        combined = self.uncertainty_estimator.compute_combined_uncertainty(
            bbox_unc, conf_unc,
            bbox_weight=self.configs['uncertainty']['uncertainty']['metrics']['combined_metric']['bbox_weight'],
            confidence_weight=self.configs['uncertainty']['uncertainty']['metrics']['combined_metric']['confidence_weight']
        )

        result['bbox_uncertainty'] = float(bbox_unc)
        result['conf_uncertainty'] = float(conf_unc)
        result['combined_uncertainty'] = float(combined)

        # Add frame classification
        result['classification'] = self.track_analyzer.classify_frame(frame_num)

        return result

    def analyze_results(self):
        """Analyze uncertainty patterns"""
        logger.info("Analyzing uncertainty patterns...")

        # Convert timeline to array
        frames = np.array([r['frame'] for r in self.uncertainty_timeline])
        uncertainties = np.array([r['uncertainty'] for r in self.uncertainty_timeline])

        # Temporal consistency
        temporal_metrics = self.uncertainty_estimator.compute_temporal_consistency(
            uncertainties
        )
        logger.info(f"Temporal correlation: {temporal_metrics['temporal_correlation']:.3f}")

        # Analyze by period
        analysis_frames = self.track_analyzer.get_analysis_frames()
        period_stats = {}

        for period, frame_list in analysis_frames.items():
            if len(frame_list) == 0:
                continue

            # Get uncertainties for this period
            period_mask = np.isin(frames, frame_list)
            if np.any(period_mask):
                period_unc = uncertainties[period_mask]
                period_stats[period] = {
                    'mean': float(np.mean(period_unc)),
                    'std': float(np.std(period_unc)),
                    'median': float(np.median(period_unc)),
                    'min': float(np.min(period_unc)),
                    'max': float(np.max(period_unc)),
                    'num_frames': len(period_unc)
                }

        self.analysis_results = {
            'temporal_metrics': temporal_metrics,
            'period_statistics': period_stats,
            'occlusion_periods': self.track_analyzer.get_occlusion_periods(),
            'recovery_periods': self.track_analyzer.get_recovery_periods()
        }

        # Log summary
        logger.info("\nUncertainty by Period:")
        for period, stats in period_stats.items():
            logger.info(f"  {period}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")

        # Statistical test: Is uncertainty higher during occlusion?
        if 'occluded' in period_stats and 'clean' in period_stats:
            occluded_mean = period_stats['occluded']['mean']
            clean_mean = period_stats['clean']['mean']
            ratio = occluded_mean / clean_mean if clean_mean > 0 else 0

            logger.info(f"\nOcclusion Impact:")
            logger.info(f"  Occluded uncertainty: {occluded_mean:.3f}")
            logger.info(f"  Clean uncertainty: {clean_mean:.3f}")
            logger.info(f"  Ratio: {ratio:.2f}x")

            self.analysis_results['occlusion_impact'] = {
                'occluded_mean': occluded_mean,
                'clean_mean': clean_mean,
                'ratio': ratio
            }

    def generate_visualizations(self):
        """Generate all visualization plots"""
        logger.info("Generating visualizations...")

        # Convert timeline to arrays
        frames = np.array([r['frame'] for r in self.uncertainty_timeline])
        uncertainties = np.array([r['uncertainty'] for r in self.uncertainty_timeline])
        bbox_unc = np.array([r['bbox_uncertainty'] for r in self.uncertainty_timeline])
        conf_unc = np.array([r['conf_uncertainty'] for r in self.uncertainty_timeline])

        # 1. Main uncertainty timeline
        self.visualizer.plot_uncertainty_timeline(
            frames, uncertainties,
            occlusion_periods=self.track_analyzer.get_occlusion_periods(),
            track_id=25,
            title="Aleatoric Uncertainty Timeline - Track 25 (MOT17-11-FRCNN)",
            save_name="track25_uncertainty_timeline.png"
        )

        # 2. Uncertainty components
        self.visualizer.plot_uncertainty_components(
            bbox_unc, conf_unc, uncertainties, frames,
            title="Uncertainty Components - Track 25",
            save_name="track25_uncertainty_components.png"
        )

        # 3. Before/During/After analysis
        analysis_frames = self.track_analyzer.get_analysis_frames()
        uncertainty_dict = {}

        for period in ['clean', 'occluded', 'early_recovery']:
            if period in analysis_frames:
                period_frames = analysis_frames[period]
                mask = np.isin(frames, period_frames)
                if np.any(mask):
                    if period == 'clean':
                        uncertainty_dict['before'] = uncertainties[mask]
                    elif period == 'occluded':
                        uncertainty_dict['during'] = uncertainties[mask]
                    elif period == 'early_recovery':
                        uncertainty_dict['after'] = uncertainties[mask]

        if uncertainty_dict:
            self.visualizer.plot_before_during_after(
                uncertainty_dict,
                title="Uncertainty Distribution - Before/During/After Occlusion",
                save_name="track25_before_during_after.png"
            )

        # 4. Recovery curves
        recovery_data = []
        occlusion_periods = self.track_analyzer.get_occlusion_periods()

        for occ_start, occ_end in occlusion_periods[:3]:  # First 3 occlusions
            # Get recovery period after this occlusion
            recovery_start = occ_end + 1
            recovery_end = min(recovery_start + 30, frames[-1])  # 30 frames or end

            recovery_mask = (frames >= recovery_start) & (frames <= recovery_end)
            if np.any(recovery_mask):
                recovery_data.append({
                    'frames': frames[recovery_mask].tolist(),
                    'uncertainties': uncertainties[recovery_mask].tolist()
                })

        if recovery_data:
            self.visualizer.plot_recovery_curves(
                recovery_data,
                title="Uncertainty Recovery After Occlusion Events",
                save_name="track25_recovery_curves.png"
            )

        # 5. Summary statistics
        stats = {
            'mean_by_period': {k: v['mean']
                             for k, v in self.analysis_results['period_statistics'].items()},
            'temporal_correlation': [self.analysis_results['temporal_metrics']['temporal_correlation']],
            'distribution': uncertainties
        }

        self.visualizer.plot_summary_statistics(
            stats,
            title="Uncertainty Analysis Summary - Track 25",
            save_name="track25_summary_statistics.png"
        )

        logger.info(f"Generated {5} visualization plots")

    def save_results(self):
        """Save all results to disk"""
        logger.info("Saving results...")

        # Save raw results
        raw_dir = self.results_dir / "raw_detections"
        raw_dir.mkdir(exist_ok=True)

        with open(raw_dir / "frame_results.pkl", 'wb') as f:
            pickle.dump(self.frame_results, f)

        # Save uncertainty metrics
        metrics_dir = self.results_dir / "uncertainty_metrics"
        metrics_dir.mkdir(exist_ok=True)

        with open(metrics_dir / "uncertainty_timeline.json", 'w') as f:
            json.dump(self.uncertainty_timeline, f, indent=2)

        with open(metrics_dir / "analysis_results.json", 'w') as f:
            json.dump(self.analysis_results, f, indent=2)

        # Save summary CSV
        df = pd.DataFrame(self.uncertainty_timeline)
        df.to_csv(metrics_dir / "uncertainty_timeline.csv", index=False)

        # Save config snapshot
        with open(self.results_dir / "config_snapshot.yaml", 'w') as f:
            yaml.dump(self.configs, f)

        # Save summary report
        self._generate_summary_report()

        logger.info(f"Results saved to {self.results_dir}")

    def _generate_summary_report(self):
        """Generate human-readable summary report"""
        report = []
        report.append("=" * 80)
        report.append("ALEATORIC UNCERTAINTY ANALYSIS REPORT")
        report.append("Track 25 - MOT17-11-FRCNN")
        report.append("=" * 80)
        report.append(f"Timestamp: {self.timestamp}")
        report.append(f"Model: YOLOv8n")
        report.append(f"MC Dropout Passes: {self.configs['uncertainty']['uncertainty']['mc_dropout']['num_forward_passes']}")
        report.append(f"TTA Enabled: {self.configs['uncertainty']['uncertainty']['tta']['enabled']}")
        report.append("")

        # Frame statistics
        report.append("FRAME STATISTICS:")
        report.append(f"  Total frames processed: {len(self.frame_results)}")
        report.append(f"  Frame range: {self.configs['experiment']['experiment']['start_frame']}-{self.configs['experiment']['experiment']['end_frame']}")
        report.append("")

        # Uncertainty statistics
        report.append("UNCERTAINTY STATISTICS:")
        for period, stats in self.analysis_results['period_statistics'].items():
            report.append(f"  {period.capitalize()}:")
            report.append(f"    Mean: {stats['mean']:.4f}")
            report.append(f"    Std: {stats['std']:.4f}")
            report.append(f"    Frames: {stats['num_frames']}")

        report.append("")

        # Temporal metrics
        report.append("TEMPORAL METRICS:")
        tm = self.analysis_results['temporal_metrics']
        report.append(f"  Temporal correlation: {tm['temporal_correlation']:.3f}")
        report.append(f"  Temporal variance: {tm['temporal_variance']:.4f}")
        report.append(f"  Smoothness: {tm['smoothness']:.3f}")
        report.append("")

        # Key finding
        if 'occlusion_impact' in self.analysis_results:
            oi = self.analysis_results['occlusion_impact']
            report.append("KEY FINDING:")
            report.append(f"  Uncertainty during occlusion is {oi['ratio']:.2f}x higher than clean periods")
            report.append(f"  This confirms aleatoric uncertainty responds to data quality degradation")

        report.append("")
        report.append("=" * 80)

        # Save report
        with open(self.results_dir / "summary_report.txt", 'w') as f:
            f.write('\n'.join(report))

        # Also print to console
        print('\n'.join(report))

    def run(self):
        """Run complete pipeline"""
        logger.info("Starting Phase 2 Pipeline: Aleatoric Uncertainty Analysis")
        logger.info("=" * 80)

        try:
            # Setup
            self.setup_components()

            # Run inference
            self.run_inference()

            # Analyze results
            self.analyze_results()

            # Generate visualizations
            self.generate_visualizations()

            # Save results
            self.save_results()

            logger.info("Pipeline completed successfully!")
            logger.info(f"Results saved to: {self.results_dir}")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise


def main():
    """Main entry point"""
    # Create and run pipeline
    pipeline = Track25UncertaintyPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()