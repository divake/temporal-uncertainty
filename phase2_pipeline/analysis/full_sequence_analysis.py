#!/usr/bin/env python3
"""
Full Sequence Analysis with Aleatoric Uncertainty
Processes all 900 frames and saves detailed per-frame data
"""

import os
import sys
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append('/ssd_4TB/divake/temporal_uncertainty/phase2_pipeline')
sys.path.append('/ssd_4TB/divake/temporal_uncertainty')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FullSequenceAnalyzer:
    """Run full 900-frame analysis with uncertainty quantification"""

    def __init__(self):
        """Initialize analyzer"""
        self.config_path = Path("/ssd_4TB/divake/temporal_uncertainty/phase2_pipeline/config/experiment.yaml")
        self.results_dir = Path("/ssd_4TB/divake/temporal_uncertainty/phase2_pipeline/analysis/full_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load config
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Update config for full sequence
        self.config['experiment']['start_frame'] = 1
        self.config['experiment']['end_frame'] = 900  # Full sequence
        self.config['experiment']['num_mc_passes'] = 30  # Keep MC passes
        self.config['experiment']['enable_tta'] = True  # Keep TTA

        # Save updated config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.results_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        config_snapshot = self.run_dir / "config.yaml"
        with open(config_snapshot, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def run_full_pipeline(self):
        """Run the full pipeline with updated config"""
        logger.info("Running full sequence analysis (900 frames)...")

        # Create temporary config file
        temp_config = Path("/tmp/full_sequence_config.yaml")
        with open(temp_config, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

        # Run the pipeline with custom config
        import subprocess
        cmd = [
            "python",
            "/ssd_4TB/divake/temporal_uncertainty/phase2_pipeline/scripts/run_pipeline.py",
            "--config", str(temp_config),
            "--output_dir", str(self.run_dir)
        ]

        # Note: We need to modify run_pipeline.py to accept command-line arguments
        # For now, let's import and run directly
        self._run_pipeline_directly()

    def _run_pipeline_directly(self):
        """Run pipeline directly by importing"""
        # Temporarily modify sys.argv to pass config
        import sys
        original_argv = sys.argv.copy()

        try:
            # Import pipeline
            from scripts.run_pipeline import UncertaintyPipeline

            # Create pipeline with modified config
            pipeline = UncertaintyPipeline(self.config)

            # Override results directory
            pipeline.results_dir = self.run_dir

            # Run inference
            pipeline.run_inference()

            # Analyze results
            pipeline.analyze_results()

            # Save all results
            pipeline.save_results()

            # Generate visualizations
            pipeline.generate_visualizations()

            # Generate report
            pipeline.generate_report()

            logger.info(f"Full sequence analysis complete. Results in {self.run_dir}")

            # Extract and save per-frame data
            self._save_per_frame_data(pipeline)

        finally:
            sys.argv = original_argv

    def _save_per_frame_data(self, pipeline):
        """Save detailed per-frame data in multiple formats"""
        logger.info("Saving per-frame data...")

        # Create per-frame directory
        per_frame_dir = self.run_dir / "per_frame_data"
        per_frame_dir.mkdir(exist_ok=True)

        # Save uncertainty timeline as JSON
        timeline_json = per_frame_dir / "uncertainty_timeline.json"
        with open(timeline_json, 'w') as f:
            json.dump(pipeline.uncertainty_timeline, f, indent=2)

        # Save as CSV for easy analysis
        timeline_csv = per_frame_dir / "uncertainty_timeline.csv"
        df = pd.DataFrame(pipeline.uncertainty_timeline)
        df.to_csv(timeline_csv, index=False)

        # Save detailed frame results
        if hasattr(pipeline, 'frame_results'):
            detailed_json = per_frame_dir / "detailed_frame_results.json"

            # Convert to serializable format
            serializable_results = []
            for result in pipeline.frame_results:
                frame_data = {
                    'frame': result['frame'],
                    'classification': result['classification'],
                    'combined_uncertainty': result['combined_uncertainty'],
                    'bbox_uncertainty': result.get('bbox_uncertainty', 0),
                    'conf_uncertainty': result.get('conf_uncertainty', 0),
                    'mc_dropout': {
                        'found': result['mc_dropout'].get('found', False),
                        'num_detections': result['mc_dropout'].get('num_detections', 0),
                        'detection_rate': result['mc_dropout'].get('detection_rate', 0)
                    }
                }

                # Add TTA results if available
                if 'tta' in result and result['tta']:
                    frame_data['tta'] = {
                        'detection_rate': result['tta'].get('detection_rate', 0),
                        'num_augmentations': result['tta'].get('num_augmentations', 5)
                    }

                serializable_results.append(frame_data)

            with open(detailed_json, 'w') as f:
                json.dump(serializable_results, f, indent=2)

        # Create summary statistics
        self._create_summary_stats(df, per_frame_dir)

        logger.info(f"Per-frame data saved to {per_frame_dir}")

    def _create_summary_stats(self, df: pd.DataFrame, save_dir: Path):
        """Create summary statistics file"""
        summary = {
            'total_frames': len(df),
            'frame_range': [int(df['frame'].min()), int(df['frame'].max())],
            'uncertainty_stats': {
                'mean': float(df['uncertainty'].mean()),
                'std': float(df['uncertainty'].std()),
                'min': float(df['uncertainty'].min()),
                'max': float(df['uncertainty'].max()),
                'median': float(df['uncertainty'].median()),
                'q25': float(df['uncertainty'].quantile(0.25)),
                'q75': float(df['uncertainty'].quantile(0.75))
            },
            'by_occlusion': {
                'occluded': {
                    'frames': int(df['is_occluded'].sum()),
                    'mean_uncertainty': float(df[df['is_occluded']]['uncertainty'].mean())
                        if df['is_occluded'].sum() > 0 else 0,
                    'std_uncertainty': float(df[df['is_occluded']]['uncertainty'].std())
                        if df['is_occluded'].sum() > 0 else 0
                },
                'clean': {
                    'frames': int((~df['is_occluded']).sum()),
                    'mean_uncertainty': float(df[~df['is_occluded']]['uncertainty'].mean())
                        if (~df['is_occluded']).sum() > 0 else 0,
                    'std_uncertainty': float(df[~df['is_occluded']]['uncertainty'].std())
                        if (~df['is_occluded']).sum() > 0 else 0
                }
            }
        }

        # Calculate ratio
        if summary['by_occlusion']['clean']['mean_uncertainty'] > 0:
            summary['occlusion_impact_ratio'] = (
                summary['by_occlusion']['occluded']['mean_uncertainty'] /
                summary['by_occlusion']['clean']['mean_uncertainty']
            )

        # Save summary
        summary_path = save_dir / "summary_statistics.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Also save as readable text
        summary_txt = save_dir / "summary_statistics.txt"
        with open(summary_txt, 'w') as f:
            f.write("UNCERTAINTY ANALYSIS SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total Frames: {summary['total_frames']}\n")
            f.write(f"Frame Range: {summary['frame_range'][0]}-{summary['frame_range'][1]}\n\n")

            f.write("Overall Uncertainty Statistics:\n")
            f.write(f"  Mean: {summary['uncertainty_stats']['mean']:.2f}\n")
            f.write(f"  Std: {summary['uncertainty_stats']['std']:.2f}\n")
            f.write(f"  Min: {summary['uncertainty_stats']['min']:.2f}\n")
            f.write(f"  Max: {summary['uncertainty_stats']['max']:.2f}\n")
            f.write(f"  Median: {summary['uncertainty_stats']['median']:.2f}\n\n")

            f.write("Occlusion Analysis:\n")
            f.write(f"  Occluded Frames: {summary['by_occlusion']['occluded']['frames']}\n")
            f.write(f"    Mean Uncertainty: {summary['by_occlusion']['occluded']['mean_uncertainty']:.2f}\n")
            f.write(f"  Clean Frames: {summary['by_occlusion']['clean']['frames']}\n")
            f.write(f"    Mean Uncertainty: {summary['by_occlusion']['clean']['mean_uncertainty']:.2f}\n")

            if 'occlusion_impact_ratio' in summary:
                f.write(f"\nOcclusion Impact: {summary['occlusion_impact_ratio']:.2f}x higher during occlusions\n")

        logger.info(f"Summary statistics saved to {summary_path}")


def main():
    """Run full sequence analysis"""
    logger.info("Starting full sequence analysis...")

    # Check if we need to modify run_pipeline.py first
    pipeline_path = Path("/ssd_4TB/divake/temporal_uncertainty/phase2_pipeline/scripts/run_pipeline.py")

    # For now, let's create a simpler approach - directly modify and run
    print("\nTo run full 900-frame analysis, execute the following:")
    print("-"*60)
    print("1. Update config file:")
    print("   Edit phase2_pipeline/config/experiment.yaml")
    print("   Set end_frame: 900")
    print("\n2. Run pipeline:")
    print("   cd /ssd_4TB/divake/temporal_uncertainty")
    print("   python phase2_pipeline/scripts/run_pipeline.py")
    print("-"*60)

    # Create a modified config for full run
    config_path = Path("/ssd_4TB/divake/temporal_uncertainty/phase2_pipeline/config/experiment_full.yaml")

    with open("/ssd_4TB/divake/temporal_uncertainty/phase2_pipeline/config/experiment.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Update for full sequence
    config['experiment']['end_frame'] = 900

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\nFull sequence config created at: {config_path}")
    print("You can run: python phase2_pipeline/scripts/run_pipeline.py")


if __name__ == "__main__":
    main()