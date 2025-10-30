#!/usr/bin/env python3
"""
Combined Visualization: Naive Detection + Aleatoric Uncertainty
Shows relationship between raw detections and uncertainty quantification
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Dict, List, Tuple
import logging

# Add parent directory to path
sys.path.append('/ssd_4TB/divake/temporal_uncertainty/phase2_pipeline')
sys.path.append('/ssd_4TB/divake/temporal_uncertainty')

from src.data.track_extractor import Track25Analyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")


class CombinedVisualizer:
    """Create combined visualizations of detection and uncertainty"""

    def __init__(self):
        """Initialize visualizer"""
        self.metadata_root = Path("/ssd_4TB/divake/temporal_uncertainty/metadata/raw_outputs")
        self.results_root = Path("/ssd_4TB/divake/temporal_uncertainty/phase2_pipeline")
        self.analysis_dir = self.results_root / "analysis"
        self.viz_dir = self.analysis_dir / "combined_visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # Load Track 25 analyzer
        self.track_analyzer = Track25Analyzer(str(self.metadata_root))

        # Colors for visualization
        self.colors = {
            'occluded': '#ffcccc',
            'clean': '#ccffcc',
            'recovery': '#ccccff',
            'uncertainty': '#1f77b4',
            'detection': '#ff7f0e',
            'confidence': '#2ca02c'
        }

    def load_latest_results(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load latest naive detection and uncertainty results"""
        # Find latest naive detection results
        naive_files = list((self.analysis_dir / "results").glob("naive_detections_*.json"))
        if not naive_files:
            raise FileNotFoundError("No naive detection results found. Run naive_detection_analysis.py first.")

        latest_naive = max(naive_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Loading naive detection results from {latest_naive}")

        with open(latest_naive, 'r') as f:
            naive_data = json.load(f)
        naive_df = pd.DataFrame(naive_data)

        # Find latest uncertainty results
        unc_dirs = list((self.results_root / "results").glob("seq11_yolov8n_track25_*"))
        if not unc_dirs:
            raise FileNotFoundError("No uncertainty results found. Run pipeline first.")

        latest_unc_dir = max(unc_dirs, key=lambda x: x.stat().st_mtime)
        unc_file = latest_unc_dir / "uncertainty_metrics" / "uncertainty_timeline.json"

        logger.info(f"Loading uncertainty results from {unc_file}")

        with open(unc_file, 'r') as f:
            unc_data = json.load(f)
        unc_df = pd.DataFrame(unc_data)

        return naive_df, unc_df

    def create_comprehensive_plot(self, naive_df: pd.DataFrame, unc_df: pd.DataFrame):
        """Create comprehensive combined visualization"""
        # Create figure with custom layout
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(6, 2, figure=fig, height_ratios=[1, 1, 1, 1, 1, 0.5])

        # Get occlusion periods
        occlusion_periods = self.track_analyzer.get_occlusion_periods()

        # Ensure frames are aligned
        max_frame = min(max(naive_df['frame'].max(), unc_df['frame'].max()), 900)
        frames = range(1, max_frame + 1)

        # 1. Top plot: Naive Detection Confidence + Occlusions
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_detection_confidence(ax1, naive_df, occlusion_periods)

        # 2. Detection Success (Binary)
        ax2 = fig.add_subplot(gs[1, :])
        self._plot_detection_success(ax2, naive_df, occlusion_periods)

        # 3. Aleatoric Uncertainty
        ax3 = fig.add_subplot(gs[2, :])
        self._plot_uncertainty(ax3, unc_df, occlusion_periods)

        # 4. Combined: Detection Quality (IoU) vs Uncertainty
        ax4 = fig.add_subplot(gs[3, :])
        self._plot_iou_vs_uncertainty(ax4, naive_df, unc_df, occlusion_periods)

        # 5. Statistical comparison
        ax5 = fig.add_subplot(gs[4, 0])
        self._plot_statistics_comparison(ax5, naive_df, unc_df)

        # 6. Correlation analysis
        ax6 = fig.add_subplot(gs[4, 1])
        self._plot_correlation(ax6, naive_df, unc_df)

        # 7. Summary text
        ax7 = fig.add_subplot(gs[5, :])
        self._add_summary_text(ax7, naive_df, unc_df)

        # Overall title
        fig.suptitle('Combined Analysis: Naive Detection vs Aleatoric Uncertainty\nMOT17-11-FRCNN Track 25',
                     fontsize=16, fontweight='bold')

        plt.tight_layout()

        # Save figure
        save_path = self.viz_dir / "combined_analysis.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved combined visualization to {save_path}")
        return save_path

    def _plot_detection_confidence(self, ax, naive_df, occlusion_periods):
        """Plot detection confidence over time"""
        # Mark occlusion periods
        for start, end in occlusion_periods:
            ax.axvspan(start, end, alpha=0.2, color=self.colors['occluded'],
                      label='Occlusion' if start == occlusion_periods[0][0] else '')

        # Plot confidence
        frames = naive_df['frame'].values
        confidence = naive_df['match_confidence'].values
        ax.plot(frames, confidence, color=self.colors['confidence'], alpha=0.7,
                linewidth=1.5, label='Detection Confidence')

        # Add rolling mean
        window = 10
        if len(confidence) > window:
            rolling_mean = pd.Series(confidence).rolling(window=window, center=True).mean()
            ax.plot(frames, rolling_mean, 'k--', alpha=0.5, linewidth=1,
                   label=f'{window}-frame rolling mean')

        ax.set_ylabel('Confidence', fontsize=10)
        ax.set_title('Naive YOLO Detection Confidence', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([frames[0], frames[-1]])
        ax.set_ylim([0, 1.05])

    def _plot_detection_success(self, ax, naive_df, occlusion_periods):
        """Plot binary detection success"""
        for start, end in occlusion_periods:
            ax.axvspan(start, end, alpha=0.2, color=self.colors['occluded'])

        frames = naive_df['frame'].values
        detected = naive_df['detection_found'].astype(int).values

        # Create filled area plot
        ax.fill_between(frames, 0, detected, color=self.colors['detection'],
                       alpha=0.5, label='Track Detected')
        ax.plot(frames, detected, color=self.colors['detection'], alpha=0.7, linewidth=0.5)

        ax.set_ylabel('Detected', fontsize=10)
        ax.set_title('Track 25 Detection Success (Binary)', fontsize=11, fontweight='bold')
        ax.set_ylim([-0.05, 1.05])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Not Found', 'Found'])
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([frames[0], frames[-1]])

    def _plot_uncertainty(self, ax, unc_df, occlusion_periods):
        """Plot aleatoric uncertainty"""
        for start, end in occlusion_periods:
            ax.axvspan(start, end, alpha=0.2, color=self.colors['occluded'])

        frames = unc_df['frame'].values
        uncertainty = unc_df['uncertainty'].values

        ax.plot(frames, uncertainty, color=self.colors['uncertainty'],
                alpha=0.7, linewidth=1.5, label='Aleatoric Uncertainty')

        # Add threshold lines
        mean_unc = np.mean(uncertainty)
        std_unc = np.std(uncertainty)
        ax.axhline(y=mean_unc, color='r', linestyle='--', alpha=0.5, label=f'Mean ({mean_unc:.1f})')
        ax.axhline(y=mean_unc + std_unc, color='orange', linestyle=':', alpha=0.5,
                  label=f'+1 STD ({mean_unc + std_unc:.1f})')

        ax.set_ylabel('Uncertainty', fontsize=10)
        ax.set_title('Aleatoric Uncertainty (MC Dropout + TTA)', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([frames[0], frames[-1]])

    def _plot_iou_vs_uncertainty(self, ax, naive_df, unc_df, occlusion_periods):
        """Plot IoU and uncertainty together"""
        for start, end in occlusion_periods:
            ax.axvspan(start, end, alpha=0.2, color=self.colors['occluded'])

        # Merge dataframes on frame
        merged = pd.merge(naive_df[['frame', 'match_iou']],
                         unc_df[['frame', 'uncertainty']],
                         on='frame', how='inner')

        frames = merged['frame'].values

        # Normalize for dual y-axis
        ax2 = ax.twinx()

        # Plot IoU
        l1 = ax.plot(frames, merged['match_iou'].values, 'g-', alpha=0.7,
                    linewidth=1.5, label='Detection IoU')
        ax.set_ylabel('IoU with GT', color='g', fontsize=10)
        ax.tick_params(axis='y', labelcolor='g')

        # Plot uncertainty
        l2 = ax2.plot(frames, merged['uncertainty'].values, 'b-', alpha=0.7,
                     linewidth=1.5, label='Uncertainty')
        ax2.set_ylabel('Uncertainty', color='b', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='b')

        # Combine legends
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right', fontsize=9)

        ax.set_xlabel('Frame Number', fontsize=10)
        ax.set_title('Detection Quality vs Uncertainty', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([frames[0], frames[-1]])

    def _plot_statistics_comparison(self, ax, naive_df, unc_df):
        """Plot statistical comparison by period"""
        # Calculate statistics by classification
        stats = []
        classifications = ['occluded', 'clean', 'early_recovery']

        for cls in classifications:
            naive_cls = naive_df[naive_df['classification'] == cls]
            unc_cls = unc_df[unc_df['frame'].isin(naive_cls['frame'])]

            if len(naive_cls) > 0:
                stats.append({
                    'period': cls.replace('_', '\n'),
                    'detection_rate': naive_cls['detection_found'].mean(),
                    'avg_confidence': naive_cls['match_confidence'].mean(),
                    'avg_uncertainty': unc_cls['uncertainty'].mean() if len(unc_cls) > 0 else 0
                })

        if not stats:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            return

        stats_df = pd.DataFrame(stats)

        # Create grouped bar plot
        x = np.arange(len(stats_df))
        width = 0.25

        ax.bar(x - width, stats_df['detection_rate'], width, label='Detection Rate',
               color='green', alpha=0.7)
        ax.bar(x, stats_df['avg_confidence'], width, label='Avg Confidence',
               color='blue', alpha=0.7)

        # Normalize uncertainty for visualization
        norm_unc = stats_df['avg_uncertainty'] / stats_df['avg_uncertainty'].max()
        ax.bar(x + width, norm_unc, width, label='Normalized Uncertainty',
               color='red', alpha=0.7)

        ax.set_xlabel('Period', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_title('Statistics by Period', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(stats_df['period'])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_correlation(self, ax, naive_df, unc_df):
        """Plot correlation between detection quality and uncertainty"""
        # Merge dataframes
        merged = pd.merge(
            naive_df[['frame', 'match_iou', 'match_confidence', 'detection_found']],
            unc_df[['frame', 'uncertainty']],
            on='frame', how='inner'
        )

        # Only use frames where detection was found
        detected = merged[merged['detection_found']]

        if len(detected) > 10:
            # Scatter plot
            scatter = ax.scatter(detected['match_iou'], detected['uncertainty'],
                               c=detected['match_confidence'], cmap='viridis',
                               alpha=0.6, s=20)

            # Add trend line
            z = np.polyfit(detected['match_iou'], detected['uncertainty'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(detected['match_iou'].min(), detected['match_iou'].max(), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.5, linewidth=2)

            # Calculate correlation
            correlation = detected['match_iou'].corr(detected['uncertainty'])
            ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                   transform=ax.transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.colorbar(scatter, ax=ax, label='Confidence')
            ax.set_xlabel('Detection IoU', fontsize=10)
            ax.set_ylabel('Uncertainty', fontsize=10)
            ax.set_title('IoU vs Uncertainty Correlation', fontsize=11, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Insufficient data for correlation',
                   ha='center', va='center')

        ax.grid(True, alpha=0.3)

    def _add_summary_text(self, ax, naive_df, unc_df):
        """Add summary statistics text"""
        ax.axis('off')

        # Calculate key metrics
        occluded_frames = naive_df[naive_df['is_occluded']]
        clean_frames = naive_df[~naive_df['is_occluded']]

        summary_text = "KEY FINDINGS:\n"
        summary_text += "-" * 80 + "\n"

        # Detection statistics
        if len(occluded_frames) > 0:
            occ_detect_rate = occluded_frames['detection_found'].mean()
            summary_text += f"• Detection rate during occlusion: {occ_detect_rate:.1%}\n"

        if len(clean_frames) > 0:
            clean_detect_rate = clean_frames['detection_found'].mean()
            summary_text += f"• Detection rate during clean periods: {clean_detect_rate:.1%}\n"

        # Uncertainty statistics
        occ_unc_frames = unc_df[unc_df['is_occluded']]
        clean_unc_frames = unc_df[~unc_df['is_occluded']]

        if len(occ_unc_frames) > 0 and len(clean_unc_frames) > 0:
            occ_unc_mean = occ_unc_frames['uncertainty'].mean()
            clean_unc_mean = clean_unc_frames['uncertainty'].mean()
            unc_ratio = occ_unc_mean / clean_unc_mean if clean_unc_mean > 0 else 0

            summary_text += f"• Uncertainty during occlusion: {occ_unc_mean:.1f} (±{occ_unc_frames['uncertainty'].std():.1f})\n"
            summary_text += f"• Uncertainty during clean: {clean_unc_mean:.1f} (±{clean_unc_frames['uncertainty'].std():.1f})\n"
            summary_text += f"• Uncertainty ratio (occluded/clean): {unc_ratio:.2f}x\n"

        # Add insight
        summary_text += "\nINSIGHT: "
        if unc_ratio > 5:
            summary_text += "Strong correlation between occlusion and uncertainty - aleatoric uncertainty effectively captures data quality degradation"
        elif unc_ratio > 2:
            summary_text += "Moderate correlation between occlusion and uncertainty - system responds to visibility changes"
        else:
            summary_text += "Weak correlation - investigate uncertainty quantification methodology"

        ax.text(0.05, 0.5, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='center',
               fontfamily='monospace')

    def create_paper_ready_plot(self, naive_df: pd.DataFrame, unc_df: pd.DataFrame):
        """Create publication-ready visualization for paper"""
        # Set publication style
        plt.style.use('seaborn-paper')
        sns.set_palette("husl")

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # Get occlusion periods
        occlusion_periods = self.track_analyzer.get_occlusion_periods()

        # Common frame range
        max_frame = min(naive_df['frame'].max(), unc_df['frame'].max())
        frames = range(1, max_frame + 1)

        # 1. Detection Quality
        ax1 = axes[0]
        for start, end in occlusion_periods[:3]:  # Show first 3 occlusions
            ax1.axvspan(start, end, alpha=0.15, color='red')

        ax1.plot(naive_df['frame'], naive_df['match_iou'], 'b-', alpha=0.8, linewidth=1.2)
        ax1.set_ylabel('IoU with GT', fontsize=11)
        ax1.set_title('Tracking Performance and Uncertainty During Occlusions', fontsize=13, pad=10)
        ax1.grid(True, alpha=0.3, linestyle=':')
        ax1.set_ylim([0, 1.05])

        # 2. Detection Confidence
        ax2 = axes[1]
        for start, end in occlusion_periods[:3]:
            ax2.axvspan(start, end, alpha=0.15, color='red')

        ax2.plot(naive_df['frame'], naive_df['match_confidence'], 'g-', alpha=0.8, linewidth=1.2)
        ax2.set_ylabel('Confidence', fontsize=11)
        ax2.grid(True, alpha=0.3, linestyle=':')
        ax2.set_ylim([0, 1.05])

        # 3. Aleatoric Uncertainty
        ax3 = axes[2]
        for start, end in occlusion_periods[:3]:
            ax3.axvspan(start, end, alpha=0.15, color='red',
                       label='Occlusion' if start == occlusion_periods[0][0] else '')

        ax3.plot(unc_df['frame'], unc_df['uncertainty'], 'r-', alpha=0.8, linewidth=1.2,
                label='Aleatoric Uncertainty')
        ax3.set_ylabel('Uncertainty', fontsize=11)
        ax3.set_xlabel('Frame Number', fontsize=11)
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, alpha=0.3, linestyle=':')

        # Adjust layout
        plt.tight_layout()

        # Save paper-ready figure
        paper_path = self.viz_dir / "paper_figure.pdf"
        plt.savefig(paper_path, dpi=300, bbox_inches='tight', format='pdf')

        png_path = self.viz_dir / "paper_figure.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight')

        plt.close()

        logger.info(f"Saved paper-ready figure to {paper_path}")
        return paper_path


def main():
    """Create combined visualizations"""
    visualizer = CombinedVisualizer()

    try:
        # Load results
        naive_df, unc_df = visualizer.load_latest_results()

        # Create comprehensive plot
        comp_path = visualizer.create_comprehensive_plot(naive_df, unc_df)
        print(f"Created comprehensive visualization: {comp_path}")

        # Create paper-ready plot
        paper_path = visualizer.create_paper_ready_plot(naive_df, unc_df)
        print(f"Created paper-ready figure: {paper_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run naive_detection_analysis.py and the uncertainty pipeline first.")


if __name__ == "__main__":
    main()