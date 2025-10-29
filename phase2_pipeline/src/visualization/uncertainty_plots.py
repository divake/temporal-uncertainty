"""
Uncertainty Visualization Plots
Borrowed concepts from uncertainty-toolbox/uncertainty_toolbox/viz.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class UncertaintyVisualizer:
    """Create uncertainty visualizations"""

    def __init__(self, save_dir: str = None):
        """
        Initialize visualizer

        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_uncertainty_timeline(
        self,
        frame_numbers: np.ndarray,
        uncertainties: np.ndarray,
        occlusion_periods: List[Tuple[int, int]] = None,
        track_id: int = None,
        title: str = None,
        save_name: str = None
    ) -> plt.Figure:
        """
        Plot uncertainty over time with occlusion periods highlighted

        Args:
            frame_numbers: Frame numbers
            uncertainties: Uncertainty values
            occlusion_periods: List of (start, end) tuples for occlusions
            track_id: Track ID for title
            title: Custom title
            save_name: Filename to save

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(15, 6))

        # Main uncertainty line
        ax.plot(frame_numbers, uncertainties, 'b-', linewidth=2,
                label='Aleatoric Uncertainty', alpha=0.8)

        # Highlight occlusion periods
        if occlusion_periods:
            for start, end in occlusion_periods:
                ax.axvspan(start, end, alpha=0.3, color='red',
                          label='Occlusion' if start == occlusion_periods[0][0] else "")

        # Add horizontal lines for thresholds
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5,
                  label='High Uncertainty Threshold')
        ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5,
                  label='Low Uncertainty Threshold')

        # Labels and title
        ax.set_xlabel('Frame Number', fontsize=12)
        ax.set_ylabel('Uncertainty', fontsize=12)

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        elif track_id:
            ax.set_title(f'Aleatoric Uncertainty Timeline - Track {track_id}',
                        fontsize=14, fontweight='bold')

        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")

        return fig

    def plot_before_during_after(
        self,
        uncertainties_dict: Dict[str, np.ndarray],
        title: str = "Uncertainty Distribution",
        save_name: str = None
    ) -> plt.Figure:
        """
        Plot uncertainty distributions for different periods

        Args:
            uncertainties_dict: Dict with keys like 'before', 'during', 'after'
            title: Plot title
            save_name: Filename to save

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        periods = ['before', 'during', 'after']
        colors = ['green', 'red', 'blue']

        for idx, (period, color) in enumerate(zip(periods, colors)):
            if period in uncertainties_dict:
                data = uncertainties_dict[period]
                ax = axes[idx]

                # Histogram
                ax.hist(data, bins=20, color=color, alpha=0.6, edgecolor='black')
                ax.axvline(np.mean(data), color='black', linestyle='--',
                          linewidth=2, label=f'Mean: {np.mean(data):.3f}')

                ax.set_xlabel('Uncertainty', fontsize=11)
                ax.set_ylabel('Frequency', fontsize=11)
                ax.set_title(f'{period.capitalize()} Occlusion', fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")

        return fig

    def plot_recovery_curves(
        self,
        recovery_data: List[Dict],
        title: str = "Uncertainty Recovery After Occlusion",
        save_name: str = None
    ) -> plt.Figure:
        """
        Plot recovery curves after occlusion events

        Args:
            recovery_data: List of dicts with 'frames' and 'uncertainties'
            title: Plot title
            save_name: Filename to save

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        for idx, recovery in enumerate(recovery_data):
            frames = recovery['frames']
            uncertainties = recovery['uncertainties']

            # Normalize frames to start from 0
            norm_frames = np.array(frames) - frames[0]

            ax.plot(norm_frames, uncertainties, linewidth=2,
                   alpha=0.7, label=f'Occlusion {idx + 1}')

        ax.set_xlabel('Frames After Occlusion End', fontsize=12)
        ax.set_ylabel('Uncertainty', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")

        return fig

    def plot_uncertainty_heatmap(
        self,
        spatial_uncertainty: np.ndarray,
        frame_num: int = None,
        title: str = None,
        save_name: str = None
    ) -> plt.Figure:
        """
        Plot spatial uncertainty heatmap

        Args:
            spatial_uncertainty: 2D array of spatial uncertainties
            frame_num: Frame number for title
            title: Custom title
            save_name: Filename to save

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(spatial_uncertainty, cmap='hot', aspect='auto')
        plt.colorbar(im, ax=ax, label='Uncertainty')

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        elif frame_num:
            ax.set_title(f'Spatial Uncertainty Heatmap - Frame {frame_num}',
                        fontsize=14, fontweight='bold')

        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)

        plt.tight_layout()

        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")

        return fig

    def plot_calibration_curve(
        self,
        confidences: np.ndarray,
        accuracies: np.ndarray,
        num_bins: int = 10,
        title: str = "Calibration Curve",
        save_name: str = None
    ) -> plt.Figure:
        """
        Plot calibration curve (borrowed from uncertainty-toolbox)

        Args:
            confidences: Predicted confidences
            accuracies: Binary accuracy indicators
            num_bins: Number of bins
            title: Plot title
            save_name: Filename to save

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        # Compute calibration
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        bin_confidences = []
        bin_accuracies = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

            if in_bin.sum() > 0:
                bin_confidences.append(confidences[in_bin].mean())
                bin_accuracies.append(accuracies[in_bin].mean())

        # Plot calibration curve
        ax.plot(bin_confidences, bin_accuracies, 'bo-', linewidth=2,
               markersize=8, label='Calibration')

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')

        ax.set_xlabel('Mean Predicted Confidence', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")

        return fig

    def plot_uncertainty_components(
        self,
        bbox_uncertainties: np.ndarray,
        conf_uncertainties: np.ndarray,
        combined_uncertainties: np.ndarray,
        frame_numbers: np.ndarray,
        title: str = "Uncertainty Components",
        save_name: str = None
    ) -> plt.Figure:
        """
        Plot different uncertainty components

        Args:
            bbox_uncertainties: Bbox uncertainty values
            conf_uncertainties: Confidence uncertainty values
            combined_uncertainties: Combined uncertainty values
            frame_numbers: Frame numbers
            title: Plot title
            save_name: Filename to save

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

        # Bbox uncertainty
        axes[0].plot(frame_numbers, bbox_uncertainties, 'b-', linewidth=2)
        axes[0].set_ylabel('Bbox Uncertainty', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Bounding Box Uncertainty', fontsize=12)

        # Confidence uncertainty
        axes[1].plot(frame_numbers, conf_uncertainties, 'g-', linewidth=2)
        axes[1].set_ylabel('Confidence Uncertainty', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title('Confidence Score Uncertainty', fontsize=12)

        # Combined uncertainty
        axes[2].plot(frame_numbers, combined_uncertainties, 'r-', linewidth=2)
        axes[2].set_xlabel('Frame Number', fontsize=12)
        axes[2].set_ylabel('Combined Uncertainty', fontsize=11)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_title('Combined Aleatoric Uncertainty', fontsize=12)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")

        return fig

    def plot_summary_statistics(
        self,
        stats: Dict,
        title: str = "Uncertainty Analysis Summary",
        save_name: str = None
    ) -> plt.Figure:
        """
        Plot summary statistics

        Args:
            stats: Dictionary with statistics
            title: Plot title
            save_name: Filename to save

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Mean uncertainties by period
        if 'mean_by_period' in stats:
            periods = list(stats['mean_by_period'].keys())
            means = list(stats['mean_by_period'].values())

            axes[0, 0].bar(periods, means, color=['green', 'red', 'blue', 'orange'])
            axes[0, 0].set_ylabel('Mean Uncertainty', fontsize=11)
            axes[0, 0].set_title('Mean Uncertainty by Period', fontsize=12)
            axes[0, 0].grid(True, alpha=0.3)

        # Temporal correlation
        if 'temporal_correlation' in stats:
            lags = list(range(1, len(stats['temporal_correlation']) + 1))
            corr = stats['temporal_correlation']

            axes[0, 1].plot(lags, corr, 'bo-', linewidth=2, markersize=6)
            axes[0, 1].set_xlabel('Lag (frames)', fontsize=11)
            axes[0, 1].set_ylabel('Correlation', fontsize=11)
            axes[0, 1].set_title('Temporal Autocorrelation', fontsize=12)
            axes[0, 1].grid(True, alpha=0.3)

        # Recovery rate
        if 'recovery_rates' in stats:
            occlusion_ids = list(range(1, len(stats['recovery_rates']) + 1))
            rates = stats['recovery_rates']

            axes[1, 0].bar(occlusion_ids, rates, color='purple')
            axes[1, 0].set_xlabel('Occlusion Event', fontsize=11)
            axes[1, 0].set_ylabel('Recovery Rate', fontsize=11)
            axes[1, 0].set_title('Uncertainty Recovery Rates', fontsize=12)
            axes[1, 0].grid(True, alpha=0.3)

        # Distribution summary
        if 'distribution' in stats:
            data = stats['distribution']
            axes[1, 1].boxplot(data)
            axes[1, 1].set_ylabel('Uncertainty', fontsize=11)
            axes[1, 1].set_title('Uncertainty Distribution', fontsize=12)
            axes[1, 1].grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")

        return fig


if __name__ == "__main__":
    # Test visualization
    import sys
    sys.path.append('/ssd_4TB/divake/temporal_uncertainty')

    # Create visualizer
    viz = UncertaintyVisualizer(save_dir="/tmp/test_plots")

    # Generate dummy data
    frames = np.arange(1, 101)
    uncertainties = np.random.random(100) * 0.5 + 0.2
    uncertainties[20:30] = np.random.random(10) * 0.3 + 0.6  # Spike during occlusion

    # Test timeline plot
    fig = viz.plot_uncertainty_timeline(
        frames, uncertainties,
        occlusion_periods=[(20, 30), (60, 65)],
        track_id=25,
        save_name="test_timeline.png"
    )

    print("Created test uncertainty timeline plot")

    plt.close('all')