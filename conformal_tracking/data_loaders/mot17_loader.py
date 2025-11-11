"""
MOT17 Dataset Loader for Uncertainty Quantification

This module provides dataset-specific loading, filtering, and splitting
functionality for MOT17 YOLO cache files.

Author: Enhanced CACD Team
Date: 2025-11-10
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import sys

# Add parent directory to path to import data_loader
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from data_loader import YOLOCacheLoader


class MOT17DataLoader:
    """
    MOT17-specific data loader with filtering and splitting.

    Handles:
    - Loading YOLO cache (features, IoUs, confidences)
    - Filtering by confidence threshold
    - Computing conformity scores (1 - IoU)
    - Splitting into calibration/test sets
    """

    def __init__(self,
                 cache_path: Path,
                 layer_id: int = 21,
                 load_all_layers: bool = False,
                 conf_threshold: float = 0.5,
                 split_ratio: float = 0.5,
                 random_seed: int = 42):
        """
        Initialize MOT17 data loader.

        Args:
            cache_path: Path to .npz cache file
            layer_id: YOLO layer to use for features (default: 21)
            conf_threshold: Minimum confidence threshold (default: 0.5)
            split_ratio: Fraction for calibration set (default: 0.5)
            random_seed: Random seed for reproducibility (default: 42)
        """
        self.cache_path = Path(cache_path)
        self.layer_id = layer_id
        self.load_all_layers = load_all_layers
        self.conf_threshold = conf_threshold
        self.split_ratio = split_ratio
        self.random_seed = random_seed

        # Load cache
        print(f"\n{'='*60}")
        print(f"LOADING MOT17 CACHE")
        print(f"{'='*60}")
        print(f"Path: {self.cache_path}")
        print(f"Sequence: {self.cache_path.stem}")

        self.cache = YOLOCacheLoader(self.cache_path)

        # Load raw data
        self._load_raw_data()

        # Apply filters
        self._apply_filters()

        # Split data
        self._split_data()

        print(f"{'='*60}")
        print(f"LOADING COMPLETE ✓")
        print(f"{'='*60}\n")

    def _load_raw_data(self):
        """Load raw data from cache."""
        print(f"\n1. Loading raw data...")

        # Load features (Layer 21 by default)
        self.features_raw = self.cache.get_features(layer_id=self.layer_id)  # [N, D]
        N, D = self.features_raw.shape

        # Load all layers if requested
        if self.load_all_layers:
            self.features_all_layers = {}
            for layer in [4, 9, 15, 21]:
                self.features_all_layers[layer] = self.cache.get_features(layer_id=layer)
            print(f"   Loaded all layers: {list(self.features_all_layers.keys())}")
        else:
            self.features_all_layers = None

        # Load IoUs
        self.ious_raw = self.cache.get_ious()  # [N_matched]

        # Load confidences
        self.confidences_raw = self.cache.get_confidences()  # [N]

        # Get matched mask (True for matched detections)
        self.matched_mask = self.cache.get_matched_mask()  # [N]

        print(f"   Total detections: {N}")
        print(f"   Feature dimension: {D}")
        print(f"   Matched detections: {np.sum(self.matched_mask)} ({np.sum(self.matched_mask)/N*100:.1f}%)")
        print(f"   Unmatched detections: {np.sum(~self.matched_mask)} ({np.sum(~self.matched_mask)/N*100:.1f}%)")

    def _apply_filters(self):
        """Apply confidence threshold and extract matched data."""
        print(f"\n2. Applying filters...")
        print(f"   Confidence threshold: {self.conf_threshold}")

        # Filter: Only matched detections
        features_matched = self.features_raw[self.matched_mask]
        confidences_matched = self.confidences_raw[self.matched_mask]
        ious_matched = self.ious_raw

        print(f"   After matching filter: {len(features_matched)} samples")

        # Filter: Confidence >= threshold
        conf_filter = confidences_matched >= self.conf_threshold
        self.features = features_matched[conf_filter]
        self.confidences = confidences_matched[conf_filter]
        self.ious = ious_matched[conf_filter]

        print(f"   After confidence filter: {len(self.features)} samples")

        # Compute conformity scores (1 - IoU)
        self.conformity_scores = 1.0 - self.ious

        # Statistics
        print(f"\n   Feature statistics:")
        print(f"     Norm - Mean: {np.mean(np.linalg.norm(self.features, axis=1)):.4f}")
        print(f"     Norm - Std:  {np.std(np.linalg.norm(self.features, axis=1)):.4f}")

        print(f"\n   IoU statistics:")
        print(f"     Mean: {np.mean(self.ious):.4f}")
        print(f"     Std:  {np.std(self.ious):.4f}")
        print(f"     Min:  {np.min(self.ious):.4f}")
        print(f"     Max:  {np.max(self.ious):.4f}")

        print(f"\n   Conformity score statistics:")
        print(f"     Mean: {np.mean(self.conformity_scores):.4f}")
        print(f"     Std:  {np.std(self.conformity_scores):.4f}")
        print(f"     Min:  {np.min(self.conformity_scores):.4f}")
        print(f"     Max:  {np.max(self.conformity_scores):.4f}")

        print(f"\n   Confidence statistics:")
        print(f"     Mean: {np.mean(self.confidences):.4f}")
        print(f"     Std:  {np.std(self.confidences):.4f}")
        print(f"     Min:  {np.min(self.confidences):.4f}")
        print(f"     Max:  {np.max(self.confidences):.4f}")

    def _split_data(self):
        """Split data into calibration and test sets."""
        print(f"\n3. Splitting data...")
        print(f"   Split ratio (calibration): {self.split_ratio}")
        print(f"   Random seed: {self.random_seed}")

        N = len(self.features)
        N_cal = int(N * self.split_ratio)
        N_test = N - N_cal

        # Shuffle indices
        np.random.seed(self.random_seed)
        indices = np.random.permutation(N)

        # Split indices
        cal_indices = indices[:N_cal]
        test_indices = indices[N_cal:]

        # Split all arrays
        self.X_cal = self.features[cal_indices]
        self.X_test = self.features[test_indices]

        self.conformity_cal = self.conformity_scores[cal_indices]
        self.conformity_test = self.conformity_scores[test_indices]

        self.ious_cal = self.ious[cal_indices]
        self.ious_test = self.ious[test_indices]

        self.conf_cal = self.confidences[cal_indices]
        self.conf_test = self.confidences[test_indices]

        # Split multi-layer features if loaded
        if self.features_all_layers is not None:
            self.X_cal_layers = {layer: features[cal_indices]
                                for layer, features in self.features_all_layers.items()}
            self.X_test_layers = {layer: features[test_indices]
                                 for layer, features in self.features_all_layers.items()}
        else:
            self.X_cal_layers = None
            self.X_test_layers = None

        print(f"\n   Calibration set: {N_cal} samples")
        print(f"   Test set:        {N_test} samples")

        # Verify splits are similar
        print(f"\n   Calibration statistics:")
        print(f"     IoU Mean: {np.mean(self.ious_cal):.4f}")
        print(f"     Conformity Mean: {np.mean(self.conformity_cal):.4f}")

        print(f"\n   Test statistics:")
        print(f"     IoU Mean: {np.mean(self.ious_test):.4f}")
        print(f"     Conformity Mean: {np.mean(self.conformity_test):.4f}")

    def get_calibration_data(self) -> Dict[str, np.ndarray]:
        """
        Get calibration data.

        Returns:
            Dictionary with:
                - 'features': Features [N_cal, D]
                - 'conformity_scores': 1 - IoU [N_cal]
                - 'ious': IoU values [N_cal]
                - 'confidences': Confidence scores [N_cal]
        """
        return {
            'features': self.X_cal,
            'conformity_scores': self.conformity_cal,
            'ious': self.ious_cal,
            'confidences': self.conf_cal
        }

    def get_test_data(self) -> Dict[str, np.ndarray]:
        """
        Get test data.

        Returns:
            Dictionary with:
                - 'features': Features [N_test, D]
                - 'conformity_scores': 1 - IoU [N_test]
                - 'ious': IoU values [N_test]
                - 'confidences': Confidence scores [N_test]
        """
        return {
            'features': self.X_test,
            'conformity_scores': self.conformity_test,
            'ious': self.ious_test,
            'confidences': self.conf_test
        }

    def get_calibration_layers(self) -> Optional[Dict[int, np.ndarray]]:
        """Get calibration data from all layers (if loaded)."""
        return self.X_cal_layers

    def get_test_layers(self) -> Optional[Dict[int, np.ndarray]]:
        """Get test data from all layers (if loaded)."""
        return self.X_test_layers

    def plot_data_distributions(self, save_dir: Optional[Path] = None, prefix: str = ""):
        """
        Plot data distributions for visualization.

        Args:
            save_dir: Directory to save plots (if None, don't save)
            prefix: Prefix for saved filenames
        """
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Plot 1: Feature norms
        norms_cal = np.linalg.norm(self.X_cal, axis=1)
        norms_test = np.linalg.norm(self.X_test, axis=1)

        axes[0, 0].hist(norms_cal, bins=50, alpha=0.6, label='Calibration', edgecolor='black')
        axes[0, 0].hist(norms_test, bins=50, alpha=0.6, label='Test', edgecolor='black')
        axes[0, 0].set_xlabel('Feature Norm', fontsize=12)
        axes[0, 0].set_ylabel('Count', fontsize=12)
        axes[0, 0].set_title('Feature Norms Distribution', fontsize=13, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: IoU distribution
        axes[0, 1].hist(self.ious_cal, bins=50, alpha=0.6, label='Calibration', edgecolor='black')
        axes[0, 1].hist(self.ious_test, bins=50, alpha=0.6, label='Test', edgecolor='black')
        axes[0, 1].set_xlabel('IoU', fontsize=12)
        axes[0, 1].set_ylabel('Count', fontsize=12)
        axes[0, 1].set_title('IoU Distribution', fontsize=13, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Conformity score distribution
        axes[0, 2].hist(self.conformity_cal, bins=50, alpha=0.6, label='Calibration', edgecolor='black')
        axes[0, 2].hist(self.conformity_test, bins=50, alpha=0.6, label='Test', edgecolor='black')
        axes[0, 2].set_xlabel('Conformity Score (1 - IoU)', fontsize=12)
        axes[0, 2].set_ylabel('Count', fontsize=12)
        axes[0, 2].set_title('Conformity Score Distribution', fontsize=13, fontweight='bold')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Plot 4: Confidence distribution
        axes[1, 0].hist(self.conf_cal, bins=50, alpha=0.6, label='Calibration', edgecolor='black')
        axes[1, 0].hist(self.conf_test, bins=50, alpha=0.6, label='Test', edgecolor='black')
        axes[1, 0].set_xlabel('Confidence', fontsize=12)
        axes[1, 0].set_ylabel('Count', fontsize=12)
        axes[1, 0].set_title('Confidence Distribution', fontsize=13, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 5: Confidence vs IoU scatter
        # Subsample for clarity (max 2000 points)
        n_plot = min(2000, len(self.ious_test))
        idx_plot = np.random.choice(len(self.ious_test), n_plot, replace=False)

        axes[1, 1].scatter(self.conf_test[idx_plot], self.ious_test[idx_plot],
                          alpha=0.3, s=10, c='blue')
        axes[1, 1].set_xlabel('Confidence', fontsize=12)
        axes[1, 1].set_ylabel('IoU', fontsize=12)
        axes[1, 1].set_title(f'Confidence vs IoU (Test, n={n_plot})', fontsize=13, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        # Add correlation
        from scipy.stats import pearsonr
        corr, _ = pearsonr(self.conf_test, self.ious_test)
        axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.4f}',
                       transform=axes[1, 1].transAxes,
                       fontsize=11, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Plot 6: IoU quality breakdown
        excellent = np.sum(self.ious_test >= 0.7) / len(self.ious_test) * 100
        good = np.sum((self.ious_test >= 0.5) & (self.ious_test < 0.7)) / len(self.ious_test) * 100
        poor = np.sum(self.ious_test < 0.5) / len(self.ious_test) * 100

        categories = ['Excellent\n(IoU≥0.7)', 'Good\n(0.5≤IoU<0.7)', 'Poor\n(IoU<0.5)']
        percentages = [excellent, good, poor]
        colors = ['green', 'orange', 'red']

        axes[1, 2].bar(categories, percentages, color=colors, alpha=0.7, edgecolor='black')
        axes[1, 2].set_ylabel('Percentage (%)', fontsize=12)
        axes[1, 2].set_title('IoU Quality Breakdown (Test)', fontsize=13, fontweight='bold')
        axes[1, 2].grid(True, alpha=0.3, axis='y')

        # Add percentage labels on bars
        for i, (cat, pct) in enumerate(zip(categories, percentages)):
            axes[1, 2].text(i, pct + 2, f'{pct:.1f}%',
                           ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()

        if save_dir is not None:
            save_path = save_dir / f"{prefix}data_distributions.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n  Saved: {save_path}")
        else:
            plt.show()

        plt.close()

        print(f"\n  Data distribution plots complete ✓")
