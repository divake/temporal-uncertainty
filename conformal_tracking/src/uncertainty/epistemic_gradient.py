"""
Inter-Layer Feature Divergence for Epistemic Uncertainty

This module implements Method 3: Gradient-based epistemic uncertainty
by measuring how features diverge across YOLO layers.

Author: Enhanced CACD Team
Date: 2025-11-11
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Tuple
from scipy.spatial.distance import cosine


class GradientDivergenceDetector:
    """
    Detect epistemic uncertainty via inter-layer feature divergence

    Measures how smoothly features evolve through the network.
    Large divergence = Model struggling = High epistemic uncertainty

    Uses features from multiple YOLO layers:
    - Layer 4:  64D (early)
    - Layer 9:  256D (middle)
    - Layer 15: 64D (late)
    - Layer 21: 256D (final)
    """

    def __init__(self,
                 layer_pairs: list = [(4, 9), (9, 15), (15, 21)],
                 aggregation: str = 'mean',
                 verbose: bool = True):
        """
        Initialize gradient divergence detector

        Args:
            layer_pairs: Which layer pairs to compare
            aggregation: How to combine divergences ('mean', 'max', 'weighted')
            verbose: Print progress
        """
        self.layer_pairs = layer_pairs
        self.aggregation = aggregation
        self.verbose = verbose

        # Will be set during fitting
        self.is_fitted = False
        self.divergence_min = None
        self.divergence_max = None
        self.layer_weights = None  # For weighted aggregation

    def fit(self, X_cal_layers: Dict[int, np.ndarray],
            mahalanobis_model=None,
            save_dir: Optional[Path] = None) -> 'GradientDivergenceDetector':
        """
        Fit the gradient divergence detector on calibration data

        Args:
            X_cal_layers: Dictionary mapping layer_id -> features [N, D]
                         e.g., {4: [N, 64], 9: [N, 256], 15: [N, 64], 21: [N, 256]}
            mahalanobis_model: Not used, for API consistency
            save_dir: Directory to save diagnostic plots

        Returns:
            self
        """
        if self.verbose:
            print("\n" + "="*60)
            print("FITTING GRADIENT DIVERGENCE DETECTOR")
            print("="*60)
            n_samples = len(X_cal_layers[list(X_cal_layers.keys())[0]])
            print(f"Calibration samples: {n_samples}")
            print(f"Layer pairs: {self.layer_pairs}")

        # Compute divergences on calibration set
        divergences = []
        for i in range(min(500, n_samples)):  # Sample for efficiency
            div = self._compute_divergence(
                {layer: X_cal_layers[layer][i] for layer in X_cal_layers.keys()}
            )
            divergences.append(div)

        divergences = np.array(divergences)

        # Store normalization bounds
        self.divergence_min = np.percentile(divergences, 5)  # 5th percentile
        self.divergence_max = np.percentile(divergences, 95)  # 95th percentile

        # Compute layer-specific weights if using weighted aggregation
        if self.aggregation == 'weighted':
            # Weight by variance contribution
            div_per_pair = []
            for pair in self.layer_pairs:
                pair_divs = []
                for i in range(len(divergences)):
                    layers_dict = {layer: X_cal_layers[layer][i] for layer in X_cal_layers.keys()}
                    div = self._compute_single_pair_divergence(
                        layers_dict[pair[0]], layers_dict[pair[1]]
                    )
                    pair_divs.append(div)
                div_per_pair.append(np.std(pair_divs))

            # Normalize to sum to 1
            self.layer_weights = np.array(div_per_pair)
            self.layer_weights = self.layer_weights / self.layer_weights.sum()

        self.is_fitted = True

        if self.verbose:
            print(f"\nCalibration Divergence Statistics:")
            print(f"  Mean: {divergences.mean():.4f}")
            print(f"  Std:  {divergences.std():.4f}")
            print(f"  Range: [{self.divergence_min:.4f}, {self.divergence_max:.4f}]")

            if self.layer_weights is not None:
                print(f"\nLayer Pair Weights:")
                for pair, weight in zip(self.layer_pairs, self.layer_weights):
                    print(f"  Layers {pair[0]}-{pair[1]}: {weight:.3f}")

        # Generate diagnostic plots
        if save_dir is not None:
            self._plot_calibration_diagnostics(divergences, save_dir)

        if self.verbose:
            print("="*60)
            print("GRADIENT FITTING COMPLETE ✓")
            print("="*60 + "\n")

        return self

    def predict(self, X_test_layers: Dict[int, np.ndarray],
                return_diagnostics: bool = False) -> Dict[str, np.ndarray]:
        """
        Predict epistemic uncertainty for test samples

        Args:
            X_test_layers: Dictionary of test features per layer
            return_diagnostics: Return additional diagnostic info

        Returns:
            Dictionary with:
                - 'epistemic': Normalized uncertainty [0, 1]
                - 'raw_divergence': Raw divergence values (if diagnostics)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        n_test = len(X_test_layers[list(X_test_layers.keys())[0]])

        divergences = []
        for i in range(n_test):
            div = self._compute_divergence(
                {layer: X_test_layers[layer][i] for layer in X_test_layers.keys()}
            )
            divergences.append(div)

        divergences = np.array(divergences)

        # Normalize to [0, 1]
        epistemic = (divergences - self.divergence_min) / (
            self.divergence_max - self.divergence_min + 1e-10
        )
        epistemic = np.clip(epistemic, 0, 1)

        results = {'epistemic': epistemic}

        if return_diagnostics:
            results['raw_divergence'] = divergences

        return results

    def _compute_divergence(self, features_dict: Dict[int, np.ndarray]) -> float:
        """
        Compute divergence for a single sample across all layer pairs

        Args:
            features_dict: {layer_id: feature_vector}

        Returns:
            Aggregated divergence value
        """
        divergences = []

        for layer1, layer2 in self.layer_pairs:
            f1 = features_dict[layer1]
            f2 = features_dict[layer2]

            div = self._compute_single_pair_divergence(f1, f2)
            divergences.append(div)

        # Aggregate divergences
        if self.aggregation == 'mean':
            return np.mean(divergences)
        elif self.aggregation == 'max':
            return np.max(divergences)
        elif self.aggregation == 'weighted':
            return np.sum(np.array(divergences) * self.layer_weights)
        else:
            return np.mean(divergences)

    def _compute_single_pair_divergence(self, f1: np.ndarray, f2: np.ndarray) -> float:
        """
        Compute divergence between two feature vectors

        Uses 1 - cosine_similarity, which is scale-invariant
        and works for different dimensions

        Args:
            f1: Features from layer 1 (any dimension)
            f2: Features from layer 2 (any dimension)

        Returns:
            Divergence in [0, 2] (0 = same direction, 2 = opposite)
        """
        # Normalize vectors (L2 norm)
        f1_norm = f1 / (np.linalg.norm(f1) + 1e-10)
        f2_norm = f2 / (np.linalg.norm(f2) + 1e-10)

        # Cosine similarity (works for different dimensions via padding)
        # Pad shorter vector with zeros
        if len(f1_norm) < len(f2_norm):
            f1_padded = np.pad(f1_norm, (0, len(f2_norm) - len(f1_norm)))
            f2_padded = f2_norm
        elif len(f2_norm) < len(f1_norm):
            f1_padded = f1_norm
            f2_padded = np.pad(f2_norm, (0, len(f1_norm) - len(f2_norm)))
        else:
            f1_padded = f1_norm
            f2_padded = f2_norm

        # Cosine similarity
        cos_sim = np.dot(f1_padded, f2_padded)

        # Convert to divergence: 1 - cos_sim
        # cos_sim = 1 → same direction → divergence = 0
        # cos_sim = 0 → orthogonal → divergence = 1
        # cos_sim = -1 → opposite → divergence = 2
        divergence = 1 - cos_sim

        return divergence

    def _plot_calibration_diagnostics(self, divergences: np.ndarray, save_dir: Path):
        """Generate diagnostic plots for calibration"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Gradient Divergence - Calibration Diagnostics', fontsize=14, fontweight='bold')

        # 1. Divergence distribution
        ax1 = axes[0, 0]
        ax1.hist(divergences, bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax1.axvline(divergences.mean(), color='red', linestyle='--',
                   label=f'Mean: {divergences.mean():.3f}')
        ax1.axvline(self.divergence_min, color='green', linestyle='--',
                   label=f'5th pct: {self.divergence_min:.3f}')
        ax1.axvline(self.divergence_max, color='orange', linestyle='--',
                   label=f'95th pct: {self.divergence_max:.3f}')
        ax1.set_xlabel('Raw Divergence')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Divergence Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Normalized divergence
        ax2 = axes[0, 1]
        normalized = (divergences - self.divergence_min) / (self.divergence_max - self.divergence_min)
        normalized = np.clip(normalized, 0, 1)
        ax2.hist(normalized, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(normalized.mean(), color='red', linestyle='--',
                   label=f'Mean: {normalized.mean():.3f}')
        ax2.set_xlabel('Normalized Divergence [0, 1]')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Normalized Epistemic Uncertainty')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Layer pair contributions (if weighted)
        ax3 = axes[1, 0]
        if self.layer_weights is not None:
            pair_labels = [f"{p[0]}-{p[1]}" for p in self.layer_pairs]
            ax3.bar(pair_labels, self.layer_weights, alpha=0.7, color='blue')
            ax3.set_xlabel('Layer Pair')
            ax3.set_ylabel('Weight')
            ax3.set_title('Layer Pair Weights')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Equal weighting used', ha='center', va='center')
            ax3.set_title('Aggregation: ' + self.aggregation)

        # 4. Statistics summary
        ax4 = axes[1, 1]
        ax4.axis('off')

        stats_text = f"""
        GRADIENT DIVERGENCE STATISTICS
        ==============================

        Calibration samples: {len(divergences)}
        Layer pairs: {len(self.layer_pairs)}

        Raw Divergence:
          Mean: {divergences.mean():.4f}
          Std:  {divergences.std():.4f}
          Min:  {divergences.min():.4f}
          Max:  {divergences.max():.4f}

        Normalization:
          Min bound: {self.divergence_min:.4f}
          Max bound: {self.divergence_max:.4f}

        Aggregation: {self.aggregation}
        """

        ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center')

        plt.tight_layout()
        save_path = save_dir / 'gradient_calibration_diagnostics.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {save_path}")

    def plot_test_diagnostics(self, results: Dict[str, np.ndarray],
                              save_dir: Path, prefix: str = ""):
        """
        Generate diagnostic plots for test predictions

        Args:
            results: Dictionary with epistemic and optionally diagnostics
            save_dir: Directory to save plots
            prefix: Prefix for plot filename
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        epistemic = results['epistemic']

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Gradient Divergence - Test Diagnostics', fontsize=14, fontweight='bold')

        # 1. Epistemic uncertainty distribution
        ax1 = axes[0]
        ax1.hist(epistemic, bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax1.axvline(epistemic.mean(), color='red', linestyle='--',
                   label=f'Mean: {epistemic.mean():.3f}')
        ax1.set_xlabel('Epistemic Uncertainty')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Epistemic Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Statistics text
        ax2 = axes[1]
        ax2.axis('off')

        stats_text = f"""
        GRADIENT EPISTEMIC STATISTICS
        =============================
        Test samples: {len(epistemic)}

        Epistemic Uncertainty:
          Mean: {epistemic.mean():.4f}
          Std:  {epistemic.std():.4f}
          Min:  {epistemic.min():.4f}
          Max:  {epistemic.max():.4f}

        Percentiles:
          25th: {np.percentile(epistemic, 25):.4f}
          50th: {np.percentile(epistemic, 50):.4f}
          75th: {np.percentile(epistemic, 75):.4f}
          95th: {np.percentile(epistemic, 95):.4f}
        """

        ax2.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center')

        plt.tight_layout()
        save_path = save_dir / f'{prefix}gradient_test_diagnostics.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        if self.verbose:
            print(f"  Saved: {save_path}")
