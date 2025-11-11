"""
Conformal Prediction with Combined Uncertainty Score

This module implements our novel conformal calibration method:
- Stage 1: Combined score calibration (aleatoric + epistemic)
- Stage 2: Local scaling via decision trees
- Coverage guarantee: P(Y ∈ I(X)) ≥ 1-α

Author: Enhanced CACD Team
Date: November 11, 2025
"""

import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple


class CombinedConformalCalibrator:
    """
    Conformal prediction with combined aleatoric + epistemic uncertainty

    Novel approach: Combine uncertainties BEFORE calibration (not after)
    This maintains coverage guarantee while adapting intervals
    """

    def __init__(self, alpha: float = 0.1, use_local_scaling: bool = True,
                 max_depth: int = 5, min_samples_leaf: int = 10, verbose: bool = True):
        """
        Args:
            alpha: Miscoverage level (0.1 = 90% coverage)
            use_local_scaling: Whether to use decision tree for local scaling
            max_depth: Max depth of decision tree
            min_samples_leaf: Min samples per leaf in tree
            verbose: Print diagnostics
        """
        self.alpha = alpha
        self.use_local_scaling = use_local_scaling
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.verbose = verbose

        # Fitted parameters
        self.q_hat = None
        self.scaling_tree = None
        self.is_fitted = False

        # Calibration statistics
        self.cal_scores = None
        self.cal_combined_uncertainty = None

    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray, y_pred_cal: np.ndarray,
            sigma_alea_cal: np.ndarray, sigma_epis_cal: np.ndarray):
        """
        Calibrate conformal prediction intervals

        Args:
            X_cal: Calibration features [n_cal, d]
            y_cal: Calibration targets (e.g., IoU) [n_cal]
            y_pred_cal: Calibration predictions [n_cal]
            sigma_alea_cal: Aleatoric uncertainty [n_cal]
            sigma_epis_cal: Epistemic uncertainty [n_cal]
        """
        if self.verbose:
            print("="*80)
            print("FITTING COMBINED CONFORMAL CALIBRATION")
            print("="*80)
            print(f"Calibration samples: {len(X_cal)}")
            print(f"Miscoverage level α: {self.alpha}")
            print(f"Target coverage: {1-self.alpha:.1%}")

        # Stage 1: Combined uncertainty
        sigma_combined = np.sqrt(sigma_alea_cal**2 + sigma_epis_cal**2)
        self.cal_combined_uncertainty = sigma_combined

        if self.verbose:
            print(f"\nCombined Uncertainty Statistics:")
            print(f"  Mean: {sigma_combined.mean():.4f}")
            print(f"  Std:  {sigma_combined.std():.4f}")
            print(f"  Min:  {sigma_combined.min():.4f}")
            print(f"  Max:  {sigma_combined.max():.4f}")

        # Stage 2: Nonconformity scores
        residuals = np.abs(y_cal - y_pred_cal)
        scores = residuals / (sigma_combined + 1e-10)
        self.cal_scores = scores

        if self.verbose:
            print(f"\nNonconformity Scores:")
            print(f"  Mean: {scores.mean():.4f}")
            print(f"  Std:  {scores.std():.4f}")
            print(f"  Min:  {scores.min():.4f}")
            print(f"  Max:  {scores.max():.4f}")

        # Stage 3: Global quantile (with finite-sample correction)
        n = len(scores)
        quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat = np.quantile(scores, quantile_level)

        if self.verbose:
            print(f"\nGlobal Quantile:")
            print(f"  Quantile level: {quantile_level:.4f}")
            print(f"  q̂: {self.q_hat:.4f}")

        # Stage 4: Local quantiles via stratification (optional)
        if self.use_local_scaling:
            if self.verbose:
                print(f"\nFitting Locally Adaptive Quantiles...")

            # Determine max depth based on sample size
            adaptive_depth = min(self.max_depth, int(np.log2(n / 20)))

            # Use tree to stratify samples (group similar contexts)
            # Train on combined uncertainty to identify difficulty regions
            from sklearn.tree import DecisionTreeClassifier

            # Create bins based on normalized scores for stratification
            n_bins = min(10, int(n / 50))  # At least 50 samples per bin
            score_bins = np.linspace(0, np.quantile(scores, 0.95), n_bins)
            bin_labels = np.digitize(scores, score_bins)

            # Train classifier to predict difficulty stratum
            self.scaling_tree = DecisionTreeClassifier(
                max_depth=adaptive_depth,
                min_samples_split=20,
                min_samples_leaf=self.min_samples_leaf,
                random_state=42
            )
            self.scaling_tree.fit(X_cal, bin_labels)

            # Get leaf assignments
            leaf_ids = self.scaling_tree.apply(X_cal)
            unique_leaves = np.unique(leaf_ids)

            # Compute quantile per leaf (with coverage correction)
            self.leaf_quantiles = {}

            for leaf_id in unique_leaves:
                leaf_mask = leaf_ids == leaf_id
                leaf_scores = scores[leaf_mask]
                n_leaf = len(leaf_scores)

                if n_leaf >= self.min_samples_leaf:
                    # Compute quantile with finite-sample correction
                    # Use slightly higher quantile for small leaves to maintain coverage
                    if n_leaf < 50:
                        # More conservative for small leaves
                        adjusted_alpha = self.alpha * 0.8
                    else:
                        adjusted_alpha = self.alpha

                    q_level = np.ceil((n_leaf + 1) * (1 - adjusted_alpha)) / n_leaf
                    q_level = min(q_level, 1.0)  # Ensure valid quantile
                    leaf_q = np.quantile(leaf_scores, q_level)
                    self.leaf_quantiles[leaf_id] = leaf_q
                else:
                    # Fall back to global quantile for small leaves
                    self.leaf_quantiles[leaf_id] = self.q_hat

            if self.verbose:
                print(f"  Tree depth: {self.scaling_tree.get_depth()}")
                print(f"  Num leaves: {len(unique_leaves)}")
                print(f"  Leaf quantiles:")
                q_values = list(self.leaf_quantiles.values())
                print(f"    Mean: {np.mean(q_values):.4f}")
                print(f"    Std:  {np.std(q_values):.4f}")
                print(f"    Min:  {np.min(q_values):.4f}")
                print(f"    Max:  {np.max(q_values):.4f}")
                print(f"    Ratio to global: {np.mean(q_values)/self.q_hat:.2f}x")

        self.is_fitted = True

        if self.verbose:
            print("="*80)
            print("CALIBRATION COMPLETE ✓")
            print("="*80 + "\n")

        return self

    def predict(self, X_test: np.ndarray, y_pred_test: np.ndarray,
                sigma_alea_test: np.ndarray, sigma_epis_test: np.ndarray
                ) -> Dict[str, np.ndarray]:
        """
        Generate prediction intervals for test set

        Args:
            X_test: Test features [n_test, d]
            y_pred_test: Test predictions [n_test]
            sigma_alea_test: Aleatoric uncertainty [n_test]
            sigma_epis_test: Epistemic uncertainty [n_test]

        Returns:
            Dictionary with:
                - 'lower': Lower bounds [n_test]
                - 'upper': Upper bounds [n_test]
                - 'width': Interval widths [n_test]
                - 'combined_uncertainty': Combined uncertainty [n_test]
                - 'local_quantiles': Leaf-specific quantiles [n_test]
                - 'scaling_factors': Ratio to global quantile [n_test]
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before prediction")

        # Combined uncertainty
        sigma_combined = np.sqrt(sigma_alea_test**2 + sigma_epis_test**2)

        # Local quantiles (or global if not using local scaling)
        if self.use_local_scaling:
            # Get leaf assignments for test samples
            leaf_ids = self.scaling_tree.apply(X_test)

            # Get quantile for each sample's leaf
            q_local = np.array([
                self.leaf_quantiles.get(leaf_id, self.q_hat)
                for leaf_id in leaf_ids
            ])
        else:
            q_local = np.full(len(X_test), self.q_hat)

        # Prediction intervals: I(x) = ŷ ± q_local × σ_combined
        width = q_local * sigma_combined
        lower = y_pred_test - width
        upper = y_pred_test + width

        # Clip to valid range [0, 1] for IoU
        lower = np.clip(lower, 0, 1)
        upper = np.clip(upper, 0, 1)

        return {
            'lower': lower,
            'upper': upper,
            'width': width,
            'combined_uncertainty': sigma_combined,
            'local_quantiles': q_local,
            'scaling_factors': q_local / self.q_hat  # Ratio to global quantile
        }

    def evaluate_coverage(self, y_true: np.ndarray, intervals: Dict[str, np.ndarray],
                         stratify_by: Optional[np.ndarray] = None
                         ) -> Dict[str, float]:
        """
        Evaluate empirical coverage

        Args:
            y_true: True values [n]
            intervals: Output from predict()
            stratify_by: Optional stratification variable (e.g., leaf index)

        Returns:
            Dictionary with coverage statistics
        """
        lower = intervals['lower']
        upper = intervals['upper']

        # Overall coverage
        covered = (y_true >= lower) & (y_true <= upper)
        coverage = covered.mean()

        results = {
            'coverage': coverage,
            'n_samples': len(y_true),
            'target_coverage': 1 - self.alpha
        }

        # Stratified coverage (if provided)
        if stratify_by is not None:
            unique_strata = np.unique(stratify_by)
            strata_coverage = {}

            for stratum in unique_strata:
                mask = stratify_by == stratum
                if mask.sum() > 0:
                    strata_cov = covered[mask].mean()
                    strata_coverage[int(stratum)] = {
                        'coverage': strata_cov,
                        'n_samples': mask.sum()
                    }

            results['stratified_coverage'] = strata_coverage

        return results

    def plot_diagnostics(self, y_true: np.ndarray, intervals: Dict[str, np.ndarray],
                        save_dir: Path, prefix: str = ""):
        """
        Generate diagnostic plots

        Args:
            y_true: True values [n]
            intervals: Output from predict()
            save_dir: Directory to save plots
            prefix: Prefix for filenames
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Conformal Prediction Diagnostics', fontsize=16, fontweight='bold')

        lower = intervals['lower']
        upper = intervals['upper']
        width = intervals['width']
        covered = (y_true >= lower) & (y_true <= upper)

        # Plot 1: Calibration Scores Distribution
        ax1 = axes[0, 0]
        if self.cal_scores is not None:
            ax1.hist(self.cal_scores, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax1.axvline(self.q_hat, color='red', linestyle='--', linewidth=2,
                       label=f'q̂ = {self.q_hat:.3f}')
            ax1.axvline(np.quantile(self.cal_scores, 1-self.alpha), color='orange',
                       linestyle=':', linewidth=2, label=f'{1-self.alpha:.0%} quantile')
            ax1.set_xlabel('Nonconformity Score', fontsize=12)
            ax1.set_ylabel('Frequency', fontsize=12)
            ax1.set_title('Calibration Scores Distribution', fontsize=13, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # Plot 2: Coverage vs Uncertainty
        ax2 = axes[0, 1]
        uncertainty_bins = np.linspace(width.min(), width.max(), 20)
        bin_indices = np.digitize(width, uncertainty_bins)

        coverage_by_bin = []
        bin_centers = []

        for i in range(1, len(uncertainty_bins)):
            mask = bin_indices == i
            if mask.sum() > 10:  # At least 10 samples
                coverage_by_bin.append(covered[mask].mean())
                bin_centers.append((uncertainty_bins[i-1] + uncertainty_bins[i]) / 2)

        ax2.plot(bin_centers, coverage_by_bin, 'o-', color='blue', linewidth=2, markersize=8)
        ax2.axhline(1-self.alpha, color='red', linestyle='--', linewidth=2,
                   label=f'Target: {1-self.alpha:.0%}')
        ax2.set_xlabel('Interval Width', fontsize=12)
        ax2.set_ylabel('Empirical Coverage', fontsize=12)
        ax2.set_title('Coverage vs Interval Width', fontsize=13, fontweight='bold')
        ax2.set_ylim([0.75, 1.0])
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Interval Width Distribution
        ax3 = axes[1, 0]
        ax3.hist(width, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax3.axvline(width.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {width.mean():.3f}')
        ax3.axvline(np.median(width), color='orange', linestyle=':', linewidth=2,
                   label=f'Median: {np.median(width):.3f}')
        ax3.set_xlabel('Interval Width', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title('Prediction Interval Width Distribution', fontsize=13, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Coverage Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')

        overall_coverage = covered.mean()
        n_covered = covered.sum()
        n_total = len(covered)

        stats_text = f"""
        COVERAGE STATISTICS
        ==================
        Target Coverage: {1-self.alpha:.1%}
        Empirical Coverage: {overall_coverage:.1%}

        Samples Covered: {n_covered} / {n_total}

        Interval Width:
          Mean:   {width.mean():.4f}
          Median: {np.median(width):.4f}
          Std:    {width.std():.4f}
          Min:    {width.min():.4f}
          Max:    {width.max():.4f}

        Combined Uncertainty:
          Mean: {intervals['combined_uncertainty'].mean():.4f}
          Std:  {intervals['combined_uncertainty'].std():.4f}
        """

        if self.use_local_scaling:
            xi = intervals['scaling_factors']
            stats_text += f"""
        Local Scaling (ξ):
          Mean: {xi.mean():.4f}
          Std:  {xi.std():.4f}
          Min:  {xi.min():.4f}
          Max:  {xi.max():.4f}
            """

        ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center')

        plt.tight_layout()
        save_path = save_dir / f'{prefix}conformal_diagnostics.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        if self.verbose:
            print(f"  Saved: {save_path}")


class VanillaConformal:
    """
    Baseline: Standard conformal prediction (constant width)
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.q_hat = None
        self.is_fitted = False

    def fit(self, y_cal: np.ndarray, y_pred_cal: np.ndarray):
        """Calibrate vanilla conformal"""
        residuals = np.abs(y_cal - y_pred_cal)
        n = len(residuals)
        quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat = np.quantile(residuals, quantile_level)
        self.is_fitted = True
        return self

    def predict(self, y_pred_test: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate constant-width intervals"""
        if not self.is_fitted:
            raise ValueError("Must fit before predict")

        width = np.full(len(y_pred_test), self.q_hat)
        lower = np.clip(y_pred_test - width, 0, 1)
        upper = np.clip(y_pred_test + width, 0, 1)

        return {
            'lower': lower,
            'upper': upper,
            'width': width
        }
