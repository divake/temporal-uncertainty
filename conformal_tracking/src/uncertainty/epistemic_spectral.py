"""
Spectral Feature Collapse Detection for Epistemic Uncertainty

This module implements epistemic uncertainty quantification via eigenspectrum analysis
of local feature manifolds. When features collapse to lower dimensions, it indicates
the model lacks discriminative knowledge.

Author: Enhanced CACD Team
Date: 2025-11-10
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, Optional
import seaborn as sns


class SpectralCollapseDetector:
    """
    Epistemic uncertainty via spectral analysis of feature manifold

    Core idea: When model lacks knowledge, features collapse to lower-dimensional
    manifolds. This is measurable via eigenspectrum entropy.
    """

    def __init__(self, k_neighbors: int = 50, verbose: bool = False):
        """
        Initialize spectral collapse detector

        Args:
            k_neighbors: Number of neighbors for local analysis
            verbose: Whether to print debug information
        """
        self.k_neighbors = k_neighbors
        self.verbose = verbose

        # Normalization statistics (computed during fit)
        self.cal_min_entropy = None
        self.cal_max_entropy = None
        self.cal_mean_eff_rank = None
        self.feature_dim = None

        # Store calibration data for analysis
        self.X_cal = None
        self.cal_diagnostics = {}

    def fit(self, X_calibration: np.ndarray,
            mahalanobis_model=None,
            plot_diagnostics: bool = True,
            save_dir: Optional[Path] = None):
        """
        Fit spectral model on calibration data

        Args:
            X_calibration: Calibration features [N_cal, D]
            mahalanobis_model: Pre-fitted Mahalanobis model for distance computation
            plot_diagnostics: Whether to generate diagnostic plots
            save_dir: Directory to save plots
        """
        self.X_cal = X_calibration
        self.feature_dim = X_calibration.shape[1]
        self.mahalanobis_model = mahalanobis_model

        if self.verbose:
            print("\n" + "="*60)
            print("FITTING SPECTRAL COLLAPSE DETECTOR")
            print("="*60)
            print(f"Calibration samples: {len(X_calibration)}")
            print(f"Feature dimension: {self.feature_dim}")

        # Compute spectral statistics on calibration set
        cal_entropies = []
        cal_eff_ranks = []
        cal_top_eigenvalues = []

        # Sample subset for efficiency
        n_samples = min(500, len(X_calibration))
        sample_idx = np.random.choice(len(X_calibration), n_samples, replace=False)

        if self.verbose:
            print(f"\nComputing spectral statistics on {n_samples} samples...")

        for i, idx in enumerate(sample_idx):
            if i % 100 == 0 and self.verbose:
                print(f"  Processing sample {i}/{n_samples}")

            # Compute spectral metrics for this calibration point
            entropy, eff_rank, eigenvalues = self._compute_spectral_metrics(
                X_calibration[idx], X_calibration, return_eigenvalues=True
            )

            cal_entropies.append(entropy)
            cal_eff_ranks.append(eff_rank)
            cal_top_eigenvalues.append(eigenvalues[-5:])  # Top 5

        # Store normalization statistics
        cal_entropies = np.array(cal_entropies)
        cal_eff_ranks = np.array(cal_eff_ranks)

        self.cal_min_entropy = np.percentile(cal_entropies, 5)
        self.cal_max_entropy = np.percentile(cal_entropies, 95)
        self.cal_mean_eff_rank = np.mean(cal_eff_ranks)

        # Store diagnostics
        self.cal_diagnostics = {
            'entropies': cal_entropies,
            'effective_ranks': cal_eff_ranks,
            'top_eigenvalues': np.array(cal_top_eigenvalues),
            'mean_entropy': np.mean(cal_entropies),
            'std_entropy': np.std(cal_entropies),
            'mean_eff_rank': np.mean(cal_eff_ranks),
            'std_eff_rank': np.std(cal_eff_ranks)
        }

        if self.verbose:
            print(f"\nCalibration Statistics:")
            print(f"  Entropy range: [{self.cal_min_entropy:.3f}, {self.cal_max_entropy:.3f}]")
            print(f"  Mean effective rank: {self.cal_mean_eff_rank:.1f} / {self.feature_dim}")
            print(f"  Rank utilization: {self.cal_mean_eff_rank/self.feature_dim*100:.1f}%")

        # Generate diagnostic plots
        if plot_diagnostics and save_dir is not None:
            self._plot_calibration_diagnostics(save_dir)

        if self.verbose:
            print("\n" + "="*60)
            print("SPECTRAL FITTING COMPLETE ✓")
            print("="*60 + "\n")

    def predict(self, X_test: np.ndarray,
                return_diagnostics: bool = True) -> Dict[str, np.ndarray]:
        """
        Predict spectral epistemic uncertainty for test samples

        Args:
            X_test: Test features [N_test, D]
            return_diagnostics: Whether to return detailed diagnostics

        Returns:
            Dictionary with 'epistemic' and optionally diagnostic information
        """
        n_test = len(X_test) if len(X_test.shape) > 1 else 1
        if len(X_test.shape) == 1:
            X_test = X_test.reshape(1, -1)

        epistemic = np.zeros(n_test)
        eff_ranks = np.zeros(n_test)
        entropies = np.zeros(n_test)

        for i in range(n_test):
            # Compute spectral metrics
            entropy, eff_rank, _ = self._compute_spectral_metrics(
                X_test[i], self.X_cal, return_eigenvalues=False
            )

            # Normalize entropy to [0, 1]
            epistemic_normalized = self._normalize_entropy(entropy)

            epistemic[i] = epistemic_normalized
            eff_ranks[i] = eff_rank
            entropies[i] = entropy

        results = {'epistemic': epistemic}

        if return_diagnostics:
            results['effective_ranks'] = eff_ranks
            results['entropies'] = entropies
            results['rank_utilization'] = eff_ranks / self.feature_dim

        return results

    def _compute_spectral_metrics(self, x_test: np.ndarray,
                                   X_reference: np.ndarray,
                                   return_eigenvalues: bool = False) -> Tuple:
        """
        Compute spectral collapse metrics for a test point

        Args:
            x_test: Test feature vector [D]
            X_reference: Reference features for neighborhood [N, D]
            return_eigenvalues: Whether to return eigenvalues

        Returns:
            (entropy, effective_rank, eigenvalues)
        """
        # Find k nearest neighbors
        # Always use Euclidean distance for finding neighbors
        # (Mahalanobis is for distance from the mean, not between points)
        distances = np.linalg.norm(X_reference - x_test, axis=1)

        # Get k nearest neighbors
        k = min(self.k_neighbors, len(X_reference))
        neighbor_idx = np.argsort(distances)[:k]
        X_local = X_reference[neighbor_idx]

        # Center the local features
        mu_local = X_local.mean(axis=0)
        X_centered = X_local - mu_local

        # Compute local covariance
        # Use more stable computation for small k
        if k < self.feature_dim:
            # Add small regularization for numerical stability
            Sigma_local = (X_centered.T @ X_centered) / k
            Sigma_local += np.eye(self.feature_dim) * 1e-8
        else:
            Sigma_local = np.cov(X_centered.T)

        # Eigendecomposition
        try:
            eigenvalues = np.linalg.eigvalsh(Sigma_local)
            eigenvalues = np.maximum(eigenvalues, 1e-10)  # Numerical stability
        except:
            # Fallback for numerical issues
            eigenvalues = np.ones(self.feature_dim) / self.feature_dim

        # Normalize eigenvalues
        lambda_norm = eigenvalues / (eigenvalues.sum() + 1e-10)

        # Compute spectral entropy
        # H = -sum(p * log(p))
        entropy = -np.sum(lambda_norm * np.log(lambda_norm + 1e-10))

        # Effective rank = exp(entropy)
        # This gives the "effective number of dimensions"
        effective_rank = np.exp(entropy)

        if return_eigenvalues:
            return entropy, effective_rank, eigenvalues
        else:
            return entropy, effective_rank, None

    def _normalize_entropy(self, entropy: float) -> float:
        """
        Normalize entropy to [0, 1] epistemic uncertainty

        Low entropy (collapsed) → High epistemic
        High entropy (rich) → Low epistemic
        """
        # Clip to calibration range
        entropy_clipped = np.clip(entropy, self.cal_min_entropy, self.cal_max_entropy)

        # Normalize to [0, 1]
        entropy_norm = (entropy_clipped - self.cal_min_entropy) / \
                      (self.cal_max_entropy - self.cal_min_entropy + 1e-10)

        # Invert: low entropy = high epistemic
        # But not completely inverted - we want moderate relationship
        # High entropy still indicates some uncertainty (too spread out)

        # Use a parabolic transformation centered at 0.5
        # This gives high uncertainty at both extremes
        epistemic = 1.0 - 2.0 * np.abs(entropy_norm - 0.5)

        return np.clip(epistemic, 0, 1)

    def _plot_calibration_diagnostics(self, save_dir: Path):
        """
        Generate diagnostic plots for spectral analysis
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Plot 1: Entropy distribution
        axes[0, 0].hist(self.cal_diagnostics['entropies'], bins=30,
                       edgecolor='black', alpha=0.7, color='blue')
        axes[0, 0].axvline(self.cal_min_entropy, color='red', linestyle='--',
                          label=f'5th percentile: {self.cal_min_entropy:.3f}')
        axes[0, 0].axvline(self.cal_max_entropy, color='red', linestyle='--',
                          label=f'95th percentile: {self.cal_max_entropy:.3f}')
        axes[0, 0].set_xlabel('Spectral Entropy', fontsize=12)
        axes[0, 0].set_ylabel('Count', fontsize=12)
        axes[0, 0].set_title('Calibration Entropy Distribution', fontsize=13, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Effective rank distribution
        axes[0, 1].hist(self.cal_diagnostics['effective_ranks'], bins=30,
                       edgecolor='black', alpha=0.7, color='green')
        axes[0, 1].axvline(self.cal_mean_eff_rank, color='red', linestyle='--',
                          label=f'Mean: {self.cal_mean_eff_rank:.1f}')
        axes[0, 1].set_xlabel('Effective Rank', fontsize=12)
        axes[0, 1].set_ylabel('Count', fontsize=12)
        axes[0, 1].set_title(f'Effective Rank Distribution (max={self.feature_dim})',
                           fontsize=13, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Rank utilization
        rank_util = self.cal_diagnostics['effective_ranks'] / self.feature_dim * 100
        axes[0, 2].hist(rank_util, bins=30, edgecolor='black', alpha=0.7, color='orange')
        axes[0, 2].set_xlabel('Rank Utilization (%)', fontsize=12)
        axes[0, 2].set_ylabel('Count', fontsize=12)
        axes[0, 2].set_title('Feature Space Utilization', fontsize=13, fontweight='bold')
        axes[0, 2].axvline(rank_util.mean(), color='red', linestyle='--',
                          label=f'Mean: {rank_util.mean():.1f}%')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Plot 4: Entropy vs Effective Rank (should be correlated)
        axes[1, 0].scatter(self.cal_diagnostics['entropies'],
                          self.cal_diagnostics['effective_ranks'],
                          alpha=0.5, s=20)
        axes[1, 0].set_xlabel('Spectral Entropy', fontsize=12)
        axes[1, 0].set_ylabel('Effective Rank', fontsize=12)
        axes[1, 0].set_title('Entropy vs Effective Rank', fontsize=13, fontweight='bold')

        # Add correlation
        corr = np.corrcoef(self.cal_diagnostics['entropies'],
                          self.cal_diagnostics['effective_ranks'])[0, 1]
        axes[1, 0].text(0.05, 0.95, f'Correlation: {corr:.3f}',
                       transform=axes[1, 0].transAxes, fontsize=11,
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 5: Top eigenvalues distribution
        top_eigs = self.cal_diagnostics['top_eigenvalues']
        mean_top_eigs = top_eigs.mean(axis=0)
        std_top_eigs = top_eigs.std(axis=0)

        x_pos = np.arange(5)
        axes[1, 1].bar(x_pos, mean_top_eigs, yerr=std_top_eigs,
                      capsize=5, color='purple', alpha=0.7)
        axes[1, 1].set_xlabel('Eigenvalue Rank (largest to smallest)', fontsize=12)
        axes[1, 1].set_ylabel('Eigenvalue Magnitude', fontsize=12)
        axes[1, 1].set_title('Top 5 Eigenvalues (mean ± std)', fontsize=13, fontweight='bold')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(['1st', '2nd', '3rd', '4th', '5th'])
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        # Plot 6: Epistemic transformation function
        entropy_range = np.linspace(self.cal_min_entropy, self.cal_max_entropy, 100)
        epistemic_values = [self._normalize_entropy(e) for e in entropy_range]

        axes[1, 2].plot(entropy_range, epistemic_values, linewidth=2, color='red')
        axes[1, 2].set_xlabel('Spectral Entropy', fontsize=12)
        axes[1, 2].set_ylabel('Epistemic Uncertainty [0,1]', fontsize=12)
        axes[1, 2].set_title('Entropy → Epistemic Transformation', fontsize=13, fontweight='bold')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_ylim([-0.05, 1.05])

        # Add markers for key points
        axes[1, 2].scatter([self.cal_min_entropy, self.cal_max_entropy],
                          [self._normalize_entropy(self.cal_min_entropy),
                           self._normalize_entropy(self.cal_max_entropy)],
                          color='blue', s=50, zorder=5)

        plt.suptitle('Spectral Collapse Calibration Diagnostics', fontsize=15, fontweight='bold')
        plt.tight_layout()

        save_path = save_dir / 'spectral_calibration_diagnostics.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

        plt.close()

    def plot_test_diagnostics(self, results: Dict, save_dir: Path, prefix: str = ""):
        """
        Generate diagnostic plots for test predictions
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Epistemic distribution
        axes[0, 0].hist(results['epistemic'], bins=30,
                       edgecolor='black', alpha=0.7, color='red')
        axes[0, 0].set_xlabel('Spectral Epistemic Uncertainty', fontsize=12)
        axes[0, 0].set_ylabel('Count', fontsize=12)
        axes[0, 0].set_title('Test Epistemic Distribution', fontsize=13, fontweight='bold')
        axes[0, 0].axvline(results['epistemic'].mean(), color='blue', linestyle='--',
                          label=f"Mean: {results['epistemic'].mean():.3f}")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Effective rank distribution
        axes[0, 1].hist(results['effective_ranks'], bins=30,
                       edgecolor='black', alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Effective Rank', fontsize=12)
        axes[0, 1].set_ylabel('Count', fontsize=12)
        axes[0, 1].set_title('Test Effective Rank Distribution', fontsize=13, fontweight='bold')
        axes[0, 1].axvline(results['effective_ranks'].mean(), color='red', linestyle='--',
                          label=f"Mean: {results['effective_ranks'].mean():.1f}")
        axes[0, 1].axvline(self.cal_mean_eff_rank, color='blue', linestyle='--',
                          label=f"Cal Mean: {self.cal_mean_eff_rank:.1f}")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Rank utilization comparison
        test_util = results['rank_utilization'] * 100
        cal_util = self.cal_diagnostics['effective_ranks'] / self.feature_dim * 100

        axes[1, 0].violinplot([cal_util, test_util], positions=[1, 2],
                             showmeans=True, showmedians=True)
        axes[1, 0].set_xticks([1, 2])
        axes[1, 0].set_xticklabels(['Calibration', 'Test'])
        axes[1, 0].set_ylabel('Rank Utilization (%)', fontsize=12)
        axes[1, 0].set_title('Calibration vs Test Rank Utilization', fontsize=13, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # Plot 4: Entropy vs Epistemic
        axes[1, 1].scatter(results['entropies'], results['epistemic'],
                          alpha=0.5, s=20, c=results['epistemic'], cmap='RdYlBu_r')
        axes[1, 1].set_xlabel('Spectral Entropy', fontsize=12)
        axes[1, 1].set_ylabel('Epistemic Uncertainty', fontsize=12)
        axes[1, 1].set_title('Entropy → Epistemic Mapping', fontsize=13, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        # Add calibration range
        axes[1, 1].axvline(self.cal_min_entropy, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].axvline(self.cal_max_entropy, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].fill_betweenx([0, 1], self.cal_min_entropy, self.cal_max_entropy,
                                 alpha=0.1, color='gray', label='Cal range')
        axes[1, 1].legend()

        plt.suptitle(f'{prefix}Spectral Epistemic Test Diagnostics', fontsize=15, fontweight='bold')
        plt.tight_layout()

        save_path = save_dir / f'{prefix}spectral_test_diagnostics.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

        plt.close()

        # Print summary statistics
        print("\n" + "="*60)
        print("SPECTRAL EPISTEMIC TEST STATISTICS")
        print("="*60)
        print(f"Epistemic Uncertainty:")
        print(f"  Mean: {results['epistemic'].mean():.4f}")
        print(f"  Std:  {results['epistemic'].std():.4f}")
        print(f"  Min:  {results['epistemic'].min():.4f}")
        print(f"  Max:  {results['epistemic'].max():.4f}")
        print(f"\nEffective Rank:")
        print(f"  Mean: {results['effective_ranks'].mean():.1f} / {self.feature_dim}")
        print(f"  Std:  {results['effective_ranks'].std():.1f}")
        print(f"  Utilization: {results['rank_utilization'].mean()*100:.1f}%")
        print("="*60)