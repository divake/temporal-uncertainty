"""
Mahalanobis Distance-Based Aleatoric Uncertainty

This module implements aleatoric uncertainty quantification using Mahalanobis distance
from a multivariate Gaussian distribution fitted to calibration features.

Based on: "Mahalanobis Distance-based Multivariate Gaussian Distribution-based
          Aleatoric Uncertainty" method

Author: Enhanced CACD Team
Date: 2025-11-10
"""

import numpy as np
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path


class MahalanobisUncertainty:
    """
    Mahalanobis distance-based aleatoric uncertainty estimator.

    Fits a multivariate Gaussian to calibration features and computes
    Mahalanobis distance for test samples as uncertainty measure.

    Attributes:
        mean_: Mean vector (μ) of calibration features [D]
        cov_: Covariance matrix (Σ) of calibration features [D, D]
        cov_inv_: Inverse covariance matrix (Σ⁻¹) [D, D]
        log_M_min_: Min log Mahalanobis distance from calibration (for normalization)
        log_M_max_: Max log Mahalanobis distance from calibration (for normalization)
        is_fitted_: Whether the model has been fitted
    """

    def __init__(self, reg_lambda: float = 1e-4, eps: float = 1e-10):
        """
        Initialize Mahalanobis uncertainty estimator.

        Args:
            reg_lambda: Regularization parameter for covariance matrix
            eps: Small constant to avoid log(0) and sqrt(negative)
        """
        self.reg_lambda = reg_lambda
        self.eps = eps

        # Will be set during fit()
        self.mean_ = None
        self.cov_ = None
        self.cov_inv_ = None
        self.log_M_min_ = None
        self.log_M_max_ = None
        self.is_fitted_ = False

        # Store calibration distances for analysis
        self.cal_mahal_distances_ = None

    def fit(self, X_cal: np.ndarray, verbose: bool = True) -> 'MahalanobisUncertainty':
        """
        Fit multivariate Gaussian to calibration features.

        Args:
            X_cal: Calibration features [N_cal, D]
            verbose: Print statistics

        Returns:
            self: Fitted estimator
        """
        N, D = X_cal.shape

        if verbose:
            print(f"\n{'='*60}")
            print(f"FITTING MAHALANOBIS UNCERTAINTY")
            print(f"{'='*60}")
            print(f"Calibration samples: {N}")
            print(f"Feature dimension: {D}")

        # Step 1: Compute mean vector (Equation 1)
        self.mean_ = np.mean(X_cal, axis=0)  # [D]

        if verbose:
            print(f"\n1. Mean vector (μ):")
            print(f"   Shape: {self.mean_.shape}")
            print(f"   Mean norm: {np.linalg.norm(self.mean_):.4f}")

        # Step 2: Compute covariance matrix (Equation 2)
        self.cov_ = np.cov(X_cal, rowvar=False)  # [D, D]

        if verbose:
            print(f"\n2. Covariance matrix (Σ):")
            print(f"   Shape: {self.cov_.shape}")
            print(f"   Trace: {np.trace(self.cov_):.4f}")
            print(f"   Condition number (before reg): {np.linalg.cond(self.cov_):.2e}")

        # Step 3: Regularize covariance (prevent singular matrix)
        trace_val = np.trace(self.cov_)
        reg_val = self.reg_lambda * (trace_val / D)
        self.cov_ = self.cov_ + reg_val * np.eye(D)

        if verbose:
            print(f"\n3. Regularization:")
            print(f"   Lambda: {self.reg_lambda}")
            print(f"   Reg value: {reg_val:.6f}")
            print(f"   Condition number (after reg): {np.linalg.cond(self.cov_):.2e}")

        # Step 4: Invert covariance matrix
        try:
            self.cov_inv_ = np.linalg.inv(self.cov_)
        except np.linalg.LinAlgError:
            raise ValueError("Covariance matrix is singular even after regularization. "
                           "Try increasing reg_lambda.")

        if verbose:
            print(f"\n4. Inverse covariance (Σ⁻¹):")
            print(f"   Shape: {self.cov_inv_.shape}")
            print(f"   Max element: {np.max(np.abs(self.cov_inv_)):.4f}")

        # Step 5: Compute calibration distances for normalization
        self.cal_mahal_distances_ = self._compute_mahalanobis(X_cal)  # [N_cal]

        log_M_cal = np.log(self.cal_mahal_distances_ + self.eps)
        self.log_M_min_ = np.min(log_M_cal)
        self.log_M_max_ = np.max(log_M_cal)

        if verbose:
            print(f"\n5. Calibration Mahalanobis distances:")
            print(f"   Min: {np.min(self.cal_mahal_distances_):.4f}")
            print(f"   Mean: {np.mean(self.cal_mahal_distances_):.4f}")
            print(f"   Median: {np.median(self.cal_mahal_distances_):.4f}")
            print(f"   Max: {np.max(self.cal_mahal_distances_):.4f}")
            print(f"   Std: {np.std(self.cal_mahal_distances_):.4f}")
            print(f"\n   Log normalization range:")
            print(f"   log_M_min: {self.log_M_min_:.4f}")
            print(f"   log_M_max: {self.log_M_max_:.4f}")

        self.is_fitted_ = True

        if verbose:
            print(f"\n{'='*60}")
            print(f"FITTING COMPLETE ✓")
            print(f"{'='*60}\n")

        return self

    def _compute_mahalanobis(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Mahalanobis distance for samples.

        Args:
            X: Features [N, D]

        Returns:
            M: Mahalanobis distances [N]
        """
        # Center the data
        X_centered = X - self.mean_  # [N, D]

        # Compute M(x) = sqrt[(x - μ)ᵀ Σ⁻¹ (x - μ)]
        # For efficiency: M² = (X_centered @ Σ_inv) * X_centered, then sum over D
        M_squared = np.sum((X_centered @ self.cov_inv_) * X_centered, axis=1)  # [N]

        # Ensure non-negative before sqrt (numerical stability)
        M_squared = np.maximum(M_squared, 0.0)
        M = np.sqrt(M_squared)

        return M

    def predict_raw(self, X_test: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        Predict raw Mahalanobis distance (for correlation with errors).

        Args:
            X_test: Test features [N_test, D]
            verbose: Print statistics

        Returns:
            uncertainty_raw: Raw Mahalanobis distances [N_test]
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")

        M = self._compute_mahalanobis(X_test)

        if verbose:
            print(f"\nRaw Mahalanobis distances (test):")
            print(f"  Min: {np.min(M):.4f}")
            print(f"  Mean: {np.mean(M):.4f}")
            print(f"  Median: {np.median(M):.4f}")
            print(f"  Max: {np.max(M):.4f}")
            print(f"  Std: {np.std(M):.4f}")

        return M

    def predict_normalized(self, X_test: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        Predict normalized uncertainty in [0, 1] (for interpretation).

        Uses log transform + min-max normalization based on calibration range.

        Args:
            X_test: Test features [N_test, D]
            verbose: Print statistics

        Returns:
            uncertainty_normalized: Normalized scores in [0, 1] [N_test]
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")

        M = self._compute_mahalanobis(X_test)

        # Log transform + min-max normalization (Equation 4)
        log_M = np.log(M + self.eps)
        d_norm = (log_M - self.log_M_min_) / (self.log_M_max_ - self.log_M_min_ + self.eps)

        # Clip to [0, 1] (some test samples might be outside calibration range)
        d_norm = np.clip(d_norm, 0.0, 1.0)

        if verbose:
            print(f"\nNormalized uncertainty (test):")
            print(f"  Min: {np.min(d_norm):.4f}")
            print(f"  Mean: {np.mean(d_norm):.4f}")
            print(f"  Median: {np.median(d_norm):.4f}")
            print(f"  Max: {np.max(d_norm):.4f}")
            print(f"  Std: {np.std(d_norm):.4f}")

            # Category distribution
            low = np.sum(d_norm < 0.3) / len(d_norm) * 100
            medium = np.sum((d_norm >= 0.3) & (d_norm < 0.7)) / len(d_norm) * 100
            high = np.sum(d_norm >= 0.7) / len(d_norm) * 100
            print(f"\n  Category distribution:")
            print(f"    Low (0-0.3):     {low:5.1f}%")
            print(f"    Medium (0.3-0.7): {medium:5.1f}%")
            print(f"    High (0.7-1.0):   {high:5.1f}%")

        return d_norm

    def predict(self, X_test: np.ndarray, verbose: bool = True) -> Dict[str, np.ndarray]:
        """
        Predict both raw and normalized uncertainty.

        Args:
            X_test: Test features [N_test, D]
            verbose: Print statistics

        Returns:
            Dictionary with:
                - 'raw': Raw Mahalanobis distances [N_test]
                - 'normalized': Normalized scores in [0, 1] [N_test]
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"PREDICTING UNCERTAINTY")
            print(f"{'='*60}")
            print(f"Test samples: {len(X_test)}")

        raw = self.predict_raw(X_test, verbose=verbose)
        normalized = self.predict_normalized(X_test, verbose=verbose)

        if verbose:
            print(f"\n{'='*60}")
            print(f"PREDICTION COMPLETE ✓")
            print(f"{'='*60}\n")

        return {
            'raw': raw,
            'normalized': normalized
        }

    def plot_diagnostics(self, save_dir: Optional[Path] = None, prefix: str = ""):
        """
        Plot diagnostic visualizations of the fitted Gaussian and distances.

        Args:
            save_dir: Directory to save plots (if None, don't save)
            prefix: Prefix for saved filenames
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        # Plot 1: Mahalanobis distance distribution
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1a. Raw Mahalanobis distances (histogram)
        axes[0, 0].hist(self.cal_mahal_distances_, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(np.mean(self.cal_mahal_distances_), color='red',
                          linestyle='--', linewidth=2, label=f'Mean: {np.mean(self.cal_mahal_distances_):.2f}')
        axes[0, 0].axvline(np.median(self.cal_mahal_distances_), color='orange',
                          linestyle='--', linewidth=2, label=f'Median: {np.median(self.cal_mahal_distances_):.2f}')
        axes[0, 0].set_xlabel('Mahalanobis Distance', fontsize=12)
        axes[0, 0].set_ylabel('Count', fontsize=12)
        axes[0, 0].set_title('Calibration: Raw Mahalanobis Distance Distribution', fontsize=13, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 1b. Log Mahalanobis distances (histogram)
        log_M = np.log(self.cal_mahal_distances_ + self.eps)
        axes[0, 1].hist(log_M, bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[0, 1].axvline(np.mean(log_M), color='red',
                          linestyle='--', linewidth=2, label=f'Mean: {np.mean(log_M):.2f}')
        axes[0, 1].axvline(self.log_M_min_, color='green',
                          linestyle='--', linewidth=2, label=f'Min: {self.log_M_min_:.2f}')
        axes[0, 1].axvline(self.log_M_max_, color='purple',
                          linestyle='--', linewidth=2, label=f'Max: {self.log_M_max_:.2f}')
        axes[0, 1].set_xlabel('log(Mahalanobis Distance)', fontsize=12)
        axes[0, 1].set_ylabel('Count', fontsize=12)
        axes[0, 1].set_title('Calibration: Log Mahalanobis Distance Distribution', fontsize=13, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 1c. Normalized scores (histogram)
        # Compute normalized scores for calibration
        log_M_cal = np.log(self.cal_mahal_distances_ + self.eps)
        d_norm_cal = (log_M_cal - self.log_M_min_) / (self.log_M_max_ - self.log_M_min_ + self.eps)
        d_norm_cal = np.clip(d_norm_cal, 0.0, 1.0)

        axes[1, 0].hist(d_norm_cal, bins=50, edgecolor='black', alpha=0.7, color='green')
        axes[1, 0].axvline(0.3, color='blue', linestyle='--', linewidth=2, label='Low/Med boundary')
        axes[1, 0].axvline(0.7, color='red', linestyle='--', linewidth=2, label='Med/High boundary')
        axes[1, 0].set_xlabel('Normalized Uncertainty Score', fontsize=12)
        axes[1, 0].set_ylabel('Count', fontsize=12)
        axes[1, 0].set_title('Calibration: Normalized Uncertainty Distribution', fontsize=13, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim(-0.05, 1.05)

        # 1d. Covariance matrix heatmap (first 50x50 for visibility)
        D = min(50, self.cov_.shape[0])
        im = axes[1, 1].imshow(self.cov_[:D, :D], cmap='viridis', aspect='auto')
        axes[1, 1].set_xlabel('Feature Dimension', fontsize=12)
        axes[1, 1].set_ylabel('Feature Dimension', fontsize=12)
        axes[1, 1].set_title(f'Covariance Matrix (first {D}x{D})', fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=axes[1, 1])

        plt.tight_layout()

        if save_dir is not None:
            save_path = save_dir / f"{prefix}mahalanobis_diagnostics.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        else:
            plt.show()

        plt.close()

        # Plot 2: Feature statistics
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 2a. Mean vector
        axes[0].plot(self.mean_, linewidth=2)
        axes[0].set_xlabel('Feature Dimension', fontsize=12)
        axes[0].set_ylabel('Mean Value', fontsize=12)
        axes[0].set_title(f'Mean Vector (μ) - Norm: {np.linalg.norm(self.mean_):.2f}',
                         fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # 2b. Covariance diagonal (variances)
        variances = np.diag(self.cov_)
        axes[1].plot(variances, linewidth=2, color='orange')
        axes[1].set_xlabel('Feature Dimension', fontsize=12)
        axes[1].set_ylabel('Variance', fontsize=12)
        axes[1].set_title(f'Covariance Diagonal (Variances) - Mean: {np.mean(variances):.4f}',
                         fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_dir is not None:
            save_path = save_dir / f"{prefix}gaussian_parameters.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        else:
            plt.show()

        plt.close()
