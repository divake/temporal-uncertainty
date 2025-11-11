"""
Combined Epistemic Uncertainty: Triple-S Framework

This module combines Spectral collapse, Spatial (repulsive), and Statistical methods
to provide comprehensive epistemic uncertainty quantification.

Author: Enhanced CACD Team
Date: 2025-11-10
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from scipy.optimize import minimize
import json

from epistemic_spectral import SpectralCollapseDetector
from epistemic_repulsive import RepulsiveVoidDetector
from epistemic_gradient import GradientDivergenceDetector


class EpistemicUncertainty:
    """
    Combined epistemic uncertainty using multiple orthogonal sources

    Sources:
    1. Spectral: Feature manifold collapse detection
    2. Repulsive: Void detection via force fields
    3. Gradient: Inter-layer feature divergence (optional)
    """

    def __init__(self,
                 k_neighbors_spectral: int = 50,
                 k_neighbors_repulsive: int = 100,
                 temperature: float = 1.0,
                 weights: Union[str, list] = 'equal',
                 verbose: bool = False):
        """
        Initialize combined epistemic uncertainty estimator

        Args:
            k_neighbors_spectral: Neighbors for spectral analysis
            k_neighbors_repulsive: Neighbors for repulsive forces
            temperature: Temperature for repulsive forces
            weights: 'equal', 'optimize', or list of weights [spectral, repulsive, gradient]
            verbose: Whether to print debug information
        """
        # Initialize component detectors
        self.spectral_detector = SpectralCollapseDetector(
            k_neighbors=k_neighbors_spectral,
            verbose=verbose
        )

        self.repulsive_detector = RepulsiveVoidDetector(
            k_neighbors=k_neighbors_repulsive,
            temperature=temperature,
            verbose=verbose
        )

        self.gradient_detector = GradientDivergenceDetector(
            verbose=verbose
        )

        # Weight configuration
        if weights == 'equal':
            self.weights = [0.5, 0.5, 0.0]  # No gradient by default
        elif weights == 'optimize':
            self.weights = None  # Will be optimized during fit
            self.optimize_weights_flag = True
        elif isinstance(weights, list) and len(weights) == 3:
            self.weights = weights
            self.optimize_weights_flag = False
        else:
            raise ValueError("weights must be 'equal', 'optimize', or list of 3 values")

        self.verbose = verbose
        self.is_fitted = False

        # Store calibration results for analysis
        self.cal_results = {}
        self.fit_diagnostics = {}

    def fit(self, X_calibration: np.ndarray,
            X_cal_layers: Optional[Dict[int, np.ndarray]] = None,
            mahalanobis_model=None,
            aleatoric_cal: Optional[np.ndarray] = None,
            conformity_cal: Optional[np.ndarray] = None,
            plot_diagnostics: bool = True,
            save_dir: Optional[Path] = None):
        """
        Fit combined epistemic model on calibration data

        Args:
            X_calibration: Calibration features [N_cal, D]
            mahalanobis_model: Pre-fitted Mahalanobis model
            aleatoric_cal: Aleatoric uncertainty on calibration (for orthogonality)
            conformity_cal: Conformity scores (for weight optimization)
            plot_diagnostics: Whether to generate diagnostic plots
            save_dir: Directory to save plots
        """
        if self.verbose:
            print("\n" + "="*80)
            print("FITTING COMBINED EPISTEMIC UNCERTAINTY MODEL")
            print("="*80)

        # Create save directory if needed
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Fit individual components
        if self.verbose:
            print("\n[1/3] Fitting Spectral Collapse Detector...")

        self.spectral_detector.fit(
            X_calibration,
            mahalanobis_model=mahalanobis_model,
            plot_diagnostics=plot_diagnostics,
            save_dir=save_dir
        )

        if self.verbose:
            print("\n[2/3] Fitting Repulsive Void Detector...")

        self.repulsive_detector.fit(
            X_calibration,
            mahalanobis_model=mahalanobis_model,
            plot_diagnostics=plot_diagnostics,
            save_dir=save_dir
        )

        # Fit gradient detector if multi-layer data provided
        if X_cal_layers is not None:
            if self.verbose:
                print("\n[2.5/3] Fitting Gradient Divergence Detector...")

            self.gradient_detector.fit(
                X_cal_layers,
                mahalanobis_model=mahalanobis_model,
                save_dir=save_dir
            )
            gradient_available = True
        else:
            gradient_available = False
            if self.verbose:
                print("\n⚠️  No multi-layer data provided, gradient method disabled")

        # Step 2: Compute epistemic on calibration set for weight optimization
        if self.verbose:
            print("\n[3/3] Computing calibration epistemic components...")

        spectral_cal = self.spectral_detector.predict(X_calibration, return_diagnostics=False)
        repulsive_cal = self.repulsive_detector.predict(X_calibration, return_diagnostics=False)

        if gradient_available:
            gradient_cal = self.gradient_detector.predict(X_cal_layers, return_diagnostics=False)
            gradient_uncertainty = gradient_cal['epistemic']
        else:
            gradient_uncertainty = np.zeros(len(X_calibration))

        self.cal_results = {
            'spectral': spectral_cal['epistemic'],
            'repulsive': repulsive_cal['epistemic'],
            'gradient': gradient_uncertainty
        }
        self.gradient_available = gradient_available

        # Step 3: Optimize weights if requested
        if hasattr(self, 'optimize_weights_flag') and self.optimize_weights_flag:
            if aleatoric_cal is not None:
                self._optimize_weights(aleatoric_cal, conformity_cal)
            else:
                if self.verbose:
                    print("\n⚠️ Warning: No aleatoric provided, using equal weights")
                self.weights = [0.5, 0.5, 0.0]

        # Compute combined calibration epistemic
        self.cal_results['combined'] = self._combine_sources(
            self.cal_results['spectral'],
            self.cal_results['repulsive'],
            self.cal_results['gradient']
        )

        self.is_fitted = True

        # Generate diagnostic plots
        if plot_diagnostics and save_dir is not None:
            self._plot_fit_diagnostics(aleatoric_cal, conformity_cal, save_dir)

        if self.verbose:
            print("\n" + "="*80)
            print("COMBINED EPISTEMIC FITTING COMPLETE ✓")
            print("="*80)
            print(f"Final weights:")
            print(f"  Spectral:  {self.weights[0]:.3f}")
            print(f"  Repulsive: {self.weights[1]:.3f}")
            print(f"  Gradient:  {self.weights[2]:.3f}")
            print("="*80 + "\n")

    def predict(self, X_test: np.ndarray,
                X_test_layers: Optional[Dict[int, np.ndarray]] = None,
                return_components: bool = True,
                plot_diagnostics: bool = False,
                save_dir: Optional[Path] = None) -> Dict[str, np.ndarray]:
        """
        Predict combined epistemic uncertainty for test samples

        Args:
            X_test: Test features [N_test, D]
            X_test_layers: Optional dict mapping layer IDs to features for gradient method
            return_components: Whether to return individual components
            plot_diagnostics: Whether to generate diagnostic plots
            save_dir: Directory to save plots

        Returns:
            Dictionary with 'combined' and optionally component uncertainties
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Get predictions from each component
        spectral_results = self.spectral_detector.predict(X_test, return_diagnostics=True)
        repulsive_results = self.repulsive_detector.predict(X_test, return_diagnostics=True)

        # Get gradient predictions if available
        if hasattr(self, 'gradient_available') and self.gradient_available and X_test_layers is not None:
            gradient_results = self.gradient_detector.predict(X_test_layers, return_diagnostics=True)
            gradient_uncertainty = gradient_results['epistemic']
        else:
            gradient_uncertainty = np.zeros_like(spectral_results['epistemic'])
            gradient_results = None

        # Combine with weights
        combined = self._combine_sources(
            spectral_results['epistemic'],
            repulsive_results['epistemic'],
            gradient_uncertainty
        )

        results = {'combined': combined}

        if return_components:
            results['spectral'] = spectral_results['epistemic']
            results['repulsive'] = repulsive_results['epistemic']
            results['gradient'] = gradient_uncertainty

            # Add diagnostics
            results['spectral_diagnostics'] = {
                'effective_ranks': spectral_results['effective_ranks'],
                'entropies': spectral_results['entropies']
            }
            results['repulsive_diagnostics'] = {
                'force_magnitudes': repulsive_results['force_magnitudes'],
                'direction_entropies': repulsive_results['direction_entropies']
            }

            # Add gradient diagnostics if available
            if gradient_results is not None:
                results['gradient_diagnostics'] = {
                    'pair_divergences': gradient_results.get('pair_divergences', {}),
                    'mean_divergence': gradient_results.get('mean_divergence', None)
                }

        # Generate test diagnostic plots
        if plot_diagnostics and save_dir is not None:
            self._plot_test_diagnostics(results, save_dir)

            # Also plot individual component diagnostics
            self.spectral_detector.plot_test_diagnostics(
                spectral_results, save_dir, prefix="combined_"
            )
            self.repulsive_detector.plot_test_diagnostics(
                repulsive_results, save_dir, prefix="combined_"
            )

            # Plot gradient diagnostics if available
            if gradient_results is not None:
                self.gradient_detector.plot_test_diagnostics(
                    gradient_results, save_dir, prefix="combined_"
                )

        return results

    def _combine_sources(self, spectral: np.ndarray,
                        repulsive: np.ndarray,
                        gradient: np.ndarray) -> np.ndarray:
        """
        Combine epistemic sources with weights
        """
        combined = (
            self.weights[0] * spectral +
            self.weights[1] * repulsive +
            self.weights[2] * gradient
        )
        return np.clip(combined, 0, 1)

    def _optimize_weights(self, aleatoric_cal: np.ndarray,
                         conformity_cal: Optional[np.ndarray] = None):
        """
        Optimize weights to maximize orthogonality with aleatoric

        Args:
            aleatoric_cal: Aleatoric uncertainty on calibration set
            conformity_cal: Conformity scores (optional, for secondary objective)
        """
        if self.verbose:
            print("\nOptimizing weights for orthogonality with aleatoric...")

        def objective(w):
            # Compute weighted epistemic
            epistemic = (
                w[0] * self.cal_results['spectral'] +
                w[1] * self.cal_results['repulsive'] +
                w[2] * self.cal_results['gradient']
            )

            # Primary: Minimize absolute correlation with aleatoric
            corr_aleatoric = np.abs(np.corrcoef(epistemic, aleatoric_cal)[0, 1])

            # Secondary: Maximize correlation with conformity (if provided)
            if conformity_cal is not None:
                corr_conformity = np.corrcoef(epistemic, conformity_cal)[0, 1]
                # Combine objectives (orthogonality is more important)
                loss = corr_aleatoric - 0.1 * corr_conformity
            else:
                loss = corr_aleatoric

            # Add small penalty for extreme weights
            weight_penalty = 0.01 * np.std(w)

            return loss + weight_penalty

        # Constraints: weights sum to 1, all non-negative
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'ineq', 'fun': lambda w: w[0]},  # w[0] >= 0
            {'type': 'ineq', 'fun': lambda w: w[1]},  # w[1] >= 0
            {'type': 'ineq', 'fun': lambda w: w[2]}   # w[2] >= 0
        ]

        # Initial guess
        w0 = [0.45, 0.45, 0.1]  # Slight preference for spectral and repulsive

        # Optimize
        result = minimize(
            objective, w0,
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 100}
        )

        if result.success:
            self.weights = result.x
            if self.verbose:
                print(f"  Optimization successful!")
                print(f"  Optimized weights: [{self.weights[0]:.3f}, "
                      f"{self.weights[1]:.3f}, {self.weights[2]:.3f}]")

                # Report final correlations
                epistemic_opt = self._combine_sources(
                    self.cal_results['spectral'],
                    self.cal_results['repulsive'],
                    self.cal_results['gradient']
                )
                corr_final = np.corrcoef(epistemic_opt, aleatoric_cal)[0, 1]
                print(f"  Final correlation with aleatoric: {corr_final:.4f}")
        else:
            if self.verbose:
                print(f"  ⚠️ Optimization failed: {result.message}")
                print("  Using equal weights as fallback")
            self.weights = [0.5, 0.5, 0.0]

    def _plot_fit_diagnostics(self, aleatoric_cal: Optional[np.ndarray],
                             conformity_cal: Optional[np.ndarray],
                             save_dir: Path):
        """
        Generate diagnostic plots for model fitting
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Plot 1: Component distributions
        axes[0, 0].hist(self.cal_results['spectral'], bins=30, alpha=0.5,
                       label='Spectral', color='blue', edgecolor='black')
        axes[0, 0].hist(self.cal_results['repulsive'], bins=30, alpha=0.5,
                       label='Repulsive', color='red', edgecolor='black')
        axes[0, 0].hist(self.cal_results['combined'], bins=30, alpha=0.5,
                       label='Combined', color='purple', edgecolor='black')
        axes[0, 0].set_xlabel('Epistemic Uncertainty', fontsize=12)
        axes[0, 0].set_ylabel('Count', fontsize=12)
        axes[0, 0].set_title('Calibration Epistemic Components', fontsize=13, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Component correlations
        corr_matrix = np.corrcoef([
            self.cal_results['spectral'],
            self.cal_results['repulsive'],
            self.cal_results['combined']
        ])

        im = axes[0, 1].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0, 1].set_xticks([0, 1, 2])
        axes[0, 1].set_yticks([0, 1, 2])
        axes[0, 1].set_xticklabels(['Spectral', 'Repulsive', 'Combined'])
        axes[0, 1].set_yticklabels(['Spectral', 'Repulsive', 'Combined'])
        axes[0, 1].set_title('Component Correlations', fontsize=13, fontweight='bold')

        # Add correlation values
        for i in range(3):
            for j in range(3):
                axes[0, 1].text(j, i, f'{corr_matrix[i, j]:.2f}',
                              ha='center', va='center', color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')

        plt.colorbar(im, ax=axes[0, 1])

        # Plot 3: Weights visualization
        weights_labels = ['Spectral', 'Repulsive', 'Gradient']
        colors = ['blue', 'red', 'green']
        axes[0, 2].bar(weights_labels, self.weights, color=colors, alpha=0.7, edgecolor='black')
        axes[0, 2].set_ylabel('Weight', fontsize=12)
        axes[0, 2].set_title('Optimized Weights', fontsize=13, fontweight='bold')
        axes[0, 2].set_ylim([0, 1])
        axes[0, 2].grid(True, alpha=0.3, axis='y')

        # Add values on bars
        for i, (label, weight) in enumerate(zip(weights_labels, self.weights)):
            axes[0, 2].text(i, weight + 0.02, f'{weight:.3f}',
                          ha='center', fontsize=11, fontweight='bold')

        # Plot 4: Spectral vs Repulsive scatter
        axes[1, 0].scatter(self.cal_results['spectral'], self.cal_results['repulsive'],
                          alpha=0.5, s=20)
        axes[1, 0].set_xlabel('Spectral Epistemic', fontsize=12)
        axes[1, 0].set_ylabel('Repulsive Epistemic', fontsize=12)
        axes[1, 0].set_title('Spectral vs Repulsive Components', fontsize=13, fontweight='bold')

        # Add correlation
        corr_sr = np.corrcoef(self.cal_results['spectral'], self.cal_results['repulsive'])[0, 1]
        axes[1, 0].text(0.05, 0.95, f'Correlation: {corr_sr:.3f}',
                       transform=axes[1, 0].transAxes, fontsize=11,
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 5: Orthogonality check (if aleatoric provided)
        if aleatoric_cal is not None:
            axes[1, 1].scatter(aleatoric_cal, self.cal_results['combined'],
                             alpha=0.5, s=20, c=self.cal_results['combined'], cmap='RdYlBu_r')
            axes[1, 1].set_xlabel('Aleatoric Uncertainty', fontsize=12)
            axes[1, 1].set_ylabel('Epistemic Uncertainty', fontsize=12)
            axes[1, 1].set_title('Aleatoric vs Epistemic (Orthogonality Check)', fontsize=13, fontweight='bold')

            # Add correlation
            corr_ae = np.corrcoef(aleatoric_cal, self.cal_results['combined'])[0, 1]
            axes[1, 1].text(0.05, 0.95, f'Correlation: {corr_ae:.3f}',
                           transform=axes[1, 1].transAxes, fontsize=11,
                           bbox=dict(boxstyle='round',
                                   facecolor='green' if abs(corr_ae) < 0.3 else 'red',
                                   alpha=0.5))
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No aleatoric data provided',
                          ha='center', va='center', fontsize=14)
            axes[1, 1].set_title('Orthogonality Check (N/A)', fontsize=13)

        # Plot 6: Epistemic vs Conformity (if provided)
        if conformity_cal is not None:
            axes[1, 2].scatter(self.cal_results['combined'], conformity_cal,
                             alpha=0.5, s=20)
            axes[1, 2].set_xlabel('Combined Epistemic', fontsize=12)
            axes[1, 2].set_ylabel('Conformity Score', fontsize=12)
            axes[1, 2].set_title('Epistemic vs Conformity', fontsize=13, fontweight='bold')

            # Add correlation
            corr_ec = np.corrcoef(self.cal_results['combined'], conformity_cal)[0, 1]
            axes[1, 2].text(0.05, 0.95, f'Correlation: {corr_ec:.3f}',
                           transform=axes[1, 2].transAxes, fontsize=11,
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].text(0.5, 0.5, 'No conformity data provided',
                          ha='center', va='center', fontsize=14)
            axes[1, 2].set_title('Conformity Correlation (N/A)', fontsize=13)

        plt.suptitle('Combined Epistemic Model Fitting Diagnostics', fontsize=15, fontweight='bold')
        plt.tight_layout()

        save_path = save_dir / 'combined_epistemic_fit_diagnostics.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

        plt.close()

    def _plot_test_diagnostics(self, results: Dict, save_dir: Path):
        """
        Generate diagnostic plots for test predictions
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Component distributions
        axes[0, 0].hist(results['spectral'], bins=30, alpha=0.5,
                       label='Spectral', color='blue', edgecolor='black')
        axes[0, 0].hist(results['repulsive'], bins=30, alpha=0.5,
                       label='Repulsive', color='red', edgecolor='black')
        axes[0, 0].hist(results['combined'], bins=30, alpha=0.5,
                       label='Combined', color='purple', edgecolor='black')
        axes[0, 0].set_xlabel('Epistemic Uncertainty', fontsize=12)
        axes[0, 0].set_ylabel('Count', fontsize=12)
        axes[0, 0].set_title('Test Epistemic Components', fontsize=13, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Component scatter
        axes[0, 1].scatter(results['spectral'], results['repulsive'],
                          alpha=0.5, s=20, c=results['combined'], cmap='viridis')
        axes[0, 1].set_xlabel('Spectral Epistemic', fontsize=12)
        axes[0, 1].set_ylabel('Repulsive Epistemic', fontsize=12)
        axes[0, 1].set_title('Component Relationship', fontsize=13, fontweight='bold')

        corr = np.corrcoef(results['spectral'], results['repulsive'])[0, 1]
        axes[0, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}',
                       transform=axes[0, 1].transAxes, fontsize=11,
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        axes[0, 1].grid(True, alpha=0.3)

        cbar = plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1])
        cbar.set_label('Combined', fontsize=10)

        # Plot 3: Component contribution
        means = [results['spectral'].mean(), results['repulsive'].mean(), results['gradient'].mean()]
        stds = [results['spectral'].std(), results['repulsive'].std(), results['gradient'].std()]
        labels = ['Spectral', 'Repulsive', 'Gradient']
        colors = ['blue', 'red', 'green']

        x_pos = np.arange(len(labels))
        axes[1, 0].bar(x_pos, means, yerr=stds, capsize=5,
                      color=colors, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(labels)
        axes[1, 0].set_ylabel('Mean Epistemic', fontsize=12)
        axes[1, 0].set_title('Component Contributions (mean ± std)', fontsize=13, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # Plot 4: Combined distribution with statistics
        axes[1, 1].hist(results['combined'], bins=30, edgecolor='black',
                       alpha=0.7, color='purple')
        axes[1, 1].set_xlabel('Combined Epistemic Uncertainty', fontsize=12)
        axes[1, 1].set_ylabel('Count', fontsize=12)
        axes[1, 1].set_title('Combined Epistemic Distribution', fontsize=13, fontweight='bold')

        # Add statistics
        mean_val = results['combined'].mean()
        std_val = results['combined'].std()
        axes[1, 1].axvline(mean_val, color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {mean_val:.3f}')
        axes[1, 1].axvline(mean_val - std_val, color='orange', linestyle='--',
                          label=f'±1 std: {std_val:.3f}')
        axes[1, 1].axvline(mean_val + std_val, color='orange', linestyle='--')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle('Combined Epistemic Test Diagnostics', fontsize=15, fontweight='bold')
        plt.tight_layout()

        save_path = save_dir / 'combined_epistemic_test_diagnostics.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

        plt.close()

        # Print summary
        print("\n" + "="*60)
        print("COMBINED EPISTEMIC TEST STATISTICS")
        print("="*60)
        print(f"Combined Epistemic:")
        print(f"  Mean: {results['combined'].mean():.4f}")
        print(f"  Std:  {results['combined'].std():.4f}")
        print(f"  Min:  {results['combined'].min():.4f}")
        print(f"  Max:  {results['combined'].max():.4f}")
        print(f"\nComponent Means:")
        print(f"  Spectral:  {results['spectral'].mean():.4f}")
        print(f"  Repulsive: {results['repulsive'].mean():.4f}")
        print(f"  Gradient:  {results['gradient'].mean():.4f}")
        print(f"\nWeighted Contribution:")
        print(f"  Spectral:  {self.weights[0] * results['spectral'].mean():.4f} "
              f"({self.weights[0]*100:.1f}%)")
        print(f"  Repulsive: {self.weights[1] * results['repulsive'].mean():.4f} "
              f"({self.weights[1]*100:.1f}%)")
        print(f"  Gradient:  {self.weights[2] * results['gradient'].mean():.4f} "
              f"({self.weights[2]*100:.1f}%)")
        print("="*60)

    def save_model(self, save_path: Path):
        """
        Save model parameters and weights
        """
        save_dict = {
            'weights': self.weights.tolist() if isinstance(self.weights, np.ndarray) else self.weights,
            'spectral_params': {
                'k_neighbors': self.spectral_detector.k_neighbors,
                'cal_min_entropy': float(self.spectral_detector.cal_min_entropy),
                'cal_max_entropy': float(self.spectral_detector.cal_max_entropy),
                'cal_mean_eff_rank': float(self.spectral_detector.cal_mean_eff_rank)
            },
            'repulsive_params': {
                'k_neighbors': self.repulsive_detector.k_neighbors,
                'temperature': self.repulsive_detector.temperature,
                'mean_repulsive_force': float(self.repulsive_detector.mean_repulsive_force),
                'std_repulsive_force': float(self.repulsive_detector.std_repulsive_force)
            }
        }

        with open(save_path, 'w') as f:
            json.dump(save_dict, f, indent=2)

        print(f"Model saved to: {save_path}")