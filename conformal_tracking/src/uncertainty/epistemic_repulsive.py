"""
Repulsive Void Detection for Epistemic Uncertainty

This module implements epistemic uncertainty quantification via physics-inspired
repulsive force fields. Points in knowledge voids between training clusters
experience high repulsive forces from all directions.

Author: Enhanced CACD Team
Date: 2025-11-10
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, Optional
import seaborn as sns
from scipy.spatial.distance import cdist


class RepulsiveVoidDetector:
    """
    Epistemic uncertainty via repulsive force field analysis

    Core idea: Knowledge gaps create "voids" in feature space where test points
    experience repulsive forces from all directions, indicating high epistemic uncertainty.
    """

    def __init__(self, k_neighbors: int = 100, temperature: float = 1.0, verbose: bool = False):
        """
        Initialize repulsive void detector

        Args:
            k_neighbors: Number of neighbors for force computation
            temperature: Controls force decay rate (higher = longer range)
            verbose: Whether to print debug information
        """
        self.k_neighbors = k_neighbors
        self.temperature = temperature
        self.verbose = verbose

        # Normalization statistics
        self.mean_repulsive_force = None
        self.std_repulsive_force = None
        self.max_repulsive_force = None

        # Calibration data
        self.X_cal = None
        self.cal_diagnostics = {}

    def fit(self, X_calibration: np.ndarray,
            mahalanobis_model=None,
            plot_diagnostics: bool = True,
            save_dir: Optional[Path] = None):
        """
        Fit repulsive model on calibration data

        Args:
            X_calibration: Calibration features [N_cal, D]
            mahalanobis_model: Pre-fitted Mahalanobis model for distance computation
            plot_diagnostics: Whether to generate diagnostic plots
            save_dir: Directory to save plots
        """
        self.X_cal = X_calibration
        self.mahalanobis_model = mahalanobis_model
        self.feature_dim = X_calibration.shape[1]

        if self.verbose:
            print("\n" + "="*60)
            print("FITTING REPULSIVE VOID DETECTOR")
            print("="*60)
            print(f"Calibration samples: {len(X_calibration)}")
            print(f"Feature dimension: {self.feature_dim}")
            print(f"K neighbors: {self.k_neighbors}")
            print(f"Temperature: {self.temperature}")

        # Compute repulsive forces on calibration subset
        n_samples = min(500, len(X_calibration))
        sample_idx = np.random.choice(len(X_calibration), n_samples, replace=False)

        cal_forces = []
        cal_magnitudes = []
        cal_entropies = []
        cal_components = []

        if self.verbose:
            print(f"\nComputing repulsive forces on {n_samples} samples...")

        for i, idx in enumerate(sample_idx):
            if i % 100 == 0 and self.verbose:
                print(f"  Processing sample {i}/{n_samples}")

            # Compute repulsive force for this calibration point
            force_mag, force_vec, diagnostics = self._compute_repulsive_force(
                X_calibration[idx], X_calibration, return_components=True
            )

            cal_magnitudes.append(force_mag)
            cal_forces.append(force_vec)
            cal_entropies.append(diagnostics['direction_entropy'])
            cal_components.append(diagnostics['force_components'][:10])  # Top 10

        cal_magnitudes = np.array(cal_magnitudes)
        cal_entropies = np.array(cal_entropies)

        # Compute normalization statistics
        self.mean_repulsive_force = np.mean(cal_magnitudes)
        self.std_repulsive_force = np.std(cal_magnitudes)
        self.max_repulsive_force = np.percentile(cal_magnitudes, 95)

        # Store diagnostics
        self.cal_diagnostics = {
            'force_magnitudes': cal_magnitudes,
            'direction_entropies': cal_entropies,
            'force_vectors': np.array(cal_forces),
            'force_components': np.array(cal_components),
            'mean_magnitude': self.mean_repulsive_force,
            'std_magnitude': self.std_repulsive_force
        }

        if self.verbose:
            print(f"\nCalibration Statistics:")
            print(f"  Mean force magnitude: {self.mean_repulsive_force:.4f}")
            print(f"  Std force magnitude:  {self.std_repulsive_force:.4f}")
            print(f"  95th percentile:      {self.max_repulsive_force:.4f}")
            print(f"  Mean direction entropy: {np.mean(cal_entropies):.3f}")

        # Generate diagnostic plots
        if plot_diagnostics and save_dir is not None:
            self._plot_calibration_diagnostics(save_dir)

        if self.verbose:
            print("\n" + "="*60)
            print("REPULSIVE FITTING COMPLETE ✓")
            print("="*60 + "\n")

    def predict(self, X_test: np.ndarray,
                return_diagnostics: bool = True) -> Dict[str, np.ndarray]:
        """
        Predict repulsive epistemic uncertainty for test samples

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
        force_magnitudes = np.zeros(n_test)
        direction_entropies = np.zeros(n_test)
        force_vectors = []

        for i in range(n_test):
            # Compute repulsive force
            force_mag, force_vec, diagnostics = self._compute_repulsive_force(
                X_test[i], self.X_cal, return_components=True
            )

            # Normalize to [0, 1]
            epistemic_normalized = self._normalize_force(force_mag)

            epistemic[i] = epistemic_normalized
            force_magnitudes[i] = force_mag
            direction_entropies[i] = diagnostics['direction_entropy']
            force_vectors.append(force_vec)

        results = {'epistemic': epistemic}

        if return_diagnostics:
            results['force_magnitudes'] = force_magnitudes
            results['direction_entropies'] = direction_entropies
            results['force_vectors'] = np.array(force_vectors)
            results['normalized_magnitudes'] = force_magnitudes / self.mean_repulsive_force

        return results

    def _compute_repulsive_force(self, x_test: np.ndarray,
                                  X_reference: np.ndarray,
                                  return_components: bool = False) -> Tuple:
        """
        Compute repulsive force field at test point

        Args:
            x_test: Test feature vector [D]
            X_reference: Reference features [N, D]
            return_components: Whether to return force components

        Returns:
            (force_magnitude, force_vector, diagnostics)
        """
        # Compute distances
        # Always use Euclidean distance for finding neighbors
        # (Mahalanobis is for distance from the mean, not between points)
        distances = np.linalg.norm(X_reference - x_test, axis=1)

        # Avoid self-repulsion if test point is in reference set
        distances[distances < 1e-10] = np.inf

        # Get k nearest neighbors
        k = min(self.k_neighbors, len(X_reference))
        nearest_idx = np.argsort(distances)[:k]

        # Compute repulsive forces (Coulomb-like with temperature)
        forces = []
        force_magnitudes = []

        for idx in nearest_idx:
            x_i = X_reference[idx]
            d_i = distances[idx]

            # Direction vector (from neighbor to test point)
            direction = x_test - x_i
            direction_norm = np.linalg.norm(direction)

            if direction_norm > 1e-10:
                direction = direction / direction_norm
            else:
                # Random direction for identical points
                direction = np.random.randn(len(direction))
                direction = direction / np.linalg.norm(direction)

            # Coulomb-like force with temperature modulation
            # F = exp(-d/T) / d²
            magnitude = np.exp(-d_i / self.temperature) / (d_i**2 + 1e-6)

            force = direction * magnitude
            forces.append(force)
            force_magnitudes.append(magnitude)

        forces = np.array(forces)
        force_magnitudes = np.array(force_magnitudes)

        # Net repulsive force
        net_force = np.sum(forces, axis=0)
        net_magnitude = np.linalg.norm(net_force)

        # Compute direction entropy (high entropy = forces from all directions = in void)
        direction_entropy = self._compute_direction_entropy(forces)

        diagnostics = {
            'direction_entropy': direction_entropy,
            'force_components': force_magnitudes,
            'n_neighbors_used': k,
            'mean_distance': np.mean(distances[nearest_idx]),
            'min_distance': np.min(distances[nearest_idx])
        }

        if return_components:
            return net_magnitude, net_force, diagnostics
        else:
            return net_magnitude, net_force, None

    def _compute_direction_entropy(self, forces: np.ndarray) -> float:
        """
        Compute entropy of force directions

        High entropy = forces from many directions = in void
        Low entropy = forces from similar directions = near cluster edge
        """
        if len(forces) < 2:
            return 0.0

        # Normalize force vectors
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        normalized = forces / (force_norms + 1e-10)

        # Compute pairwise cosine similarities
        similarities = normalized @ normalized.T

        # Extract upper triangle (avoid diagonal and duplicates)
        upper_triangle = np.triu(similarities, k=1)
        pairwise_sims = upper_triangle[upper_triangle != 0]

        if len(pairwise_sims) == 0:
            return 0.0

        # Convert similarities to angles
        # cos(θ) ranges from -1 to 1
        # θ ranges from 0 to π
        angles = np.arccos(np.clip(pairwise_sims, -1, 1))

        # Compute entropy of angle distribution
        # Normalize angles to [0, 1]
        angles_norm = angles / np.pi

        # Create histogram for entropy calculation
        hist, _ = np.histogram(angles_norm, bins=10, range=(0, 1))
        hist = hist + 1  # Add pseudocount
        probs = hist / hist.sum()

        # Shannon entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        return entropy

    def _normalize_force(self, force_magnitude: float) -> float:
        """
        Normalize force magnitude to [0, 1] epistemic uncertainty
        """
        # Z-score normalization with clipping
        z_score = (force_magnitude - self.mean_repulsive_force) / \
                  (self.std_repulsive_force + 1e-10)

        # Convert to [0, 1] using sigmoid-like transformation
        # Maps z=-2 to ~0.02, z=0 to 0.5, z=2 to ~0.98
        epistemic = 1 / (1 + np.exp(-z_score))

        return np.clip(epistemic, 0, 1)

    def _plot_calibration_diagnostics(self, save_dir: Path):
        """
        Generate diagnostic plots for repulsive force analysis
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Plot 1: Force magnitude distribution
        axes[0, 0].hist(self.cal_diagnostics['force_magnitudes'], bins=30,
                       edgecolor='black', alpha=0.7, color='red')
        axes[0, 0].axvline(self.mean_repulsive_force, color='blue', linestyle='--',
                          label=f'Mean: {self.mean_repulsive_force:.4f}')
        axes[0, 0].axvline(self.max_repulsive_force, color='green', linestyle='--',
                          label=f'95th %ile: {self.max_repulsive_force:.4f}')
        axes[0, 0].set_xlabel('Force Magnitude', fontsize=12)
        axes[0, 0].set_ylabel('Count', fontsize=12)
        axes[0, 0].set_title('Calibration Force Magnitude Distribution', fontsize=13, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Direction entropy distribution
        axes[0, 1].hist(self.cal_diagnostics['direction_entropies'], bins=30,
                       edgecolor='black', alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Direction Entropy', fontsize=12)
        axes[0, 1].set_ylabel('Count', fontsize=12)
        axes[0, 1].set_title('Force Direction Entropy Distribution', fontsize=13, fontweight='bold')
        axes[0, 1].axvline(self.cal_diagnostics['direction_entropies'].mean(),
                          color='red', linestyle='--',
                          label=f"Mean: {self.cal_diagnostics['direction_entropies'].mean():.3f}")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Force magnitude vs direction entropy
        axes[0, 2].scatter(self.cal_diagnostics['force_magnitudes'],
                          self.cal_diagnostics['direction_entropies'],
                          alpha=0.5, s=20)
        axes[0, 2].set_xlabel('Force Magnitude', fontsize=12)
        axes[0, 2].set_ylabel('Direction Entropy', fontsize=12)
        axes[0, 2].set_title('Force Magnitude vs Direction Diversity', fontsize=13, fontweight='bold')

        # Add correlation
        corr = np.corrcoef(self.cal_diagnostics['force_magnitudes'],
                          self.cal_diagnostics['direction_entropies'])[0, 1]
        axes[0, 2].text(0.05, 0.95, f'Correlation: {corr:.3f}',
                       transform=axes[0, 2].transAxes, fontsize=11,
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        axes[0, 2].grid(True, alpha=0.3)

        # Plot 4: Top force components (average)
        mean_components = self.cal_diagnostics['force_components'].mean(axis=0)
        std_components = self.cal_diagnostics['force_components'].std(axis=0)

        x_pos = np.arange(len(mean_components))
        axes[1, 0].bar(x_pos, mean_components, yerr=std_components,
                      capsize=5, color='purple', alpha=0.7)
        axes[1, 0].set_xlabel('Neighbor Rank', fontsize=12)
        axes[1, 0].set_ylabel('Force Component', fontsize=12)
        axes[1, 0].set_title('Average Force Components (Top 10 Neighbors)', fontsize=13, fontweight='bold')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels([f'{i+1}' for i in range(len(mean_components))])
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # Plot 5: Force vector magnitudes (2D projection)
        # Use PCA for visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        force_2d = pca.fit_transform(self.cal_diagnostics['force_vectors'])

        scatter = axes[1, 1].scatter(force_2d[:, 0], force_2d[:, 1],
                                    c=self.cal_diagnostics['force_magnitudes'],
                                    cmap='RdYlBu_r', alpha=0.6, s=30)
        axes[1, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        axes[1, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        axes[1, 1].set_title('Force Vectors (PCA Projection)', fontsize=13, fontweight='bold')
        plt.colorbar(scatter, ax=axes[1, 1], label='Force Magnitude')

        # Plot 6: Normalization function
        force_range = np.linspace(0, self.max_repulsive_force * 1.5, 100)
        epistemic_values = [self._normalize_force(f) for f in force_range]

        axes[1, 2].plot(force_range, epistemic_values, linewidth=2, color='red')
        axes[1, 2].set_xlabel('Force Magnitude', fontsize=12)
        axes[1, 2].set_ylabel('Epistemic Uncertainty [0,1]', fontsize=12)
        axes[1, 2].set_title('Force → Epistemic Transformation', fontsize=13, fontweight='bold')
        axes[1, 2].grid(True, alpha=0.3)

        # Add key points
        axes[1, 2].axvline(self.mean_repulsive_force, color='blue', linestyle='--', alpha=0.5,
                          label='Mean force')
        axes[1, 2].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        axes[1, 2].legend()

        plt.suptitle('Repulsive Void Calibration Diagnostics', fontsize=15, fontweight='bold')
        plt.tight_layout()

        save_path = save_dir / 'repulsive_calibration_diagnostics.png'
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
        axes[0, 0].set_xlabel('Repulsive Epistemic Uncertainty', fontsize=12)
        axes[0, 0].set_ylabel('Count', fontsize=12)
        axes[0, 0].set_title('Test Epistemic Distribution', fontsize=13, fontweight='bold')
        axes[0, 0].axvline(results['epistemic'].mean(), color='blue', linestyle='--',
                          label=f"Mean: {results['epistemic'].mean():.3f}")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Force magnitude distribution
        axes[0, 1].hist(results['force_magnitudes'], bins=30,
                       edgecolor='black', alpha=0.7, color='orange')
        axes[0, 1].set_xlabel('Force Magnitude', fontsize=12)
        axes[0, 1].set_ylabel('Count', fontsize=12)
        axes[0, 1].set_title('Test Force Magnitude Distribution', fontsize=13, fontweight='bold')
        axes[0, 1].axvline(results['force_magnitudes'].mean(), color='red', linestyle='--',
                          label=f"Test Mean: {results['force_magnitudes'].mean():.4f}")
        axes[0, 1].axvline(self.mean_repulsive_force, color='blue', linestyle='--',
                          label=f"Cal Mean: {self.mean_repulsive_force:.4f}")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Direction entropy comparison
        test_entropy = results['direction_entropies']
        cal_entropy = self.cal_diagnostics['direction_entropies']

        axes[1, 0].violinplot([cal_entropy, test_entropy], positions=[1, 2],
                             showmeans=True, showmedians=True)
        axes[1, 0].set_xticks([1, 2])
        axes[1, 0].set_xticklabels(['Calibration', 'Test'])
        axes[1, 0].set_ylabel('Direction Entropy', fontsize=12)
        axes[1, 0].set_title('Calibration vs Test Direction Entropy', fontsize=13, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # Plot 4: Force magnitude vs epistemic
        axes[1, 1].scatter(results['force_magnitudes'], results['epistemic'],
                          alpha=0.5, s=20, c=results['epistemic'], cmap='RdYlBu_r')
        axes[1, 1].set_xlabel('Force Magnitude', fontsize=12)
        axes[1, 1].set_ylabel('Epistemic Uncertainty', fontsize=12)
        axes[1, 1].set_title('Force → Epistemic Mapping', fontsize=13, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        # Add reference lines
        axes[1, 1].axvline(self.mean_repulsive_force, color='blue', linestyle='--', alpha=0.5,
                          label='Cal mean')
        axes[1, 1].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        axes[1, 1].legend()

        plt.suptitle(f'{prefix}Repulsive Epistemic Test Diagnostics', fontsize=15, fontweight='bold')
        plt.tight_layout()

        save_path = save_dir / f'{prefix}repulsive_test_diagnostics.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

        plt.close()

        # Print summary statistics
        print("\n" + "="*60)
        print("REPULSIVE EPISTEMIC TEST STATISTICS")
        print("="*60)
        print(f"Epistemic Uncertainty:")
        print(f"  Mean: {results['epistemic'].mean():.4f}")
        print(f"  Std:  {results['epistemic'].std():.4f}")
        print(f"  Min:  {results['epistemic'].min():.4f}")
        print(f"  Max:  {results['epistemic'].max():.4f}")
        print(f"\nForce Magnitudes:")
        print(f"  Mean: {results['force_magnitudes'].mean():.4f}")
        print(f"  Std:  {results['force_magnitudes'].std():.4f}")
        print(f"  Normalized mean: {results['normalized_magnitudes'].mean():.3f}")
        print(f"\nDirection Entropy:")
        print(f"  Mean: {results['direction_entropies'].mean():.3f}")
        print(f"  Std:  {results['direction_entropies'].std():.3f}")
        print("="*60)