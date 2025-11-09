"""
Epistemic uncertainty estimation using multiple complementary sources.
Includes density, distance, and entropy-based methods.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky, LinAlgError
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity
import warnings
from .uncertainty_base import UncertaintyModel


class InverseDensityEpistemic(UncertaintyModel):
    """
    Epistemic uncertainty based on inverse density.
    High uncertainty in regions with few calibration samples.
    """

    def __init__(self, bandwidth='scott', name="inverse_density_epistemic"):
        super().__init__(name)
        self.bandwidth = bandwidth
        self.kde = None
        self.max_density = None

    def fit(self, X_cal, y_cal=None, y_pred_cal=None, **kwargs):
        """Fit KDE on calibration features."""
        # Fit KDE
        self.kde = KernelDensity(bandwidth=self.bandwidth, kernel='gaussian')
        self.kde.fit(X_cal)

        # Compute max density for normalization
        log_densities = self.kde.score_samples(X_cal)
        densities = np.exp(log_densities)
        self.max_density = densities.max()

        self.is_fitted = True
        self.metadata = {
            'bandwidth': self.bandwidth if isinstance(self.bandwidth, (int, float)) else str(self.bandwidth),
            'n_cal': len(X_cal),
            'max_density': float(self.max_density)
        }

        return self

    def predict(self, X_test):
        """Compute inverse density epistemic uncertainty."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Compute densities
        log_densities = self.kde.score_samples(X_test)
        densities = np.exp(log_densities)

        # Inverse density (normalized)
        # High density -> low epistemic, low density -> high epistemic
        epistemic = (self.max_density - densities) / (densities + 1e-6)

        # Clip to reasonable range
        epistemic = np.clip(epistemic, 0, 100)

        return epistemic


class MinDistanceEpistemic(UncertaintyModel):
    """
    Epistemic uncertainty based on distance to nearest calibration point.
    Direct measure of novelty.
    """

    def __init__(self, use_mahalanobis=True, regularization=1e-4, name="min_distance_epistemic"):
        super().__init__(name)
        self.use_mahalanobis = use_mahalanobis
        self.regularization = regularization
        self.X_cal = None
        self.L_inv = None
        self.Sigma_inv = None

    def fit(self, X_cal, y_cal=None, y_pred_cal=None, **kwargs):
        """Store calibration data and compute covariance if using Mahalanobis."""
        self.X_cal = X_cal

        if self.use_mahalanobis:
            # Compute covariance matrix
            Sigma = np.cov(X_cal.T)

            # Regularize
            reg_value = self.regularization * np.trace(Sigma) / X_cal.shape[1]
            Sigma_reg = Sigma + reg_value * np.eye(Sigma.shape[0])

            # Try Cholesky decomposition for efficiency
            try:
                L = cholesky(Sigma_reg, lower=True)
                self.L_inv = np.linalg.inv(L)
            except LinAlgError:
                warnings.warn("Cholesky decomposition failed, using direct inverse")
                self.Sigma_inv = np.linalg.inv(Sigma_reg)
                self.L_inv = None

        self.is_fitted = True
        self.metadata = {
            'use_mahalanobis': self.use_mahalanobis,
            'regularization': self.regularization,
            'n_cal': len(X_cal),
            'n_features': X_cal.shape[1]
        }

        return self

    def predict(self, X_test):
        """Compute minimum distance to calibration set."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if self.use_mahalanobis:
            distances = self._mahalanobis_distance(X_test, self.X_cal)
        else:
            distances = cdist(X_test, self.X_cal, metric='euclidean')

        # Minimum distance for each test point
        min_distances = distances.min(axis=1)

        return min_distances

    def _mahalanobis_distance(self, X1, X2):
        """Compute Mahalanobis distance between X1 and X2."""
        if self.L_inv is not None:
            # Efficient computation using Cholesky
            Z1 = X1 @ self.L_inv.T
            Z2 = X2 @ self.L_inv.T
            distances = cdist(Z1, Z2, metric='euclidean')
        else:
            # Direct computation
            distances = np.zeros((X1.shape[0], X2.shape[0]))
            for i in range(X1.shape[0]):
                diff = X1[i:i+1] - X2
                distances[i] = np.sqrt(np.sum(diff @ self.Sigma_inv * diff, axis=1))

        return distances


class EntropyEpistemic(UncertaintyModel):
    """
    Epistemic uncertainty based on feature space entropy.
    High entropy = confused (similar to many different calibration points).
    """

    def __init__(self, temperature='adaptive', use_mahalanobis=True, regularization=1e-4,
                 name="entropy_epistemic"):
        super().__init__(name)
        self.temperature = temperature
        self.use_mahalanobis = use_mahalanobis
        self.regularization = regularization
        self.X_cal = None
        self.L_inv = None
        self.Sigma_inv = None
        self.T = None  # Temperature parameter

    def fit(self, X_cal, y_cal=None, y_pred_cal=None, **kwargs):
        """Fit entropy model on calibration data."""
        self.X_cal = X_cal

        if self.use_mahalanobis:
            # Compute covariance matrix
            Sigma = np.cov(X_cal.T)

            # Regularize
            reg_value = self.regularization * np.trace(Sigma) / X_cal.shape[1]
            Sigma_reg = Sigma + reg_value * np.eye(Sigma.shape[0])

            # Try Cholesky decomposition
            try:
                L = cholesky(Sigma_reg, lower=True)
                self.L_inv = np.linalg.inv(L)
            except LinAlgError:
                warnings.warn("Cholesky decomposition failed, using direct inverse")
                self.Sigma_inv = np.linalg.inv(Sigma_reg)
                self.L_inv = None

        # Compute adaptive temperature
        if self.temperature == 'adaptive':
            # Use median of pairwise distances
            n_samples = min(100, len(X_cal))
            idx = np.random.choice(len(X_cal), n_samples, replace=False)
            X_sample = X_cal[idx]

            if self.use_mahalanobis:
                sample_distances = self._mahalanobis_distance(X_sample, X_sample)
            else:
                sample_distances = cdist(X_sample, X_sample, metric='euclidean')

            # Exclude diagonal (self-distances)
            np.fill_diagonal(sample_distances, np.nan)
            self.T = np.nanmedian(sample_distances)
        else:
            self.T = self.temperature

        self.is_fitted = True
        self.metadata = {
            'temperature': float(self.T),
            'use_mahalanobis': self.use_mahalanobis,
            'n_cal': len(X_cal)
        }

        return self

    def predict(self, X_test):
        """Compute entropy-based epistemic uncertainty."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Compute distances to all calibration points
        if self.use_mahalanobis:
            distances = self._mahalanobis_distance(X_test, self.X_cal)
        else:
            distances = cdist(X_test, self.X_cal, metric='euclidean')

        # Compute softmax probabilities
        # p_k(x) = exp(-d(x, x_k) / T) / sum_j exp(-d(x, x_j) / T)
        exp_values = np.exp(-distances / self.T)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        # Compute entropy
        # H(x) = -sum_k p_k(x) * log(p_k(x))
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        entropy = -np.sum(probabilities * np.log(probabilities + epsilon), axis=1)

        return entropy

    def _mahalanobis_distance(self, X1, X2):
        """Compute Mahalanobis distance between X1 and X2."""
        if self.L_inv is not None:
            Z1 = X1 @ self.L_inv.T
            Z2 = X2 @ self.L_inv.T
            distances = cdist(Z1, Z2, metric='euclidean')
        else:
            distances = np.zeros((X1.shape[0], X2.shape[0]))
            for i in range(X1.shape[0]):
                diff = X1[i:i+1] - X2
                distances[i] = np.sqrt(np.sum(diff @ self.Sigma_inv * diff, axis=1))

        return distances


class MultiSourceEpistemic(UncertaintyModel):
    """
    Multi-source epistemic ensemble combining density, distance, and entropy.
    Weights are learned via optimization.
    """

    def __init__(self, sources=None, use_mahalanobis=True, regularization=1e-4,
                 name="multi_source_epistemic"):
        super().__init__(name)

        # Initialize sources if not provided
        if sources is None:
            self.sources = {
                'density': InverseDensityEpistemic(),
                'distance': MinDistanceEpistemic(use_mahalanobis=use_mahalanobis,
                                                  regularization=regularization),
                'entropy': EntropyEpistemic(use_mahalanobis=use_mahalanobis,
                                            regularization=regularization)
            }
        else:
            self.sources = sources

        self.weights = None
        self.normalization_params = {}

    def fit(self, X_cal, y_cal, y_pred_cal, learn_weights=True, **kwargs):
        """
        Fit all epistemic sources and learn optimal weights.

        Args:
            X_cal: Calibration features
            y_cal: True calibration targets
            y_pred_cal: Predicted calibration targets
            learn_weights: Whether to learn weights via optimization
        """
        # Fit all sources
        for name, source in self.sources.items():
            source.fit(X_cal, y_cal, y_pred_cal)

        # Get predictions on calibration set
        cal_predictions = {}
        for name, source in self.sources.items():
            cal_predictions[name] = source.predict(X_cal)

        # Compute normalization parameters (min-max)
        for name, preds in cal_predictions.items():
            self.normalization_params[name] = {
                'min': preds.min(),
                'max': preds.max(),
                'mean': preds.mean(),
                'std': preds.std()
            }

        # Learn weights
        if learn_weights:
            self.weights = self._learn_weights(X_cal, y_cal, y_pred_cal, cal_predictions)
        else:
            # Equal weights
            n_sources = len(self.sources)
            self.weights = {name: 1.0/n_sources for name in self.sources.keys()}

        self.is_fitted = True
        self.metadata = {
            'n_sources': len(self.sources),
            'weights': self.weights,
            'normalization': self.normalization_params,
            'n_cal': len(X_cal)
        }

        return self

    def _learn_weights(self, X_cal, y_cal, y_pred_cal, cal_predictions):
        """
        Learn optimal weights via SLSQP optimization.
        Maximize correlation with OOD proxy (high-error samples).
        """
        # Create OOD proxy: high-error samples
        errors = np.abs(y_cal - y_pred_cal)
        threshold = np.percentile(errors, 80)  # Top 20% errors
        ood_proxy = (errors > threshold).astype(float)

        # Normalize predictions
        normalized_preds = []
        source_names = []
        for name, preds in cal_predictions.items():
            # Min-max normalization to [0, 1]
            min_val = self.normalization_params[name]['min']
            max_val = self.normalization_params[name]['max']
            normalized = (preds - min_val) / (max_val - min_val + 1e-10)
            normalized_preds.append(normalized)
            source_names.append(name)

        normalized_preds = np.array(normalized_preds).T  # Shape: (n_cal, n_sources)

        # Optimization objective: maximize correlation with OOD proxy
        def objective(weights):
            # Normalize weights
            weights = weights / weights.sum()
            # Compute weighted combination
            combined = normalized_preds @ weights
            # Negative correlation (we're minimizing)
            corr = np.corrcoef(combined, ood_proxy)[0, 1]
            return -corr

        # Constraints: weights sum to 1, all weights >= 0
        constraints = [
            {'type': 'eq', 'fun': lambda w: w.sum() - 1}
        ]
        bounds = [(0, 1) for _ in range(len(self.sources))]

        # Initial guess: equal weights
        x0 = np.ones(len(self.sources)) / len(self.sources)

        # Optimize
        result = minimize(
            objective,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            warnings.warn(f"Weight optimization did not converge: {result.message}")

        # Convert to dictionary
        weights = {name: weight for name, weight in zip(source_names, result.x)}

        print(f"Learned epistemic weights:")
        for name, weight in weights.items():
            print(f"  {name}: {weight:.3f}")

        return weights

    def predict(self, X_test):
        """Compute multi-source epistemic uncertainty."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Get predictions from all sources
        predictions = {}
        for name, source in self.sources.items():
            predictions[name] = source.predict(X_test)

        # Normalize
        normalized_preds = {}
        for name, preds in predictions.items():
            min_val = self.normalization_params[name]['min']
            max_val = self.normalization_params[name]['max']
            normalized = (preds - min_val) / (max_val - min_val + 1e-10)
            normalized = np.clip(normalized, 0, 1)  # Ensure [0, 1]
            normalized_preds[name] = normalized

        # Weighted combination
        combined = np.zeros(len(X_test))
        for name, weight in self.weights.items():
            combined += weight * normalized_preds[name]

        return combined

    def predict_all_sources(self, X_test):
        """
        Get predictions from all individual sources plus combined.
        Useful for analysis and comparison.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        results = {}

        # Individual sources (raw)
        for name, source in self.sources.items():
            results[f'{name}_raw'] = source.predict(X_test)

        # Individual sources (normalized)
        for name in self.sources.keys():
            min_val = self.normalization_params[name]['min']
            max_val = self.normalization_params[name]['max']
            normalized = (results[f'{name}_raw'] - min_val) / (max_val - min_val + 1e-10)
            results[f'{name}_norm'] = np.clip(normalized, 0, 1)

        # Combined
        results['combined'] = self.predict(X_test)

        # Add weights for reference
        results['weights'] = self.weights

        return results