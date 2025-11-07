"""
Aleatoric uncertainty estimation using KNN approaches.
Includes both baseline (Euclidean) and enhanced (Mahalanobis) versions.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import cholesky, LinAlgError
from scipy.spatial.distance import cdist
import warnings


class BaselineAleatoric:
    """
    Original Method D aleatoric uncertainty using Euclidean KNN.
    This serves as our baseline for comparison.
    """

    def __init__(self, k_neighbors=10):
        self.k = k_neighbors
        self.knn = None
        self.residuals_cal = None

    def fit(self, X_cal, y_cal, y_pred_cal):
        """
        Fit the KNN model on calibration data.

        Args:
            X_cal: Calibration features (already scaled)
            y_cal: True calibration targets
            y_pred_cal: Predicted calibration targets
        """
        # Store residuals for each calibration point
        self.residuals_cal = y_cal - y_pred_cal

        # Fit KNN with Euclidean distance
        self.knn = NearestNeighbors(
            n_neighbors=self.k,
            algorithm='auto',
            metric='euclidean'
        )
        self.knn.fit(X_cal)

        return self

    def predict(self, X_test):
        """
        Compute aleatoric uncertainty for test points.

        Args:
            X_test: Test features (already scaled)

        Returns:
            Array of aleatoric uncertainties
        """
        # Find K nearest neighbors
        distances, indices = self.knn.kneighbors(X_test)

        # Compute local variance of residuals
        aleatoric = []
        for i, neighbor_idx in enumerate(indices):
            neighbor_residuals = self.residuals_cal[neighbor_idx]
            # Use standard deviation as uncertainty
            local_std = np.std(neighbor_residuals)
            aleatoric.append(local_std)

        return np.array(aleatoric)

    def predict_with_details(self, X_test):
        """
        Compute aleatoric with additional details for analysis.

        Returns dictionary with:
            - aleatoric: uncertainty values
            - distances: distances to neighbors
            - indices: indices of neighbors
            - weights: uniform weights (1/K)
        """
        distances, indices = self.knn.kneighbors(X_test)

        aleatoric = []
        all_weights = []

        for i, neighbor_idx in enumerate(indices):
            neighbor_residuals = self.residuals_cal[neighbor_idx]
            local_std = np.std(neighbor_residuals)
            aleatoric.append(local_std)

            # Uniform weights for baseline
            weights = np.ones(self.k) / self.k
            all_weights.append(weights)

        return {
            'aleatoric': np.array(aleatoric),
            'distances': distances,
            'indices': indices,
            'weights': np.array(all_weights)
        }


class EnhancedAleatoric:
    """
    Enhanced aleatoric uncertainty using Mahalanobis distance and softmax weighting.
    Key improvements:
    1. Mahalanobis distance accounts for feature correlations
    2. Softmax weights give more importance to closer neighbors
    3. Weighted variance computation
    """

    def __init__(self, k_neighbors=10, regularization=1e-4):
        self.k = k_neighbors
        self.regularization = regularization
        self.residuals_cal = None
        self.X_cal = None
        self.Sigma_inv = None
        self.L_inv = None  # For efficient Mahalanobis computation
        self.bandwidth = None

    def fit(self, X_cal, y_cal, y_pred_cal):
        """
        Fit the enhanced aleatoric model.

        Args:
            X_cal: Calibration features (already scaled)
            y_cal: True calibration targets
            y_pred_cal: Predicted calibration targets
        """
        self.X_cal = X_cal
        self.residuals_cal = y_cal - y_pred_cal

        # Compute covariance matrix
        self.Sigma = np.cov(X_cal.T)

        # Regularize for numerical stability
        reg_value = self.regularization * np.trace(self.Sigma) / X_cal.shape[1]
        self.Sigma_reg = self.Sigma + reg_value * np.eye(self.Sigma.shape[0])

        # Compute Cholesky decomposition for efficient Mahalanobis distance
        try:
            L = cholesky(self.Sigma_reg, lower=True)
            self.L_inv = np.linalg.inv(L)
        except LinAlgError:
            warnings.warn("Cholesky decomposition failed, falling back to direct inverse")
            self.Sigma_inv = np.linalg.inv(self.Sigma_reg)
            self.L_inv = None

        # Compute adaptive bandwidth using median heuristic
        self._compute_bandwidth()

        return self

    def _compute_bandwidth(self, n_samples=100):
        """Compute adaptive bandwidth using median of pairwise distances."""
        # Sample random pairs to estimate median distance
        n_cal = self.X_cal.shape[0]
        idx1 = np.random.choice(n_cal, min(n_samples, n_cal), replace=True)
        idx2 = np.random.choice(n_cal, min(n_samples, n_cal), replace=True)

        distances = []
        for i1, i2 in zip(idx1, idx2):
            if i1 != i2:
                dist = self._mahalanobis_distance(
                    self.X_cal[i1:i1+1],
                    self.X_cal[i2:i2+1]
                )[0]
                distances.append(dist)

        if distances:
            self.bandwidth = np.median(distances) / np.sqrt(2)
        else:
            self.bandwidth = 1.0

    def _mahalanobis_distance(self, X1, X2):
        """
        Compute Mahalanobis distance between X1 and X2.

        Args:
            X1: First set of points (n1, d)
            X2: Second set of points (n2, d)

        Returns:
            Distance matrix (n1, n2)
        """
        if self.L_inv is not None:
            # Efficient computation using Cholesky
            Z1 = X1 @ self.L_inv.T
            Z2 = X2 @ self.L_inv.T
            distances = cdist(Z1, Z2, metric='euclidean')
        else:
            # Fallback to direct computation
            distances = np.zeros((X1.shape[0], X2.shape[0]))
            for i in range(X1.shape[0]):
                diff = X1[i:i+1] - X2
                distances[i] = np.sqrt(np.sum(diff @ self.Sigma_inv * diff, axis=1))

        return distances

    def _find_k_nearest(self, X_test):
        """Find K nearest neighbors using Mahalanobis distance."""
        distances = self._mahalanobis_distance(X_test, self.X_cal)

        # Get K nearest neighbors for each test point
        indices = np.argsort(distances, axis=1)[:, :self.k]

        # Extract corresponding distances
        k_distances = np.array([
            distances[i, indices[i]]
            for i in range(X_test.shape[0])
        ])

        return k_distances, indices

    def _compute_softmax_weights(self, distances):
        """
        Compute softmax weights based on Mahalanobis distances.

        Args:
            distances: Array of distances to K neighbors (n_test, K)

        Returns:
            Softmax weights (n_test, K)
        """
        # Use squared distances in exponential (Gaussian kernel)
        exp_values = np.exp(-distances**2 / (2 * self.bandwidth**2))

        # Normalize to get weights
        weights = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        return weights

    def predict(self, X_test):
        """
        Compute enhanced aleatoric uncertainty for test points.

        Args:
            X_test: Test features (already scaled)

        Returns:
            Array of aleatoric uncertainties
        """
        # Find K nearest neighbors using Mahalanobis distance
        distances, indices = self._find_k_nearest(X_test)

        # Compute softmax weights
        weights = self._compute_softmax_weights(distances)

        # Compute weighted aleatoric uncertainty
        aleatoric = []
        for i, (neighbor_idx, w) in enumerate(zip(indices, weights)):
            neighbor_residuals = self.residuals_cal[neighbor_idx]

            # Weighted variance computation
            weighted_mean = np.sum(w * neighbor_residuals)
            weighted_var = np.sum(w * (neighbor_residuals - weighted_mean)**2)
            aleatoric.append(np.sqrt(weighted_var))

        return np.array(aleatoric)

    def predict_with_details(self, X_test):
        """
        Compute aleatoric with additional details for analysis.

        Returns dictionary with comprehensive details.
        """
        distances, indices = self._find_k_nearest(X_test)
        weights = self._compute_softmax_weights(distances)

        aleatoric = []
        weighted_means = []
        weighted_vars = []

        for i, (neighbor_idx, w) in enumerate(zip(indices, weights)):
            neighbor_residuals = self.residuals_cal[neighbor_idx]

            weighted_mean = np.sum(w * neighbor_residuals)
            weighted_var = np.sum(w * (neighbor_residuals - weighted_mean)**2)

            aleatoric.append(np.sqrt(weighted_var))
            weighted_means.append(weighted_mean)
            weighted_vars.append(weighted_var)

        return {
            'aleatoric': np.array(aleatoric),
            'distances': distances,
            'indices': indices,
            'weights': weights,
            'weighted_means': np.array(weighted_means),
            'weighted_vars': np.array(weighted_vars),
            'bandwidth': self.bandwidth
        }


class HybridAleatoric:
    """
    Hybrid approach: Mahalanobis distance with uniform weights.
    This helps isolate the contribution of distance metric vs weighting.
    """

    def __init__(self, k_neighbors=10, regularization=1e-4):
        self.enhanced = EnhancedAleatoric(k_neighbors, regularization)
        self.k = k_neighbors

    def fit(self, X_cal, y_cal, y_pred_cal):
        """Fit using enhanced distance computation."""
        self.enhanced.fit(X_cal, y_cal, y_pred_cal)
        return self

    def predict(self, X_test):
        """Predict using Mahalanobis distance but uniform weights."""
        # Find neighbors using Mahalanobis
        distances, indices = self.enhanced._find_k_nearest(X_test)

        # But use uniform weights
        aleatoric = []
        for neighbor_idx in indices:
            neighbor_residuals = self.enhanced.residuals_cal[neighbor_idx]
            local_std = np.std(neighbor_residuals)
            aleatoric.append(local_std)

        return np.array(aleatoric)