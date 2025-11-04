"""
Method D: Hybrid Approach

Key idea: Don't try to decompose conformal scores at all.
Instead:
1. Use vanilla CP for coverage (guaranteed!)
2. Compute aleatoric from LOCAL DATA properties (variance, heteroscedasticity)
3. Compute epistemic from MODEL properties (density, distance to training)
4. Report both independently without forcing them to sum to quantile

This is more principled: each uncertainty uses the RIGHT method for its concept.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KernelDensity


class MethodD_CACD:
    """Hybrid CACD using different methods for each uncertainty"""

    def __init__(self, alpha=0.1, k_neighbors=10, kde_bandwidth='scott'):
        self.alpha = alpha
        self.k_neighbors = k_neighbors
        self.kde_bandwidth = kde_bandwidth

        self.scaler = StandardScaler()
        self.vanilla_quantile = None
        self.X_cal_scaled = None
        self.y_cal = None

    def calibrate(self, X_cal, y_cal, y_pred_cal, verbose=True):
        """
        Calibrate using vanilla CP + fit KNN/KDE

        No neural network training needed!
        """
        if verbose:
            print(f"\n{'='*70}")
            print("METHOD D: HYBRID APPROACH")
            print(f"{'='*70}")

        # Conformal scores
        cal_scores = np.abs(y_cal - y_pred_cal)

        # Vanilla quantile (coverage guarantee)
        self.vanilla_quantile = np.quantile(cal_scores, 1 - self.alpha)

        if verbose:
            print(f"\nVanilla Quantile: {self.vanilla_quantile:.4f}")

        # Store calibration data for test-time computation
        self.X_cal_scaled = self.scaler.fit_transform(X_cal)
        self.y_cal = y_cal
        self.residuals_cal = y_cal - y_pred_cal

        # Fit KDE for density estimation
        if verbose:
            print(f"Fitting KDE for epistemic uncertainty...")

        self.kde = KernelDensity(bandwidth=self.kde_bandwidth, kernel='gaussian')
        self.kde.fit(self.X_cal_scaled)

        # Fit KNN for local variance
        if verbose:
            print(f"Fitting KNN (k={self.k_neighbors}) for aleatoric uncertainty...")

        self.knn = NearestNeighbors(n_neighbors=self.k_neighbors, algorithm='auto')
        self.knn.fit(self.X_cal_scaled)

        if verbose:
            print(f"Calibration complete!")

        return self

    def _compute_aleatoric(self, X_test_scaled):
        """
        Compute aleatoric uncertainty from local variance

        For each test point:
        1. Find K nearest neighbors in calibration set
        2. Compute variance of their RESIDUALS
        3. This estimates local noise level
        """
        distances, indices = self.knn.kneighbors(X_test_scaled)

        aleatoric = []
        for neighbor_idx in indices:
            # Get residuals of neighbors
            neighbor_residuals = self.residuals_cal[neighbor_idx]
            # Variance = aleatoric noise
            local_var = np.var(neighbor_residuals)
            aleatoric.append(np.sqrt(local_var))  # Convert to std

        return np.array(aleatoric)

    def _compute_epistemic(self, X_test_scaled):
        """
        Compute epistemic uncertainty from density

        For each test point:
        1. Compute density using KDE
        2. Inverse density = epistemic uncertainty
        3. High uncertainty in sparse regions
        """
        log_densities = self.kde.score_samples(X_test_scaled)
        densities = np.exp(log_densities)

        # Inverse density (normalized)
        max_density = densities.max()
        epistemic = (max_density / (densities + 1e-6)) - 1.0  # Subtract 1 so dense regions have ~0

        return epistemic

    def predict(self, X_test, y_pred_test):
        """
        Predict with uncertainty decomposition

        Key difference: aleatoric and epistemic are INDEPENDENT
        They don't need to sum to vanilla_quantile!
        """
        X_test_scaled = self.scaler.transform(X_test)

        # Compute aleatoric (from local variance)
        aleatoric_raw = self._compute_aleatoric(X_test_scaled)

        # Compute epistemic (from density)
        epistemic_raw = self._compute_epistemic(X_test_scaled)

        # Normalize each independently to [0, 1]
        aleatoric_norm = (aleatoric_raw - aleatoric_raw.min()) / (aleatoric_raw.max() - aleatoric_raw.min() + 1e-6)
        epistemic_norm = (epistemic_raw - epistemic_raw.min()) / (epistemic_raw.max() - epistemic_raw.min() + 1e-6)

        # Scale to reasonable magnitude (use quantile as reference)
        aleatoric = aleatoric_norm * self.vanilla_quantile
        epistemic = epistemic_norm * self.vanilla_quantile

        # Intervals use vanilla quantile (coverage guarantee!)
        lower = y_pred_test - self.vanilla_quantile
        upper = y_pred_test + self.vanilla_quantile

        return {
            'lower': lower,
            'upper': upper,
            'aleatoric': aleatoric,
            'epistemic': epistemic,
            'mu_s': aleatoric_raw,  # For compatibility
            'sigma_s': epistemic_raw
        }

    def evaluate(self, X_test, y_test, y_pred_test):
        """Evaluate performance"""
        results = self.predict(X_test, y_pred_test)

        coverage = np.mean((y_test >= results['lower']) & (y_test <= results['upper']))
        width = np.mean(results['upper'] - results['lower'])
        correlation = np.corrcoef(results['aleatoric'], results['epistemic'])[0, 1]

        errors = np.abs(y_test - y_pred_test)
        # How well does aleatoric correlate with errors?
        alea_quality = np.corrcoef(errors, results['aleatoric'])[0, 1]
        # How well does epistemic correlate with errors?
        epis_quality = np.corrcoef(errors, results['epistemic'])[0, 1]

        return {
            'coverage': coverage,
            'width': width,
            'correlation': correlation,
            'mu_s_quality': alea_quality,  # Aleatoric quality
            'sigma_s_quality': epis_quality,  # Epistemic quality
            'aleatoric_mean': results['aleatoric'].mean(),
            'epistemic_mean': results['epistemic'].mean()
        }


class MethodD_v2_CACD:
    """
    Variant: Use conformal scores directly instead of residuals

    Aleatoric = local variance of CONFORMAL SCORES
    Epistemic = density-based
    """

    def __init__(self, alpha=0.1, k_neighbors=10, kde_bandwidth='scott'):
        self.alpha = alpha
        self.k_neighbors = k_neighbors
        self.kde_bandwidth = kde_bandwidth

        self.scaler = StandardScaler()
        self.vanilla_quantile = None
        self.X_cal_scaled = None
        self.cal_scores = None

    def calibrate(self, X_cal, y_cal, y_pred_cal, verbose=True):
        """Calibrate using conformal scores"""

        if verbose:
            print(f"\n{'='*70}")
            print("METHOD D (v2): HYBRID WITH SCORE VARIANCE")
            print(f"{'='*70}")

        # Conformal scores
        cal_scores = np.abs(y_cal - y_pred_cal)

        # Vanilla quantile
        self.vanilla_quantile = np.quantile(cal_scores, 1 - self.alpha)

        if verbose:
            print(f"\nVanilla Quantile: {self.vanilla_quantile:.4f}")

        # Store calibration data
        self.X_cal_scaled = self.scaler.fit_transform(X_cal)
        self.cal_scores = cal_scores

        # Fit KDE for density
        if verbose:
            print(f"Fitting KDE for epistemic uncertainty...")

        self.kde = KernelDensity(bandwidth=self.kde_bandwidth, kernel='gaussian')
        self.kde.fit(self.X_cal_scaled)

        # Fit KNN for local score variance
        if verbose:
            print(f"Fitting KNN (k={self.k_neighbors}) for aleatoric uncertainty...")

        self.knn = NearestNeighbors(n_neighbors=self.k_neighbors, algorithm='auto')
        self.knn.fit(self.X_cal_scaled)

        if verbose:
            print(f"Calibration complete!")

        return self

    def _compute_aleatoric(self, X_test_scaled):
        """Compute from local variance of conformal scores"""
        distances, indices = self.knn.kneighbors(X_test_scaled)

        aleatoric = []
        for neighbor_idx in indices:
            neighbor_scores = self.cal_scores[neighbor_idx]
            local_std = np.std(neighbor_scores)
            aleatoric.append(local_std)

        return np.array(aleatoric)

    def _compute_epistemic(self, X_test_scaled):
        """Compute from density"""
        log_densities = self.kde.score_samples(X_test_scaled)
        densities = np.exp(log_densities)

        max_density = densities.max()
        epistemic = (max_density / (densities + 1e-6)) - 1.0

        return epistemic

    def predict(self, X_test, y_pred_test):
        """Predict with decomposition"""
        X_test_scaled = self.scaler.transform(X_test)

        aleatoric_raw = self._compute_aleatoric(X_test_scaled)
        epistemic_raw = self._compute_epistemic(X_test_scaled)

        # Normalize independently
        aleatoric_norm = (aleatoric_raw - aleatoric_raw.min()) / (aleatoric_raw.max() - aleatoric_raw.min() + 1e-6)
        epistemic_norm = (epistemic_raw - epistemic_raw.min()) / (epistemic_raw.max() - epistemic_raw.min() + 1e-6)

        # Scale by quantile
        aleatoric = aleatoric_norm * self.vanilla_quantile
        epistemic = epistemic_norm * self.vanilla_quantile

        # Intervals
        lower = y_pred_test - self.vanilla_quantile
        upper = y_pred_test + self.vanilla_quantile

        return {
            'lower': lower,
            'upper': upper,
            'aleatoric': aleatoric,
            'epistemic': epistemic,
            'mu_s': aleatoric_raw,
            'sigma_s': epistemic_raw
        }

    def evaluate(self, X_test, y_test, y_pred_test):
        """Evaluate performance"""
        results = self.predict(X_test, y_pred_test)

        coverage = np.mean((y_test >= results['lower']) & (y_test <= results['upper']))
        width = np.mean(results['upper'] - results['lower'])
        correlation = np.corrcoef(results['aleatoric'], results['epistemic'])[0, 1]

        errors = np.abs(y_test - y_pred_test)
        alea_quality = np.corrcoef(errors, results['aleatoric'])[0, 1]
        epis_quality = np.corrcoef(errors, results['epistemic'])[0, 1]

        return {
            'coverage': coverage,
            'width': width,
            'correlation': correlation,
            'mu_s_quality': alea_quality,
            'sigma_s_quality': epis_quality,
            'aleatoric_mean': results['aleatoric'].mean(),
            'epistemic_mean': results['epistemic'].mean()
        }
