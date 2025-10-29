"""
Aleatoric Uncertainty Computation
Core uncertainty quantification methods borrowed from GitHub repos
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AleatricUncertaintyEstimator:
    """
    Compute aleatoric uncertainty from multiple predictions
    Based on concepts from:
    - Bayesian-Neural-Networks/src/MC_dropout/model.py
    - uncertainty-toolbox/uncertainty_toolbox/metrics.py
    """

    def __init__(self, combine_method: str = "variance"):
        """
        Initialize uncertainty estimator

        Args:
            combine_method: How to combine predictions ("variance", "entropy", "mutual_info")
        """
        self.combine_method = combine_method

    def compute_bbox_uncertainty(
        self,
        bbox_predictions: np.ndarray,
        return_components: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Compute uncertainty for bounding box predictions

        Args:
            bbox_predictions: Array of shape (N, 4) with N predictions of [x, y, w, h]
            return_components: Whether to return per-component variance

        Returns:
            Dictionary with uncertainty metrics
        """
        if len(bbox_predictions) == 0:
            return {
                'mean': np.zeros(4),
                'variance': np.zeros(4),
                'std': np.zeros(4),
                'total_uncertainty': 0.0
            }

        # Compute statistics
        mean = np.mean(bbox_predictions, axis=0)
        variance = np.var(bbox_predictions, axis=0)
        std = np.std(bbox_predictions, axis=0)

        # Total uncertainty (average variance across components)
        total_uncertainty = np.mean(variance)

        result = {
            'mean': mean,
            'variance': variance,
            'std': std,
            'total_uncertainty': total_uncertainty
        }

        if return_components:
            # Add per-component uncertainties
            result['x_variance'] = variance[0]
            result['y_variance'] = variance[1]
            result['w_variance'] = variance[2]
            result['h_variance'] = variance[3]

            # Compute center and size uncertainties
            result['center_uncertainty'] = np.mean(variance[:2])  # x, y
            result['size_uncertainty'] = np.mean(variance[2:])    # w, h

        return result

    def compute_confidence_uncertainty(
        self,
        confidence_predictions: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute uncertainty for confidence predictions

        Args:
            confidence_predictions: Array of shape (N,) with N confidence scores

        Returns:
            Dictionary with confidence uncertainty metrics
        """
        if len(confidence_predictions) == 0:
            return {
                'mean': 0.0,
                'variance': 0.0,
                'std': 0.0,
                'entropy': 0.0
            }

        mean = np.mean(confidence_predictions)
        variance = np.var(confidence_predictions)
        std = np.std(confidence_predictions)

        # Compute entropy of confidence distribution
        # Treat as probability distribution
        probs = confidence_predictions / (np.sum(confidence_predictions) + 1e-8)
        entropy = -np.sum(probs * np.log(probs + 1e-8))

        return {
            'mean': float(mean),
            'variance': float(variance),
            'std': float(std),
            'entropy': float(entropy)
        }

    def compute_combined_uncertainty(
        self,
        bbox_variance: float,
        confidence_variance: float,
        bbox_weight: float = 0.7,
        confidence_weight: float = 0.3
    ) -> float:
        """
        Compute combined uncertainty metric

        Args:
            bbox_variance: Bounding box variance
            confidence_variance: Confidence score variance
            bbox_weight: Weight for bbox component
            confidence_weight: Weight for confidence component

        Returns:
            Combined uncertainty score
        """
        # Normalize weights
        total_weight = bbox_weight + confidence_weight
        bbox_weight = bbox_weight / total_weight
        confidence_weight = confidence_weight / total_weight

        # Weighted combination
        combined = bbox_weight * bbox_variance + confidence_weight * confidence_variance

        return float(combined)

    def compute_predictive_variance(
        self,
        predictions: np.ndarray,
        axis: int = 0
    ) -> np.ndarray:
        """
        Compute predictive variance across multiple predictions
        From uncertainty-toolbox/uncertainty_toolbox/metrics.py

        Args:
            predictions: Array of predictions
            axis: Axis along which to compute variance

        Returns:
            Predictive variance
        """
        return np.var(predictions, axis=axis)

    def compute_mutual_information(
        self,
        predictions: np.ndarray,
        epsilon: float = 1e-8
    ) -> float:
        """
        Compute mutual information for uncertainty
        MI = H(E[p]) - E[H(p)]

        Args:
            predictions: Array of predictions (N, K) for N samples, K predictions
            epsilon: Small value for numerical stability

        Returns:
            Mutual information
        """
        # Convert to probabilities if not already
        if predictions.min() < 0 or predictions.max() > 1:
            # Normalize to [0, 1]
            predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min() + epsilon)

        # Mean prediction across samples
        mean_pred = np.mean(predictions, axis=0)

        # Entropy of mean prediction
        entropy_of_mean = -np.sum(mean_pred * np.log(mean_pred + epsilon))

        # Mean of entropies
        entropies = []
        for pred in predictions:
            entropy = -np.sum(pred * np.log(pred + epsilon))
            entropies.append(entropy)
        mean_of_entropy = np.mean(entropies)

        # Mutual information
        mi = entropy_of_mean - mean_of_entropy

        return float(mi)

    def compute_temporal_consistency(
        self,
        uncertainty_sequence: np.ndarray,
        window_size: int = 5
    ) -> Dict[str, float]:
        """
        Compute temporal consistency metrics for uncertainty

        Args:
            uncertainty_sequence: Time series of uncertainty values
            window_size: Window size for smoothing

        Returns:
            Temporal consistency metrics
        """
        if len(uncertainty_sequence) < 2:
            return {
                'temporal_correlation': 0.0,
                'temporal_variance': 0.0,
                'smoothness': 0.0
            }

        # Temporal correlation (autocorrelation at lag 1)
        if len(uncertainty_sequence) > 1:
            temporal_corr = np.corrcoef(
                uncertainty_sequence[:-1],
                uncertainty_sequence[1:]
            )[0, 1]
        else:
            temporal_corr = 0.0

        # Temporal variance
        temporal_var = np.var(uncertainty_sequence)

        # Smoothness (inverse of average absolute difference)
        diffs = np.abs(np.diff(uncertainty_sequence))
        smoothness = 1.0 / (np.mean(diffs) + 1e-8)

        return {
            'temporal_correlation': float(temporal_corr) if not np.isnan(temporal_corr) else 0.0,
            'temporal_variance': float(temporal_var),
            'smoothness': float(smoothness)
        }

    def apply_temporal_smoothing(
        self,
        uncertainty_sequence: np.ndarray,
        alpha: float = 0.3
    ) -> np.ndarray:
        """
        Apply exponential smoothing to uncertainty sequence

        Args:
            uncertainty_sequence: Time series of uncertainty values
            alpha: Smoothing factor (0 = no smoothing, 1 = no memory)

        Returns:
            Smoothed uncertainty sequence
        """
        if len(uncertainty_sequence) == 0:
            return uncertainty_sequence

        smoothed = np.zeros_like(uncertainty_sequence)
        smoothed[0] = uncertainty_sequence[0]

        for t in range(1, len(uncertainty_sequence)):
            smoothed[t] = alpha * uncertainty_sequence[t] + (1 - alpha) * smoothed[t - 1]

        return smoothed


class UncertaintyMetrics:
    """
    Additional uncertainty metrics borrowed from uncertainty-toolbox
    """

    @staticmethod
    def expected_calibration_error(
        confidences: np.ndarray,
        accuracies: np.ndarray,
        num_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error (ECE)
        From uncertainty-toolbox/uncertainty_toolbox/metrics_calibration.py

        Args:
            confidences: Predicted confidences
            accuracies: Binary accuracy indicators
            num_bins: Number of bins for calibration

        Returns:
            ECE value
        """
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return float(ece)

    @staticmethod
    def sharpness(y_std: np.ndarray) -> float:
        """
        Compute sharpness (average predictive standard deviation)
        From uncertainty-toolbox/uncertainty_toolbox/metrics_calibration.py

        Args:
            y_std: Predicted standard deviations

        Returns:
            Sharpness value
        """
        return float(np.mean(y_std))

    @staticmethod
    def compute_iou_variance(
        boxes1: np.ndarray,
        boxes2: np.ndarray
    ) -> float:
        """
        Compute variance of IoU scores between two sets of boxes

        Args:
            boxes1: First set of boxes (N, 4)
            boxes2: Second set of boxes (N, 4)

        Returns:
            IoU variance
        """
        ious = []
        for b1, b2 in zip(boxes1, boxes2):
            iou = UncertaintyMetrics._compute_single_iou(b1, b2)
            ious.append(iou)

        return float(np.var(ious))

    @staticmethod
    def _compute_single_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0] + box1[2], box2[0] + box2[2])
        y2 = min(box1[1] + box1[3], box2[1] + box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union = area1 + area2 - intersection

        return intersection / (union + 1e-8)


if __name__ == "__main__":
    # Test uncertainty computation
    import sys
    sys.path.append('/ssd_4TB/divake/temporal_uncertainty')

    # Create estimator
    estimator = AleatricUncertaintyEstimator()

    # Test bbox uncertainty
    bbox_preds = np.array([
        [100, 100, 50, 50],
        [102, 98, 48, 52],
        [99, 101, 51, 49],
        [101, 99, 49, 51],
        [100, 100, 50, 50]
    ])

    bbox_uncertainty = estimator.compute_bbox_uncertainty(bbox_preds)
    print("Bbox Uncertainty:")
    print(f"  Mean: {bbox_uncertainty['mean']}")
    print(f"  Variance: {bbox_uncertainty['variance']}")
    print(f"  Total: {bbox_uncertainty['total_uncertainty']:.4f}")

    # Test confidence uncertainty
    conf_preds = np.array([0.8, 0.75, 0.82, 0.79, 0.81])
    conf_uncertainty = estimator.compute_confidence_uncertainty(conf_preds)
    print(f"\nConfidence Uncertainty:")
    print(f"  Mean: {conf_uncertainty['mean']:.3f}")
    print(f"  Variance: {conf_uncertainty['variance']:.4f}")
    print(f"  Entropy: {conf_uncertainty['entropy']:.4f}")

    # Test combined uncertainty
    combined = estimator.compute_combined_uncertainty(
        bbox_uncertainty['total_uncertainty'],
        conf_uncertainty['variance']
    )
    print(f"\nCombined Uncertainty: {combined:.4f}")

    # Test temporal consistency
    uncertainty_seq = np.array([0.1, 0.12, 0.15, 0.14, 0.16, 0.13, 0.11, 0.10])
    temporal = estimator.compute_temporal_consistency(uncertainty_seq)
    print(f"\nTemporal Consistency:")
    print(f"  Correlation: {temporal['temporal_correlation']:.3f}")
    print(f"  Smoothness: {temporal['smoothness']:.3f}")