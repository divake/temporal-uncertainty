"""
Save Per-Detection Uncertainty Data for Real Plotting

This script re-runs the uncertainty analysis and saves the actual per-detection
values instead of just summary statistics, enabling real (not synthetic) plots.

Author: Analysis Team
Date: 2025-11-11
"""

import numpy as np
import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import YOLOCacheLoader
from src.uncertainty.mahalanobis import MahalanobisUncertainty
from src.uncertainty.epistemic_combined import EpistemicUncertainty


def save_per_detection_data(sequence_name: str = 'MOT17-05'):
    """
    Recompute uncertainties and save per-detection values

    Args:
        sequence_name: MOT17 sequence to process
    """

    print(f"\n{'='*80}")
    print(f"SAVING PER-DETECTION DATA FOR {sequence_name}")
    print(f"{'='*80}\n")

    # Create output directory
    output_dir = Path(__file__).parent.parent / 'results' / f'per_detection_{sequence_name.lower()}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data from cache
    print(f"[1/4] Loading data from cache...")
    cache_path = Path(f'/ssd_4TB/divake/temporal_uncertainty/yolo_cache/data/mot17/yolov8n/{sequence_name}-FRCNN.npz')

    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    loader = YOLOCacheLoader(cache_path)
    print(f"  Loaded: {loader}")

    # Get features and ground truth
    features = loader.get_features(layer_id=9)  # Use layer 9
    ious = loader.get_ious()
    confidences = loader.get_confidences()
    matched_mask = loader.get_matched_mask()

    # Filter to only matched detections
    features = features[matched_mask]
    ious = ious
    confidences = confidences[matched_mask]

    print(f"  Total detections: {matched_mask.sum()}")
    print(f"  Features shape: {features.shape}")

    # Split calibration/test (60/40)
    n_total = len(ious)
    n_cal = int(0.6 * n_total)

    indices = np.arange(n_total)
    np.random.seed(42)
    np.random.shuffle(indices)

    cal_idx = indices[:n_cal]
    test_idx = indices[n_cal:]

    cal_data = {
        'features': features[cal_idx],
        'ious': ious[cal_idx],
        'confidences': confidences[cal_idx]
    }

    test_data = {
        'features': features[test_idx],
        'ious': ious[test_idx],
        'confidences': confidences[test_idx]
    }

    print(f"  Calibration: {len(cal_idx)} samples")
    print(f"  Test: {len(test_idx)} samples")

    # Compute aleatoric uncertainty
    print(f"\n[2/4] Computing aleatoric uncertainty...")
    alea_model = MahalanobisUncertainty(
        reg_lambda=1e-4,
        eps=1e-10
    )

    alea_model.fit(cal_data['features'])
    alea_result = alea_model.predict(test_data['features'])

    # Extract normalized uncertainty array from result dict
    if isinstance(alea_result, dict):
        alea_test = alea_result['normalized']  # Use normalized values in [0,1]
    else:
        alea_test = alea_result

    print(f"  Aleatoric: mean={alea_test.mean():.3f}, std={alea_test.std():.3f}")
    print(f"  Range: [{alea_test.min():.3f}, {alea_test.max():.3f}]")

    # Compute epistemic uncertainty
    print(f"\n[3/4] Computing epistemic uncertainty...")
    epistemic_model = EpistemicUncertainty(
        k_neighbors_spectral=50,
        k_neighbors_repulsive=100,
        temperature=1.0,
        weights='equal',
        verbose=True
    )

    epistemic_model.fit(
        X_calibration=cal_data['features'],
        conformity_cal=1 - cal_data['ious'],  # Higher conformity = worse tracking
        plot_diagnostics=False
    )

    epis_results = epistemic_model.predict(
        test_data['features'],
        return_components=True,
        plot_diagnostics=False
    )

    epis_test = epis_results['combined']

    print(f"  Epistemic: mean={epis_test.mean():.3f}, std={epis_test.std():.3f}")
    print(f"  Range: [{epis_test.min():.3f}, {epis_test.max():.3f}]")

    # Save per-detection data
    print(f"\n[4/4] Saving per-detection data...")

    # Prepare data arrays
    per_detection_data = {
        'n_detections': len(test_idx),
        'sequence': sequence_name,
        'ious': test_data['ious'].tolist(),
        'confidences': test_data['confidences'].tolist(),
        'aleatoric': alea_test.tolist(),
        'epistemic': epis_test.tolist(),
        'spectral': epis_results['spectral'].tolist(),
        'repulsive': epis_results['repulsive'].tolist(),
        'total_uncertainty': (alea_test + epis_test).tolist(),
        'conformity': (1 - test_data['ious']).tolist()
    }

    # Save as JSON
    json_path = output_dir / 'per_detection_data.json'
    with open(json_path, 'w') as f:
        json.dump(per_detection_data, f, indent=2)

    print(f"  ✓ Saved JSON: {json_path}")

    # Also save as NumPy arrays for faster loading
    npz_path = output_dir / 'per_detection_data.npz'
    np.savez(
        npz_path,
        ious=test_data['ious'],
        confidences=test_data['confidences'],
        aleatoric=alea_test,
        epistemic=epis_test,
        spectral=epis_results['spectral'],
        repulsive=epis_results['repulsive'],
        total=alea_test + epis_test,
        conformity=1 - test_data['ious']
    )

    print(f"  ✓ Saved NPZ: {npz_path}")

    # Compute and display summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")

    print(f"\nDataset: {len(test_idx)} detections")

    # IoU categories
    excellent_mask = test_data['ious'] > 0.8
    good_mask = (test_data['ious'] >= 0.5) & (test_data['ious'] <= 0.8)
    poor_mask = test_data['ious'] < 0.5

    print(f"\nIoU Distribution:")
    print(f"  Excellent (>0.8):  {excellent_mask.sum():4d} detections")
    print(f"  Good (0.5-0.8):    {good_mask.sum():4d} detections")
    print(f"  Poor (<0.5):       {poor_mask.sum():4d} detections")

    print(f"\nAleatoric Uncertainty by IoU:")
    print(f"  Excellent: {alea_test[excellent_mask].mean():.3f} ± {alea_test[excellent_mask].std():.3f}")
    print(f"  Good:      {alea_test[good_mask].mean():.3f} ± {alea_test[good_mask].std():.3f}")
    print(f"  Poor:      {alea_test[poor_mask].mean():.3f} ± {alea_test[poor_mask].std():.3f}")

    print(f"\nEpistemic Uncertainty by IoU:")
    print(f"  Excellent: {epis_test[excellent_mask].mean():.3f} ± {epis_test[excellent_mask].std():.3f}")
    print(f"  Good:      {epis_test[good_mask].mean():.3f} ± {epis_test[good_mask].std():.3f}")
    print(f"  Poor:      {epis_test[poor_mask].mean():.3f} ± {epis_test[poor_mask].std():.3f}")

    # Correlations
    from scipy.stats import pearsonr

    r_alea_iou, p_alea = pearsonr(alea_test, test_data['ious'])
    r_epis_iou, p_epis = pearsonr(epis_test, test_data['ious'])
    r_alea_epis, p_ortho = pearsonr(alea_test, epis_test)

    print(f"\nCorrelations:")
    print(f"  Aleatoric ↔ IoU:      r = {r_alea_iou:+.4f} (p = {p_alea:.2e})")
    print(f"  Epistemic ↔ IoU:      r = {r_epis_iou:+.4f} (p = {p_epis:.2e})")
    print(f"  Aleatoric ↔ Epistemic: r = {r_alea_epis:+.4f} (p = {p_ortho:.2e})")

    if abs(r_alea_epis) < 0.1:
        print(f"\n  ✅ Orthogonality achieved! |r| = {abs(r_alea_epis):.4f} < 0.1")
    else:
        print(f"\n  ⚠️  Not perfectly orthogonal: |r| = {abs(r_alea_epis):.4f}")

    print(f"\n{'='*80}")
    print(f"DATA SAVED SUCCESSFULLY ✓")
    print(f"{'='*80}\n")

    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Save per-detection uncertainty data')
    parser.add_argument('--sequence', type=str, default='MOT17-05',
                        help='MOT17 sequence name (default: MOT17-05)')

    args = parser.parse_args()

    output_dir = save_per_detection_data(args.sequence)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"\nYou can now create REAL plots using the saved data!")
