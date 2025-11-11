"""
Single Object Tracking with Temporal Uncertainty Decomposition

This script:
1. Loads precomputed aleatoric + epistemic uncertainties from run_epistemic_mot17.py
2. Filters for a single track ID
3. Visualizes temporal evolution

Key: Reuses the WORKING uncertainty computation from Method 3!

Author: Enhanced CACD Team
Date: November 11, 2025
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from collections import defaultdict

# Run the epistemic experiment first if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

def load_or_compute_uncertainties(sequence_name='MOT17-11-FRCNN'):
    """
    Recompute uncertainties using the same method as run_epistemic_mot17.py

    Returns:
        Dictionary with all detection data and uncertainties
    """
    # Load cache to get detection details
    cache_path = Path('/ssd_4TB/divake/temporal_uncertainty/yolo_cache/data/mot17/yolov8n') / f'{sequence_name}.npz'
    cache = np.load(cache_path, allow_pickle=True)

    # Load MOT17 data using the same loader
    print("[1/4] Loading MOT17 data...")
    from data_loaders.mot17_loader import MOT17DataLoader

    loader = MOT17DataLoader(
        cache_path=cache_path,
        load_all_layers=True,  # Need all layers for epistemic
        conf_threshold=0.3,
        split_ratio=1.0  # Use all data (no split needed for single track analysis)
    )
    data = loader.get_calibration_data()

    # Now compute uncertainties using the SAME method as run_epistemic_mot17.py
    print("\n[2/4] Computing aleatoric uncertainty...")
    from src.uncertainty.mahalanobis import MahalanobisUncertainty

    mahalanobis_model = MahalanobisUncertainty()
    mahalanobis_model.fit(data['features'], verbose=False)

    # Get aleatoric for all detections
    aleatoric_results = mahalanobis_model.predict(data['features'], verbose=False)
    aleatoric = aleatoric_results['normalized']

    print("\n[3/4] Computing epistemic uncertainty...")
    from src.uncertainty.epistemic_combined import EpistemicUncertainty

    # Prepare multi-layer features using the loader
    cal_layers = loader.get_calibration_layers()

    epistemic_model = EpistemicUncertainty(
        k_neighbors_spectral=50,
        k_neighbors_repulsive=100,
        temperature=1.0,
        weights='optimize',  # Will optimize during fit
        verbose=True
    )

    epistemic_model.fit(
        data['features'],
        X_cal_layers=cal_layers,
        mahalanobis_model=mahalanobis_model,
        aleatoric_cal=aleatoric,
        conformity_cal=data['conformity_scores'],
        plot_diagnostics=False,  # Skip plots during fitting
        save_dir=None
    )

    epistemic_results = epistemic_model.predict(
        data['features'],
        X_test_layers=cal_layers,
        return_components=True,
        plot_diagnostics=False
    )
    epistemic = epistemic_results['combined']

    total = aleatoric + epistemic

    # Get additional data from cache
    # The loader already filtered by matched detections and conf >= 0.3
    # The indices in gt_matching arrays correspond to matched detections
    all_track_ids = cache['gt_matching/gt_track_ids']
    matched_det_indices = cache['gt_matching/det_indices']  # Detection indices for matches
    all_confidences = cache['detections/confidences']
    all_frame_ids = cache['detections/frame_ids']
    all_bboxes = cache['detections/bboxes']

    # Apply confidence filter to matched detections
    conf_filter = all_confidences[matched_det_indices] >= 0.3

    return {
        'aleatoric': aleatoric,
        'epistemic': epistemic,
        'total': total,
        'track_ids': all_track_ids[conf_filter],
        'frame_ids': all_frame_ids[matched_det_indices[conf_filter]],
        'ious': data['ious'],
        'confidences': data['confidences'],
        'bboxes': all_bboxes[matched_det_indices[conf_filter]]
    }


def analyze_track(data, track_id=1):
    """Filter data for single track and create visualizations"""

    print(f"\n[4/4] Analyzing Track ID {track_id}...")

    # Filter for this track
    track_mask = data['track_ids'] == track_id

    track_frames = data['frame_ids'][track_mask]
    track_alea = data['aleatoric'][track_mask]
    track_epis = data['epistemic'][track_mask]
    track_total = data['total'][track_mask]
    track_ious = data['ious'][track_mask]
    track_conf = data['confidences'][track_mask]

    # Sort by frame
    sort_idx = np.argsort(track_frames)
    track_frames = track_frames[sort_idx]
    track_alea = track_alea[sort_idx]
    track_epis = track_epis[sort_idx]
    track_total = track_total[sort_idx]
    track_ious = track_ious[sort_idx]

    print(f"  Found {len(track_frames)} detections")
    print(f"  Frame range: {track_frames.min()}-{track_frames.max()}")
    print(f"\n  Uncertainty statistics:")
    print(f"    Aleatoric: mean={track_alea.mean():.3f}, std={track_alea.std():.3f}")
    print(f"    Epistemic: mean={track_epis.mean():.3f}, std={track_epis.std():.3f}")
    print(f"    Total:     mean={track_total.mean():.3f}, std={track_total.std():.3f}")

    # Calculate epistemic fraction
    epis_fraction = track_epis / (track_total + 1e-10)
    print(f"    Epistemic fraction: {epis_fraction.mean():.1%} (± {epis_fraction.std():.1%})")

    # Create visualization
    results_dir = Path(__file__).parent.parent / 'results' / 'track_temporal_analysis'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Main plot: Temporal evolution
    fig, axes = plt.subplots(2, 1, figsize=(18, 12))

    # Plot 1: Uncertainty curves
    ax = axes[0]
    ax.plot(track_frames, track_alea, 'g-', linewidth=2.5, label='Aleatoric (Data Noise)', alpha=0.8)
    ax.plot(track_frames, track_epis, 'b-', linewidth=2.5, label='Epistemic (Model Uncertainty)', alpha=0.8)
    ax.plot(track_frames, track_total, 'r-', linewidth=3, label='Total Uncertainty', alpha=0.9)

    ax.set_xlabel('Frame Number', fontsize=14, fontweight='bold')
    ax.set_ylabel('Uncertainty', fontsize=14, fontweight='bold')
    ax.set_title(f'Temporal Uncertainty Evolution - Track ID {track_id}',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=13, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Plot 2: IoU overlay
    ax2 = axes[1]
    ax2_twin = ax2.twinx()

    ax2.plot(track_frames, track_total, 'r-', linewidth=2.5, label='Total Uncertainty')
    ax2_twin.plot(track_frames, track_ious, 'b-', linewidth=2.5, label='Ground Truth IoU', alpha=0.7)

    ax2.set_xlabel('Frame Number', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Uncertainty', fontsize=13, fontweight='bold', color='r')
    ax2_twin.set_ylabel('IoU', fontsize=13, fontweight='bold', color='b')
    ax2.set_title('Uncertainty vs Tracking Quality (IoU)', fontsize=16, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2_twin.tick_params(axis='y', labelcolor='b')
    ax2.grid(True, alpha=0.3)

    # Add correlation
    corr = np.corrcoef(track_total, track_ious)[0, 1]
    ax2.text(0.02, 0.98, f'Correlation: {corr:.3f}',
             transform=ax2.transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    save_path = results_dir / f'track_{track_id}_temporal_uncertainty.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {save_path}")

    # Save results
    results = {
        'track_id': int(track_id),
        'num_frames': len(track_frames),
        'frame_range': [int(track_frames.min()), int(track_frames.max())],
        'uncertainty_stats': {
            'aleatoric': {'mean': float(track_alea.mean()), 'std': float(track_alea.std())},
            'epistemic': {'mean': float(track_epis.mean()), 'std': float(track_epis.std())},
            'total': {'mean': float(track_total.mean()), 'std': float(track_total.std())},
            'epistemic_fraction': {'mean': float(epis_fraction.mean()), 'std': float(epis_fraction.std())}
        },
        'correlation_total_iou': float(corr),
        'temporal_data': {
            'frames': track_frames.tolist(),
            'aleatoric': track_alea.tolist(),
            'epistemic': track_epis.tolist(),
            'total': track_total.tolist(),
            'ious': track_ious.tolist()
        }
    }

    results_path = results_dir / f'track_{track_id}_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved: {results_path}")

    return results


if __name__ == '__main__':
    sequence = 'MOT17-11-FRCNN'
    track_id = 1

    if len(sys.argv) > 1:
        sequence = sys.argv[1]
    if len(sys.argv) > 2:
        track_id = int(sys.argv[2])

    print("="*80)
    print("SINGLE TRACK TEMPORAL UNCERTAINTY ANALYSIS")
    print("="*80)
    print(f"Sequence: {sequence}")
    print(f"Track ID: {track_id}\n")

    # Load/compute uncertainties
    data = load_or_compute_uncertainties(sequence)

    # Analyze specific track
    results = analyze_track(data, track_id)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE ✓")
    print("="*80)
