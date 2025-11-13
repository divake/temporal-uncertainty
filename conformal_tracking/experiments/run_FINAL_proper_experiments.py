"""
FINAL Proper Experiments with Real Triple-S Framework

This script:
1. Loads cached YOLO features from npz files
2. Computes REAL aleatoric uncertainty (Mahalanobis)
3. Computes REAL epistemic uncertainty (Triple-S with SLSQP optimization)
4. Saves epistemic weights (Spectral, Repulsive, Gradient)
5. Runs conformal prediction properly
6. Generates publication-ready results for table filling

Usage:
    # Run all experiments
    python run_FINAL_proper_experiments.py

    # Run specific model
    python run_FINAL_proper_experiments.py --model yolov8s

    # Run specific dataset
    python run_FINAL_proper_experiments.py --dataset mot17

Author: CVPR Paper Team
Date: 2025-11-13
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from uncertainty.mahalanobis import MahalanobisUncertainty
from uncertainty.epistemic_combined import EpistemicUncertainty
from uncertainty.conformal_calibration import CombinedConformalCalibrator


# ============================================================================
# Configuration
# ============================================================================

CACHE_ROOT = Path("/ssd_4TB/divake/temporal_uncertainty/yolo_cache/data")
RESULTS_DIR = Path(__file__).parent.parent / "results" / "FINAL_experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Experimental configuration matching PAPER_TABLE_UNIFIED.md
EXPERIMENTS = {
    'mot17': {
        'sequences': ['MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-11-FRCNN'],
        'models': ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
    },
    'mot20': {
        'sequences': ['MOT20-05'],
        'models': ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
    },
    'dancetrack': {
        'sequences': ['dancetrack0019'],
        'models': ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
    }
}

# Hyperparameters from CLAUDE.md
HYPERPARAMS = {
    'aleatoric': {
        'k_neighbors': 15,
        'reg_lambda': 1e-4
    },
    'epistemic': {
        'k_neighbors_spectral': 50,
        'k_neighbors_repulsive': 100,
        'temperature': 1.0,
        'weights': 'optimize'  # SLSQP optimization
    },
    'conformal': {
        'alpha': 0.1,  # 90% coverage
        'method': 'combined_local'
    }
}


# ============================================================================
# Data Loading
# ============================================================================

def load_cached_features(dataset: str, sequence: str, model: str) -> Dict:
    """
    Load cached features from npz file

    Returns:
        Dictionary with:
        - features_dict: {layer_id: features array}
        - ious: IoU with ground truth
        - confidences: YOLO confidence scores
        - frame_ids: Frame indices
    """
    cache_path = CACHE_ROOT / dataset / model / f"{sequence}.npz"

    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")

    print(f"Loading cache: {cache_path}")
    data = np.load(cache_path, allow_pickle=True)

    # Extract multi-layer features for ALL detections
    features_all_dict = {}
    for layer in [4, 9, 15, 21]:
        key = f'features/layer_{layer}'
        if key in data:
            features_all_dict[layer] = data[key]

    # Extract matched detection info
    ious_matched = data['gt_matching/iou']  # shape (12947,)
    det_indices = data['gt_matching/det_indices']  # shape (12947,) - indices into full detection array
    conf_all = data['detections/confidences']  # shape (48923,)

    # Get confidences for matched detections
    conf_matched = conf_all[det_indices]

    # Filter: IoU > 0.3 AND confidence > 0.3
    valid_mask = (ious_matched > 0.3) & (conf_matched > 0.3)

    # Apply filter to matched data
    ious = ious_matched[valid_mask]
    confidences = conf_matched[valid_mask]
    valid_det_indices = det_indices[valid_mask]

    # Extract features for valid detections only
    features_dict_filtered = {}
    for layer, feats_all in features_all_dict.items():
        features_dict_filtered[layer] = feats_all[valid_det_indices]

    print(f"  Total detections: {len(conf_all)}")
    print(f"  Matched detections: {len(ious_matched)}")
    print(f"  Valid detections (IoU>0.3, conf>0.3): {len(ious)}")

    result = {
        'features_dict': features_dict_filtered,
        'ious': ious,
        'confidences': confidences,
        'sequence': sequence,
        'model': model,
        'dataset': dataset
    }

    return result


# ============================================================================
# Experiment Runner
# ============================================================================

def run_single_experiment(dataset: str, sequence: str, model: str) -> Dict:
    """
    Run ONE experiment with REAL Triple-S framework

    Returns:
        Dictionary with all results including epistemic weights
    """

    print("\n" + "="*80)
    print(f"EXPERIMENT: {dataset} | {sequence} | {model}")
    print("="*80)

    exp_start = datetime.now()

    # Step 1: Load cached features
    print("\n[1/5] Loading cached features...")
    try:
        data = load_cached_features(dataset, sequence, model)
    except FileNotFoundError as e:
        print(f"✗ {e}")
        return None

    features_dict = data['features_dict']
    ious = data['ious']
    confidences = data['confidences']

    # Primary layer for aleatoric and epistemic (layer 9)
    features_primary = features_dict[9]

    # Check sufficient data
    if len(ious) < 200:
        print(f"✗ Insufficient data: {len(ious)} samples (need >= 200)")
        return None

    # Step 2: Split into calibration and test (50/50)
    print("\n[2/5] Splitting into calibration and test...")
    np.random.seed(42)
    indices = np.arange(len(ious))
    np.random.shuffle(indices)

    split = len(indices) // 2
    cal_idx = indices[:split]
    test_idx = indices[split:]

    # Calibration data
    features_cal_dict = {k: v[cal_idx] for k, v in features_dict.items()}
    features_cal = features_primary[cal_idx]
    ious_cal = ious[cal_idx]
    conf_cal = confidences[cal_idx]

    # Test data
    features_test_dict = {k: v[test_idx] for k, v in features_dict.items()}
    features_test = features_primary[test_idx]
    ious_test = ious[test_idx]
    conf_test = confidences[test_idx]

    print(f"  Calibration: {len(cal_idx)} samples")
    print(f"  Test: {len(test_idx)} samples")

    # Step 3: Compute REAL Aleatoric Uncertainty
    print("\n[3/5] Computing aleatoric uncertainty (Mahalanobis)...")
    mahal = MahalanobisUncertainty(
        reg_lambda=HYPERPARAMS['aleatoric']['reg_lambda']
    )

    mahal.fit(features_cal, verbose=False)

    alea_cal_dict = mahal.predict(features_cal)
    alea_test_dict = mahal.predict(features_test)

    # Extract normalized uncertainty (1D arrays)
    alea_cal = alea_cal_dict['normalized']
    alea_test = alea_test_dict['normalized']

    # Correlation with IoU
    r_alea_cal = np.corrcoef(alea_cal, ious_cal)[0, 1]
    r_alea_test = np.corrcoef(alea_test, ious_test)[0, 1]

    print(f"  Aleatoric (cal):  mean={alea_cal.mean():.3f}, std={alea_cal.std():.3f}, r(IoU)={r_alea_cal:.3f}")
    print(f"  Aleatoric (test): mean={alea_test.mean():.3f}, std={alea_test.std():.3f}, r(IoU)={r_alea_test:.3f}")

    # Step 4: Compute REAL Epistemic Uncertainty (Triple-S with SLSQP)
    print("\n[4/5] Computing epistemic uncertainty (Triple-S + SLSQP)...")

    epis_model = EpistemicUncertainty(
        k_neighbors_spectral=HYPERPARAMS['epistemic']['k_neighbors_spectral'],
        k_neighbors_repulsive=HYPERPARAMS['epistemic']['k_neighbors_repulsive'],
        temperature=HYPERPARAMS['epistemic']['temperature'],
        weights='optimize',  # SLSQP optimization
        verbose=False
    )

    # Fit with multi-layer features for gradient method
    epis_model.fit(
        X_calibration=features_cal,
        X_cal_layers=features_cal_dict,
        aleatoric_cal=alea_cal,
        plot_diagnostics=False
    )

    # Get optimized weights
    weights = epis_model.weights
    print(f"  Optimized weights:")
    print(f"    Spectral:  {weights[0]:.3f}")
    print(f"    Repulsive: {weights[1]:.3f}")
    print(f"    Gradient:  {weights[2]:.3f}")

    # Predict on calibration and test
    epis_results_cal = epis_model.predict(
        X_test=features_cal,
        X_test_layers=features_cal_dict,
        return_components=True
    )

    epis_results_test = epis_model.predict(
        X_test=features_test,
        X_test_layers=features_test_dict,
        return_components=True
    )

    epis_cal = epis_results_cal['combined']
    epis_test = epis_results_test['combined']

    # Check orthogonality
    r_epis_alea = np.corrcoef(epis_cal, alea_cal)[0, 1]
    r_epis_iou_cal = np.corrcoef(epis_cal, ious_cal)[0, 1]
    r_epis_iou_test = np.corrcoef(epis_test, ious_test)[0, 1]

    print(f"  Epistemic (cal):  mean={epis_cal.mean():.3f}, std={epis_cal.std():.3f}")
    print(f"  Epistemic (test): mean={epis_test.mean():.3f}, std={epis_test.std():.3f}")
    print(f"  Orthogonality |r(epis, alea)|: {abs(r_epis_alea):.3f} (target: <0.3)")
    print(f"  r(epis, IoU) cal:  {r_epis_iou_cal:.3f}")
    print(f"  r(epis, IoU) test: {r_epis_iou_test:.3f}")

    # Step 5: Conformal Prediction
    print("\n[5/5] Running conformal prediction...")

    # Combined uncertainty
    sigma_combined_cal = np.sqrt(alea_cal**2 + epis_cal**2)
    sigma_combined_test = np.sqrt(alea_test**2 + epis_test**2)

    # Conformal calibration
    calibrator = CombinedConformalCalibrator(
        alpha=HYPERPARAMS['conformal']['alpha'],
        use_local_scaling=True,  # Use decision tree for local scaling
        verbose=False
    )

    # Use confidence as proxy for predicted IoU
    calibrator.fit(
        X_cal=features_cal,
        y_cal=ious_cal,
        y_pred_cal=conf_cal,
        sigma_alea_cal=alea_cal,
        sigma_epis_cal=epis_cal
    )

    # Predict intervals
    intervals_dict = calibrator.predict(
        X_test=features_test,
        y_pred_test=conf_test,
        sigma_alea_test=alea_test,
        sigma_epis_test=epis_test
    )

    # Coverage
    lower = intervals_dict['lower']
    upper = intervals_dict['upper']
    coverage = np.mean((ious_test >= lower) & (ious_test <= upper))
    mean_width = np.mean(upper - lower)

    print(f"  Coverage: {coverage*100:.1f}% (target: 90%)")
    print(f"  Mean interval width: {mean_width:.3f}")

    # Prepare results
    elapsed = (datetime.now() - exp_start).total_seconds()

    results = {
        'experiment': {
            'dataset': dataset,
            'sequence': sequence,
            'model': model,
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': elapsed
        },
        'data_stats': {
            'n_total': len(ious),
            'n_calibration': len(cal_idx),
            'n_test': len(test_idx)
        },
        'aleatoric': {
            'mean_cal': float(alea_cal.mean()),
            'std_cal': float(alea_cal.std()),
            'mean_test': float(alea_test.mean()),
            'std_test': float(alea_test.std()),
            'iou_r_cal': float(r_alea_cal),
            'iou_r_test': float(r_alea_test)
        },
        'epistemic': {
            'mean_cal': float(epis_cal.mean()),
            'std_cal': float(epis_cal.std()),
            'mean_test': float(epis_test.mean()),
            'std_test': float(epis_test.std()),
            'iou_r_cal': float(r_epis_iou_cal),
            'iou_r_test': float(r_epis_iou_test),
            'orthogonality_r': float(r_epis_alea),
            'weights': {
                'spectral': float(weights[0]),
                'repulsive': float(weights[1]),
                'gradient': float(weights[2])
            },
            'components_cal': {
                'spectral_mean': float(epis_results_cal['spectral'].mean()),
                'repulsive_mean': float(epis_results_cal['repulsive'].mean()),
                'gradient_mean': float(epis_results_cal['gradient'].mean())
            },
            'components_test': {
                'spectral_mean': float(epis_results_test['spectral'].mean()),
                'repulsive_mean': float(epis_results_test['repulsive'].mean()),
                'gradient_mean': float(epis_results_test['gradient'].mean())
            }
        },
        'conformal': {
            'coverage': float(coverage),
            'mean_width': float(mean_width),
            'target_coverage': 0.9,
            'alpha': HYPERPARAMS['conformal']['alpha']
        }
    }

    print(f"\n✓ Experiment completed in {elapsed:.1f}s")

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run FINAL proper experiments with real Triple-S')
    parser.add_argument('--model', type=str, default=None,
                       help='Specific model to run (default: all)')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Specific dataset to run (default: all)')

    args = parser.parse_args()

    # Determine which experiments to run
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = list(EXPERIMENTS.keys())

    # Count total experiments
    total_exp = 0
    for dataset in datasets:
        sequences = EXPERIMENTS[dataset]['sequences']
        if args.model:
            models = [args.model]
        else:
            models = EXPERIMENTS[dataset]['models']
        total_exp += len(sequences) * len(models)

    print("\n" + "#"*80)
    print("# FINAL PROPER EXPERIMENTS WITH REAL TRIPLE-S")
    print("#"*80)
    print(f"\nTotal experiments to run: {total_exp}")
    print(f"Results will be saved to: {RESULTS_DIR}")
    print("\nHyperparameters:")
    print(f"  Aleatoric: k={HYPERPARAMS['aleatoric']['k_neighbors']}, λ={HYPERPARAMS['aleatoric']['reg_lambda']}")
    print(f"  Epistemic: k_spectral={HYPERPARAMS['epistemic']['k_neighbors_spectral']}, k_repulsive={HYPERPARAMS['epistemic']['k_neighbors_repulsive']}, T={HYPERPARAMS['epistemic']['temperature']}")
    print(f"  Conformal: α={HYPERPARAMS['conformal']['alpha']} (90% coverage)")
    print("#"*80 + "\n")

    # Run experiments
    overall_start = datetime.now()
    completed = 0
    failed = 0
    all_results = []

    for dataset in datasets:
        sequences = EXPERIMENTS[dataset]['sequences']
        if args.model:
            models = [args.model]
        else:
            models = EXPERIMENTS[dataset]['models']

        for sequence in sequences:
            for model in models:
                exp_name = f"{dataset}_{sequence}_{model}"

                # Run experiment
                results = run_single_experiment(dataset, sequence, model)

                if results is not None:
                    # Save individual result
                    output_file = RESULTS_DIR / f"{exp_name}.json"
                    with open(output_file, 'w') as f:
                        json.dump(results, f, indent=2)

                    all_results.append(results)
                    completed += 1
                    print(f"✓ Saved: {output_file}")
                else:
                    failed += 1
                    print(f"✗ Failed: {exp_name}")

                # Progress update
                done = completed + failed
                pct = (done / total_exp) * 100
                elapsed_min = (datetime.now() - overall_start).total_seconds() / 60

                print(f"\nProgress: {done}/{total_exp} ({pct:.1f}%) | Completed: {completed} | Failed: {failed}")
                print(f"Elapsed: {elapsed_min:.1f} min")
                if done > 0:
                    avg_time = elapsed_min / done
                    remaining = (total_exp - done) * avg_time
                    print(f"Estimated remaining: {remaining:.1f} min ({remaining/60:.1f} hours)\n")

    # Save aggregated results
    total_elapsed = (datetime.now() - overall_start).total_seconds() / 60

    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_experiments': total_exp,
        'completed': completed,
        'failed': failed,
        'elapsed_minutes': total_elapsed,
        'hyperparameters': HYPERPARAMS,
        'results': all_results
    }

    summary_file = RESULTS_DIR / "FINAL_all_experiments.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*80)
    print(f"Total experiments: {total_exp}")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(completed/total_exp)*100:.1f}%")
    print(f"Total time: {total_elapsed:.1f} min ({total_elapsed/60:.1f} hours)")
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Summary file: {summary_file}")
    print("="*80 + "\n")

    print("Next steps:")
    print("1. Validate results (check epistemic variance, weights sum to 1, orthogonality)")
    print("2. Fill PAPER_TABLE_UNIFIED.md with experimental values")
    print("3. Fill TRIPLE_S_WEIGHTS_TABLE.md with weight distributions")


if __name__ == '__main__':
    main()
