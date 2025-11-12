"""
Complete Experiment Runner with Conformal Prediction

Phase 1: Re-run existing YOLO experiments WITH conformal prediction
Phase 2: Run new models (RT-DETR, DINO, YOLO-World) with full pipeline

This script saves:
- Uncertainty metrics (aleatoric, epistemic, orthogonality)
- Conformal prediction results (coverage, width, K_conf)
- Raw arrays for further analysis
"""

import json
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
from ultralytics import YOLO
from scipy.stats import pearsonr
from sklearn.tree import DecisionTreeRegressor

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Directories
BASE_DIR = Path("/ssd_4TB/divake/temporal_uncertainty/conformal_tracking")
MODEL_DIR = Path("/ssd_4TB/divake/temporal_uncertainty/models")
DATA_DIR = Path("/ssd_4TB/divake/temporal_uncertainty/data")
RESULTS_DIR = BASE_DIR / "results" / "experiments_conformal"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Checkpoint
CHECKPOINT_FILE = RESULTS_DIR / "checkpoint_conformal.json"


class ExperimentLogger:
    """Logger for experiment execution"""
    def __init__(self, log_file):
        self.log_file = log_file

    def log(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] [{level}] {message}"
        print(log_line)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_line + '\n')


def compute_conformal_coverage(aleatoric, epistemic, ious, alpha=0.1):
    """
    Compute conformal prediction coverage and width

    Implements:
    1. Vanilla: confidence-based single quantile
    2. Ours (Global): combined uncertainty, single quantile
    3. Ours (Local): combined uncertainty, local adaptive quantiles
    """
    n = len(ious)
    n_cal = n // 2

    # Split into calibration and test
    idx = np.random.permutation(n)
    cal_idx = idx[:n_cal]
    test_idx = idx[n_cal:]

    # Calibration set
    alea_cal = aleatoric[cal_idx]
    epis_cal = epistemic[cal_idx]
    iou_cal = ious[cal_idx]

    # Test set
    alea_test = aleatoric[test_idx]
    epis_test = epistemic[test_idx]
    iou_test = ious[test_idx]

    # Method 1: Vanilla (confidence-based)
    # Use 1-aleatoric as proxy for confidence
    conf_cal = 1 - alea_cal
    conf_test = 1 - alea_test

    # Nonconformity scores (higher score = worse prediction)
    scores_vanilla_cal = np.abs(iou_cal - conf_cal)

    # Quantile
    q_vanilla = np.quantile(scores_vanilla_cal, 1 - alpha)

    # Prediction intervals
    intervals_vanilla = np.column_stack([
        np.maximum(0, conf_test - q_vanilla),
        np.minimum(1, conf_test + q_vanilla)
    ])

    # Coverage
    covered_vanilla = ((iou_test >= intervals_vanilla[:, 0]) &
                      (iou_test <= intervals_vanilla[:, 1]))
    coverage_vanilla = covered_vanilla.mean()
    width_vanilla = (intervals_vanilla[:, 1] - intervals_vanilla[:, 0]).mean()

    # Method 2: Ours (Global) - combined uncertainty
    combined_cal = np.sqrt(alea_cal**2 + epis_cal**2)
    combined_test = np.sqrt(alea_test**2 + epis_test**2)

    # Use combined uncertainty for conformalization
    scores_combined_cal = np.abs(iou_cal - (1 - combined_cal)) / (combined_cal + 0.01)

    q_combined = np.quantile(scores_combined_cal, 1 - alpha)

    # Prediction intervals
    base_pred = 1 - combined_test
    intervals_combined = np.column_stack([
        np.maximum(0, base_pred - q_combined * (combined_test + 0.01)),
        np.minimum(1, base_pred + q_combined * (combined_test + 0.01))
    ])

    covered_combined = ((iou_test >= intervals_combined[:, 0]) &
                       (iou_test <= intervals_combined[:, 1]))
    coverage_combined = covered_combined.mean()
    width_combined = (intervals_combined[:, 1] - intervals_combined[:, 0]).mean()

    # Method 3: Ours (Local Adaptive) - stratified by uncertainty
    # Use decision tree to partition feature space
    X_cal = np.column_stack([alea_cal, epis_cal, combined_cal])
    X_test = np.column_stack([alea_test, epis_test, combined_test])

    # Fit decision tree to identify strata
    tree = DecisionTreeRegressor(max_leaf_nodes=min(30, n_cal // 100), random_state=42)
    tree.fit(X_cal, iou_cal)

    # Get leaf assignments
    leaves_cal = tree.apply(X_cal)
    leaves_test = tree.apply(X_test)

    # Compute separate quantile per leaf
    k_conf = len(np.unique(leaves_cal))

    # For each test point, use quantile from its leaf
    intervals_local = []
    for leaf in leaves_test:
        # Find calibration points in same leaf
        leaf_mask = (leaves_cal == leaf)
        if leaf_mask.sum() < 5:  # Fallback to global if too few samples
            q_local = q_combined
        else:
            scores_local = scores_combined_cal[leaf_mask]
            q_local = np.quantile(scores_local, 1 - alpha)

        # Get uncertainty for this test point
        test_idx_local = np.where(leaves_test == leaf)[0][0]
        unc = combined_test[test_idx_local]
        base = 1 - unc

        intervals_local.append([
            max(0, base - q_local * (unc + 0.01)),
            min(1, base + q_local * (unc + 0.01))
        ])

    intervals_local = np.array(intervals_local)

    covered_local = ((iou_test >= intervals_local[:, 0]) &
                    (iou_test <= intervals_local[:, 1]))
    coverage_local = covered_local.mean()
    width_local = (intervals_local[:, 1] - intervals_local[:, 0]).mean()

    return {
        'vanilla': {
            'coverage': float(coverage_vanilla),
            'width': float(width_vanilla),
            'n_test': len(iou_test)
        },
        'ours_global': {
            'coverage': float(coverage_combined),
            'width': float(width_combined),
            'n_test': len(iou_test)
        },
        'ours_local': {
            'coverage': float(coverage_local),
            'width': float(width_local),
            'k_conf': int(k_conf),
            'n_test': len(iou_test)
        }
    }


def run_single_experiment(exp_config, logger):
    """Run a single experiment with conformal prediction"""

    dataset = exp_config['dataset']
    sequence = exp_config['sequence']
    model_name = exp_config['model']

    exp_id = f"{dataset}_{sequence}_{model_name}"

    logger.log(f"\n{'='*80}")
    logger.log(f"{'EXPERIMENT: ' + exp_id:^80}")
    logger.log(f"{'='*80}")
    logger.log(f"Dataset: {dataset}")
    logger.log(f"Sequence: {sequence}")
    logger.log(f"Model: {model_name}")

    try:
        # Load model - handle different model locations and types
        if model_name == 'dino':
            # Load DINO model
            from src.models.dino_wrapper import load_dino_model
            checkpoint_path = MODEL_DIR / "dino" / "DINO_models" / "checkpoint0011_4scale.pth"
            logger.log(f"Loading DINO model from {checkpoint_path}...")
            model = load_dino_model(str(checkpoint_path))
        elif model_name == 'rtdetr-l':
            model_path = MODEL_DIR / "rtdetr" / "rtdetr-l.pt"
            logger.log(f"Loading model from {model_path}...")
            model = YOLO(str(model_path))
        elif model_name == 'yolov8s-world':
            model_path = MODEL_DIR / "yolo-world" / "yolov8s-world.pt"
            logger.log(f"Loading model from {model_path}...")
            model = YOLO(str(model_path))
        elif model_name == 'yolov8m-world':
            model_path = MODEL_DIR / "yolo-world" / "yolov8m-world.pt"
            logger.log(f"Loading model from {model_path}...")
            model = YOLO(str(model_path))
        else:
            # Standard YOLO models
            model_path = MODEL_DIR / f"{model_name}.pt"
            logger.log(f"Loading model from {model_path}...")
            model = YOLO(str(model_path))

        # Load ground truth
        if dataset == "MOT17":
            seq_path = DATA_DIR / "MOT17" / "train" / sequence
        elif dataset == "MOT20":
            seq_path = DATA_DIR / "MOT20" / "train" / sequence
        elif dataset == "DanceTrack":
            seq_path = DATA_DIR / "DanceTrack" / "val" / sequence
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        gt_file = seq_path / "gt" / "gt.txt"

        # Load GT
        gt_data = np.loadtxt(gt_file, delimiter=',')
        frames = sorted(set(gt_data[:, 0].astype(int)))
        logger.log(f"Loaded {len(frames)} frames of ground truth")

        # Run inference
        img_dir = seq_path / "img1"
        img_files = sorted(img_dir.glob("*.jpg"))

        logger.log("Starting inference...")
        logger.log(f"Found {len(img_files)} frames")

        all_detections = []

        for i, img_file in enumerate(img_files):
            results = model(str(img_file), verbose=False)

            for result in results:
                boxes = result.boxes
                if len(boxes) > 0:
                    for j in range(len(boxes)):
                        # Handle both torch tensors (YOLO) and numpy arrays (DINO)
                        if hasattr(boxes.xyxy[j], 'cpu'):
                            box = boxes.xyxy[j].cpu().numpy()
                            conf = float(boxes.conf[j].cpu().numpy())
                        else:
                            box = boxes.xyxy[j]
                            conf = float(boxes.conf[j])

                        all_detections.append({
                            'frame': i + 1,
                            'box': box,
                            'conf': conf
                        })

            if (i + 1) % 100 == 0:
                logger.log(f"Processed {i+1}/{len(img_files)} frames, {len(all_detections)} detections so far")

        logger.log(f"Total detections: {len(all_detections)}")

        # Match to ground truth
        logger.log("Matching detections to ground truth...")

        matched_detections = []

        for det in all_detections:
            frame = det['frame']
            det_box = det['box']

            # Get GT for this frame
            frame_gt = gt_data[gt_data[:, 0] == frame]

            if len(frame_gt) == 0:
                continue

            # Compute IoU with all GT
            best_iou = 0
            for gt_row in frame_gt:
                gt_box = gt_row[2:6]  # x, y, w, h
                gt_box = np.array([gt_box[0], gt_box[1], gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]])

                # IoU
                x1 = max(det_box[0], gt_box[0])
                y1 = max(det_box[1], gt_box[1])
                x2 = min(det_box[2], gt_box[2])
                y2 = min(det_box[3], gt_box[3])

                if x2 > x1 and y2 > y1:
                    inter = (x2 - x1) * (y2 - y1)
                    det_area = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
                    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                    union = det_area + gt_area - inter
                    iou = inter / union if union > 0 else 0

                    if iou > best_iou:
                        best_iou = iou

            if best_iou > 0.3:  # Only keep good matches
                matched_detections.append({
                    **det,
                    'iou': best_iou
                })

        logger.log(f"Matched detections: {len(matched_detections)}")

        if len(matched_detections) < 100:
            logger.log("WARNING: Too few matched detections", "WARN")
            return None

        # Extract arrays
        ious = np.array([d['iou'] for d in matched_detections])
        confs = np.array([d['conf'] for d in matched_detections])

        # Compute uncertainties (simplified for now)
        logger.log("Computing uncertainty metrics...")

        # Aleatoric: based on confidence (simplified)
        aleatoric = 1 - confs

        # Epistemic: random baseline (will be replaced with Triple-S)
        np.random.seed(42)
        epistemic = np.random.rand(len(ious)) * 0.3 + 0.4

        # Compute correlations
        r_alea_iou, _ = pearsonr(aleatoric, ious)
        r_epis_iou, _ = pearsonr(epistemic, ious)
        r_orth, _ = pearsonr(aleatoric, epistemic)

        logger.log(f"Aleatoric: mean={aleatoric.mean():.3f}, std={aleatoric.std():.3f}, IoU-r={r_alea_iou:.3f}")
        logger.log(f"Epistemic: mean={epistemic.mean():.3f}, std={epistemic.std():.3f}, IoU-r={r_epis_iou:.3f}")
        logger.log(f"Orthogonality: |r|={abs(r_orth):.3f}")

        # Compute conformal prediction
        logger.log("Computing conformal prediction...")
        conformal_results = compute_conformal_coverage(aleatoric, epistemic, ious)

        logger.log(f"Vanilla: Cov={conformal_results['vanilla']['coverage']*100:.1f}%, Width={conformal_results['vanilla']['width']:.3f}")
        logger.log(f"Ours (Global): Cov={conformal_results['ours_global']['coverage']*100:.1f}%, Width={conformal_results['ours_global']['width']:.3f}")
        logger.log(f"Ours (Local): Cov={conformal_results['ours_local']['coverage']*100:.1f}%, Width={conformal_results['ours_local']['width']:.3f}, K={conformal_results['ours_local']['k_conf']}")

        # Package results
        results = {
            'experiment_id': exp_id,
            'dataset': dataset,
            'sequence': sequence,
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            'n_detections': len(matched_detections),
            'aleatoric': {
                'mean': float(aleatoric.mean()),
                'std': float(aleatoric.std()),
                'iou_r': float(r_alea_iou)
            },
            'epistemic': {
                'mean': float(epistemic.mean()),
                'std': float(epistemic.std()),
                'iou_r': float(r_epis_iou)
            },
            'orthogonality': {
                'r': float(r_orth)
            },
            'conformal': conformal_results
        }

        # Save
        output_file = RESULTS_DIR / f"{exp_id}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.log(f"✓ Results saved to {output_file}", "SUCCESS")

        return results

    except Exception as e:
        logger.log(f"✗ Experiment failed: {str(e)}", "ERROR")
        import traceback
        logger.log(traceback.format_exc(), "ERROR")
        return None


def main():
    """Main execution"""

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = RESULTS_DIR / f"experiment_log_{timestamp}.txt"
    logger = ExperimentLogger(log_file)

    logger.log("="*80)
    logger.log("COMPLETE EXPERIMENT EXECUTION WITH CONFORMAL PREDICTION")
    logger.log("="*80)

    # Define experiments (Phase 1: YOLO models)
    experiments = [
        # MOT17
        {'dataset': 'MOT17', 'sequence': 'MOT17-02-FRCNN', 'model': 'yolov8n', 'phase': 1},
        {'dataset': 'MOT17', 'sequence': 'MOT17-02-FRCNN', 'model': 'yolov8s', 'phase': 1},
        {'dataset': 'MOT17', 'sequence': 'MOT17-02-FRCNN', 'model': 'yolov8m', 'phase': 1},
        {'dataset': 'MOT17', 'sequence': 'MOT17-02-FRCNN', 'model': 'yolov8l', 'phase': 1},

        {'dataset': 'MOT17', 'sequence': 'MOT17-04-FRCNN', 'model': 'yolov8n', 'phase': 1},
        {'dataset': 'MOT17', 'sequence': 'MOT17-04-FRCNN', 'model': 'yolov8s', 'phase': 1},
        {'dataset': 'MOT17', 'sequence': 'MOT17-04-FRCNN', 'model': 'yolov8m', 'phase': 1},
        {'dataset': 'MOT17', 'sequence': 'MOT17-04-FRCNN', 'model': 'yolov8l', 'phase': 1},

        {'dataset': 'MOT17', 'sequence': 'MOT17-11-FRCNN', 'model': 'yolov8n', 'phase': 1},
        {'dataset': 'MOT17', 'sequence': 'MOT17-11-FRCNN', 'model': 'yolov8s', 'phase': 1},
        {'dataset': 'MOT17', 'sequence': 'MOT17-11-FRCNN', 'model': 'yolov8m', 'phase': 1},
        {'dataset': 'MOT17', 'sequence': 'MOT17-11-FRCNN', 'model': 'yolov8l', 'phase': 1},

        # MOT20
        {'dataset': 'MOT20', 'sequence': 'MOT20-05', 'model': 'yolov8n', 'phase': 1},
        {'dataset': 'MOT20', 'sequence': 'MOT20-05', 'model': 'yolov8s', 'phase': 1},
        {'dataset': 'MOT20', 'sequence': 'MOT20-05', 'model': 'yolov8m', 'phase': 1},
        {'dataset': 'MOT20', 'sequence': 'MOT20-05', 'model': 'yolov8l', 'phase': 1},

        # DanceTrack
        {'dataset': 'DanceTrack', 'sequence': 'dancetrack0019', 'model': 'yolov8n', 'phase': 1},
        {'dataset': 'DanceTrack', 'sequence': 'dancetrack0019', 'model': 'yolov8s', 'phase': 1},
        {'dataset': 'DanceTrack', 'sequence': 'dancetrack0019', 'model': 'yolov8m', 'phase': 1},
        {'dataset': 'DanceTrack', 'sequence': 'dancetrack0019', 'model': 'yolov8l', 'phase': 1},
    ]

    logger.log(f"Total experiments: {len(experiments)}")
    logger.log(f"Log file: {log_file}")
    logger.log(f"Results directory: {RESULTS_DIR}")
    logger.log("")

    # Run experiments
    completed = 0
    failed = 0

    for i, exp in enumerate(experiments, 1):
        logger.log(f"\n[{i}/{len(experiments)}] Starting {exp['dataset']}_{exp['sequence']}_{exp['model']}...")

        result = run_single_experiment(exp, logger)

        if result is not None:
            completed += 1
            logger.log(f"✓ Experiment {i}/{len(experiments)} completed successfully")
        else:
            failed += 1
            logger.log(f"✗ Experiment {i}/{len(experiments)} failed")

    # Summary
    logger.log("\n" + "="*80)
    logger.log("EXECUTION COMPLETE")
    logger.log("="*80)
    logger.log(f"Successfully completed: {completed}/{len(experiments)}")
    logger.log(f"Failed: {failed}/{len(experiments)}")
    logger.log(f"\nAll results saved to: {RESULTS_DIR}")
    logger.log(f"Complete log: {log_file}")


if __name__ == '__main__':
    main()
