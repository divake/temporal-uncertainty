"""
Phase 2: Run RT-DETR and YOLO-World experiments with conformal prediction

Models:
- RT-DETR (rtdetr-l.pt)
- YOLO-World (yolov8s-world.pt)

Sequences: 5 total
- MOT17: MOT17-02-FRCNN, MOT17-04-FRCNN, MOT17-11-FRCNN
- MOT20: MOT20-05
- DanceTrack: dancetrack0019

Total: 2 models × 5 sequences = 10 experiments
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import experiment runner
from experiments.run_complete_with_conformal import run_single_experiment, ExperimentLogger

# Setup paths
RESULTS_DIR = project_root / "results" / "experiments_conformal"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Create logger
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = RESULTS_DIR / f"phase2_log_{timestamp}.txt"
logger = ExperimentLogger(LOG_FILE)

# Define Phase 2 experiments
experiments = [
    # MOT17 sequences (3 × 2 models = 6 experiments)
    {'dataset': 'MOT17', 'sequence': 'MOT17-02-FRCNN', 'model': 'rtdetr-l'},
    {'dataset': 'MOT17', 'sequence': 'MOT17-02-FRCNN', 'model': 'yolov8s-world'},

    {'dataset': 'MOT17', 'sequence': 'MOT17-04-FRCNN', 'model': 'rtdetr-l'},
    {'dataset': 'MOT17', 'sequence': 'MOT17-04-FRCNN', 'model': 'yolov8s-world'},

    {'dataset': 'MOT17', 'sequence': 'MOT17-11-FRCNN', 'model': 'rtdetr-l'},
    {'dataset': 'MOT17', 'sequence': 'MOT17-11-FRCNN', 'model': 'yolov8s-world'},

    # MOT20 sequence (1 × 2 models = 2 experiments)
    {'dataset': 'MOT20', 'sequence': 'MOT20-05', 'model': 'rtdetr-l'},
    {'dataset': 'MOT20', 'sequence': 'MOT20-05', 'model': 'yolov8s-world'},

    # DanceTrack sequence (1 × 2 models = 2 experiments)
    {'dataset': 'DanceTrack', 'sequence': 'dancetrack0019', 'model': 'rtdetr-l'},
    {'dataset': 'DanceTrack', 'sequence': 'dancetrack0019', 'model': 'yolov8s-world'},
]

logger.log("="*80)
logger.log("PHASE 2: RT-DETR AND YOLO-WORLD EXPERIMENTS")
logger.log("="*80)
logger.log(f"Total experiments: {len(experiments)}")
logger.log(f"Models: RT-DETR (rtdetr-l), YOLO-World (yolov8s-world)")
logger.log(f"Results directory: {RESULTS_DIR}")
logger.log(f"Log file: {LOG_FILE}")
logger.log("")

success_count = 0
failed_experiments = []

for i, exp in enumerate(experiments, 1):
    logger.log(f"\n[{i}/{len(experiments)}] Starting {exp['dataset']}_{exp['sequence']}_{exp['model']}...")

    success = run_single_experiment(exp, logger)

    if success:
        logger.log(f"✓ Experiment {i}/{len(experiments)} completed successfully", level="SUCCESS")
        success_count += 1
    else:
        logger.log(f"✗ Experiment {i}/{len(experiments)} FAILED", level="ERROR")
        failed_experiments.append(exp)

logger.log("")
logger.log("="*80)
logger.log("PHASE 2 EXECUTION COMPLETE")
logger.log("="*80)
logger.log(f"Successfully completed: {success_count}/{len(experiments)}")
logger.log(f"Failed: {len(failed_experiments)}/{len(experiments)}")

if failed_experiments:
    logger.log("")
    logger.log("FAILED EXPERIMENTS:")
    for exp in failed_experiments:
        logger.log(f"  - {exp['dataset']}_{exp['sequence']}_{exp['model']}")

logger.log("")
logger.log(f"All results saved to: {RESULTS_DIR}")
logger.log(f"Complete log: {LOG_FILE}")
logger.log("")

# Exit with appropriate code
sys.exit(0 if len(failed_experiments) == 0 else 1)
