"""
Run DINO experiments on 3 sequences (1 per dataset)

DINO: Pure Vision Transformer (ViT) based object detector
- Architecture: DETR with Swin Transformer backbone
- Parameters: ~540M (4-scale model)
- Type: End-to-end transformer detector

Sequences:
- MOT17-02-FRCNN (600 frames, representative MOT17)
- MOT20-05 (3315 frames, crowded scene)
- dancetrack0019 (2402 frames, uniform appearance)

Total: 3 experiments
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
LOG_FILE = RESULTS_DIR / f"dino_log_{timestamp}.txt"
logger = ExperimentLogger(LOG_FILE)

# Define DINO experiments - 1 sequence per dataset
experiments = [
    {'dataset': 'MOT17', 'sequence': 'MOT17-02-FRCNN', 'model': 'dino'},
    {'dataset': 'MOT20', 'sequence': 'MOT20-05', 'model': 'dino'},
    {'dataset': 'DanceTrack', 'sequence': 'dancetrack0019', 'model': 'dino'},
]

logger.log("="*80)
logger.log("DINO EXPERIMENTS (PURE TRANSFORMER ARCHITECTURE)")
logger.log("="*80)
logger.log(f"Total experiments: {len(experiments)}")
logger.log(f"Model: DINO 4-scale with ResNet-50 backbone")
logger.log(f"Architecture: DETR + Improved DeNoising Anchor Boxes")
logger.log(f"Sequences: 1 per dataset (3 total)")
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
logger.log("DINO EXECUTION COMPLETE")
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
