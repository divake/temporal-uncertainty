#!/usr/bin/env python
"""
Run Epistemic Uncertainty Experiments on All MOT17 Sequences

This script runs the complete epistemic uncertainty pipeline on all available
MOT17 sequences, generates visualizations, and creates a comprehensive report.

Author: Enhanced CACD Team
Date: 2025-11-10
"""

import subprocess
import numpy as np
from pathlib import Path
import json
import time
import sys


def check_sequence_exists(cache_dir, sequence_name):
    """Check if a sequence cache file exists."""
    cache_path = cache_dir / f"{sequence_name}.npz"
    return cache_path.exists()


def run_epistemic_experiment(sequence_name, python_path):
    """Run epistemic experiment for a single sequence."""
    print(f"\n{'='*80}")
    print(f"Running experiment for {sequence_name}")
    print(f"{'='*80}")

    cmd = [
        python_path,
        "experiments/run_epistemic_mot17.py",
        sequence_name
    ]

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ SUCCESS - Completed in {elapsed:.1f}s")
            # Extract key results from output
            for line in result.stdout.split('\n'):
                if 'Orthogonality:' in line and '|r|' in line:
                    print(f"  {line.strip()}")
                elif 'Aleatoric correlation:' in line:
                    print(f"  {line.strip()}")
                elif 'Epistemic correlation:' in line:
                    print(f"  {line.strip()}")
            return True
        else:
            print(f"❌ FAILED - Error after {elapsed:.1f}s")
            print("Error output:")
            print(result.stderr[:500])  # First 500 chars of error
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT - Exceeded 600s limit")
        return False
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False


def run_visualizations(sequence_name, python_path):
    """Run visualization for a single sequence."""
    print(f"  Generating visualizations for {sequence_name}...")

    cmd = [
        python_path,
        "experiments/visualize_uncertainty_decomposition.py",
        sequence_name
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )

        if result.returncode == 0:
            print(f"  ✅ Visualizations complete")
            return True
        else:
            print(f"  ❌ Visualization failed")
            return False

    except subprocess.TimeoutExpired:
        print(f"  ⏰ Visualization timeout")
        return False
    except Exception as e:
        print(f"  ❌ Visualization exception: {e}")
        return False


def main():
    """Main execution function."""

    # Setup paths
    cache_dir = Path("/ssd_4TB/divake/temporal_uncertainty/yolo_cache/data/mot17/yolov8n")
    results_dir = Path("/ssd_4TB/divake/temporal_uncertainty/conformal_tracking/results")
    python_path = "/home/divake/miniconda3/envs/env_py311/bin/python"

    # All MOT17 test sequences
    all_sequences = [
        "MOT17-01-FRCNN",
        "MOT17-03-FRCNN",
        "MOT17-06-FRCNN",
        "MOT17-07-FRCNN",
        "MOT17-08-FRCNN",
        "MOT17-12-FRCNN",
        "MOT17-14-FRCNN",
        # Train sequences (already done some)
        "MOT17-02-FRCNN",
        "MOT17-04-FRCNN",
        "MOT17-05-FRCNN",
        "MOT17-09-FRCNN",
        "MOT17-10-FRCNN",
        "MOT17-11-FRCNN",  # Already done
        "MOT17-13-FRCNN",  # Already done
    ]

    # Check which sequences are available
    available_sequences = []
    for seq in all_sequences:
        if check_sequence_exists(cache_dir, seq):
            available_sequences.append(seq)
        else:
            print(f"⚠️  Sequence {seq} not found in cache")

    print(f"\n{'='*80}")
    print("EPISTEMIC UNCERTAINTY - BATCH EXPERIMENT")
    print(f"{'='*80}")
    print(f"Found {len(available_sequences)} sequences in cache")
    print(f"Sequences: {', '.join(available_sequences)}")

    # Check which already have results
    completed = []
    pending = []

    for seq in available_sequences:
        seq_num = seq.split('-')[1]
        result_file = results_dir / f"epistemic_mot17_{seq_num}" / "results.json"
        if result_file.exists():
            completed.append(seq)
        else:
            pending.append(seq)

    print(f"\n✅ Already completed: {len(completed)}")
    if completed:
        print(f"   {', '.join(completed)}")

    print(f"\n⏳ Pending: {len(pending)}")
    if pending:
        print(f"   {', '.join(pending)}")

    # Run experiments on pending sequences
    if pending:
        print(f"\n{'='*80}")
        print("RUNNING EXPERIMENTS")
        print(f"{'='*80}")

        successful = []
        failed = []

        for i, seq in enumerate(pending, 1):
            print(f"\n[{i}/{len(pending)}] Processing {seq}")

            success = run_epistemic_experiment(seq, python_path)

            if success:
                successful.append(seq)
                # Also run visualization
                run_visualizations(seq, python_path)
            else:
                failed.append(seq)

        print(f"\n{'='*80}")
        print("BATCH SUMMARY")
        print(f"{'='*80}")
        print(f"✅ Successful: {len(successful)}/{len(pending)}")
        if successful:
            print(f"   {', '.join(successful)}")
        if failed:
            print(f"❌ Failed: {len(failed)}/{len(pending)}")
            print(f"   {', '.join(failed)}")

    # Generate comprehensive report for all completed sequences
    print(f"\n{'='*80}")
    print("GENERATING COMPREHENSIVE REPORT")
    print(f"{'='*80}")

    all_completed = completed + [s for s in pending if s in successful] if 'successful' in locals() else completed

    report_data = {}

    for seq in all_completed:
        seq_num = seq.split('-')[1]
        result_file = results_dir / f"epistemic_mot17_{seq_num}" / "results.json"

        if result_file.exists():
            with open(result_file, 'r') as f:
                data = json.load(f)

            report_data[seq] = {
                'n_samples': data['data']['n_samples'],
                'aleatoric_r': data['correlations']['aleatoric']['pearson']['r'],
                'epistemic_r': data['correlations']['epistemic']['pearson']['r'],
                'total_r': data['correlations']['total']['pearson']['r'],
                'orthogonality': abs(data['correlations']['orthogonality']['aleatoric_epistemic_corr']),
                'epistemic_fraction': data['epistemic_fraction']['mean'],
                'spectral_mean': data['epistemic_components']['spectral']['mean'],
                'repulsive_mean': data['epistemic_components']['repulsive']['mean']
            }

    # Save comprehensive report
    report_path = results_dir / "epistemic_all_sequences_report.json"
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)

    print(f"Report saved to: {report_path}")

    # Print summary table
    if report_data:
        print("\n" + "="*100)
        print("ALL SEQUENCES SUMMARY")
        print("="*100)
        print(f"{'Sequence':<15} {'Samples':<8} {'Aleat r':<10} {'Epist r':<10} "
              f"{'Ortho':<8} {'Epist %':<10} {'Status'}")
        print("-"*100)

        for seq in sorted(report_data.keys()):
            d = report_data[seq]
            status = '✅' if d['orthogonality'] < 0.3 else '⚠️'
            print(f"{seq:<15} {d['n_samples']:<8} {d['aleatoric_r']:>9.3f} "
                  f"{d['epistemic_r']:>9.3f} {d['orthogonality']:>7.3f} "
                  f"{d['epistemic_fraction']*100:>8.1f}% {status:>8}")

        # Overall statistics
        print("-"*100)
        ortho_vals = [d['orthogonality'] for d in report_data.values()]
        epist_frac = [d['epistemic_fraction'] for d in report_data.values()]
        print(f"{'MEAN':<15} {'':<8} {'':<10} {'':<10} "
              f"{np.mean(ortho_vals):>7.3f} {np.mean(epist_frac)*100:>8.1f}%")
        print(f"{'STD':<15} {'':<8} {'':<10} {'':<10} "
              f"{np.std(ortho_vals):>7.3f} {np.std(epist_frac)*100:>8.1f}%")

    print(f"\n{'='*80}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()