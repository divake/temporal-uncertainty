"""
Run Experiment 1C (IoU>=0.3, conf>=0.3) on All MOT17 Sequences

This script runs the complete pipeline on all 7 MOT17 sequences:
- MOT17-02, MOT17-04, MOT17-05, MOT17-09, MOT17-10, MOT17-11, MOT17-13

Each sequence gets its own results directory:
- results/aleatoric_mot17_02/
- results/aleatoric_mot17_04/
- etc.

Author: Enhanced CACD Team
Date: 2025-11-10
"""

import subprocess
import yaml
from pathlib import Path
import sys
import json
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent


def update_config_sequence(sequence_name: str):
    """Update datasets.yaml to select a specific sequence."""
    config_path = PROJECT_ROOT / 'config' / 'datasets.yaml'

    # Read config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update selected sequence
    config['mot17']['selected_sequence'] = sequence_name

    # Write back
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"  ✓ Updated config: selected_sequence = {sequence_name}")


def update_experiment_output_dirs(sequence_num: str):
    """Update experiment.yaml output directories for current sequence."""
    config_path = PROJECT_ROOT / 'config' / 'experiment.yaml'

    # Read config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update output directories
    config['experiment']['output']['results_dir'] = f'results/aleatoric_mot17_{sequence_num}'
    config['experiment']['output']['plots_dir'] = f'results/aleatoric_mot17_{sequence_num}/plots'

    # Write back
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"  ✓ Updated output dirs: results/aleatoric_mot17_{sequence_num}/")


def run_experiment(sequence_name: str, sequence_num: str):
    """Run experiment for a specific sequence."""
    print("\n" + "="*80)
    print(f"RUNNING EXPERIMENT: {sequence_name}")
    print("="*80)

    # Update configs
    print("\n[Config] Updating configuration files...")
    update_config_sequence(sequence_name)
    update_experiment_output_dirs(sequence_num)

    # Run experiment
    print(f"\n[Experiment] Running aleatoric uncertainty experiment...")
    script_path = PROJECT_ROOT / 'experiments' / 'run_aleatoric_mot17_11.py'
    python_path = '/home/divake/miniconda3/envs/env_py311/bin/python'

    try:
        result = subprocess.run(
            [python_path, str(script_path)],
            cwd=str(PROJECT_ROOT),
            check=True,
            capture_output=False,
            text=True
        )

        print(f"\n✓ Experiment completed successfully for {sequence_name}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Experiment failed for {sequence_name}")
        print(f"Error: {e}")
        return False


def collect_results_summary():
    """Collect results from all sequences and create summary."""
    print("\n" + "="*80)
    print("COLLECTING RESULTS SUMMARY")
    print("="*80)

    sequences = ['02', '04', '05', '09', '10', '11', '13']
    results_summary = {
        'experiment_name': 'Experiment 1C - All Sequences',
        'config': 'IoU >= 0.3, Confidence >= 0.3',
        'timestamp': datetime.now().isoformat(),
        'sequences': {}
    }

    for seq_num in sequences:
        results_path = PROJECT_ROOT / 'results' / f'aleatoric_mot17_{seq_num}' / 'results.json'

        if results_path.exists():
            with open(results_path, 'r') as f:
                seq_results = json.load(f)

            # Extract key metrics
            pearson_r = seq_results['correlations']['normalized_uncertainty']['pearson']['r']
            p_value = seq_results['correlations']['normalized_uncertainty']['pearson']['p_value']
            n_samples = seq_results['data']['n_samples']

            iou_stats = seq_results['uncertainty_by_iou_quality']

            results_summary['sequences'][f'MOT17-{seq_num}'] = {
                'pearson_r': pearson_r,
                'p_value': p_value,
                'n_samples': n_samples,
                'variance_explained': pearson_r ** 2,
                'excellent_count': iou_stats['excellent_iou_0.7']['count'],
                'good_count': iou_stats['good_iou_0.5_0.7']['count'],
                'poor_count': iou_stats['poor_iou_below_0.5']['count'],
                'excellent_unc': iou_stats['excellent_iou_0.7']['mean_uncertainty'],
                'good_unc': iou_stats['good_iou_0.5_0.7']['mean_uncertainty'],
                'poor_unc': iou_stats['poor_iou_below_0.5']['mean_uncertainty']
            }

            print(f"\n  MOT17-{seq_num}:")
            print(f"    Pearson r = {pearson_r:.4f} (p = {p_value:.2e})")
            print(f"    Samples: {n_samples} (Excellent: {iou_stats['excellent_iou_0.7']['count']}, "
                  f"Good: {iou_stats['good_iou_0.5_0.7']['count']}, "
                  f"Poor: {iou_stats['poor_iou_below_0.5']['count']})")
        else:
            print(f"\n  MOT17-{seq_num}: Results not found")

    # Save summary
    summary_path = PROJECT_ROOT / 'results' / 'all_sequences_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n✓ Summary saved to: {summary_path}")

    # Print aggregate statistics
    if results_summary['sequences']:
        all_rs = [v['pearson_r'] for v in results_summary['sequences'].values()]
        print(f"\n{'='*60}")
        print(f"AGGREGATE STATISTICS")
        print(f"{'='*60}")
        print(f"  Mean Pearson r: {sum(all_rs)/len(all_rs):.4f}")
        print(f"  Min Pearson r:  {min(all_rs):.4f}")
        print(f"  Max Pearson r:  {max(all_rs):.4f}")
        print(f"{'='*60}\n")

    return results_summary


def main():
    """Main execution pipeline."""
    print("\n" + "="*80)
    print("RUNNING EXPERIMENT 1C ON ALL MOT17 SEQUENCES")
    print("Configuration: IoU >= 0.3, Confidence >= 0.3")
    print("="*80)

    # Define sequences
    sequences = [
        ('MOT17-02-FRCNN', '02'),
        ('MOT17-04-FRCNN', '04'),
        ('MOT17-05-FRCNN', '05'),
        ('MOT17-09-FRCNN', '09'),
        ('MOT17-10-FRCNN', '10'),
        ('MOT17-11-FRCNN', '11'),
        ('MOT17-13-FRCNN', '13')
    ]

    successful = []
    failed = []

    # Run experiments
    for sequence_name, sequence_num in sequences:
        success = run_experiment(sequence_name, sequence_num)

        if success:
            successful.append(sequence_name)
        else:
            failed.append(sequence_name)

    # Collect and summarize results
    print("\n" + "="*80)
    summary = collect_results_summary()

    # Final report
    print("\n" + "="*80)
    print("FINAL REPORT")
    print("="*80)
    print(f"\n✓ Successful: {len(successful)}/{len(sequences)}")
    for seq in successful:
        print(f"    - {seq}")

    if failed:
        print(f"\n✗ Failed: {len(failed)}/{len(sequences)}")
        for seq in failed:
            print(f"    - {seq}")

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*80)
    print(f"\nResults available in: {PROJECT_ROOT / 'results'}/")
    print(f"Summary: {PROJECT_ROOT / 'results' / 'all_sequences_summary.json'}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
