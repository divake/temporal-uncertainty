"""
Run conformal prediction experiments on all MOT17 sequences

This script:
1. Runs conformal prediction on all 7 MOT17 sequences
2. Aggregates results into comprehensive JSON
3. Generates summary plots and tables
4. Creates publishable figures

Author: Enhanced CACD Team
Date: November 11, 2025
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.run_conformal_mot17 import run_conformal_experiment

# All MOT17 sequences
SEQUENCES = [
    'MOT17-02-FRCNN',
    'MOT17-04-FRCNN',
    'MOT17-05-FRCNN',
    'MOT17-09-FRCNN',
    'MOT17-10-FRCNN',
    'MOT17-11-FRCNN',
    'MOT17-13-FRCNN'
]

def run_all_sequences():
    """Run conformal prediction on all sequences"""
    print("="*80)
    print("RUNNING CONFORMAL PREDICTION ON ALL MOT17 SEQUENCES")
    print("="*80)
    print(f"Total sequences: {len(SEQUENCES)}")
    print(f"Sequences: {', '.join(SEQUENCES)}\n")

    results = {}

    for i, seq in enumerate(SEQUENCES, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(SEQUENCES)}] Processing {seq}")
        print(f"{'='*80}\n")

        try:
            # Run experiment
            result = run_conformal_experiment(seq)
            results[seq] = result

            print(f"\n✓ {seq} completed successfully")
            print(f"  Vanilla coverage: {result['vanilla']['coverage']:.1%}")
            print(f"  Global coverage: {result['global']['coverage']:.1%}")
            print(f"  Local coverage: {result['local']['coverage']:.1%}")

        except Exception as e:
            print(f"\n✗ {seq} failed: {e}")
            results[seq] = {'error': str(e)}

    return results


def aggregate_results(results):
    """Aggregate results into comprehensive summary"""
    print("\n" + "="*80)
    print("AGGREGATING RESULTS")
    print("="*80)

    # Collect statistics
    aggregated = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'num_sequences': len(SEQUENCES),
            'sequences': SEQUENCES
        },
        'per_sequence': {},
        'summary': {
            'vanilla': {'coverages': [], 'widths': [], 'n_samples': []},
            'global': {'coverages': [], 'widths': [], 'n_samples': []},
            'local': {'coverages': [], 'widths': [], 'n_samples': []}
        }
    }

    for seq, result in results.items():
        if 'error' in result:
            continue

        # Per-sequence details
        aggregated['per_sequence'][seq] = {
            'vanilla': {
                'coverage': result['methods']['vanilla']['coverage'],
                'mean_width': result['methods']['vanilla']['mean_width'],
                'n_test': result['n_test']
            },
            'global': {
                'coverage': result['methods']['combined_global']['coverage'],
                'mean_width': result['methods']['combined_global']['mean_width'],
                'n_test': result['n_test'],
                'q_hat': result['methods']['combined_global']['q_hat'],
                'width_reduction_pct': result['width_reduction']['global_vs_vanilla']
            },
            'local': {
                'coverage': result['methods']['combined_local']['coverage'],
                'mean_width': result['methods']['combined_local']['mean_width'],
                'n_test': result['n_test'],
                'q_hat': result['methods']['combined_local']['q_hat'],
                'width_reduction_pct': result['width_reduction']['local_vs_vanilla'],
                'num_leaves': result['methods']['combined_local']['tree_leaves']
            },
            'calibration': {
                'n_cal': result['n_calibration']
            }
        }

        # Summary statistics
        aggregated['summary']['vanilla']['coverages'].append(result['methods']['vanilla']['coverage'])
        aggregated['summary']['vanilla']['widths'].append(result['methods']['vanilla']['mean_width'])
        aggregated['summary']['vanilla']['n_samples'].append(result['n_test'])

        aggregated['summary']['global']['coverages'].append(result['methods']['combined_global']['coverage'])
        aggregated['summary']['global']['widths'].append(result['methods']['combined_global']['mean_width'])
        aggregated['summary']['global']['n_samples'].append(result['n_test'])

        aggregated['summary']['local']['coverages'].append(result['methods']['combined_local']['coverage'])
        aggregated['summary']['local']['widths'].append(result['methods']['combined_local']['mean_width'])
        aggregated['summary']['local']['n_samples'].append(result['n_test'])

    # Compute overall statistics
    for method in ['vanilla', 'global', 'local']:
        data = aggregated['summary'][method]
        if data['coverages']:
            data['mean_coverage'] = float(np.mean(data['coverages']))
            data['std_coverage'] = float(np.std(data['coverages']))
            data['mean_width'] = float(np.mean(data['widths']))
            data['std_width'] = float(np.std(data['widths']))
            data['total_samples'] = sum(data['n_samples'])

    # Compute overall efficiency gains
    if aggregated['summary']['vanilla']['widths']:
        vanilla_mean = aggregated['summary']['vanilla']['mean_width']
        global_mean = aggregated['summary']['global']['mean_width']
        local_mean = aggregated['summary']['local']['mean_width']

        aggregated['summary']['global']['avg_width_reduction_pct'] = \
            (1 - global_mean / vanilla_mean) * 100
        aggregated['summary']['local']['avg_width_reduction_pct'] = \
            (1 - local_mean / vanilla_mean) * 100

    return aggregated


def create_summary_table(aggregated):
    """Create summary table for paper"""
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)

    # Create pandas DataFrame
    rows = []
    for seq in SEQUENCES:
        if seq not in aggregated['per_sequence']:
            continue

        seq_data = aggregated['per_sequence'][seq]
        rows.append({
            'Sequence': seq.replace('MOT17-', '').replace('-FRCNN', ''),
            'N_test': seq_data['vanilla']['n_test'],
            'V_Cov': f"{seq_data['vanilla']['coverage']:.1%}",
            'V_Width': f"{seq_data['vanilla']['mean_width']:.3f}",
            'G_Cov': f"{seq_data['global']['coverage']:.1%}",
            'G_Width': f"{seq_data['global']['mean_width']:.3f}",
            'G_Δ%': f"{seq_data['global']['width_reduction_pct']:+.1f}%",
            'L_Cov': f"{seq_data['local']['coverage']:.1%}",
            'L_Width': f"{seq_data['local']['mean_width']:.3f}",
            'L_Δ%': f"{seq_data['local']['width_reduction_pct']:+.1f}%"
        })

    df = pd.DataFrame(rows)

    # Print table
    print("\nPer-Sequence Results:")
    print(df.to_string(index=False))

    # Overall statistics
    print("\n" + "-"*80)
    print("Overall Statistics (across all sequences):")
    print("-"*80)

    for method_name, method_key in [('Vanilla', 'vanilla'),
                                     ('Combined (Global)', 'global'),
                                     ('Combined (Local)', 'local')]:
        data = aggregated['summary'][method_key]
        print(f"\n{method_name}:")
        print(f"  Mean Coverage: {data['mean_coverage']:.1%} (± {data['std_coverage']:.1%})")
        print(f"  Mean Width:    {data['mean_width']:.4f} (± {data['std_width']:.4f})")
        print(f"  Total Samples: {data['total_samples']}")

        if 'avg_width_reduction_pct' in data:
            print(f"  Width Reduction: {data['avg_width_reduction_pct']:+.1f}%")

    return df


def create_summary_plots(aggregated, save_dir):
    """Create comprehensive summary plots"""
    print("\n" + "="*80)
    print("GENERATING SUMMARY PLOTS")
    print("="*80)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    sequences = [seq for seq in SEQUENCES if seq in aggregated['per_sequence']]
    seq_labels = [s.replace('MOT17-', '').replace('-FRCNN', '') for s in sequences]

    # Extract data
    vanilla_cov = [aggregated['per_sequence'][s]['vanilla']['coverage'] for s in sequences]
    global_cov = [aggregated['per_sequence'][s]['global']['coverage'] for s in sequences]
    local_cov = [aggregated['per_sequence'][s]['local']['coverage'] for s in sequences]

    vanilla_width = [aggregated['per_sequence'][s]['vanilla']['mean_width'] for s in sequences]
    global_width = [aggregated['per_sequence'][s]['global']['mean_width'] for s in sequences]
    local_width = [aggregated['per_sequence'][s]['local']['mean_width'] for s in sequences]

    # Plot 1: Coverage comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Coverage bar plot
    ax = axes[0]
    x = np.arange(len(sequences))
    width = 0.25

    ax.bar(x - width, vanilla_cov, width, label='Vanilla', color='gray', alpha=0.8)
    ax.bar(x, global_cov, width, label='Combined (Global)', color='blue', alpha=0.8)
    ax.bar(x + width, local_cov, width, label='Combined (Local)', color='green', alpha=0.8)

    ax.axhline(0.9, color='red', linestyle='--', linewidth=2, label='Target (90%)')
    ax.set_xlabel('Sequence', fontsize=12, fontweight='bold')
    ax.set_ylabel('Empirical Coverage', fontsize=12, fontweight='bold')
    ax.set_title('Coverage Comparison Across Sequences', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(seq_labels, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.75, 1.0])

    # Width bar plot
    ax = axes[1]
    ax.bar(x - width, vanilla_width, width, label='Vanilla', color='gray', alpha=0.8)
    ax.bar(x, global_width, width, label='Combined (Global)', color='blue', alpha=0.8)
    ax.bar(x + width, local_width, width, label='Combined (Local)', color='green', alpha=0.8)

    ax.set_xlabel('Sequence', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Interval Width', fontsize=12, fontweight='bold')
    ax.set_title('Interval Width Comparison Across Sequences', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(seq_labels, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = save_dir / 'summary_coverage_width_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

    # Plot 2: Efficiency vs Coverage tradeoff
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot
    ax.scatter(vanilla_width, vanilla_cov, s=200, marker='o',
               color='gray', alpha=0.7, label='Vanilla', zorder=3)
    ax.scatter(global_width, global_cov, s=200, marker='s',
               color='blue', alpha=0.7, label='Combined (Global)', zorder=3)
    ax.scatter(local_width, local_cov, s=200, marker='^',
               color='green', alpha=0.7, label='Combined (Local)', zorder=3)

    # Add sequence labels
    for i, label in enumerate(seq_labels):
        ax.annotate(label, (vanilla_width[i], vanilla_cov[i]),
                   fontsize=8, alpha=0.6, xytext=(5, 5), textcoords='offset points')

    ax.axhline(0.9, color='red', linestyle='--', linewidth=2,
               label='Target Coverage (90%)', zorder=1)
    ax.set_xlabel('Mean Interval Width', fontsize=13, fontweight='bold')
    ax.set_ylabel('Empirical Coverage', fontsize=13, fontweight='bold')
    ax.set_title('Efficiency-Coverage Tradeoff', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / 'summary_efficiency_coverage_tradeoff.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

    # Plot 3: Overall summary statistics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Overall Summary Statistics', fontsize=16, fontweight='bold')

    methods = ['vanilla', 'global', 'local']
    method_names = ['Vanilla', 'Combined\n(Global)', 'Combined\n(Local)']
    colors = ['gray', 'blue', 'green']

    # Coverage statistics
    ax = axes[0, 0]
    means = [aggregated['summary'][m]['mean_coverage'] for m in methods]
    stds = [aggregated['summary'][m]['std_coverage'] for m in methods]
    ax.bar(method_names, means, yerr=stds, color=colors, alpha=0.7, capsize=10)
    ax.axhline(0.9, color='red', linestyle='--', linewidth=2)
    ax.set_ylabel('Coverage', fontsize=11, fontweight='bold')
    ax.set_title('Mean Coverage (± std)', fontsize=12, fontweight='bold')
    ax.set_ylim([0.85, 0.95])
    ax.grid(True, alpha=0.3, axis='y')

    # Width statistics
    ax = axes[0, 1]
    means = [aggregated['summary'][m]['mean_width'] for m in methods]
    stds = [aggregated['summary'][m]['std_width'] for m in methods]
    ax.bar(method_names, means, yerr=stds, color=colors, alpha=0.7, capsize=10)
    ax.set_ylabel('Mean Width', fontsize=11, fontweight='bold')
    ax.set_title('Mean Interval Width (± std)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Width reduction
    ax = axes[1, 0]
    reductions = [0,
                  aggregated['summary']['global']['avg_width_reduction_pct'],
                  aggregated['summary']['local']['avg_width_reduction_pct']]
    bars = ax.bar(method_names, reductions, color=colors, alpha=0.7)
    for bar, val in zip(bars, reductions):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:+.1f}%', ha='center', va='bottom' if val >= 0 else 'top',
                fontweight='bold')
    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel('Width Reduction (%)', fontsize=11, fontweight='bold')
    ax.set_title('Efficiency Gain vs Vanilla', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Sample sizes
    ax = axes[1, 1]
    totals = [aggregated['summary'][m]['total_samples'] for m in methods]
    ax.bar(method_names, totals, color=colors, alpha=0.7)
    ax.set_ylabel('Total Test Samples', fontsize=11, fontweight='bold')
    ax.set_title('Total Samples Evaluated', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = save_dir / 'summary_overall_statistics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    """Main execution"""
    # Output directory
    results_dir = Path(__file__).parent.parent / 'results' / 'conformal_all_sequences'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = results_dir / f'run_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}\n")

    # Run experiments
    results = run_all_sequences()

    # Aggregate results
    aggregated = aggregate_results(results)

    # Save raw results
    raw_path = output_dir / 'raw_results.json'
    with open(raw_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved raw results: {raw_path}")

    # Save aggregated results
    agg_path = output_dir / 'aggregated_results.json'
    with open(agg_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"✓ Saved aggregated results: {agg_path}")

    # Create summary table
    df = create_summary_table(aggregated)
    table_path = output_dir / 'summary_table.csv'
    df.to_csv(table_path, index=False)
    print(f"\n✓ Saved summary table: {table_path}")

    # Create plots
    create_summary_plots(aggregated, output_dir / 'plots')

    # Create symlink to latest
    latest_link = results_dir / 'latest'
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(output_dir.name)

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE ✓")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"Latest results: {latest_link}")

    return aggregated


if __name__ == '__main__':
    main()
