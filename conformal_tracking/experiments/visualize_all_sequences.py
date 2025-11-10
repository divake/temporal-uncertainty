"""
Comprehensive Multi-Sequence Visualization for Epistemic Uncertainty

This script creates comparative visualizations across all MOT17 sequences,
showing patterns, trends, and insights from the epistemic uncertainty analysis.

Author: Enhanced CACD Team
Date: 2025-11-10
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import seaborn as sns
import pandas as pd
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_all_results(results_dir):
    """Load results from all available sequences."""
    all_results = {}

    # Find all epistemic result directories
    for dir_path in results_dir.glob("epistemic_mot17_*"):
        if dir_path.is_dir():
            results_file = dir_path / "results.json"
            if results_file.exists():
                seq_num = dir_path.name.split('_')[-1]
                seq_name = f"MOT17-{seq_num}"

                with open(results_file, 'r') as f:
                    all_results[seq_name] = json.load(f)

    return all_results


def create_overview_dashboard(all_results, save_path):
    """Create comprehensive overview dashboard."""

    fig = plt.figure(figsize=(24, 16))
    fig.suptitle('Epistemic Uncertainty Analysis - All MOT17 Sequences',
                 fontsize=22, fontweight='bold')

    sequences = sorted(all_results.keys())
    n_seq = len(sequences)

    # Extract data for plotting
    aleatoric_r = []
    epistemic_r = []
    orthogonality = []
    epistemic_fraction = []
    n_samples = []
    spectral_mean = []
    repulsive_mean = []

    for seq in sequences:
        data = all_results[seq]
        aleatoric_r.append(data['correlations']['aleatoric']['pearson']['r'])
        epistemic_r.append(data['correlations']['epistemic']['pearson']['r'])
        orthogonality.append(abs(data['correlations']['orthogonality']['aleatoric_epistemic_corr']))
        epistemic_fraction.append(data['epistemic_fraction']['mean'])
        n_samples.append(data['data']['n_samples'])
        spectral_mean.append(data['epistemic_components']['spectral']['mean'])
        repulsive_mean.append(data['epistemic_components']['repulsive']['mean'])

    # Short sequence names for x-axis
    seq_labels = [s.replace('MOT17-', '') for s in sequences]

    # 1. Correlation Comparison
    ax1 = plt.subplot(3, 4, 1)
    x = np.arange(len(sequences))
    width = 0.35
    bars1 = ax1.bar(x - width/2, aleatoric_r, width, label='Aleatoric', color='blue', alpha=0.7)
    bars2 = ax1.bar(x + width/2, epistemic_r, width, label='Epistemic', color='red', alpha=0.7)

    ax1.set_xlabel('Sequence')
    ax1.set_ylabel('Correlation with Conformity')
    ax1.set_title('Uncertainty Correlations')
    ax1.set_xticks(x)
    ax1.set_xticklabels(seq_labels, rotation=45)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars1, aleatoric_r):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom' if val > 0 else 'top', fontsize=8)

    for bar, val in zip(bars2, epistemic_r):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom' if val > 0 else 'top', fontsize=8)

    # 2. Orthogonality Achievement
    ax2 = plt.subplot(3, 4, 2)
    colors = ['green' if o < 0.3 else 'orange' for o in orthogonality]
    bars = ax2.bar(seq_labels, orthogonality, color=colors, alpha=0.7)
    ax2.axhline(y=0.3, color='red', linestyle='--', label='Target (|r| < 0.3)')
    ax2.set_xlabel('Sequence')
    ax2.set_ylabel('|Correlation(A, E)|')
    ax2.set_title('Orthogonality Achievement')
    ax2.set_xticklabels(seq_labels, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, orthogonality):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    # 3. Epistemic Fraction
    ax3 = plt.subplot(3, 4, 3)
    ax3.bar(seq_labels, [f*100 for f in epistemic_fraction], color='purple', alpha=0.7)
    ax3.axhline(y=np.mean(epistemic_fraction)*100, color='red', linestyle='--',
               label=f'Mean: {np.mean(epistemic_fraction)*100:.1f}%')
    ax3.set_xlabel('Sequence')
    ax3.set_ylabel('Epistemic Fraction (%)')
    ax3.set_title('Epistemic Contribution to Total Uncertainty')
    ax3.set_xticklabels(seq_labels, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Sample Size Effect
    ax4 = plt.subplot(3, 4, 4)
    scatter = ax4.scatter(n_samples, orthogonality, s=100, alpha=0.7,
                         c=range(len(n_samples)), cmap='viridis')
    ax4.set_xlabel('Number of Samples')
    ax4.set_ylabel('Orthogonality |r|')
    ax4.set_title('Sample Size vs Orthogonality')
    ax4.axhline(y=0.3, color='red', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3)

    # Add sequence labels
    for i, seq in enumerate(seq_labels):
        ax4.annotate(seq, (n_samples[i], orthogonality[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    # 5. Correlation Scatter
    ax5 = plt.subplot(3, 4, 5)
    scatter = ax5.scatter(aleatoric_r, epistemic_r, s=100, alpha=0.7,
                         c=orthogonality, cmap='RdYlGn_r', vmin=0, vmax=0.5)
    ax5.set_xlabel('Aleatoric Correlation')
    ax5.set_ylabel('Epistemic Correlation')
    ax5.set_title('Correlation Space')
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax5.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='Orthogonality')

    # Add sequence labels
    for i, seq in enumerate(seq_labels):
        ax5.annotate(seq, (aleatoric_r[i], epistemic_r[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    # 6. Component Comparison
    ax6 = plt.subplot(3, 4, 6)
    x = np.arange(len(sequences))
    width = 0.35
    ax6.bar(x - width/2, spectral_mean, width, label='Spectral', color='green', alpha=0.7)
    ax6.bar(x + width/2, repulsive_mean, width, label='Repulsive', color='orange', alpha=0.7)
    ax6.set_xlabel('Sequence')
    ax6.set_ylabel('Mean Uncertainty')
    ax6.set_title('Epistemic Component Comparison')
    ax6.set_xticks(x)
    ax6.set_xticklabels(seq_labels, rotation=45)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. Box Plot of Correlations
    ax7 = plt.subplot(3, 4, 7)
    data_for_box = [aleatoric_r, epistemic_r, orthogonality]
    bp = ax7.boxplot(data_for_box, labels=['Aleatoric r', 'Epistemic r', 'Orthogonality'],
                     patch_artist=True)
    colors = ['blue', 'red', 'green']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax7.set_ylabel('Value')
    ax7.set_title('Distribution of Metrics')
    ax7.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax7.grid(True, alpha=0.3)

    # 8. Heatmap of All Metrics
    ax8 = plt.subplot(3, 4, 8)

    # Create matrix for heatmap
    metrics_matrix = np.array([
        aleatoric_r,
        epistemic_r,
        orthogonality,
        epistemic_fraction
    ])

    im = ax8.imshow(metrics_matrix, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax8.set_xticks(range(len(seq_labels)))
    ax8.set_xticklabels(seq_labels, rotation=45)
    ax8.set_yticks(range(4))
    ax8.set_yticklabels(['Aleatoric r', 'Epistemic r', 'Orthogonality', 'Epistemic %'])
    ax8.set_title('Metrics Heatmap')
    plt.colorbar(im, ax=ax8)

    # Add text annotations
    for i in range(4):
        for j in range(len(seq_labels)):
            text = ax8.text(j, i, f'{metrics_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)

    # 9. Uncertainty by IoU Categories (aggregate)
    ax9 = plt.subplot(3, 4, 9)

    # Aggregate IoU statistics across sequences
    excellent_aleatoric = []
    excellent_epistemic = []
    good_aleatoric = []
    good_epistemic = []
    poor_aleatoric = []
    poor_epistemic = []

    for seq, data in all_results.items():
        if 'uncertainty_by_iou' in data:
            iou_data = data['uncertainty_by_iou']
            if 'excellent' in iou_data:
                excellent_aleatoric.append(iou_data['excellent']['aleatoric']['mean'])
                excellent_epistemic.append(iou_data['excellent']['epistemic']['mean'])
            if 'good' in iou_data:
                good_aleatoric.append(iou_data['good']['aleatoric']['mean'])
                good_epistemic.append(iou_data['good']['epistemic']['mean'])
            if 'poor' in iou_data:
                poor_aleatoric.append(iou_data['poor']['aleatoric']['mean'])
                poor_epistemic.append(iou_data['poor']['epistemic']['mean'])

    categories = ['Excellent\nIoU>0.8', 'Good\n0.6-0.8', 'Poor\nIoU<0.6']
    aleatoric_means = [np.mean(excellent_aleatoric) if excellent_aleatoric else 0,
                       np.mean(good_aleatoric) if good_aleatoric else 0,
                       np.mean(poor_aleatoric) if poor_aleatoric else 0]
    epistemic_means = [np.mean(excellent_epistemic) if excellent_epistemic else 0,
                       np.mean(good_epistemic) if good_epistemic else 0,
                       np.mean(poor_epistemic) if poor_epistemic else 0]

    x = np.arange(len(categories))
    width = 0.35
    ax9.bar(x - width/2, aleatoric_means, width, label='Aleatoric', color='blue', alpha=0.7)
    ax9.bar(x + width/2, epistemic_means, width, label='Epistemic', color='red', alpha=0.7)
    ax9.set_xlabel('IoU Category')
    ax9.set_ylabel('Mean Uncertainty')
    ax9.set_title('Uncertainty by Detection Quality (All Sequences)')
    ax9.set_xticks(x)
    ax9.set_xticklabels(categories)
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    # 10. Statistical Summary
    ax10 = plt.subplot(3, 4, 10)
    ax10.axis('off')

    # Calculate statistics
    mean_ortho = np.mean(orthogonality)
    std_ortho = np.std(orthogonality)
    success_rate = sum(1 for o in orthogonality if o < 0.3) / len(orthogonality) * 100

    summary_text = f"""
    SUMMARY STATISTICS
    {'='*35}

    Total Sequences: {n_seq}
    Total Detections: {sum(n_samples):,}

    Orthogonality:
      Mean: {mean_ortho:.3f}
      Std:  {std_ortho:.3f}
      Success Rate: {success_rate:.1f}%

    Correlations:
      Aleatoric Mean: {np.mean(aleatoric_r):.3f}
      Epistemic Mean: {np.mean(epistemic_r):.3f}

    Epistemic Fraction:
      Mean: {np.mean(epistemic_fraction)*100:.1f}%
      Std:  {np.std(epistemic_fraction)*100:.1f}%

    Key Finding:
      {'✅ All sequences achieve orthogonality' if success_rate == 100 else f'⚠️ {100-success_rate:.0f}% need improvement'}
    """

    ax10.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center')

    # 11. Trend Analysis
    ax11 = plt.subplot(3, 4, 11)

    # Sort sequences by aleatoric correlation for trend
    sorted_indices = np.argsort(aleatoric_r)
    sorted_aleatoric = [aleatoric_r[i] for i in sorted_indices]
    sorted_epistemic = [epistemic_r[i] for i in sorted_indices]
    sorted_labels = [seq_labels[i] for i in sorted_indices]

    ax11.plot(range(len(sorted_aleatoric)), sorted_aleatoric, 'b-o', label='Aleatoric', linewidth=2)
    ax11.plot(range(len(sorted_epistemic)), sorted_epistemic, 'r-s', label='Epistemic', linewidth=2)
    ax11.set_xlabel('Sequences (sorted by aleatoric r)')
    ax11.set_ylabel('Correlation')
    ax11.set_title('Correlation Trends')
    ax11.set_xticks(range(len(sorted_labels)))
    ax11.set_xticklabels(sorted_labels, rotation=45)
    ax11.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax11.legend()
    ax11.grid(True, alpha=0.3)

    # 12. Success Indicators
    ax12 = plt.subplot(3, 4, 12)

    # Create success matrix
    success_matrix = []
    for seq in sequences:
        data = all_results[seq]
        ortho = abs(data['correlations']['orthogonality']['aleatoric_epistemic_corr'])
        aleat_p = data['correlations']['aleatoric']['pearson']['p_value']
        epist_p = data['correlations']['epistemic']['pearson']['p_value']

        success_row = [
            1 if ortho < 0.3 else 0,  # Orthogonality achieved
            1 if aleat_p < 0.05 else 0,  # Aleatoric significant
            1 if epist_p < 0.05 else 0,  # Epistemic significant
        ]
        success_matrix.append(success_row)

    success_matrix = np.array(success_matrix).T

    im = ax12.imshow(success_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax12.set_xticks(range(len(seq_labels)))
    ax12.set_xticklabels(seq_labels, rotation=45)
    ax12.set_yticks(range(3))
    ax12.set_yticklabels(['Orthogonal', 'Aleat Sig.', 'Epist Sig.'])
    ax12.set_title('Success Indicators')

    # Add checkmarks/crosses
    for i in range(3):
        for j in range(len(seq_labels)):
            symbol = '✓' if success_matrix[i, j] == 1 else '✗'
            color = 'white' if success_matrix[i, j] == 1 else 'black'
            ax12.text(j, i, symbol, ha="center", va="center",
                     color=color, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved overview dashboard: {save_path}")


def create_detailed_comparison(all_results, save_path):
    """Create detailed sequence-by-sequence comparison."""

    n_seq = len(all_results)
    sequences = sorted(all_results.keys())

    fig, axes = plt.subplots(n_seq, 4, figsize=(20, 4*n_seq))
    if n_seq == 1:
        axes = axes.reshape(1, -1)

    for idx, seq in enumerate(sequences):
        data = all_results[seq]

        # Extract data
        aleat_r = data['correlations']['aleatoric']['pearson']['r']
        epist_r = data['correlations']['epistemic']['pearson']['r']
        total_r = data['correlations']['total']['pearson']['r']
        ortho = abs(data['correlations']['orthogonality']['aleatoric_epistemic_corr'])
        epist_frac = data['epistemic_fraction']['mean']
        n_samp = data['data']['n_samples']

        # Plot 1: Correlations bar chart
        ax1 = axes[idx, 0]
        correlations = [aleat_r, epist_r, total_r]
        colors = ['blue', 'red', 'purple']
        bars = ax1.bar(['Aleatoric', 'Epistemic', 'Total'], correlations, color=colors, alpha=0.7)
        ax1.set_ylabel('Correlation')
        ax1.set_title(f'{seq}: Correlations')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_ylim([-0.5, 0.5])
        ax1.grid(True, alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, correlations):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom' if val > 0 else 'top')

        # Plot 2: Orthogonality gauge
        ax2 = axes[idx, 1]
        color = 'green' if ortho < 0.3 else 'orange'
        ax2.barh([0], [ortho], color=color, alpha=0.7)
        ax2.axvline(x=0.3, color='red', linestyle='--', label='Target')
        ax2.set_xlim([0, 0.5])
        ax2.set_xlabel('|Correlation(A,E)|')
        ax2.set_title(f'Orthogonality: {ortho:.3f}')
        ax2.set_yticks([])
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Epistemic components
        ax3 = axes[idx, 2]
        if 'epistemic_components' in data:
            spectral = data['epistemic_components']['spectral']['mean']
            repulsive = data['epistemic_components']['repulsive']['mean']
            ax3.bar(['Spectral', 'Repulsive'], [spectral, repulsive],
                   color=['green', 'orange'], alpha=0.7)
            ax3.set_ylabel('Mean Uncertainty')
            ax3.set_title('Epistemic Components')
            ax3.grid(True, alpha=0.3)

        # Plot 4: Info box
        ax4 = axes[idx, 3]
        ax4.axis('off')
        info_text = f"""
        {seq} Statistics
        {'='*25}
        Samples: {n_samp}
        Epistemic %: {epist_frac*100:.1f}%

        P-values:
        Aleatoric: {data['correlations']['aleatoric']['pearson']['p_value']:.2e}
        Epistemic: {data['correlations']['epistemic']['pearson']['p_value']:.2e}

        Status: {'✅ PASS' if ortho < 0.3 else '⚠️ REVIEW'}
        """
        ax4.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
                verticalalignment='center')

    plt.suptitle('Detailed Sequence-by-Sequence Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved detailed comparison: {save_path}")


def main():
    """Main execution."""

    results_dir = Path("/ssd_4TB/divake/temporal_uncertainty/conformal_tracking/results")
    viz_dir = results_dir / "all_sequences_visualizations"
    viz_dir.mkdir(exist_ok=True)

    print("\n" + "="*80)
    print("GENERATING MULTI-SEQUENCE VISUALIZATIONS")
    print("="*80)

    # Load all results
    all_results = load_all_results(results_dir)

    if not all_results:
        print("❌ No results found!")
        return

    print(f"Found results for {len(all_results)} sequences:")
    for seq in sorted(all_results.keys()):
        print(f"  - {seq}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Overview dashboard
    create_overview_dashboard(
        all_results,
        viz_dir / "overview_dashboard.png"
    )

    # 2. Detailed comparison
    create_detailed_comparison(
        all_results,
        viz_dir / "detailed_comparison.png"
    )

    print(f"\n✅ Visualizations complete!")
    print(f"   Saved to: {viz_dir}")


if __name__ == "__main__":
    main()