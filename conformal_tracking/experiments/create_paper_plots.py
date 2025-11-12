"""
Individual Publication-Quality Plots - REAL DATA
=================================================

Creates separate square plots using REAL per-detection uncertainty values
(not synthetic data generated from summary statistics).

Features:
- Times New Roman font
- Large axis numbers and labels
- 1:1 aspect ratio
- 150 DPI for paper submission
- Uses actual computed uncertainty values

Author: Analysis Team
Date: 2025-11-11
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, gaussian_kde
from scipy.ndimage import gaussian_filter, uniform_filter1d
from pathlib import Path
import matplotlib

# Set font to Times New Roman
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'paper_figures' / 'real_data'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_real_data(sequence='MOT17-05'):
    """Load real per-detection data from NPZ file"""
    data_path = PROJECT_ROOT / 'results' / f'per_detection_{sequence.lower()}' / 'per_detection_data.npz'

    if not data_path.exists():
        raise FileNotFoundError(f"Real data not found: {data_path}\nRun save_per_detection_data.py first!")

    data = np.load(data_path)

    return {
        'aleatoric': data['aleatoric'],
        'epistemic': data['epistemic'],
        'iou': data['ious'],
        'conformity': data['conformity'],
        'spectral': data['spectral'],
        'repulsive': data['repulsive'],
        'total': data['total']
    }


def plot_1_main_scatter(data):
    """
    Plot 1: Main scatter plot with density contours
    Aleatoric vs Epistemic colored by conformity
    """
    print("\n[1/6] Creating main scatter plot...")

    fig, ax = plt.subplots(figsize=(10, 10))

    # Scatter plot
    scatter = ax.scatter(data['aleatoric'], data['epistemic'],
                        c=data['conformity'], cmap='RdYlBu_r',
                        s=60, alpha=0.7, edgecolors='black',
                        linewidth=0.5, rasterized=True)

    # Density contours
    try:
        xy = np.vstack([data['aleatoric'], data['epistemic']])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = data['aleatoric'][idx], data['epistemic'][idx], z[idx]

        xi = np.linspace(data['aleatoric'].min(), data['aleatoric'].max(), 100)
        yi = np.linspace(data['epistemic'].min(), data['epistemic'].max(), 100)
        Xi, Yi = np.meshgrid(xi, yi)

        from scipy.interpolate import griddata
        Zi = griddata((x, y), z, (Xi, Yi), method='cubic')
        Zi_smooth = gaussian_filter(Zi, sigma=1.5)

        contours = ax.contour(Xi, Yi, Zi_smooth, levels=5,
                             colors='black', alpha=0.4,
                             linewidths=2, linestyles='solid')
    except:
        pass

    # Styling
    ax.set_xlabel('Aleatoric Uncertainty', fontsize=28, fontweight='bold')
    ax.set_ylabel('Epistemic Uncertainty', fontsize=28, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.set_aspect('equal', adjustable='box')

    # Manually set x-axis ticks to shift "1.0" label slightly left
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 0.98])
    ax.set_xticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label('Conformity Score', fontsize=24, fontweight='bold')
    cbar.ax.tick_params(labelsize=20)

    plt.tight_layout()
    output = OUTPUT_DIR / 'plot1_scatter_main.png'
    plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'plot1_scatter_main.pdf', bbox_inches='tight')
    print(f"  ✓ Saved: {output}")
    plt.close()


def plot_2_aleatoric_distribution(data):
    """Plot 2: Aleatoric distribution with KDE"""
    print("\n[2/6] Creating aleatoric distribution...")

    fig, ax = plt.subplots(figsize=(10, 10))

    # Histogram
    n, bins, patches = ax.hist(data['aleatoric'], bins=40, density=True,
                               alpha=0.7, color='green', edgecolor='black',
                               linewidth=1.5)

    # KDE overlay
    density = gaussian_kde(data['aleatoric'])
    xs = np.linspace(data['aleatoric'].min(), data['aleatoric'].max(), 300)
    ax.plot(xs, density(xs), 'darkgreen', linewidth=4, label='KDE')

    # Mean line
    mean_val = data['aleatoric'].mean()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=3,
              label=f'Mean = {mean_val:.3f}')

    ax.set_xlabel('Aleatoric Uncertainty', fontsize=28, fontweight='bold')
    ax.set_ylabel('Density', fontsize=28, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.legend(fontsize=22, loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1)

    plt.tight_layout()
    output = OUTPUT_DIR / 'plot2_aleatoric_dist.png'
    plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'plot2_aleatoric_dist.pdf', bbox_inches='tight')
    print(f"  ✓ Saved: {output}")
    plt.close()


def plot_3_epistemic_distribution(data):
    """Plot 3: Epistemic distribution with KDE"""
    print("\n[3/6] Creating epistemic distribution...")

    fig, ax = plt.subplots(figsize=(10, 10))

    # Histogram
    ax.hist(data['epistemic'], bins=40, density=True, alpha=0.7,
           color='blue', edgecolor='black', linewidth=1.5)

    # KDE overlay
    density = gaussian_kde(data['epistemic'])
    xs = np.linspace(data['epistemic'].min(), data['epistemic'].max(), 300)
    ax.plot(xs, density(xs), 'darkblue', linewidth=4, label='KDE')

    # Mean line
    mean_val = data['epistemic'].mean()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=3,
              label=f'Mean = {mean_val:.3f}')

    ax.set_xlabel('Epistemic Uncertainty', fontsize=28, fontweight='bold')
    ax.set_ylabel('Density', fontsize=28, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.legend(fontsize=22, loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1)

    plt.tight_layout()
    output = OUTPUT_DIR / 'plot3_epistemic_dist.png'
    plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'plot3_epistemic_dist.pdf', bbox_inches='tight')
    print(f"  ✓ Saved: {output}")
    plt.close()


def plot_4_iou_evolution(data):
    """Plot 4: Uncertainty evolution across IoU"""
    print("\n[4/6] Creating IoU evolution plot...")

    fig, ax = plt.subplots(figsize=(10, 10))

    # Bin by IoU
    iou_bins = np.linspace(0, 1, 20)
    iou_centers = (iou_bins[:-1] + iou_bins[1:]) / 2

    alea_by_iou = []
    epis_by_iou = []

    for i in range(len(iou_bins) - 1):
        mask = (data['iou'] >= iou_bins[i]) & (data['iou'] < iou_bins[i+1])
        if mask.sum() > 0:
            alea_by_iou.append(data['aleatoric'][mask].mean())
            epis_by_iou.append(data['epistemic'][mask].mean())
        else:
            alea_by_iou.append(np.nan)
            epis_by_iou.append(np.nan)

    # Smooth curves
    alea_smooth = uniform_filter1d(np.nan_to_num(alea_by_iou), size=3)
    epis_smooth = uniform_filter1d(np.nan_to_num(epis_by_iou), size=3)

    # Plot
    ax.plot(iou_centers, alea_smooth, 'o-', color='green', linewidth=4,
           markersize=12, label='Aleatoric', markeredgecolor='black',
           markeredgewidth=2)
    ax.plot(iou_centers, epis_smooth, 's-', color='blue', linewidth=4,
           markersize=12, label='Epistemic', markeredgecolor='black',
           markeredgewidth=2)

    # Set axis limits first (data starts from IoU ~0.3)
    ax.set_xlim([0.3, 1.0])
    ax.set_ylim([0, 0.7])

    # IoU category backgrounds
    ax.axvspan(0.3, 0.5, alpha=0.15, color='red')
    ax.axvspan(0.5, 0.8, alpha=0.15, color='yellow')
    ax.axvspan(0.8, 1.0, alpha=0.15, color='green')

    # Labels for regions
    ax.text(0.40, 0.67, 'Poor', fontsize=20,
           ha='center', fontweight='bold', color='darkred')
    ax.text(0.65, 0.67, 'Good', fontsize=20,
           ha='center', fontweight='bold', color='darkgoldenrod')
    ax.text(0.90, 0.67, 'Excellent', fontsize=20,
           ha='center', fontweight='bold', color='darkgreen')

    ax.set_xlabel('Ground Truth IoU', fontsize=28, fontweight='bold')
    ax.set_ylabel('Mean Uncertainty', fontsize=28, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.legend(fontsize=24, loc='lower center', framealpha=0.95, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)

    plt.tight_layout()
    output = OUTPUT_DIR / 'plot4_iou_evolution.png'
    plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'plot4_iou_evolution.pdf', bbox_inches='tight')
    print(f"  ✓ Saved: {output}")
    plt.close()


def plot_5_correlation_scatter(data):
    """Plot 5: Correlation scatter plots (3-panel)"""
    print("\n[5/6] Creating correlation scatter plots...")

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    # Panel A: Aleatoric vs IoU
    r_alea, p_alea = pearsonr(data['aleatoric'], data['iou'])
    axes[0].scatter(data['iou'], data['aleatoric'], alpha=0.5,
                   c='green', s=40, edgecolors='black', linewidth=0.3)
    axes[0].set_xlabel('Ground Truth IoU', fontsize=28, fontweight='bold')
    axes[0].set_ylabel('Aleatoric Uncertainty', fontsize=28, fontweight='bold')
    axes[0].set_title(f'Aleatoric ↔ IoU\nr = {r_alea:.3f}',
                     fontsize=24, fontweight='bold')
    axes[0].tick_params(axis='both', which='major', labelsize=24)
    axes[0].grid(True, alpha=0.3)

    # Panel B: Epistemic vs IoU
    r_epis, p_epis = pearsonr(data['epistemic'], data['iou'])
    axes[1].scatter(data['iou'], data['epistemic'], alpha=0.5,
                   c='blue', s=40, edgecolors='black', linewidth=0.3)
    axes[1].set_xlabel('Ground Truth IoU', fontsize=28, fontweight='bold')
    axes[1].set_ylabel('Epistemic Uncertainty', fontsize=28, fontweight='bold')
    axes[1].set_title(f'Epistemic ↔ IoU\nr = {r_epis:.3f}',
                     fontsize=24, fontweight='bold')
    axes[1].tick_params(axis='both', which='major', labelsize=24)
    axes[1].grid(True, alpha=0.3)

    # Panel C: Aleatoric vs Epistemic
    r_ortho, p_ortho = pearsonr(data['aleatoric'], data['epistemic'])
    axes[2].scatter(data['aleatoric'], data['epistemic'], alpha=0.5,
                   c='purple', s=40, edgecolors='black', linewidth=0.3)
    axes[2].set_xlabel('Aleatoric Uncertainty', fontsize=28, fontweight='bold')
    axes[2].set_ylabel('Epistemic Uncertainty', fontsize=28, fontweight='bold')
    axes[2].set_title(f'Orthogonality\nr = {r_ortho:.3f}',
                     fontsize=24, fontweight='bold')
    axes[2].tick_params(axis='both', which='major', labelsize=24)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_aspect('equal', adjustable='box')

    plt.tight_layout()
    output = OUTPUT_DIR / 'plot5_correlations.png'
    plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'plot5_correlations.pdf', bbox_inches='tight')
    print(f"  ✓ Saved: {output}")
    plt.close()


def plot_6_box_by_iou(data):
    """Plot 6: Box plots showing uncertainty by IoU category"""
    print("\n[6/6] Creating box plots by IoU category...")

    fig, ax = plt.subplots(figsize=(10, 10))

    # Create masks for categories
    excellent_mask = data['iou'] > 0.8
    good_mask = (data['iou'] >= 0.5) & (data['iou'] <= 0.8)
    poor_mask = data['iou'] < 0.5

    categories = ['Excellent\n(>0.8)', 'Good\n(0.5-0.8)', 'Poor\n(<0.5)']

    alea_data = [
        data['aleatoric'][excellent_mask],
        data['aleatoric'][good_mask],
        data['aleatoric'][poor_mask]
    ]

    epis_data = [
        data['epistemic'][excellent_mask],
        data['epistemic'][good_mask],
        data['epistemic'][poor_mask]
    ]

    x = np.arange(len(categories))
    width = 0.35

    # Bar plot with error bars
    alea_means = [np.mean(d) for d in alea_data]
    epis_means = [np.mean(d) for d in epis_data]
    alea_stds = [np.std(d) for d in alea_data]
    epis_stds = [np.std(d) for d in epis_data]

    ax.bar(x - width/2, alea_means, width, yerr=alea_stds, capsize=8,
          label='Aleatoric', color='green', alpha=0.8, edgecolor='black',
          linewidth=2, error_kw={'linewidth': 3})
    ax.bar(x + width/2, epis_means, width, yerr=epis_stds, capsize=8,
          label='Epistemic', color='blue', alpha=0.8, edgecolor='black',
          linewidth=2, error_kw={'linewidth': 3})

    ax.set_xlabel('Tracking Quality (IoU)', fontsize=28, fontweight='bold')
    ax.set_ylabel('Mean Uncertainty', fontsize=28, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=24)
    ax.tick_params(axis='y', labelsize=24)
    ax.legend(fontsize=24, loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1)

    plt.tight_layout()
    output = OUTPUT_DIR / 'plot6_box_by_iou.png'
    plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'plot6_box_by_iou.pdf', bbox_inches='tight')
    print(f"  ✓ Saved: {output}")
    plt.close()


def main():
    """Generate all individual plots using REAL data"""

    print("\n" + "="*80)
    print("CREATING INDIVIDUAL PUBLICATION PLOTS - REAL DATA")
    print("="*80)

    # Load real data
    print("\n[Setup] Loading REAL per-detection data...")
    data = load_real_data('MOT17-05')

    n_points = len(data['aleatoric'])
    print(f"  ✓ Loaded {n_points} REAL detections from MOT17-05")
    print(f"  Aleatoric range: [{data['aleatoric'].min():.3f}, {data['aleatoric'].max():.3f}]")
    print(f"  Epistemic range: [{data['epistemic'].min():.3f}, {data['epistemic'].max():.3f}]")

    # Compute correlations
    r_alea_iou, _ = pearsonr(data['aleatoric'], data['iou'])
    r_epis_iou, _ = pearsonr(data['epistemic'], data['iou'])
    r_ortho, _ = pearsonr(data['aleatoric'], data['epistemic'])

    print(f"\n  Correlations:")
    print(f"    Aleatoric ↔ IoU:      r = {r_alea_iou:+.4f}")
    print(f"    Epistemic ↔ IoU:      r = {r_epis_iou:+.4f}")
    print(f"    Aleatoric ↔ Epistemic: r = {r_ortho:+.4f}")

    # Create all plots
    plot_1_main_scatter(data)
    plot_2_aleatoric_distribution(data)
    plot_3_epistemic_distribution(data)
    plot_4_iou_evolution(data)
    plot_5_correlation_scatter(data)
    plot_6_box_by_iou(data)

    print("\n" + "="*80)
    print("ALL PLOTS COMPLETE ✓")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  1. plot1_scatter_main.png/pdf       - Main scatter with contours")
    print("  2. plot2_aleatoric_dist.png/pdf     - Aleatoric distribution")
    print("  3. plot3_epistemic_dist.png/pdf     - Epistemic distribution")
    print("  4. plot4_iou_evolution.png/pdf      - Uncertainty vs IoU")
    print("  5. plot5_correlations.png/pdf       - Correlation 3-panel plot")
    print("  6. plot6_box_by_iou.png/pdf         - Box plots by IoU category")
    print("\nAll plots:")
    print("  - REAL DATA (not synthetic!)")
    print("  - Square aspect ratio (10x10 inches)")
    print("  - Times New Roman font")
    print("  - Large axis labels (28pt) and ticks (24pt)")
    print("  - 150 DPI PNG + PDF vector format")
    print("="*80)


if __name__ == "__main__":
    main()
