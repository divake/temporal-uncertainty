"""
Comprehensive Uncertainty Decomposition Visualization

This script creates extensive visualizations showing how total uncertainty
is decomposed into aleatoric and epistemic components at the detection level,
with separate analysis for each epistemic method.

Author: Enhanced CACD Team
Date: 2025-11-10
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / 'src' / 'uncertainty'))
sys.path.append(str(PROJECT_ROOT / 'src'))
sys.path.append(str(PROJECT_ROOT / 'data_loaders'))

from mot17_loader import MOT17DataLoader
from mahalanobis import MahalanobisUncertainty
from epistemic_combined import EpistemicUncertainty


def create_detection_level_decomposition(aleatoric, epistemic_spectral, epistemic_repulsive,
                                        epistemic_combined, conformity, iou,
                                        save_path, sequence_name):
    """Create comprehensive detection-level uncertainty decomposition plot."""

    fig = plt.figure(figsize=(24, 20))
    fig.suptitle(f'Detection-Level Uncertainty Decomposition - {sequence_name}',
                 fontsize=20, fontweight='bold')

    # Sample subset for detailed view (first 500 detections)
    n_detailed = min(500, len(aleatoric))
    indices = np.arange(n_detailed)

    # =================
    # Row 1: Component Breakdown
    # =================

    # 1.1: Aleatoric vs Epistemic Stacked
    ax1 = plt.subplot(5, 3, 1)
    ax1.bar(indices, aleatoric[:n_detailed], label='Aleatoric', alpha=0.7, color='blue')
    ax1.bar(indices, epistemic_combined[:n_detailed], bottom=aleatoric[:n_detailed],
            label='Epistemic', alpha=0.7, color='red')
    ax1.set_xlabel('Detection Index')
    ax1.set_ylabel('Uncertainty')
    ax1.set_title('Stacked Uncertainty Components')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 1.2: Epistemic Methods Comparison
    ax2 = plt.subplot(5, 3, 2)
    ax2.plot(indices, epistemic_spectral[:n_detailed], label='Spectral', alpha=0.7, linewidth=1)
    ax2.plot(indices, epistemic_repulsive[:n_detailed], label='Repulsive', alpha=0.7, linewidth=1)
    ax2.plot(indices, epistemic_combined[:n_detailed], label='Combined', alpha=0.9,
             linewidth=2, color='black')
    ax2.set_xlabel('Detection Index')
    ax2.set_ylabel('Epistemic Uncertainty')
    ax2.set_title('Epistemic Method Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 1.3: Uncertainty Ratio (Epistemic / Total)
    ax3 = plt.subplot(5, 3, 3)
    total = aleatoric + epistemic_combined
    ratio = epistemic_combined / (total + 1e-10)
    ax3.fill_between(indices, 0, ratio[:n_detailed], alpha=0.5, color='purple')
    ax3.axhline(y=ratio.mean(), color='red', linestyle='--',
                label=f'Mean: {ratio.mean():.2%}')
    ax3.set_xlabel('Detection Index')
    ax3.set_ylabel('Epistemic Fraction')
    ax3.set_title('Epistemic Contribution to Total Uncertainty')
    ax3.set_ylim([0, 1])
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # =================
    # Row 2: Correlation with Ground Truth
    # =================

    # 2.1: Uncertainty vs IoU
    ax4 = plt.subplot(5, 3, 4)
    scatter1 = ax4.scatter(iou, aleatoric, alpha=0.3, s=10, label='Aleatoric')
    scatter2 = ax4.scatter(iou, epistemic_combined, alpha=0.3, s=10, label='Epistemic')

    # Add trend lines
    z1 = np.polyfit(iou, aleatoric, 1)
    p1 = np.poly1d(z1)
    z2 = np.polyfit(iou, epistemic_combined, 1)
    p2 = np.poly1d(z2)

    iou_sorted = np.sort(iou)
    ax4.plot(iou_sorted, p1(iou_sorted), "b-", alpha=0.8, linewidth=2)
    ax4.plot(iou_sorted, p2(iou_sorted), "r-", alpha=0.8, linewidth=2)

    ax4.set_xlabel('IoU with Ground Truth')
    ax4.set_ylabel('Uncertainty')
    ax4.set_title('Uncertainty vs Detection Quality')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 2.2: Uncertainty vs Conformity Score
    ax5 = plt.subplot(5, 3, 5)
    scatter3 = ax5.scatter(conformity, aleatoric, alpha=0.3, s=10, label='Aleatoric')
    scatter4 = ax5.scatter(conformity, epistemic_combined, alpha=0.3, s=10, label='Epistemic')

    # Correlations
    r_aleatoric, _ = pearsonr(conformity, aleatoric)
    r_epistemic, _ = pearsonr(conformity, epistemic_combined)

    ax5.set_xlabel('Conformity Score (1 - IoU)')
    ax5.set_ylabel('Uncertainty')
    ax5.set_title(f'Uncertainty vs Conformity\n(r_a={r_aleatoric:.3f}, r_e={r_epistemic:.3f})')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 2.3: Orthogonality Check
    ax6 = plt.subplot(5, 3, 6)
    ax6.scatter(aleatoric, epistemic_combined, alpha=0.3, s=10)

    # Correlation and orthogonality line
    r_ortho, _ = pearsonr(aleatoric, epistemic_combined)
    ax6.set_xlabel('Aleatoric Uncertainty')
    ax6.set_ylabel('Epistemic Uncertainty')
    ax6.set_title(f'Orthogonality Check\n(r={r_ortho:.3f})')

    # Add reference lines
    ax6.axhline(y=epistemic_combined.mean(), color='red', linestyle='--', alpha=0.5)
    ax6.axvline(x=aleatoric.mean(), color='blue', linestyle='--', alpha=0.5)
    ax6.grid(True, alpha=0.3)

    # =================
    # Row 3: Spectral Method Analysis
    # =================

    # 3.1: Spectral Uncertainty Distribution
    ax7 = plt.subplot(5, 3, 7)
    ax7.hist(epistemic_spectral, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax7.axvline(x=epistemic_spectral.mean(), color='red', linestyle='--',
                label=f'Mean: {epistemic_spectral.mean():.3f}')
    ax7.set_xlabel('Spectral Uncertainty')
    ax7.set_ylabel('Frequency')
    ax7.set_title('Spectral Collapse Detection Distribution')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 3.2: Spectral vs IoU Categories
    ax8 = plt.subplot(5, 3, 8)
    categories = ['Excellent\n(IoU>0.8)', 'Good\n(0.6-0.8)', 'Poor\n(IoU<0.6)']
    excellent_mask = iou > 0.8
    good_mask = (iou >= 0.6) & (iou <= 0.8)
    poor_mask = iou < 0.6

    spectral_by_category = [
        epistemic_spectral[excellent_mask],
        epistemic_spectral[good_mask],
        epistemic_spectral[poor_mask]
    ]

    bp1 = ax8.boxplot(spectral_by_category, labels=categories, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('green')
        patch.set_alpha(0.7)

    ax8.set_ylabel('Spectral Uncertainty')
    ax8.set_title('Spectral Method by Detection Quality')
    ax8.grid(True, alpha=0.3)

    # 3.3: Spectral Temporal Evolution
    ax9 = plt.subplot(5, 3, 9)
    window_size = 50
    spectral_rolling = np.convolve(epistemic_spectral,
                                   np.ones(window_size)/window_size,
                                   mode='valid')
    ax9.plot(spectral_rolling, color='green', linewidth=2)
    ax9.fill_between(range(len(spectral_rolling)), 0, spectral_rolling,
                     alpha=0.3, color='green')
    ax9.set_xlabel('Detection Index')
    ax9.set_ylabel('Rolling Mean Spectral Uncertainty')
    ax9.set_title(f'Spectral Evolution (window={window_size})')
    ax9.grid(True, alpha=0.3)

    # =================
    # Row 4: Repulsive Method Analysis
    # =================

    # 4.1: Repulsive Uncertainty Distribution
    ax10 = plt.subplot(5, 3, 10)
    ax10.hist(epistemic_repulsive, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax10.axvline(x=epistemic_repulsive.mean(), color='red', linestyle='--',
                 label=f'Mean: {epistemic_repulsive.mean():.3f}')
    ax10.set_xlabel('Repulsive Uncertainty')
    ax10.set_ylabel('Frequency')
    ax10.set_title('Repulsive Void Detection Distribution')
    ax10.legend()
    ax10.grid(True, alpha=0.3)

    # 4.2: Repulsive vs IoU Categories
    ax11 = plt.subplot(5, 3, 11)
    repulsive_by_category = [
        epistemic_repulsive[excellent_mask],
        epistemic_repulsive[good_mask],
        epistemic_repulsive[poor_mask]
    ]

    bp2 = ax11.boxplot(repulsive_by_category, labels=categories, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('orange')
        patch.set_alpha(0.7)

    ax11.set_ylabel('Repulsive Uncertainty')
    ax11.set_title('Repulsive Method by Detection Quality')
    ax11.grid(True, alpha=0.3)

    # 4.3: Repulsive Temporal Evolution
    ax12 = plt.subplot(5, 3, 12)
    repulsive_rolling = np.convolve(epistemic_repulsive,
                                    np.ones(window_size)/window_size,
                                    mode='valid')
    ax12.plot(repulsive_rolling, color='orange', linewidth=2)
    ax12.fill_between(range(len(repulsive_rolling)), 0, repulsive_rolling,
                     alpha=0.3, color='orange')
    ax12.set_xlabel('Detection Index')
    ax12.set_ylabel('Rolling Mean Repulsive Uncertainty')
    ax12.set_title(f'Repulsive Evolution (window={window_size})')
    ax12.grid(True, alpha=0.3)

    # =================
    # Row 5: Combined Analysis
    # =================

    # 5.1: Method Contributions
    ax13 = plt.subplot(5, 3, 13)
    # Note: We'll approximate contributions based on correlation
    spectral_contrib = np.abs(np.corrcoef(epistemic_spectral, epistemic_combined)[0,1])
    repulsive_contrib = np.abs(np.corrcoef(epistemic_repulsive, epistemic_combined)[0,1])
    total_contrib = spectral_contrib + repulsive_contrib

    sizes = [spectral_contrib/total_contrib, repulsive_contrib/total_contrib]
    colors = ['green', 'orange']
    labels = [f'Spectral\n({sizes[0]:.1%})', f'Repulsive\n({sizes[1]:.1%})']

    ax13.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
             startangle=90, textprops={'fontsize': 10})
    ax13.set_title('Method Contributions to Combined')

    # 5.2: Uncertainty Heatmap
    ax14 = plt.subplot(5, 3, 14)

    # Create 2D histogram
    H, xedges, yedges = np.histogram2d(aleatoric, epistemic_combined, bins=30)
    H = H.T

    im = ax14.imshow(H, origin='lower', aspect='auto', cmap='YlOrRd',
                     extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    ax14.set_xlabel('Aleatoric Uncertainty')
    ax14.set_ylabel('Epistemic Uncertainty')
    ax14.set_title('Joint Uncertainty Distribution')
    plt.colorbar(im, ax=ax14, label='Detection Count')

    # 5.3: Summary Statistics
    ax15 = plt.subplot(5, 3, 15)
    ax15.axis('off')

    summary_text = f"""
    SUMMARY STATISTICS - {sequence_name}
    {'='*40}

    Aleatoric Uncertainty:
      Mean: {aleatoric.mean():.3f}
      Std:  {aleatoric.std():.3f}
      Correlation with IoU: {pearsonr(iou, aleatoric)[0]:.3f}

    Epistemic Uncertainty (Combined):
      Mean: {epistemic_combined.mean():.3f}
      Std:  {epistemic_combined.std():.3f}
      Correlation with IoU: {pearsonr(iou, epistemic_combined)[0]:.3f}

    Epistemic Components:
      Spectral Mean:  {epistemic_spectral.mean():.3f}
      Repulsive Mean: {epistemic_repulsive.mean():.3f}

    Orthogonality:
      Correlation(A,E): {r_ortho:.3f}
      Status: {'✅ ORTHOGONAL' if abs(r_ortho) < 0.3 else '⚠️ CORRELATED'}

    Epistemic Fraction:
      Mean: {ratio.mean():.1%}
      Std:  {ratio.std():.1%}
    """

    ax15.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved comprehensive decomposition plot: {save_path}")

    return {
        'aleatoric_mean': aleatoric.mean(),
        'epistemic_mean': epistemic_combined.mean(),
        'spectral_mean': epistemic_spectral.mean(),
        'repulsive_mean': epistemic_repulsive.mean(),
        'orthogonality': r_ortho,
        'epistemic_fraction_mean': ratio.mean()
    }


def create_frame_level_analysis(detections_per_frame, uncertainties_per_frame,
                               save_path, sequence_name):
    """Create frame-level aggregated uncertainty analysis."""

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'Frame-Level Uncertainty Analysis - {sequence_name}',
                 fontsize=18, fontweight='bold')

    frames = list(detections_per_frame.keys())
    n_frames = len(frames)

    # Aggregate uncertainties per frame
    aleatoric_mean = []
    epistemic_mean = []
    total_mean = []
    n_detections = []

    for frame in frames:
        if frame in uncertainties_per_frame:
            unc = uncertainties_per_frame[frame]
            aleatoric_mean.append(unc['aleatoric'].mean())
            epistemic_mean.append(unc['epistemic'].mean())
            total_mean.append(unc['total'].mean())
            n_detections.append(len(unc['aleatoric']))
        else:
            aleatoric_mean.append(0)
            epistemic_mean.append(0)
            total_mean.append(0)
            n_detections.append(0)

    # 1. Uncertainty Evolution
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(frames, aleatoric_mean, label='Aleatoric', alpha=0.8, linewidth=2)
    ax1.plot(frames, epistemic_mean, label='Epistemic', alpha=0.8, linewidth=2)
    ax1.plot(frames, total_mean, label='Total', alpha=0.8, linewidth=2, color='black')
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('Mean Uncertainty')
    ax1.set_title('Uncertainty Evolution Across Frames')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Detection Count
    ax2 = plt.subplot(3, 2, 2)
    ax2.bar(frames, n_detections, alpha=0.7, color='gray')
    ax2.set_xlabel('Frame Number')
    ax2.set_ylabel('Number of Detections')
    ax2.set_title('Detection Count per Frame')
    ax2.grid(True, alpha=0.3)

    # 3. Stacked Area Chart
    ax3 = plt.subplot(3, 2, 3)
    ax3.fill_between(frames, 0, aleatoric_mean, alpha=0.7, color='blue', label='Aleatoric')
    ax3.fill_between(frames, aleatoric_mean,
                     np.array(aleatoric_mean) + np.array(epistemic_mean),
                     alpha=0.7, color='red', label='Epistemic')
    ax3.set_xlabel('Frame Number')
    ax3.set_ylabel('Stacked Uncertainty')
    ax3.set_title('Uncertainty Composition per Frame')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Epistemic Fraction Evolution
    ax4 = plt.subplot(3, 2, 4)
    epistemic_fraction = np.array(epistemic_mean) / (np.array(total_mean) + 1e-10)
    ax4.plot(frames, epistemic_fraction, linewidth=2, color='purple')
    ax4.fill_between(frames, 0, epistemic_fraction, alpha=0.3, color='purple')
    ax4.axhline(y=epistemic_fraction.mean(), color='red', linestyle='--',
                label=f'Mean: {epistemic_fraction.mean():.1%}')
    ax4.set_xlabel('Frame Number')
    ax4.set_ylabel('Epistemic Fraction')
    ax4.set_title('Epistemic Contribution Evolution')
    ax4.set_ylim([0, 1])
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Rolling Statistics
    ax5 = plt.subplot(3, 2, 5)
    window = 30
    aleatoric_rolling = np.convolve(aleatoric_mean, np.ones(window)/window, mode='valid')
    epistemic_rolling = np.convolve(epistemic_mean, np.ones(window)/window, mode='valid')

    ax5.plot(frames[:len(aleatoric_rolling)], aleatoric_rolling,
             label=f'Aleatoric (w={window})', linewidth=2)
    ax5.plot(frames[:len(epistemic_rolling)], epistemic_rolling,
             label=f'Epistemic (w={window})', linewidth=2)
    ax5.set_xlabel('Frame Number')
    ax5.set_ylabel('Rolling Mean Uncertainty')
    ax5.set_title('Smoothed Uncertainty Trends')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Summary
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')

    summary_text = f"""
    FRAME-LEVEL SUMMARY - {sequence_name}
    {'='*40}

    Total Frames: {n_frames}
    Total Detections: {sum(n_detections)}
    Avg Detections/Frame: {np.mean(n_detections):.1f}

    Mean Uncertainties (across frames):
      Aleatoric:  {np.mean(aleatoric_mean):.3f}
      Epistemic:  {np.mean(epistemic_mean):.3f}
      Total:      {np.mean(total_mean):.3f}

    Epistemic Fraction:
      Mean: {epistemic_fraction.mean():.1%}
      Std:  {epistemic_fraction.std():.1%}
      Min:  {epistemic_fraction.min():.1%}
      Max:  {epistemic_fraction.max():.1%}

    Variability (std across frames):
      Aleatoric:  {np.std(aleatoric_mean):.3f}
      Epistemic:  {np.std(epistemic_mean):.3f}
    """

    ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved frame-level analysis: {save_path}")


def run_comprehensive_visualization(sequence_name, cache_dir, results_base_dir):
    """Run complete visualization pipeline for a sequence."""

    print(f"\n{'='*80}")
    print(f"PROCESSING {sequence_name}")
    print(f"{'='*80}")

    # Setup paths
    cache_path = cache_dir / f"{sequence_name}.npz"
    seq_num = sequence_name.split('-')[1]

    # Check if results exist
    epistemic_results_dir = results_base_dir / f"epistemic_mot17_{seq_num}"
    if not epistemic_results_dir.exists():
        print(f"  ⚠️ No epistemic results found for {sequence_name}")
        print(f"     Please run: python experiments/run_epistemic_mot17.py {sequence_name}")
        return None

    # Create visualization directory
    viz_dir = epistemic_results_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    # Load saved model and results
    with open(epistemic_results_dir / "results.json", 'r') as f:
        results = json.load(f)

    with open(epistemic_results_dir / "epistemic_model.json", 'r') as f:
        model_data = json.load(f)

    print(f"\n[1/4] Loading data...")

    # Load data
    loader = MOT17DataLoader(
        cache_path=cache_path,
        layer_id=21,
        conf_threshold=0.3,
        split_ratio=0.5,
        random_seed=42
    )

    # Get test data
    test_data = loader.get_test_data()
    features = test_data['features']
    iou = test_data['ious']  # Changed from 'iou' to 'ious'
    conformity = test_data['conformity_scores']  # Changed from 'conformity'

    # Get frame IDs from the raw cache
    cache_data = np.load(cache_path)

    # Get detections data
    all_frame_ids = cache_data['detections/frame_ids']
    conf_values = cache_data['detections/confidences']

    # Get matching info
    matched_indices = cache_data['gt_matching/det_indices']

    # Create matched mask
    matched_mask = np.zeros(len(all_frame_ids), dtype=bool)
    matched_mask[matched_indices] = True

    # Apply same filtering as loader
    matched_frame_ids = all_frame_ids[matched_mask]
    matched_conf = conf_values[matched_mask]
    conf_mask = matched_conf >= 0.3
    filtered_frame_ids = matched_frame_ids[conf_mask]

    # Split same way as loader (second half for test)
    np.random.seed(42)
    n_samples = len(filtered_frame_ids)
    indices = np.random.permutation(n_samples)
    n_cal = n_samples // 2
    test_indices = indices[n_cal:]
    frame_ids = filtered_frame_ids[test_indices]

    print(f"  Test samples: {len(features)}")

    print(f"\n[2/4] Computing uncertainties...")

    # Fit and compute aleatoric
    mahalanobis_model = MahalanobisUncertainty()
    cal_data = loader.get_calibration_data()
    mahalanobis_model.fit(cal_data['features'], verbose=False)
    aleatoric_results = mahalanobis_model.predict(features, verbose=False)
    aleatoric = aleatoric_results['normalized']

    # Fit and compute epistemic
    epistemic_model = EpistemicUncertainty(
        k_neighbors_spectral=50,
        k_neighbors_repulsive=100,
        temperature=1.0,
        weights='optimize',
        verbose=False
    )

    epistemic_model.fit(
        cal_data['features'],
        aleatoric_cal=mahalanobis_model.predict(cal_data['features'], verbose=False)['normalized'],
        mahalanobis_model=mahalanobis_model,
        save_dir=viz_dir / "recomputed"
    )

    epistemic_results = epistemic_model.predict(
        features,
        save_dir=viz_dir / "recomputed_test"
    )

    epistemic_combined = epistemic_results['combined']
    epistemic_spectral = epistemic_results['spectral']
    epistemic_repulsive = epistemic_results['repulsive']

    print(f"\n[3/4] Creating detection-level visualizations...")

    # Detection-level decomposition
    decomp_stats = create_detection_level_decomposition(
        aleatoric=aleatoric,
        epistemic_spectral=epistemic_spectral,
        epistemic_repulsive=epistemic_repulsive,
        epistemic_combined=epistemic_combined,
        conformity=conformity,
        iou=iou,
        save_path=viz_dir / "detection_decomposition.png",
        sequence_name=sequence_name
    )

    print(f"\n[4/4] Creating frame-level analysis...")

    # Organize by frame
    detections_per_frame = {}
    uncertainties_per_frame = {}

    for i, frame_id in enumerate(frame_ids):
        if frame_id not in detections_per_frame:
            detections_per_frame[frame_id] = []
            uncertainties_per_frame[frame_id] = {
                'aleatoric': [],
                'epistemic': [],
                'total': []
            }

        detections_per_frame[frame_id].append(i)
        uncertainties_per_frame[frame_id]['aleatoric'].append(aleatoric[i])
        uncertainties_per_frame[frame_id]['epistemic'].append(epistemic_combined[i])
        uncertainties_per_frame[frame_id]['total'].append(aleatoric[i] + epistemic_combined[i])

    # Convert lists to arrays
    for frame_id in uncertainties_per_frame:
        for key in uncertainties_per_frame[frame_id]:
            uncertainties_per_frame[frame_id][key] = np.array(uncertainties_per_frame[frame_id][key])

    # Frame-level analysis
    create_frame_level_analysis(
        detections_per_frame=detections_per_frame,
        uncertainties_per_frame=uncertainties_per_frame,
        save_path=viz_dir / "frame_analysis.png",
        sequence_name=sequence_name
    )

    print(f"\n✅ Visualization complete for {sequence_name}")
    print(f"   Results saved to: {viz_dir}")

    return decomp_stats


def main():
    """Main execution."""

    # Setup paths
    cache_dir = Path("/ssd_4TB/divake/temporal_uncertainty/yolo_cache/data/mot17/yolov8n")
    results_base_dir = Path("/ssd_4TB/divake/temporal_uncertainty/conformal_tracking/results")

    # Process sequences that have results
    sequences = ["MOT17-11-FRCNN", "MOT17-13-FRCNN"]

    all_stats = {}

    for seq in sequences:
        stats = run_comprehensive_visualization(seq, cache_dir, results_base_dir)
        if stats:
            all_stats[seq] = stats

    # Create comparison summary
    if all_stats:
        print("\n" + "="*80)
        print("CROSS-SEQUENCE COMPARISON")
        print("="*80)

        for seq, stats in all_stats.items():
            print(f"\n{seq}:")
            print(f"  Aleatoric Mean:  {stats['aleatoric_mean']:.3f}")
            print(f"  Epistemic Mean:  {stats['epistemic_mean']:.3f}")
            print(f"  - Spectral:      {stats['spectral_mean']:.3f}")
            print(f"  - Repulsive:     {stats['repulsive_mean']:.3f}")
            print(f"  Orthogonality:   {abs(stats['orthogonality']):.3f}")
            print(f"  Epistemic %:     {stats['epistemic_fraction_mean']:.1%}")


if __name__ == "__main__":
    main()