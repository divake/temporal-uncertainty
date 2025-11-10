"""
Analyze YOLO Cache Before Implementation
=========================================

Before writing core code, let's understand:
1. What's in the cache files?
2. Feature ranges, distributions (Layer 21)
3. IoU distributions
4. Base YOLO performance (TP, FP, FN)
5. Confidence vs IoU relationship

This helps us understand what we're working with!

Author: Enhanced CACD Project
Date: 2025-11-09
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Simple loader (before we build the full one)
class SimpleYOLOCache:
    def __init__(self, cache_path):
        self.cache = np.load(str(cache_path))
        self.seq_name = Path(cache_path).stem

    def get_features(self, layer_id=21):
        key = f'features/layer_{layer_id}'
        return self.cache[key]

    def get_ious(self):
        return self.cache['gt_matching/iou']

    def get_confidences(self):
        return self.cache['detections/confidences']

    def get_matched_mask(self):
        n_dets = len(self.get_confidences())
        matched = np.zeros(n_dets, dtype=bool)
        det_indices = self.cache['gt_matching/det_indices']
        matched[det_indices] = True
        return matched


def analyze_single_sequence(seq_name, cache_dir):
    """Comprehensive analysis of one sequence."""

    print(f"\n{'='*80}")
    print(f"ANALYZING: {seq_name}")
    print('='*80)

    # Load cache
    cache_path = cache_dir / f'{seq_name}-FRCNN.npz'
    loader = SimpleYOLOCache(cache_path)

    # Get all data
    features_all = loader.get_features(layer_id=21)
    ious = loader.get_ious()
    confs_all = loader.get_confidences()
    matched_mask = loader.get_matched_mask()

    # Extract matched features/confs
    features_matched = features_all[matched_mask]
    confs_matched = confs_all[matched_mask]

    # Compute conformity scores
    conformity_scores = 1 - ious

    print(f"\n1. DATA COUNTS")
    print(f"   Total detections (all): {len(confs_all)}")
    print(f"   Matched (TP): {matched_mask.sum()} ({100*matched_mask.sum()/len(confs_all):.1f}%)")
    print(f"   Unmatched (FP): {(~matched_mask).sum()} ({100*(~matched_mask).sum()/len(confs_all):.1f}%)")

    print(f"\n2. LAYER 21 FEATURES (Matched detections)")
    print(f"   Shape: {features_matched.shape}")
    print(f"   Dimension: {features_matched.shape[1]}")
    print(f"   Mean: {features_matched.mean():.4f}")
    print(f"   Std: {features_matched.std():.4f}")
    print(f"   Min: {features_matched.min():.4f}")
    print(f"   Max: {features_matched.max():.4f}")

    # Feature norms
    norms = np.linalg.norm(features_matched, axis=1)
    print(f"\n   Feature L2 norms:")
    print(f"     Mean: {norms.mean():.4f}")
    print(f"     Std: {norms.std():.4f}")
    print(f"     Min: {norms.min():.4f}")
    print(f"     Max: {norms.max():.4f}")
    print(f"     Percentiles [25, 50, 75]: {np.percentile(norms, [25, 50, 75])}")

    print(f"\n3. CONFIDENCES")
    print(f"   All detections:")
    print(f"     Mean: {confs_all.mean():.4f}")
    print(f"     Std: {confs_all.std():.4f}")
    print(f"     Range: [{confs_all.min():.4f}, {confs_all.max():.4f}]")

    print(f"   Matched (TP) only:")
    print(f"     Mean: {confs_matched.mean():.4f}")
    print(f"     Std: {confs_matched.std():.4f}")
    print(f"     Range: [{confs_matched.min():.4f}, {confs_matched.max():.4f}]")

    print(f"\n4. IoU SCORES (Matched only)")
    print(f"   Mean: {ious.mean():.4f}")
    print(f"   Std: {ious.std():.4f}")
    print(f"   Min: {ious.min():.4f}")
    print(f"   Max: {ious.max():.4f}")
    print(f"   Percentiles [25, 50, 75, 95]: {np.percentile(ious, [25, 50, 75, 95])}")

    # IoU categories
    iou_excellent = (ious >= 0.7).sum()
    iou_good = ((ious >= 0.5) & (ious < 0.7)).sum()
    iou_poor = (ious < 0.5).sum()

    print(f"\n   IoU Categories:")
    print(f"     Excellent (≥0.7): {iou_excellent} ({100*iou_excellent/len(ious):.1f}%)")
    print(f"     Good (0.5-0.7): {iou_good} ({100*iou_good/len(ious):.1f}%)")
    print(f"     Poor (<0.5): {iou_poor} ({100*iou_poor/len(ious):.1f}%)")

    print(f"\n5. CONFORMITY SCORES (1 - IoU)")
    print(f"   Mean: {conformity_scores.mean():.4f}")
    print(f"   Std: {conformity_scores.std():.4f}")
    print(f"   Range: [{conformity_scores.min():.4f}, {conformity_scores.max():.4f}]")
    print(f"   Percentiles [25, 50, 75, 95]: {np.percentile(conformity_scores, [25, 50, 75, 95])}")

    print(f"\n6. CONFIDENCE vs IoU RELATIONSHIP")
    corr_conf_iou = np.corrcoef(confs_matched, ious)[0, 1]
    print(f"   Correlation(confidence, IoU): {corr_conf_iou:.4f}")

    if corr_conf_iou > 0.3:
        print(f"   → Positive correlation: Higher conf → Higher IoU (good!)")
    elif corr_conf_iou < -0.3:
        print(f"   → Negative correlation: Higher conf → Lower IoU (problem!)")
    else:
        print(f"   → Weak correlation: Confidence doesn't predict IoU well")

    print(f"\n7. CONFIDENCE FILTERING")
    for conf_thresh in [0.3, 0.5, 0.7]:
        mask = confs_matched >= conf_thresh
        n_kept = mask.sum()
        mean_iou_kept = ious[mask].mean() if n_kept > 0 else 0
        print(f"   At conf≥{conf_thresh}: {n_kept} samples ({100*n_kept/len(confs_matched):.1f}%), mean IoU={mean_iou_kept:.4f}")

    return {
        'seq_name': seq_name,
        'n_total': len(confs_all),
        'n_matched': matched_mask.sum(),
        'n_unmatched': (~matched_mask).sum(),
        'feature_dim': features_matched.shape[1],
        'feature_mean': features_matched.mean(),
        'feature_std': features_matched.std(),
        'norm_mean': norms.mean(),
        'norm_std': norms.std(),
        'conf_mean': confs_matched.mean(),
        'conf_std': confs_matched.std(),
        'iou_mean': ious.mean(),
        'iou_std': ious.std(),
        'conformity_mean': conformity_scores.mean(),
        'conformity_std': conformity_scores.std(),
        'corr_conf_iou': corr_conf_iou,
    }


def plot_distributions(seq_stats_list, output_dir):
    """Plot key distributions across sequences."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Cache Analysis: Feature & IoU Distributions',
                 fontsize=16, fontweight='bold')

    seq_names = [s['seq_name'].replace('MOT17-', '') for s in seq_stats_list]

    # Plot 1: Feature norms
    ax = axes[0, 0]
    norms = [s['norm_mean'] for s in seq_stats_list]
    colors = ['red' if 'MOT17-05' in s['seq_name'] else 'steelblue'
              for s in seq_stats_list]
    ax.bar(range(len(seq_names)), norms, color=colors, alpha=0.7)
    ax.set_xticks(range(len(seq_names)))
    ax.set_xticklabels(seq_names, rotation=45)
    ax.set_ylabel('Mean Feature Norm (Layer 21)')
    ax.set_title('Feature Magnitudes')
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: IoU distributions
    ax = axes[0, 1]
    ious = [s['iou_mean'] for s in seq_stats_list]
    ax.bar(range(len(seq_names)), ious, color=colors, alpha=0.7)
    ax.set_xticks(range(len(seq_names)))
    ax.set_xticklabels(seq_names, rotation=45)
    ax.set_ylabel('Mean IoU')
    ax.set_title('Detection Quality (IoU)')
    ax.axhline(y=0.5, color='red', linestyle='--', label='Match threshold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: Conformity scores
    ax = axes[0, 2]
    conformity = [s['conformity_mean'] for s in seq_stats_list]
    ax.bar(range(len(seq_names)), conformity, color=colors, alpha=0.7)
    ax.set_xticks(range(len(seq_names)))
    ax.set_xticklabels(seq_names, rotation=45)
    ax.set_ylabel('Mean Conformity Score (1-IoU)')
    ax.set_title('Non-Conformity Scores')
    ax.grid(axis='y', alpha=0.3)

    # Plot 4: Confidence
    ax = axes[1, 0]
    confs = [s['conf_mean'] for s in seq_stats_list]
    ax.bar(range(len(seq_names)), confs, color=colors, alpha=0.7)
    ax.set_xticks(range(len(seq_names)))
    ax.set_xticklabels(seq_names, rotation=45)
    ax.set_ylabel('Mean Confidence')
    ax.set_title('YOLO Confidence')
    ax.grid(axis='y', alpha=0.3)

    # Plot 5: Correlation (Conf vs IoU)
    ax = axes[1, 1]
    corrs = [s['corr_conf_iou'] for s in seq_stats_list]
    ax.bar(range(len(seq_names)), corrs, color=colors, alpha=0.7)
    ax.set_xticks(range(len(seq_names)))
    ax.set_xticklabels(seq_names, rotation=45)
    ax.set_ylabel('Correlation(Conf, IoU)')
    ax.set_title('Confidence Predicts IoU?')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)

    # Plot 6: Sample counts
    ax = axes[1, 2]
    counts = [s['n_matched'] for s in seq_stats_list]
    ax.bar(range(len(seq_names)), counts, color=colors, alpha=0.7)
    ax.set_xticks(range(len(seq_names)))
    ax.set_xticklabels(seq_names, rotation=45)
    ax.set_ylabel('Number of Matched Detections')
    ax.set_title('Sample Counts (True Positives)')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save
    plt.savefig(output_dir / 'cache_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'cache_analysis.pdf', bbox_inches='tight')
    print(f"\nSaved plots to {output_dir}")
    plt.close()


def main():
    """Main analysis."""

    print("="*80)
    print("YOLO CACHE ANALYSIS - BEFORE IMPLEMENTATION")
    print("="*80)
    print("\nThis analysis helps us understand:")
    print("  1. Feature ranges and distributions (Layer 21)")
    print("  2. IoU distributions (detection quality)")
    print("  3. YOLO performance (TP, FP counts)")
    print("  4. Confidence vs IoU relationship")
    print("\n" + "="*80)

    cache_dir = Path('/ssd_4TB/divake/temporal_uncertainty/yolo_cache/data/mot17/yolov8n')

    sequences = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09',
                 'MOT17-10', 'MOT17-11', 'MOT17-13']

    seq_stats_list = []
    for seq in sequences:
        stats = analyze_single_sequence(seq, cache_dir)
        seq_stats_list.append(stats)

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print('='*80)
    print(f"\n{'Seq':<10} {'Matched':<8} {'FeatNorm':<12} {'IoU':<10} {'Conformity':<12} {'Conf':<10} {'Corr(C,I)':<10}")
    print('-'*80)
    for s in seq_stats_list:
        seq_short = s['seq_name'].replace('MOT17-', '')
        print(f"{seq_short:<10} "
              f"{s['n_matched']:<8} "
              f"{s['norm_mean']:>5.2f}±{s['norm_std']:<4.2f} "
              f"{s['iou_mean']:<10.4f} "
              f"{s['conformity_mean']:>5.3f}±{s['conformity_std']:<4.3f} "
              f"{s['conf_mean']:<10.4f} "
              f"{s['corr_conf_iou']:<10.4f}")

    # Plot
    output_dir = Path('results/cache_analysis')
    plot_distributions(seq_stats_list, output_dir)

    print(f"\n{'='*80}")
    print("KEY INSIGHTS")
    print('='*80)

    # Find outliers
    norms = [s['norm_mean'] for s in seq_stats_list]
    ious = [s['iou_mean'] for s in seq_stats_list]

    min_norm_idx = np.argmin(norms)
    max_norm_idx = np.argmax(norms)
    min_iou_idx = np.argmin(ious)
    max_iou_idx = np.argmax(ious)

    print(f"\n1. Feature Norms (Layer 21):")
    print(f"   Lowest: {seq_stats_list[min_norm_idx]['seq_name']} ({norms[min_norm_idx]:.2f})")
    print(f"   Highest: {seq_stats_list[max_norm_idx]['seq_name']} ({norms[max_norm_idx]:.2f})")
    print(f"   Range: {max(norms)/min(norms):.2f}x difference")

    print(f"\n2. Detection Quality (IoU):")
    print(f"   Worst: {seq_stats_list[min_iou_idx]['seq_name']} (IoU={ious[min_iou_idx]:.4f})")
    print(f"   Best: {seq_stats_list[max_iou_idx]['seq_name']} (IoU={ious[max_iou_idx]:.4f})")

    print(f"\n3. Confidence vs IoU:")
    corrs = [s['corr_conf_iou'] for s in seq_stats_list]
    mean_corr = np.mean(corrs)
    print(f"   Mean correlation: {mean_corr:.4f}")
    if abs(mean_corr) < 0.3:
        print(f"   → YOLO confidence is WEAK predictor of IoU!")

    print(f"\n{'='*80}")
    print("Analysis complete! Review results before implementing core code.")
    print('='*80)


if __name__ == '__main__':
    main()
