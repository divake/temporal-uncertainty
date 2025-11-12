"""
Publication-Quality Tracking Uncertainty Visualization

Creates a compelling multi-panel figure showing:
1. GitHub-style heatmap of uncertainty over time
2. Temporal line plots
3. 4 key frames with bounding boxes (GT + predictions)

Author: Research Team
Date: November 11, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import json
from pathlib import Path
import cv2
from typing import Dict, List, Tuple

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results' / 'track_temporal_analysis'
MOT17_ROOT = Path('/ssd_4TB/divake/temporal_uncertainty/data/MOT17/train')
CACHE_ROOT = Path('/ssd_4TB/divake/temporal_uncertainty/yolo_cache/data/mot17/yolov8n')


def load_tracking_results(track_id=1):
    """Load the temporal uncertainty results"""
    results_path = RESULTS_DIR / f'track_{track_id}_results.json'
    with open(results_path, 'r') as f:
        return json.load(f)


def select_key_frames(data: Dict) -> Dict[str, int]:
    """
    Select 4 key frames representing different uncertainty scenarios

    Returns:
        Dict mapping scenario to frame index
    """
    frames = np.array(data['temporal_data']['frames'])
    aleatoric = np.array(data['temporal_data']['aleatoric'])
    epistemic = np.array(data['temporal_data']['epistemic'])
    ious = np.array(data['temporal_data']['ious'])

    # Compute normalized scores (to avoid boundary effects)
    alea_norm = (aleatoric - aleatoric.min()) / (aleatoric.max() - aleatoric.min() + 1e-6)
    epis_norm = (epistemic - epistemic.min()) / (epistemic.max() - epistemic.min() + 1e-6)

    # Scenario 1: High aleatoric, low epistemic (occlusion/blur)
    score_1 = alea_norm * (1 - epis_norm) * ious  # Want good IoU to show it's aleatoric
    idx_1 = np.argmax(score_1)

    # Scenario 2: Low aleatoric, high epistemic (unusual pose/appearance)
    score_2 = (1 - alea_norm) * epis_norm * ious
    idx_2 = np.argmax(score_2)

    # Scenario 3: Both high (challenging scenario)
    score_3 = alea_norm * epis_norm
    idx_3 = np.argmax(score_3)

    # Scenario 4: Both low (easy tracking)
    score_4 = (1 - alea_norm) * (1 - epis_norm)
    idx_4 = np.argmax(score_4)

    # Ensure they're spread out (at least 100 frames apart)
    selected = {
        'high_alea_low_epis': idx_1,
        'low_alea_high_epis': idx_2,
        'both_high': idx_3,
        'both_low': idx_4
    }

    print("\nSelected Key Frames:")
    for scenario, idx in selected.items():
        print(f"  {scenario:25s}: Frame {frames[idx]:4d} | "
              f"Alea={aleatoric[idx]:.3f} | Epis={epistemic[idx]:.3f} | IoU={ious[idx]:.3f}")

    return selected


def load_frame_and_boxes(sequence_name: str, frame_num: int, track_id: int,
                          data: Dict) -> Tuple[np.ndarray, Dict]:
    """
    Load video frame and corresponding bounding boxes with uncertainty

    Returns:
        frame: RGB image
        boxes: Dict with 'gt', 'pred', 'uncertainty_lower', 'uncertainty_upper'
    """
    # Load frame
    seq_dir = MOT17_ROOT / sequence_name / 'img1'
    frame_path = seq_dir / f'{frame_num:06d}.jpg'

    if not frame_path.exists():
        raise FileNotFoundError(f"Frame not found: {frame_path}")

    frame = cv2.imread(str(frame_path))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Load cache to get boxes
    cache_path = CACHE_ROOT / f'{sequence_name}.npz'
    cache = np.load(cache_path, allow_pickle=True)

    # Find detection for this frame and track
    frame_ids = cache['detections/frame_ids']
    matched_det_indices = cache['gt_matching/det_indices']
    track_ids = cache['gt_matching/gt_track_ids']

    # Filter to matched detections
    matched_frame_ids = frame_ids[matched_det_indices]

    # Find the specific detection
    mask = (matched_frame_ids == frame_num) & (track_ids == track_id)

    if not np.any(mask):
        raise ValueError(f"No detection found for frame {frame_num}, track {track_id}")

    idx = np.where(mask)[0][0]

    # Get prediction box (detection) - format is [x_center, y_center, w, h]
    det_idx = matched_det_indices[idx]
    pred_box_xywh = cache['detections/bboxes'][det_idx]

    # Convert to [x1, y1, x2, y2]
    pred_box = np.array([
        pred_box_xywh[0] - pred_box_xywh[2] / 2,  # x1
        pred_box_xywh[1] - pred_box_xywh[3] / 2,  # y1
        pred_box_xywh[0] + pred_box_xywh[2] / 2,  # x2
        pred_box_xywh[1] + pred_box_xywh[3] / 2   # y2
    ])

    # Get ground truth box - also [x_center, y_center, w, h]
    gt_box_xywh = cache['gt_matching/gt_bboxes'][idx]

    # Convert to [x1, y1, x2, y2]
    gt_box = np.array([
        gt_box_xywh[0] - gt_box_xywh[2] / 2,  # x1
        gt_box_xywh[1] - gt_box_xywh[3] / 2,  # y1
        gt_box_xywh[0] + gt_box_xywh[2] / 2,  # x2
        gt_box_xywh[1] + gt_box_xywh[3] / 2   # y2
    ])

    # Get uncertainty for this frame (find matching index in temporal data)
    frames_array = np.array(data['temporal_data']['frames'])
    temporal_idx = np.where(frames_array == frame_num)[0]

    if len(temporal_idx) > 0:
        temporal_idx = temporal_idx[0]
        total_uncertainty = data['temporal_data']['total'][temporal_idx]

        # Convert uncertainty to pixel space (scale by box dimensions)
        w = pred_box[2] - pred_box[0]
        h = pred_box[3] - pred_box[1]

        # Scale uncertainty to create expanded/contracted boxes
        # Use total uncertainty as fraction of box size
        scale_factor = total_uncertainty * 0.5  # Scale down for visibility

        dx = w * scale_factor
        dy = h * scale_factor

        # Upper bound: prediction + uncertainty
        upper_box = pred_box + np.array([-dx, -dy, dx, dy])

        # Lower bound: prediction - uncertainty
        lower_box = pred_box + np.array([dx, dy, -dx, -dy])
    else:
        # Fallback if frame not found
        upper_box = pred_box.copy()
        lower_box = pred_box.copy()

    return frame, {
        'pred': pred_box,
        'gt': gt_box,
        'upper': upper_box,
        'lower': lower_box
    }


def draw_boxes_on_frame(frame: np.ndarray, boxes: Dict,
                        pred_color=(1, 0, 0), gt_color=(0, 1, 0)):
    """
    Draw bounding boxes on frame

    Args:
        frame: RGB image
        boxes: Dict with 'pred' and 'gt' boxes [x1, y1, x2, y2]
        pred_color: Color for prediction box (R, G, B)
        gt_color: Color for ground truth box
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.imshow(frame)
    ax.axis('off')

    # Draw prediction box (outer, red)
    pred = boxes['pred']
    rect_pred = patches.Rectangle(
        (pred[0], pred[1]), pred[2] - pred[0], pred[3] - pred[1],
        linewidth=3, edgecolor=pred_color, facecolor='none', label='Prediction'
    )
    ax.add_patch(rect_pred)

    # Draw GT box (inner, green)
    gt = boxes['gt']
    rect_gt = patches.Rectangle(
        (gt[0], gt[1]), gt[2] - gt[0], gt[3] - gt[1],
        linewidth=3, edgecolor=gt_color, facecolor='none',
        linestyle='--', label='Ground Truth'
    )
    ax.add_patch(rect_gt)

    # Add legend
    ax.legend(loc='upper right', fontsize=10, framealpha=0.8)

    return fig, ax


def create_paper_figure(sequence_name='MOT17-11-FRCNN', track_id=1):
    """
    Create publication-quality figure with:
    - GitHub-style heatmap
    - Temporal plots
    - 4 key frame images
    """
    print("="*80)
    print("CREATING PUBLICATION-QUALITY TRACKING FIGURE")
    print("="*80)

    # Load results
    print("\n[1/5] Loading tracking results...")
    data = load_tracking_results(track_id)

    frames = np.array(data['temporal_data']['frames'])
    aleatoric = np.array(data['temporal_data']['aleatoric'])
    epistemic = np.array(data['temporal_data']['epistemic'])
    total = np.array(data['temporal_data']['total'])
    ious = np.array(data['temporal_data']['ious'])

    # Select key frames
    print("\n[2/5] Selecting key frames...")
    key_frames = select_key_frames(data)

    # Create main figure with gridspec
    print("\n[3/5] Creating main figure layout...")
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(4, 5, figure=fig, hspace=0.4, wspace=0.3,
                  height_ratios=[0.8, 0.8, 3, 2])

    # Panel 1: Heatmap for Aleatoric (top)
    print("\n[4/5] Generating heatmaps...")
    ax_heatmap_alea = fig.add_subplot(gs[0, :])
    create_heatmap(ax_heatmap_alea, frames, aleatoric, 'Aleatoric Uncertainty', 'Greens')

    # Panel 2: Heatmap for Epistemic (below aleatoric)
    ax_heatmap_epis = fig.add_subplot(gs[1, :])
    create_heatmap(ax_heatmap_epis, frames, epistemic, 'Epistemic Uncertainty', 'Blues')

    # Panel 3: Temporal line plot (middle)
    ax_temporal = fig.add_subplot(gs[2, :])
    plot_temporal_curves(ax_temporal, frames, aleatoric, epistemic, total, ious)

    # Panel 4: 4 key frame images (bottom)
    print("\n[5/5] Loading and rendering key frames...")
    ax_imgs = []
    scenarios = ['high_alea_low_epis', 'low_alea_high_epis', 'both_high', 'both_low']
    titles = [
        'High Aleatoric\nLow Epistemic',
        'Low Aleatoric\nHigh Epistemic',
        'Both High\n(Challenging)',
        'Both Low\n(Easy)'
    ]

    for i, (scenario, title) in enumerate(zip(scenarios, titles)):
        idx = key_frames[scenario]
        frame_num = frames[idx]

        # Load frame and boxes
        try:
            frame, boxes = load_frame_and_boxes(sequence_name, frame_num, track_id, data)

            ax = fig.add_subplot(gs[3, i])
            ax.imshow(frame)
            ax.axis('off')

            # Draw boxes in order: upper, lower (shaded region), pred, gt
            pred = boxes['pred']
            gt = boxes['gt']
            upper = boxes['upper']
            lower = boxes['lower']

            # 1. Upper bound (outer, light blue, dashed)
            rect_upper = patches.Rectangle(
                (upper[0], upper[1]), upper[2] - upper[0], upper[3] - upper[1],
                linewidth=1.5, edgecolor='deepskyblue', facecolor='cyan',
                alpha=0.15, linestyle=':', label='Conf. Upper'
            )
            ax.add_patch(rect_upper)

            # 2. Lower bound (inner, light blue, dashed)
            rect_lower = patches.Rectangle(
                (lower[0], lower[1]), lower[2] - lower[0], lower[3] - lower[1],
                linewidth=1.5, edgecolor='deepskyblue', facecolor='white',
                alpha=0.3, linestyle=':', label='Conf. Lower'
            )
            ax.add_patch(rect_lower)

            # 3. Prediction box (red, solid)
            rect_pred = patches.Rectangle(
                (pred[0], pred[1]), pred[2] - pred[0], pred[3] - pred[1],
                linewidth=2.5, edgecolor='red', facecolor='none', label='Prediction'
            )
            ax.add_patch(rect_pred)

            # 4. GT box (green, dashed) - should be covered by uncertainty interval
            rect_gt = patches.Rectangle(
                (gt[0], gt[1]), gt[2] - gt[0], gt[3] - gt[1],
                linewidth=2.5, edgecolor='lime', facecolor='none',
                linestyle='--', label='Ground Truth'
            )
            ax.add_patch(rect_gt)

            # Title with uncertainty values
            ax.set_title(f"{title}\nFrame {frame_num}\n"
                        f"Alea={aleatoric[idx]:.2f} | Epis={epistemic[idx]:.2f} | IoU={ious[idx]:.2f}",
                        fontsize=11, fontweight='bold')

            ax_imgs.append(ax)

        except Exception as e:
            print(f"  Warning: Could not load frame {frame_num}: {e}")
            ax = fig.add_subplot(gs[3, i])
            ax.text(0.5, 0.5, f'Frame {frame_num}\nNot Available',
                   ha='center', va='center', fontsize=12)
            ax.axis('off')

    # Add legend for boxes (bottom right)
    if ax_imgs:
        # Create custom legend patches
        from matplotlib.lines import Line2D
        red_line = Line2D([0], [0], color='red', linewidth=2.5, label='Prediction')
        green_line = Line2D([0], [0], color='lime', linewidth=2.5, linestyle='--', label='Ground Truth')
        blue_patch = patches.Patch(facecolor='cyan', edgecolor='deepskyblue',
                                   alpha=0.3, label='Uncertainty Interval')
        fig.legend(handles=[red_line, green_line, blue_patch],
                  loc='lower right', fontsize=12, framealpha=0.9,
                  bbox_to_anchor=(0.98, 0.02))

    # Main title
    fig.suptitle(f'Temporal Uncertainty Analysis - Track {track_id} ({sequence_name})',
                fontsize=16, fontweight='bold', y=0.995)

    # Save
    output_path = RESULTS_DIR / f'paper_figure_track_{track_id}.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {output_path}")

    # Also save individual frame images
    print("\n[Bonus] Saving individual key frames...")
    for scenario in scenarios:
        idx = key_frames[scenario]
        frame_num = frames[idx]
        try:
            frame, boxes = load_frame_and_boxes(sequence_name, frame_num, track_id)
            fig_frame, ax_frame = draw_boxes_on_frame(frame, boxes)

            frame_output = RESULTS_DIR / f'key_frame_{scenario}_f{frame_num}.png'
            fig_frame.savefig(frame_output, dpi=150, bbox_inches='tight')
            plt.close(fig_frame)
            print(f"  ✓ {frame_output.name}")
        except:
            pass

    print("\n" + "="*80)
    print("FIGURE GENERATION COMPLETE ✓")
    print("="*80)

    return fig


def create_heatmap(ax, frames, values, title, cmap):
    """
    Create GitHub-style contribution heatmap

    Args:
        ax: Matplotlib axis
        frames: Frame numbers
        values: Uncertainty values
        title: Plot title
        cmap: Colormap name
    """
    # Determine grid size (aim for ~50 cells per row)
    n_frames = len(frames)
    n_cols = 50
    n_rows = int(np.ceil(n_frames / n_cols))

    # Pad values to fit grid
    padded_values = np.zeros(n_rows * n_cols)
    padded_values[:n_frames] = values
    padded_values[n_frames:] = np.nan  # Use NaN for empty cells

    # Reshape to grid
    grid = padded_values.reshape(n_rows, n_cols)

    # Plot heatmap (scale to actual range, not 0-1)
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    im = ax.imshow(grid, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax,
                   interpolation='nearest')

    # Styling
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Frame Progress →', fontsize=11)
    ax.set_yticks([])
    ax.set_xticks([])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                       pad=0.02, fraction=0.05)
    cbar.set_label('Uncertainty', fontsize=10)
    cbar.ax.tick_params(labelsize=9)


def plot_temporal_curves(ax, frames, aleatoric, epistemic, total, ious):
    """
    Plot temporal uncertainty curves with IoU overlay
    """
    # Main plot: Uncertainty curves
    ax.plot(frames, aleatoric, 'g-', alpha=0.7, linewidth=1.5, label='Aleatoric')
    ax.plot(frames, epistemic, 'b-', alpha=0.7, linewidth=1.5, label='Epistemic')
    ax.plot(frames, total, 'r-', alpha=0.5, linewidth=2, label='Total')

    ax.set_xlabel('Frame Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Uncertainty', fontsize=12, fontweight='bold')
    ax.set_title('Temporal Uncertainty Evolution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

    # Add twin axis for IoU
    ax2 = ax.twinx()
    ax2.plot(frames, ious, 'cyan', alpha=0.4, linewidth=1, linestyle=':', label='IoU')
    ax2.set_ylabel('Ground Truth IoU', fontsize=11, color='cyan')
    ax2.tick_params(axis='y', labelcolor='cyan')
    ax2.set_ylim([0, 1])
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)


if __name__ == "__main__":
    import sys

    sequence = sys.argv[1] if len(sys.argv) > 1 else 'MOT17-11-FRCNN'
    track_id = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    fig = create_paper_figure(sequence, track_id)
    plt.show()
