#!/usr/bin/env python3
"""
MOT17 Dataset Deep Analysis Script

Purpose: Extract comprehensive metadata from all 7 MOT17 training sequences
         to inform tracking and uncertainty analysis decisions.

Author: Dataset Analysis (2025-10-29)
Usage: python analyze_mot17_dataset.py

Output:
    - Per-sequence JSON/PKL files with detailed statistics
    - Cross-sequence summary JSON
    - Hero tracks recommendations JSON
    - Visualization plots for all sequences

Runtime: ~5-10 minutes for all 7 sequences
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import cv2

# Base paths
BASE_PATH = "/ssd_4TB/divake/temporal_uncertainty"
MOT17_TRAIN_PATH = f"{BASE_PATH}/data/MOT17/train"
OUTPUT_PATH = f"{BASE_PATH}/metadata"

# Analysis parameters
MIN_TRACK_LENGTH = 30  # Minimum frames for temporal analysis
VISIBILITY_THRESHOLD = 0.3  # Below this = considered occluded
SPARSE_THRESHOLD = 5  # <5 tracks = sparse
CROWDED_THRESHOLD = 15  # >15 tracks = crowded
MOTION_THRESHOLD = 20.0  # Displacement threshold for "high motion"

# All sequences to analyze
SEQUENCES = [
    "MOT17-02-FRCNN",
    "MOT17-04-FRCNN",
    "MOT17-05-FRCNN",
    "MOT17-09-FRCNN",
    "MOT17-10-FRCNN",
    "MOT17-11-FRCNN",
    "MOT17-13-FRCNN"
]


def load_ground_truth(sequence_name: str) -> pd.DataFrame:
    """
    Load MOT17 ground truth file.

    GT Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>

    Args:
        sequence_name: Name of sequence (e.g., "MOT17-11-FRCNN")

    Returns:
        DataFrame with columns: frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility

    Raises:
        FileNotFoundError: If GT file doesn't exist
    """
    gt_path = f"{MOT17_TRAIN_PATH}/{sequence_name}/gt/gt.txt"

    if not os.path.exists(gt_path):
        raise FileNotFoundError(
            f"Ground truth file not found at {gt_path}\n"
            f"Expected sequence structure: {MOT17_TRAIN_PATH}/{sequence_name}/gt/gt.txt"
        )

    # Load GT file
    df = pd.read_csv(
        gt_path,
        header=None,
        names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'class', 'visibility']
    )

    # Filter to only considered detections (conf=1)
    df = df[df['conf'] == 1].copy()

    # Add computed fields
    df['bb_right'] = df['bb_left'] + df['bb_width']
    df['bb_bottom'] = df['bb_top'] + df['bb_height']
    df['bb_center_x'] = df['bb_left'] + df['bb_width'] / 2
    df['bb_center_y'] = df['bb_top'] + df['bb_height'] / 2
    df['bb_area'] = df['bb_width'] * df['bb_height']

    print(f"[INFO] Loaded {len(df)} detections from {sequence_name}")

    return df


def get_sequence_info(sequence_name: str) -> Dict:
    """Get basic sequence information."""
    img_dir = f"{MOT17_TRAIN_PATH}/{sequence_name}/img1"

    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    # Count frames
    frames = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    num_frames = len(frames)

    # Get resolution from first frame
    first_frame_path = f"{img_dir}/{frames[0]}"
    img = cv2.imread(first_frame_path)
    if img is None:
        raise RuntimeError(f"Could not read image: {first_frame_path}")

    height, width = img.shape[:2]

    return {
        'name': sequence_name,
        'total_frames': num_frames,
        'fps': 30,  # MOT17 standard
        'resolution': [width, height],
        'path': f"{MOT17_TRAIN_PATH}/{sequence_name}"
    }


def detect_entry_exit_point(track_data: pd.DataFrame, frame_width: int, frame_height: int) -> Tuple[str, str]:
    """
    Detect where track enters and exits frame.

    Args:
        track_data: DataFrame for single track
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels

    Returns:
        Tuple of (entry_point, exit_point) where each is one of:
        'left', 'right', 'top', 'bottom', 'center'
    """
    # First and last detection positions
    first_det = track_data.iloc[0]
    last_det = track_data.iloc[-1]

    # Margin for considering "edge" (10% of frame dimension)
    margin_x = frame_width * 0.1
    margin_y = frame_height * 0.1

    def classify_position(center_x, center_y):
        if center_x < margin_x:
            return 'left'
        elif center_x > frame_width - margin_x:
            return 'right'
        elif center_y < margin_y:
            return 'top'
        elif center_y > frame_height - margin_y:
            return 'bottom'
        else:
            return 'center'

    entry_point = classify_position(first_det['bb_center_x'], first_det['bb_center_y'])
    exit_point = classify_position(last_det['bb_center_x'], last_det['bb_center_y'])

    return entry_point, exit_point


def detect_occlusion_events(track_data: pd.DataFrame, visibility_threshold: float = VISIBILITY_THRESHOLD) -> List[Dict]:
    """
    Detect occlusion events in a track.

    An occlusion event is a contiguous sequence of frames where visibility < threshold.

    Args:
        track_data: DataFrame for single track (sorted by frame)
        visibility_threshold: Visibility below this = occluded

    Returns:
        List of dicts with keys: start_frame, end_frame, duration, min_visibility
    """
    occluded = track_data['visibility'] < visibility_threshold
    occlusion_events = []

    if not occluded.any():
        return []

    # Find contiguous occlusion regions
    in_occlusion = False
    start_frame = None

    for idx, row in track_data.iterrows():
        if row['visibility'] < visibility_threshold:
            if not in_occlusion:
                # Start of occlusion
                in_occlusion = True
                start_frame = row['frame']
                min_vis = row['visibility']
            else:
                # Continue occlusion, update min visibility
                min_vis = min(min_vis, row['visibility'])
        else:
            if in_occlusion:
                # End of occlusion
                end_frame = track_data[track_data['frame'] < row['frame']].iloc[-1]['frame']
                occlusion_events.append({
                    'start_frame': int(start_frame),
                    'end_frame': int(end_frame),
                    'duration': int(end_frame - start_frame + 1),
                    'min_visibility': float(min_vis)
                })
                in_occlusion = False

    # Handle occlusion at end of track
    if in_occlusion:
        end_frame = track_data.iloc[-1]['frame']
        occlusion_events.append({
            'start_frame': int(start_frame),
            'end_frame': int(end_frame),
            'duration': int(end_frame - start_frame + 1),
            'min_visibility': float(min_vis)
        })

    return occlusion_events


def analyze_track(track_id: int, track_data: pd.DataFrame, frame_width: int, frame_height: int) -> Dict:
    """
    Comprehensive analysis of a single track.

    Args:
        track_id: Track ID
        track_data: DataFrame for this track (sorted by frame)
        frame_width: Frame width
        frame_height: Frame height

    Returns:
        Dict with all track statistics
    """
    # Sort by frame
    track_data = track_data.sort_values('frame')

    # Basic statistics
    start_frame = int(track_data['frame'].min())
    end_frame = int(track_data['frame'].max())
    duration = end_frame - start_frame + 1
    num_detections = len(track_data)

    # Visibility statistics
    avg_visibility = float(track_data['visibility'].mean())
    min_visibility = float(track_data['visibility'].min())

    # Occlusion analysis
    occlusion_events = detect_occlusion_events(track_data)
    has_occlusion = len(occlusion_events) > 0

    # Entry/exit points
    entry_point, exit_point = detect_entry_exit_point(track_data, frame_width, frame_height)

    # Bounding box statistics
    avg_bbox_area = float(track_data['bb_area'].mean())
    bbox_variance = float(track_data['bb_area'].var())

    # Motion analysis
    displacements = []
    for i in range(len(track_data) - 1):
        curr = track_data.iloc[i]
        next_det = track_data.iloc[i + 1]

        dx = next_det['bb_center_x'] - curr['bb_center_x']
        dy = next_det['bb_center_y'] - curr['bb_center_y']
        displacement = np.sqrt(dx**2 + dy**2)
        displacements.append(displacement)

    avg_displacement = float(np.mean(displacements)) if displacements else 0.0
    max_displacement = float(np.max(displacements)) if displacements else 0.0
    trajectory_length = float(np.sum(displacements)) if displacements else 0.0

    # Classify motion
    if avg_displacement < 5.0:
        motion_category = "static"
    elif avg_displacement < 15.0:
        motion_category = "moving"
    else:
        motion_category = "fast_moving"

    # Object class
    obj_class = "pedestrian"  # MOT17 is pedestrian-only, but keep for extensibility

    return {
        'lifetime_frames': [start_frame, end_frame],
        'duration': duration,
        'num_detections': num_detections,
        'avg_visibility': avg_visibility,
        'min_visibility': min_visibility,
        'has_occlusion': has_occlusion,
        'occlusion_events': occlusion_events,
        'entry_point': entry_point,
        'exit_point': exit_point,
        'avg_bbox_area': avg_bbox_area,
        'bbox_variance': bbox_variance,
        'avg_displacement': avg_displacement,
        'max_displacement': max_displacement,
        'motion_category': motion_category,
        'trajectory_length': trajectory_length,
        'class': obj_class
    }


def analyze_frame(frame_num: int, frame_data: pd.DataFrame) -> Dict:
    """
    Analyze a single frame.

    Args:
        frame_num: Frame number
        frame_data: DataFrame of all detections in this frame

    Returns:
        Dict with frame statistics
    """
    num_tracks = len(frame_data)
    num_occluded = len(frame_data[frame_data['visibility'] < VISIBILITY_THRESHOLD])

    # Density classification
    if num_tracks < SPARSE_THRESHOLD:
        density = "sparse"
    elif num_tracks < CROWDED_THRESHOLD:
        density = "medium"
    else:
        density = "crowded"

    avg_visibility = float(frame_data['visibility'].mean()) if num_tracks > 0 else 1.0
    avg_bbox_area = float(frame_data['bb_area'].mean()) if num_tracks > 0 else 0.0

    return {
        'num_tracks': int(num_tracks),
        'num_occluded': int(num_occluded),
        'density': density,
        'avg_visibility': avg_visibility,
        'avg_bbox_area': avg_bbox_area
    }


def select_hero_tracks(tracks_metadata: Dict, min_length: int = MIN_TRACK_LENGTH) -> Dict:
    """
    Select "hero tracks" for detailed uncertainty analysis.

    Categories:
        - long_stable: Longest tracks with high avg visibility (baseline)
        - occlusion_heavy: Tracks with multiple occlusions (key analysis)
        - high_motion: Tracks with high displacement (motion blur)
        - edge_cases: Other interesting patterns

    Args:
        tracks_metadata: Dict of all track metadata
        min_length: Only consider tracks >= this length

    Returns:
        Dict with keys: long_stable, occlusion_heavy, high_motion, edge_cases
        Values are lists of track IDs
    """
    # Filter to minimum length
    eligible_tracks = {
        tid: meta for tid, meta in tracks_metadata.items()
        if meta['duration'] >= min_length
    }

    if not eligible_tracks:
        return {
            'long_stable': [],
            'occlusion_heavy': [],
            'high_motion': [],
            'edge_cases': []
        }

    # Long stable: Top 5 by duration with high visibility
    long_stable = sorted(
        eligible_tracks.items(),
        key=lambda x: (x[1]['duration'], x[1]['avg_visibility']),
        reverse=True
    )[:5]
    long_stable_ids = [int(tid) for tid, _ in long_stable]

    # Occlusion heavy: Top 5 by number of occlusion events
    occlusion_heavy = sorted(
        eligible_tracks.items(),
        key=lambda x: len(x[1]['occlusion_events']),
        reverse=True
    )[:5]
    # Only keep if they actually have occlusions
    occlusion_heavy_ids = [int(tid) for tid, meta in occlusion_heavy if meta['has_occlusion']]

    # High motion: Top 5 by max displacement
    high_motion = sorted(
        eligible_tracks.items(),
        key=lambda x: x[1]['max_displacement'],
        reverse=True
    )[:5]
    high_motion_ids = [int(tid) for tid, _ in high_motion]

    # Edge cases: Tracks with unusual patterns (e.g., very high bbox variance)
    edge_cases = sorted(
        eligible_tracks.items(),
        key=lambda x: x[1]['bbox_variance'],
        reverse=True
    )[:3]
    edge_cases_ids = [int(tid) for tid, _ in edge_cases]

    return {
        'long_stable': long_stable_ids,
        'occlusion_heavy': occlusion_heavy_ids,
        'high_motion': high_motion_ids,
        'edge_cases': edge_cases_ids
    }


def generate_tta_recommendations(tracks_metadata: Dict, frame_analysis: Dict, total_frames: int) -> Dict:
    """
    Generate recommendations for TTA experiments.

    Identifies:
        - Clean segments: Good for baseline/control
        - Occlusion segments: High uncertainty expected
        - Crowded segments: Complex interactions
        - Recommended test frames: Representative samples

    Args:
        tracks_metadata: All track metadata
        frame_analysis: Per-frame analysis
        total_frames: Total frames in sequence

    Returns:
        Dict with segment recommendations
    """
    # Find clean segments (sparse, high visibility, low occlusions)
    clean_frames = [
        f for f, meta in frame_analysis.items()
        if meta['density'] == 'sparse' and meta['avg_visibility'] > 0.9
    ]
    clean_segments = find_contiguous_segments(clean_frames, min_length=50)

    # Find occlusion segments
    occlusion_frames = [
        f for f, meta in frame_analysis.items()
        if meta['num_occluded'] > 2
    ]
    occlusion_segments = find_contiguous_segments(occlusion_frames, min_length=20)

    # Find crowded segments
    crowded_frames = [
        f for f, meta in frame_analysis.items()
        if meta['density'] == 'crowded'
    ]
    crowded_segments = find_contiguous_segments(crowded_frames, min_length=50)

    # Recommended test frames (evenly spaced + interesting events)
    test_frames = []
    step = max(1, total_frames // 10)
    for i in range(0, total_frames, step):
        test_frames.append(i + 1)  # 1-indexed

    # Add frames with interesting events
    if occlusion_frames:
        test_frames.append(occlusion_frames[len(occlusion_frames) // 2])
    if crowded_frames:
        test_frames.append(crowded_frames[len(crowded_frames) // 2])

    test_frames = sorted(list(set(test_frames)))[:10]  # Limit to 10

    return {
        'clean_segments': [
            {'start': seg[0], 'end': seg[1], 'reason': 'Low density, stable tracking'}
            for seg in clean_segments[:3]
        ],
        'occlusion_segments': [
            {'start': seg[0], 'end': seg[1], 'reason': 'Multiple occlusion events'}
            for seg in occlusion_segments[:3]
        ],
        'crowded_segments': [
            {'start': seg[0], 'end': seg[1], 'reason': 'High density, complex interactions'}
            for seg in crowded_segments[:3]
        ],
        'recommended_test_frames': test_frames
    }


def find_contiguous_segments(frames: List[int], min_length: int = 20) -> List[Tuple[int, int]]:
    """
    Find contiguous segments in a list of frame numbers.

    Args:
        frames: List of frame numbers (sorted)
        min_length: Minimum segment length

    Returns:
        List of (start, end) tuples
    """
    if not frames:
        return []

    frames = sorted(frames)
    segments = []
    start = frames[0]
    prev = frames[0]

    for frame in frames[1:]:
        if frame != prev + 1:
            # Gap detected, end current segment
            if prev - start + 1 >= min_length:
                segments.append((start, prev))
            start = frame
        prev = frame

    # Handle last segment
    if prev - start + 1 >= min_length:
        segments.append((start, prev))

    return segments


def analyze_sequence(sequence_name: str) -> Dict:
    """
    Complete analysis of a single sequence.

    Args:
        sequence_name: Name of sequence (e.g., "MOT17-11-FRCNN")

    Returns:
        Dict with all metadata for this sequence
    """
    print(f"\n{'='*60}")
    print(f"Analyzing {sequence_name}")
    print(f"{'='*60}")

    # Load data
    gt_df = load_ground_truth(sequence_name)
    seq_info = get_sequence_info(sequence_name)

    frame_width, frame_height = seq_info['resolution']
    total_frames = seq_info['total_frames']

    # Track-level analysis
    print(f"[INFO] Analyzing {gt_df['id'].nunique()} unique tracks...")
    tracks_metadata = {}
    for track_id in gt_df['id'].unique():
        track_data = gt_df[gt_df['id'] == track_id]
        tracks_metadata[str(track_id)] = analyze_track(track_id, track_data, frame_width, frame_height)

    # Frame-level analysis
    print(f"[INFO] Analyzing {total_frames} frames...")
    frame_analysis = {}
    for frame_num in range(1, total_frames + 1):
        frame_data = gt_df[gt_df['frame'] == frame_num]
        frame_analysis[frame_num] = analyze_frame(frame_num, frame_data)

    # Overall statistics
    track_durations = [meta['duration'] for meta in tracks_metadata.values()]
    tracks_meeting_min = sum(1 for d in track_durations if d >= MIN_TRACK_LENGTH)

    all_occlusion_events = [
        event for meta in tracks_metadata.values()
        for event in meta['occlusion_events']
    ]

    occlusion_durations = [event['duration'] for event in all_occlusion_events]

    # Crowding analysis
    tracks_per_frame = [meta['num_tracks'] for meta in frame_analysis.values()]
    sparse_frames = [f for f, m in frame_analysis.items() if m['density'] == 'sparse']
    medium_frames = [f for f, m in frame_analysis.items() if m['density'] == 'medium']
    crowded_frames = [f for f, m in frame_analysis.items() if m['density'] == 'crowded']

    peak_crowding_frame = max(frame_analysis.items(), key=lambda x: x[1]['num_tracks'])

    # Spatial patterns
    entry_zones = defaultdict(int)
    exit_zones = defaultdict(int)
    for meta in tracks_metadata.values():
        entry_zones[meta['entry_point']] += 1
        exit_zones[meta['exit_point']] += 1

    # Hero tracks selection
    hero_tracks = select_hero_tracks(tracks_metadata, MIN_TRACK_LENGTH)

    # TTA recommendations
    tta_recommendations = generate_tta_recommendations(tracks_metadata, frame_analysis, total_frames)

    # Compile metadata
    metadata = {
        'sequence_info': seq_info,
        'statistics': {
            'unique_track_ids': int(gt_df['id'].nunique()),
            'total_detections': len(gt_df),
            'avg_tracks_per_frame': float(np.mean(tracks_per_frame)),
            'max_simultaneous_tracks': int(max(tracks_per_frame)),
            'min_simultaneous_tracks': int(min(tracks_per_frame)),
            'tracks_meeting_min_length': tracks_meeting_min
        },
        'track_lifetimes': {
            'min': int(min(track_durations)),
            'max': int(max(track_durations)),
            'mean': float(np.mean(track_durations)),
            'median': float(np.median(track_durations)),
            'std': float(np.std(track_durations)),
            'distribution': [
                {'range': '0-30', 'count': sum(1 for d in track_durations if d < 30)},
                {'range': '30-100', 'count': sum(1 for d in track_durations if 30 <= d < 100)},
                {'range': '100-300', 'count': sum(1 for d in track_durations if 100 <= d < 300)},
                {'range': '300+', 'count': sum(1 for d in track_durations if d >= 300)}
            ]
        },
        'tracks': tracks_metadata,
        'frame_analysis': frame_analysis,
        'occlusion_analysis': {
            'total_occlusion_events': len(all_occlusion_events),
            'tracks_with_occlusions': sum(1 for m in tracks_metadata.values() if m['has_occlusion']),
            'avg_occlusion_duration': float(np.mean(occlusion_durations)) if occlusion_durations else 0.0,
            'occlusion_duration_distribution': {
                '1-10': sum(1 for d in occlusion_durations if d <= 10),
                '10-20': sum(1 for d in occlusion_durations if 10 < d <= 20),
                '20-50': sum(1 for d in occlusion_durations if 20 < d <= 50),
                '50+': sum(1 for d in occlusion_durations if d > 50)
            },
            'frames_with_occlusions': [f for f, m in frame_analysis.items() if m['num_occluded'] > 0]
        },
        'crowding_analysis': {
            'sparse_frames': sparse_frames,
            'medium_frames': medium_frames,
            'crowded_frames': crowded_frames,
            'peak_crowding': {
                'frame': peak_crowding_frame[0],
                'num_tracks': peak_crowding_frame[1]['num_tracks']
            }
        },
        'motion_analysis': {
            'static_tracks': [int(tid) for tid, m in tracks_metadata.items() if m['motion_category'] == 'static'],
            'moving_tracks': [int(tid) for tid, m in tracks_metadata.items() if m['motion_category'] == 'moving'],
            'fast_moving_tracks': [int(tid) for tid, m in tracks_metadata.items() if m['motion_category'] == 'fast_moving']
        },
        'spatial_patterns': {
            'entry_zones': dict(entry_zones),
            'exit_zones': dict(exit_zones)
        },
        'hero_tracks': hero_tracks,
        'tta_recommendations': tta_recommendations
    }

    print(f"[DONE] {sequence_name} analysis complete")
    print(f"  - {metadata['statistics']['unique_track_ids']} tracks")
    print(f"  - {metadata['statistics']['tracks_meeting_min_length']} tracks >= {MIN_TRACK_LENGTH} frames")
    print(f"  - {metadata['occlusion_analysis']['total_occlusion_events']} occlusion events")
    print(f"  - Hero tracks: {sum(len(v) for v in hero_tracks.values())} selected")

    return metadata


def save_metadata(metadata: Dict, sequence_name: str):
    """Save metadata as both JSON and pickle."""
    seq_short = sequence_name.split('-')[1]  # Extract "02" from "MOT17-02-FRCNN"

    # Save JSON (human-readable)
    json_path = f"{OUTPUT_PATH}/raw_outputs/seq{seq_short}_metadata.json"
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[SAVED] {json_path}")

    # Save pickle (Python-friendly)
    pkl_path = f"{OUTPUT_PATH}/raw_outputs/seq{seq_short}_metadata.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"[SAVED] {pkl_path}")


def generate_visualizations(metadata: Dict, sequence_name: str):
    """
    Generate all visualization plots for a sequence.

    Plots:
        1. Gantt chart (track lifetimes)
        2. Visibility heatmap
        3. Crowding over time
        4. Track length distribution
        5. Entry/exit map
        6. Occlusion timeline
    """
    seq_short = sequence_name.split('-')[1]
    viz_dir = f"{OUTPUT_PATH}/visualizations"

    print(f"[INFO] Generating visualizations for {sequence_name}...")

    # 1. Gantt Chart
    plot_gantt_chart(metadata, f"{viz_dir}/seq{seq_short}_gantt_chart.png")

    # 2. Visibility Heatmap
    plot_visibility_heatmap(metadata, f"{viz_dir}/seq{seq_short}_visibility_heatmap.png")

    # 3. Crowding Over Time
    plot_crowding(metadata, f"{viz_dir}/seq{seq_short}_crowding.png")

    # 4. Track Length Distribution
    plot_track_length_distribution(metadata, f"{viz_dir}/seq{seq_short}_track_length_dist.png")

    # 5. Entry/Exit Map
    plot_entry_exit_map(metadata, f"{viz_dir}/seq{seq_short}_entry_exit_map.png")

    # 6. Occlusion Timeline
    plot_occlusion_timeline(metadata, f"{viz_dir}/seq{seq_short}_occlusion_timeline.png")

    print(f"[DONE] Visualizations saved to {viz_dir}/")


def plot_gantt_chart(metadata: Dict, save_path: str):
    """Plot Gantt chart of track lifetimes with actual MOT track IDs on Y-axis."""
    tracks = metadata['tracks']
    total_frames = metadata['sequence_info']['total_frames']

    # Sort tracks by start frame
    sorted_tracks = sorted(tracks.items(), key=lambda x: x[1]['lifetime_frames'][0])

    # Limit to top 50 tracks for readability
    if len(sorted_tracks) > 50:
        sorted_tracks = sorted_tracks[:50]

    fig, ax = plt.subplots(figsize=(14, 10))

    # Extract track IDs for Y-axis labels
    track_ids = [track_id for track_id, _ in sorted_tracks]

    for idx, (track_id, track_meta) in enumerate(sorted_tracks):
        start, end = track_meta['lifetime_frames']
        duration = track_meta['duration']

        # Color by duration
        if duration >= 300:
            color = 'green'
        elif duration >= 100:
            color = 'orange'
        elif duration >= 30:
            color = 'blue'
        else:
            color = 'red'

        ax.barh(idx, end - start + 1, left=start, height=0.8, color=color, alpha=0.6)

    ax.set_xlabel('Frame Number', fontsize=12)
    ax.set_ylabel('MOT Track ID (sorted by appearance)', fontsize=12)
    ax.set_title(f'Track Lifetimes - {metadata["sequence_info"]["name"]}', fontsize=14, fontweight='bold')
    ax.set_xlim(0, total_frames)

    # Set Y-axis to show actual track IDs
    ax.set_yticks(range(len(track_ids)))
    ax.set_yticklabels(track_ids, fontsize=8)
    ax.set_ylim(-0.5, len(track_ids) - 0.5)

    ax.grid(axis='x', alpha=0.3)

    # Legend
    legend_elements = [
        mpatches.Patch(color='green', alpha=0.6, label='300+ frames'),
        mpatches.Patch(color='orange', alpha=0.6, label='100-300 frames'),
        mpatches.Patch(color='blue', alpha=0.6, label='30-100 frames'),
        mpatches.Patch(color='red', alpha=0.6, label='<30 frames')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Gantt chart saved")


def plot_visibility_heatmap(metadata: Dict, save_path: str):
    """Plot visibility heatmap for top tracks with actual MOT track IDs on Y-axis."""
    tracks = metadata['tracks']
    total_frames = metadata['sequence_info']['total_frames']

    # Select top 30 longest tracks
    sorted_tracks = sorted(tracks.items(), key=lambda x: x[1]['duration'], reverse=True)[:30]

    # Extract track IDs for Y-axis labels
    track_ids = [track_id for track_id, _ in sorted_tracks]

    # Create visibility matrix
    vis_matrix = np.ones((len(sorted_tracks), total_frames)) * -1  # -1 = no detection

    # Load GT data to get per-frame visibility
    seq_name = metadata['sequence_info']['name']
    gt_df = load_ground_truth(seq_name)

    for row_idx, (track_id, track_meta) in enumerate(sorted_tracks):
        track_data = gt_df[gt_df['id'] == int(track_id)]
        for _, det in track_data.iterrows():
            frame_idx = int(det['frame']) - 1  # 0-indexed
            if 0 <= frame_idx < total_frames:
                vis_matrix[row_idx, frame_idx] = det['visibility']

    # Plot
    fig, ax = plt.subplots(figsize=(16, 10))

    # Custom colormap: gray for no detection, red-yellow-green for visibility
    colors = ['gray', 'red', 'yellow', 'green']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('visibility', colors, N=n_bins)

    im = ax.imshow(vis_matrix, aspect='auto', cmap=cmap, vmin=-1, vmax=1, interpolation='nearest')

    ax.set_xlabel('Frame Number', fontsize=12)
    ax.set_ylabel('MOT Track ID (sorted by duration, longest first)', fontsize=12)
    ax.set_title(f'Visibility Heatmap - {metadata["sequence_info"]["name"]}', fontsize=14, fontweight='bold')

    # Set Y-axis to show actual track IDs
    ax.set_yticks(range(len(track_ids)))
    ax.set_yticklabels(track_ids, fontsize=9)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Visibility (gray=no detection)', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Visibility heatmap saved")


def plot_crowding(metadata: Dict, save_path: str):
    """Plot crowding over time."""
    frame_analysis = metadata['frame_analysis']
    total_frames = metadata['sequence_info']['total_frames']

    frames = sorted(frame_analysis.keys())
    num_tracks = [frame_analysis[f]['num_tracks'] for f in frames]

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(frames, num_tracks, linewidth=1.5, color='blue', alpha=0.7)

    # Shade density regions
    ax.axhspan(0, SPARSE_THRESHOLD, alpha=0.1, color='green', label='Sparse (<5)')
    ax.axhspan(SPARSE_THRESHOLD, CROWDED_THRESHOLD, alpha=0.1, color='yellow', label='Medium (5-15)')
    ax.axhspan(CROWDED_THRESHOLD, max(num_tracks), alpha=0.1, color='red', label='Crowded (>15)')

    ax.set_xlabel('Frame Number', fontsize=12)
    ax.set_ylabel('Number of Active Tracks', fontsize=12)
    ax.set_title(f'Crowding Over Time - {metadata["sequence_info"]["name"]}', fontsize=14, fontweight='bold')
    ax.set_xlim(0, total_frames)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Crowding plot saved")


def plot_track_length_distribution(metadata: Dict, save_path: str):
    """Plot track length distribution histogram."""
    tracks = metadata['tracks']
    durations = [meta['duration'] for meta in tracks.values()]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(durations, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(MIN_TRACK_LENGTH, color='red', linestyle='--', linewidth=2, label=f'Min Length ({MIN_TRACK_LENGTH})')

    ax.set_xlabel('Track Duration (frames)', fontsize=12)
    ax.set_ylabel('Number of Tracks', fontsize=12)
    ax.set_title(f'Track Length Distribution - {metadata["sequence_info"]["name"]}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Track length distribution saved")


def plot_entry_exit_map(metadata: Dict, save_path: str):
    """Plot entry/exit zones."""
    entry_zones = metadata['spatial_patterns']['entry_zones']
    exit_zones = metadata['spatial_patterns']['exit_zones']

    zones = ['left', 'right', 'top', 'bottom', 'center']
    entry_counts = [entry_zones.get(z, 0) for z in zones]
    exit_counts = [exit_zones.get(z, 0) for z in zones]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(zones, entry_counts, color='green', alpha=0.7)
    ax1.set_title('Entry Points', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Tracks', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    ax2.bar(zones, exit_counts, color='red', alpha=0.7)
    ax2.set_title('Exit Points', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Tracks', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle(f'Entry/Exit Zones - {metadata["sequence_info"]["name"]}', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Entry/exit map saved")


def plot_occlusion_timeline(metadata: Dict, save_path: str):
    """Plot occlusion events timeline with track ID labels."""
    tracks = metadata['tracks']
    total_frames = metadata['sequence_info']['total_frames']

    # Collect all occlusion events
    occlusion_events = []
    for track_id, track_meta in tracks.items():
        for event in track_meta['occlusion_events']:
            occlusion_events.append({
                'track_id': int(track_id),
                'start': event['start_frame'],
                'end': event['end_frame'],
                'duration': event['duration']
            })

    if not occlusion_events:
        # No occlusions, create empty plot
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.text(0.5, 0.5, 'No occlusion events detected', ha='center', va='center', fontsize=16)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Occlusion timeline saved (empty)")
        return

    # Sort by start frame
    occlusion_events = sorted(occlusion_events, key=lambda x: x['start'])

    # Limit to 50 for readability
    display_events = occlusion_events[:50]

    fig, ax = plt.subplots(figsize=(14, 8))

    # Create Y-axis labels showing event index and track ID
    y_labels = [f"Evt {idx} (TID:{event['track_id']})" for idx, event in enumerate(display_events)]

    for idx, event in enumerate(display_events):
        duration = event['end'] - event['start'] + 1
        ax.barh(idx, duration, left=event['start'], height=0.8, color='red', alpha=0.6)

    ax.set_xlabel('Frame Number', fontsize=12)
    ax.set_ylabel('Occlusion Event (with MOT Track ID)', fontsize=12)
    ax.set_title(f'Occlusion Events Timeline - {metadata["sequence_info"]["name"]}', fontsize=14, fontweight='bold')
    ax.set_xlim(0, total_frames)

    # Set Y-axis labels
    ax.set_yticks(range(len(display_events)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_ylim(-0.5, len(display_events) - 0.5)

    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Occlusion timeline saved")


def generate_cross_sequence_summary(all_metadata: Dict[str, Dict]) -> Dict:
    """
    Generate summary comparing all sequences.

    Args:
        all_metadata: Dict mapping sequence name to its metadata

    Returns:
        Cross-sequence summary dict
    """
    summary = {
        'all_sequences': {},
        'cross_sequence_comparison': {},
        'recommended_sequence_usage': {}
    }

    # Per-sequence summary
    for seq_name, metadata in all_metadata.items():
        seq_short = seq_name.split('-')[1]
        stats = metadata['statistics']
        occ_stats = metadata['occlusion_analysis']

        summary['all_sequences'][seq_name] = {
            'frames': metadata['sequence_info']['total_frames'],
            'unique_tracks': stats['unique_track_ids'],
            'avg_track_length': metadata['track_lifetimes']['mean'],
            'occlusion_rate': occ_stats['tracks_with_occlusions'] / max(stats['unique_track_ids'], 1),
            'avg_crowding': stats['avg_tracks_per_frame'],
            'scene_type': classify_scene_type(metadata)
        }

    # Cross-sequence comparison
    all_seq = summary['all_sequences']

    summary['cross_sequence_comparison'] = {
        'shortest_sequence': min(all_seq.items(), key=lambda x: x[1]['frames'])[0],
        'longest_sequence': max(all_seq.items(), key=lambda x: x[1]['frames'])[0],
        'most_crowded': max(all_seq.items(), key=lambda x: x[1]['avg_crowding'])[0],
        'least_crowded': min(all_seq.items(), key=lambda x: x[1]['avg_crowding'])[0],
        'most_occlusions': max(all_seq.items(), key=lambda x: x[1]['occlusion_rate'])[0],
        'cleanest': min(all_seq.items(), key=lambda x: x[1]['occlusion_rate'])[0]
    }

    # Recommendations
    summary['recommended_sequence_usage'] = {
        'baseline_testing': min(all_seq.items(), key=lambda x: (x[1]['avg_crowding'], x[1]['occlusion_rate']))[0],
        'occlusion_analysis': max(all_seq.items(), key=lambda x: x[1]['occlusion_rate'])[0],
        'crowding_analysis': max(all_seq.items(), key=lambda x: x[1]['avg_crowding'])[0],
        'robustness_testing': max(all_seq.items(), key=lambda x: x[1]['frames'])[0]
    }

    return summary


def classify_scene_type(metadata: Dict) -> str:
    """Classify scene type based on statistics."""
    stats = metadata['statistics']
    avg_crowding = stats['avg_tracks_per_frame']

    if avg_crowding < 8:
        return "street_low_density"
    elif avg_crowding < 15:
        return "street_medium_density"
    else:
        return "indoor_crowded"


def main():
    """Main analysis pipeline."""
    print("="*60)
    print("MOT17 Dataset Deep Analysis")
    print("="*60)
    print(f"Analyzing {len(SEQUENCES)} sequences...")
    print(f"Minimum track length: {MIN_TRACK_LENGTH} frames")
    print(f"Visibility threshold: {VISIBILITY_THRESHOLD}")
    print()

    # Create output directories
    os.makedirs(f"{OUTPUT_PATH}/raw_outputs", exist_ok=True)
    os.makedirs(f"{OUTPUT_PATH}/visualizations", exist_ok=True)

    # Analyze each sequence
    all_metadata = {}
    for seq_name in SEQUENCES:
        metadata = analyze_sequence(seq_name)
        save_metadata(metadata, seq_name)
        generate_visualizations(metadata, seq_name)
        all_metadata[seq_name] = metadata

    # Generate cross-sequence summary
    print(f"\n{'='*60}")
    print("Generating cross-sequence summary...")
    print(f"{'='*60}")

    summary = generate_cross_sequence_summary(all_metadata)

    with open(f"{OUTPUT_PATH}/raw_outputs/summary_all_sequences.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVED] {OUTPUT_PATH}/raw_outputs/summary_all_sequences.json")

    # Generate hero tracks summary
    hero_tracks_all = {}
    for seq_name, metadata in all_metadata.items():
        hero_tracks_all[seq_name] = {}
        for category, track_ids in metadata['hero_tracks'].items():
            for tid in track_ids[:3]:  # Top 3 per category
                track_meta = metadata['tracks'][str(tid)]
                hero_tracks_all[seq_name][f"track_{tid}"] = {
                    'category': category,
                    'lifetime': track_meta['lifetime_frames'],
                    'duration': track_meta['duration'],
                    'occlusion_events': len(track_meta['occlusion_events']),
                    'avg_visibility': track_meta['avg_visibility'],
                    'reason': get_hero_track_reason(track_meta, category)
                }

    with open(f"{OUTPUT_PATH}/raw_outputs/hero_tracks_all_sequences.json", 'w') as f:
        json.dump(hero_tracks_all, f, indent=2)
    print(f"[SAVED] {OUTPUT_PATH}/raw_outputs/hero_tracks_all_sequences.json")

    # Print summary
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    print(f"Total sequences analyzed: {len(SEQUENCES)}")
    print(f"Total frames: {sum(m['sequence_info']['total_frames'] for m in all_metadata.values())}")
    print(f"Total unique tracks: {sum(m['statistics']['unique_track_ids'] for m in all_metadata.values())}")
    print(f"\nOutputs:")
    print(f"  - Per-sequence metadata: {OUTPUT_PATH}/raw_outputs/seqXX_metadata.json")
    print(f"  - Cross-sequence summary: {OUTPUT_PATH}/raw_outputs/summary_all_sequences.json")
    print(f"  - Hero tracks: {OUTPUT_PATH}/raw_outputs/hero_tracks_all_sequences.json")
    print(f"  - Visualizations: {OUTPUT_PATH}/visualizations/")
    print()
    print(f"Recommended sequences:")
    print(f"  - Baseline: {summary['recommended_sequence_usage']['baseline_testing']}")
    print(f"  - Occlusions: {summary['recommended_sequence_usage']['occlusion_analysis']}")
    print(f"  - Crowding: {summary['recommended_sequence_usage']['crowding_analysis']}")
    print(f"  - Robustness: {summary['recommended_sequence_usage']['robustness_testing']}")
    print()


def get_hero_track_reason(track_meta: Dict, category: str) -> str:
    """Generate human-readable reason for hero track selection."""
    if category == 'long_stable':
        return f"Long stable track ({track_meta['duration']} frames, {track_meta['avg_visibility']:.2f} avg visibility)"
    elif category == 'occlusion_heavy':
        return f"Multiple occlusions ({len(track_meta['occlusion_events'])} events)"
    elif category == 'high_motion':
        return f"High motion (max displacement: {track_meta['max_displacement']:.1f} pixels)"
    elif category == 'edge_cases':
        return f"Unusual bbox variance ({track_meta['bbox_variance']:.1f})"
    else:
        return "Interesting pattern"


if __name__ == "__main__":
    main()
