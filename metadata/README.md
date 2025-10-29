# MOT17 Dataset Metadata Analysis

**Purpose**: Comprehensive dataset understanding before writing any tracking or uncertainty code.

**Created**: 2025-10-29

---

## Directory Structure

```
metadata/
├── README.md                           # This file - explains everything
├── analysis_code/
│   └── analyze_mot17_dataset.py        # Main analysis script
├── raw_outputs/                        # JSON/CSV/PKL outputs (one per sequence)
│   ├── seq02_metadata.json
│   ├── seq02_metadata.pkl
│   ├── seq04_metadata.json
│   ├── seq04_metadata.pkl
│   ├── seq05_metadata.json
│   ├── seq05_metadata.pkl
│   ├── seq09_metadata.json
│   ├── seq09_metadata.pkl
│   ├── seq10_metadata.json
│   ├── seq10_metadata.pkl
│   ├── seq11_metadata.json
│   ├── seq11_metadata.pkl
│   ├── seq13_metadata.json
│   ├── seq13_metadata.pkl
│   ├── summary_all_sequences.json      # Cross-sequence comparison
│   ├── hero_tracks_all_sequences.json  # Selected tracks for analysis
│   └── tta_recommendations.json        # Frame/track recommendations
└── visualizations/                     # PNG plots for understanding
    ├── seq02_gantt_chart.png
    ├── seq02_visibility_heatmap.png
    ├── seq02_crowding.png
    ├── seq02_track_length_dist.png
    ├── seq02_entry_exit_map.png
    ├── seq02_occlusion_timeline.png
    ├── (same for seq04, 05, 09, 10, 11, 13)
    ├── all_sequences_comparison.png
    └── recommended_hero_tracks.png
```

---

## What This Analysis Does

### Critical Questions Answered

1. **Track Demographics**:
   - How many unique person IDs per sequence?
   - Track length distribution (lifetime in frames)
   - Track birth/death rates over time
   - Entry/exit points in frame (left, right, top, bottom, center)

2. **Occlusion Patterns**:
   - Which tracks have occlusions? (visibility < threshold)
   - Occlusion duration statistics
   - Occlusion frequency over time
   - Partial vs full occlusions (visibility ranges)

3. **Crowd Density Analysis**:
   - Number of active tracks per frame
   - Density categories: sparse (<5), medium (5-15), crowded (>15)
   - Crowding peaks and valleys
   - Interaction events (track crossings/overlaps)

4. **Motion Patterns**:
   - Bbox displacement per frame (motion magnitude)
   - Static vs moving tracks
   - Sudden motion events (acceleration)
   - Average velocity per track

5. **Spatial Patterns**:
   - Bbox size distribution (width, height, area)
   - Position heatmap (where objects appear)
   - Entry/exit zones
   - Track trajectories

6. **TTA Feasibility**:
   - "Hero tracks" for detailed uncertainty analysis
   - Clean segments (stable tracking, good baseline)
   - Occlusion segments (high uncertainty expected)
   - Crowded segments (complex interactions)

---

## Ground Truth Format

MOT17 uses standard MOTChallenge format (`gt/gt.txt`):

```
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>
```

**Fields**:
- `frame`: Frame number (1-indexed)
- `id`: Person/track ID (unique within sequence)
- `bb_left`, `bb_top`: Top-left corner coordinates
- `bb_width`, `bb_height`: Bounding box dimensions
- `conf`: Confidence/considered flag (0=ignore, 1=consider)
- `class`: Object class (1=pedestrian, 2=person on vehicle, etc.)
- `visibility`: **CRITICAL** - Ratio of visible bounding box (0.0=fully occluded, 1.0=fully visible)

**Key Insight**: The `visibility` field is gold for aleatoric uncertainty - true occlusions from data!

---

## Output File Formats

### Per-Sequence Metadata (`seq{XX}_metadata.json`)

```json
{
  "sequence_info": {
    "name": "MOT17-11-FRCNN",
    "total_frames": 900,
    "fps": 30,
    "resolution": [1920, 1080],
    "path": "/ssd_4TB/divake/temporal_uncertainty/MOT17/train/MOT17-11-FRCNN"
  },

  "statistics": {
    "unique_track_ids": 75,
    "total_detections": 10823,
    "avg_tracks_per_frame": 12.03,
    "max_simultaneous_tracks": 28,
    "min_simultaneous_tracks": 2,
    "tracks_meeting_min_length": 18
  },

  "track_lifetimes": {
    "min": 5,
    "max": 687,
    "mean": 144.3,
    "median": 98,
    "std": 112.4,
    "distribution": [
      {"range": "0-30", "count": 42},
      {"range": "30-100", "count": 18},
      {"range": "100-300", "count": 12},
      {"range": "300+", "count": 3}
    ]
  },

  "tracks": {
    "1": {
      "lifetime_frames": [1, 45],
      "duration": 45,
      "num_detections": 45,
      "avg_visibility": 0.92,
      "min_visibility": 0.75,
      "has_occlusion": false,
      "occlusion_events": [],
      "entry_point": "left",
      "exit_point": "right",
      "avg_bbox_area": 15234.5,
      "avg_displacement": 12.3,
      "max_displacement": 45.7,
      "motion_category": "moving",
      "bbox_variance": 234.5,
      "trajectory_length": 523.4,
      "class": "pedestrian"
    },
    "5": {
      "lifetime_frames": [120, 687],
      "duration": 567,
      "num_detections": 545,
      "avg_visibility": 0.68,
      "min_visibility": 0.15,
      "has_occlusion": true,
      "occlusion_events": [
        {"start_frame": 234, "end_frame": 256, "duration": 22, "min_visibility": 0.15},
        {"start_frame": 489, "end_frame": 502, "duration": 13, "min_visibility": 0.28}
      ],
      "entry_point": "bottom",
      "exit_point": "top",
      "avg_bbox_area": 18932.1,
      "avg_displacement": 8.7,
      "max_displacement": 67.3,
      "motion_category": "moving",
      "bbox_variance": 1245.7,
      "trajectory_length": 4923.8,
      "class": "pedestrian"
    }
  },

  "frame_analysis": {
    "1": {
      "num_tracks": 3,
      "num_occluded": 0,
      "density": "sparse",
      "avg_visibility": 0.98,
      "avg_bbox_area": 12345.6,
      "total_motion": 23.4
    },
    "450": {
      "num_tracks": 23,
      "num_occluded": 8,
      "density": "crowded",
      "avg_visibility": 0.72,
      "avg_bbox_area": 9876.5,
      "total_motion": 234.7
    }
  },

  "occlusion_analysis": {
    "total_occlusion_events": 47,
    "tracks_with_occlusions": 23,
    "avg_occlusion_duration": 15.3,
    "occlusion_duration_distribution": {
      "1-10": 18,
      "10-20": 15,
      "20-50": 12,
      "50+": 2
    },
    "frames_with_occlusions": [145, 146, 147, "..."]
  },

  "crowding_analysis": {
    "sparse_frames": [1, 2, 3, "..."],
    "medium_frames": [45, 46, 47, "..."],
    "crowded_frames": [234, 235, 236, "..."],
    "peak_crowding": {
      "frame": 456,
      "num_tracks": 28
    }
  },

  "motion_analysis": {
    "static_tracks": [3, 7, 12],
    "moving_tracks": [1, 2, 4, "..."],
    "high_motion_events": [
      {"frame": 234, "track_id": 5, "displacement": 67.3},
      {"frame": 567, "track_id": 12, "displacement": 78.9}
    ]
  },

  "spatial_patterns": {
    "entry_zones": {
      "left": 12,
      "right": 15,
      "top": 8,
      "bottom": 23,
      "center": 5
    },
    "exit_zones": {
      "left": 14,
      "right": 13,
      "top": 9,
      "bottom": 21,
      "center": 6
    },
    "position_heatmap": "See visualization"
  },

  "hero_tracks": {
    "long_stable": [15, 23, 42],
    "occlusion_heavy": [5, 18, 31],
    "high_motion": [9, 27, 44],
    "edge_cases": [3, 56]
  },

  "tta_recommendations": {
    "clean_segments": [
      {"start": 1, "end": 100, "reason": "Low density, stable tracking"},
      {"start": 700, "end": 800, "reason": "Medium density, no occlusions"}
    ],
    "occlusion_segments": [
      {"start": 230, "end": 260, "reason": "Multiple occlusion events"},
      {"start": 485, "end": 510, "reason": "Heavy occlusions"}
    ],
    "crowded_segments": [
      {"start": 400, "end": 500, "reason": "Peak crowding, complex interactions"}
    ],
    "recommended_test_frames": [50, 150, 250, 450, 650, 850]
  }
}
```

### Cross-Sequence Summary (`summary_all_sequences.json`)

```json
{
  "all_sequences": {
    "MOT17-02-FRCNN": {
      "frames": 600,
      "unique_tracks": 54,
      "avg_track_length": 87.5,
      "occlusion_rate": 0.23,
      "avg_crowding": 6.2,
      "scene_type": "street_low_density"
    },
    "MOT17-04-FRCNN": {
      "frames": 1050,
      "unique_tracks": 83,
      "avg_track_length": 156.3,
      "occlusion_rate": 0.31,
      "avg_crowding": 18.7,
      "scene_type": "indoor_crowded"
    },
    "... (all 7 sequences)"
  },

  "cross_sequence_comparison": {
    "shortest_sequence": "MOT17-09-FRCNN (525 frames)",
    "longest_sequence": "MOT17-04-FRCNN (1050 frames)",
    "most_crowded": "MOT17-11-FRCNN (avg 12.3 tracks/frame)",
    "least_crowded": "MOT17-02-FRCNN (avg 6.2 tracks/frame)",
    "most_occlusions": "MOT17-13-FRCNN (47 events)",
    "cleanest": "MOT17-09-FRCNN (12 events)"
  },

  "recommended_sequence_usage": {
    "baseline_testing": "MOT17-02-FRCNN (clean, low density)",
    "occlusion_analysis": "MOT17-13-FRCNN (heavy occlusions)",
    "crowding_analysis": "MOT17-11-FRCNN (high density)",
    "robustness_testing": "MOT17-04-FRCNN (longest, diverse conditions)"
  }
}
```

### Hero Tracks Selection (`hero_tracks_all_sequences.json`)

```json
{
  "MOT17-11-FRCNN": {
    "track_5": {
      "category": "occlusion_heavy",
      "lifetime": [120, 687],
      "duration": 567,
      "occlusion_events": 2,
      "total_occlusion_frames": 35,
      "avg_visibility": 0.68,
      "reason": "Long track with multiple clean occlusion events - perfect for temporal uncertainty propagation",
      "recommended_analysis": [
        "Baseline uncertainty (frames 120-230)",
        "First occlusion (frames 234-256)",
        "Recovery period (frames 257-485)",
        "Second occlusion (frames 489-502)",
        "Post-occlusion (frames 503-687)"
      ]
    },
    "track_15": {
      "category": "long_stable",
      "lifetime": [1, 680],
      "duration": 680,
      "occlusion_events": 0,
      "total_occlusion_frames": 0,
      "avg_visibility": 0.97,
      "reason": "Longest clean track - baseline for low uncertainty",
      "recommended_analysis": [
        "Stable uncertainty baseline",
        "Compare to occluded tracks"
      ]
    },
    "track_27": {
      "category": "high_motion",
      "lifetime": [200, 567],
      "duration": 367,
      "max_displacement": 78.9,
      "avg_displacement": 23.4,
      "reason": "High-speed motion events - test motion-induced uncertainty",
      "recommended_analysis": [
        "Uncertainty during rapid motion",
        "Motion blur effects"
      ]
    }
  },
  "... (hero tracks for all 7 sequences)"
}
```

---

## Visualizations Generated

### Per-Sequence Plots

1. **Gantt Chart** (`seq{XX}_gantt_chart.png`):
   - X-axis: Frame number
   - Y-axis: Track IDs
   - Shows when each track exists
   - Color-coded by track length category

2. **Visibility Heatmap** (`seq{XX}_visibility_heatmap.png`):
   - X-axis: Frame number
   - Y-axis: Track IDs (top 30 longest)
   - Color: Visibility (0=black, 1=white)
   - Reveals occlusion patterns at a glance

3. **Crowding Over Time** (`seq{XX}_crowding.png`):
   - Line plot: Number of active tracks per frame
   - Shaded regions: Sparse/medium/crowded
   - Marks peak crowding events

4. **Track Length Distribution** (`seq{XX}_track_length_dist.png`):
   - Histogram of track lifetimes
   - Shows data quality (many short tracks = noisy)

5. **Entry/Exit Map** (`seq{XX}_entry_exit_map.png`):
   - Frame boundaries with entry/exit counts
   - Reveals camera field-of-view characteristics

6. **Occlusion Timeline** (`seq{XX}_occlusion_timeline.png`):
   - Timeline of all occlusion events
   - Shows temporal clustering of occlusions

### Cross-Sequence Comparisons

7. **All Sequences Comparison** (`all_sequences_comparison.png`):
   - Side-by-side statistics for all 7 sequences
   - Bar charts: Track count, avg length, occlusion rate, crowding

8. **Recommended Hero Tracks** (`recommended_hero_tracks.png`):
   - Visual summary of selected hero tracks
   - Shows why each track was selected

---

## Usage

### Running the Analysis

```bash
cd /ssd_4TB/divake/temporal_uncertainty/metadata/analysis_code
python analyze_mot17_dataset.py
```

**Expected Runtime**: 5-10 minutes for all 7 sequences

**Output**: All JSON/PKL files in `raw_outputs/`, all plots in `visualizations/`

### Using the Metadata in Code

```python
import json
import pickle

# Load metadata for a sequence
with open('metadata/raw_outputs/seq11_metadata.json', 'r') as f:
    seq11_meta = json.load(f)

# Get hero tracks
hero_tracks = seq11_meta['hero_tracks']['occlusion_heavy']
print(f"Occlusion-heavy tracks: {hero_tracks}")

# Get recommended test frames
test_frames = seq11_meta['tta_recommendations']['recommended_test_frames']

# Check if a track meets minimum length
track_5_duration = seq11_meta['tracks']['5']['duration']
if track_5_duration >= 30:
    print(f"Track 5 is suitable for analysis ({track_5_duration} frames)")

# Load cross-sequence summary
with open('metadata/raw_outputs/summary_all_sequences.json', 'r') as f:
    summary = json.load(f)

# Find best sequence for occlusion analysis
best_seq = summary['recommended_sequence_usage']['occlusion_analysis']
```

---

## Key Findings (To Be Filled After Running Analysis)

**This section will be updated after the analysis runs.**

### Overall Statistics
- Total frames across all sequences: TBD
- Total unique tracks: TBD
- Average occlusion rate: TBD
- Most challenging sequence: TBD

### Hero Track Highlights
- Best occlusion example: TBD
- Longest stable track: TBD
- Most complex interactions: TBD

### TTA Strategy Implications
- Recommended starting sequence: TBD
- Frame ranges for experiments: TBD
- Expected uncertainty patterns: TBD

---

## Maintenance

**This analysis is one-time work**. Re-run only if:
- MOT17 data changes (unlikely)
- Need additional statistics not currently extracted
- Discover new patterns requiring deeper analysis

**Update Log**:
- 2025-10-29: Initial analysis script created
- (Future updates will be logged here)

---

**End of README**
