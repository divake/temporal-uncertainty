# Implementation Decisions Log

**Project**: Temporal Aleatoric Uncertainty in Video Object Tracking
**Started**: 2025-10-29
**Last Updated**: 2025-10-29

---

## Table of Contents
1. [Core Philosophy](#core-philosophy)
2. [Analysis Strategy](#analysis-strategy)
3. [Code Structure](#code-structure)
4. [First Phase: Dataset Analysis](#first-phase-dataset-analysis)
5. [Second Phase: Single Model Pipeline](#second-phase-single-model-pipeline)
6. [Future Phases](#future-phases)
7. [Decisions Chronology](#decisions-chronology)

---

## Core Philosophy

### Development Approach
**"Horizontal First, Then Vertical"**

- Build complete end-to-end pipeline for ONE sequence, ONE model, ONE augmentation
- Get it working perfectly before scaling
- Scaling vertically (more models, sequences, augmentations) should be trivial once horizontal works

### Code Quality Principles

1. **No Fallbacks**: Code must fail loudly with clear error messages
   - Use `raise ValueError/FileNotFoundError` with detailed messages
   - No `try-except` that swallows errors
   - No default values that hide missing data
   - Example: `raise FileNotFoundError(f"GT file not found at {gt_path}. Expected format: {expected_format}")`

2. **Borrow, Don't Rewrite**: Use production-ready code from cloned GitHub repos
   - Core uncertainty: From `Bayesian-Neural-Networks`, `uncertainty-toolbox`
   - Augmentations: From `albumentations`, `ttach`
   - Tracking: From `boxmot`
   - Metrics: From `uncertainty-toolbox/metrics.py`
   - Add comments: `# Borrowed from uncertainty-toolbox/metrics.py`

3. **Separation of Concerns**: Keep stable code separate from experimental code
   - Files that never change: `data_loader.py`, `model_loader.py`
   - Files that evolve: experiment scripts, analysis notebooks
   - Clear folder structure (see below)

4. **One-Time Work**: Do comprehensive setup work once, reuse forever
   - Dataset metadata extraction (once per sequence)
   - Model weight organization (once)
   - Core utility functions (once)

---

## Analysis Strategy

### Primary Approach: Track-Centric with Smart Filtering

**Why Track-Level Analysis?**
- Aleatoric uncertainty is a per-object property that persists over time
- Temporal propagation only makes sense when following specific objects
- Enables the key result: "Track X shows correlated uncertainty across all models"
- Natural for video tracking research

**Track Categorization by Lifetime:**
- **Long tracks** (>100 frames): Full temporal analysis, highest priority
- **Medium tracks** (30-100 frames): Include if they have interesting events (occlusions, crossings)
- **Short tracks** (<30 frames): Aggregate statistics only, exclude from temporal analysis

**Track Selection Strategy:**
```
Priority 1: Tracks with occlusion events (from GT visibility < 1.0)
Priority 2: Tracks with high motion variance (large bbox displacement)
Priority 3: Long stable tracks (baseline comparison, low variance)
```

### Key Metrics to Prove Hypothesis

**Core Hypothesis**: Aleatoric uncertainty is data-inherent (not model-dependent) and exhibits temporal consistency.

**Success Criteria:**
1. **Cross-model correlation > 0.85**: Same uncertainty patterns across YOLOv8n/s/m/l/x
2. **Temporal consistency**: Uncertainty persists across frames (not random)
3. **Scene correlation**: Uncertainty correlates with occlusions, crowding, motion
4. **Zero training**: All analysis is post-hoc inference only

**Primary Metrics to Compute:**
- Predictive variance across augmentations: `σ²_aleatoric = Var_T[f(T(x_t))]`
- Expected Calibration Error (ECE): Does uncertainty correlate with actual errors?
- Temporal correlation: `corr(uncertainty[t], uncertainty[t+1])`
- Cross-model correlation matrix: 5×5 matrix for all YOLO pairs

### Handling Track Birth/Death

**Track Birth:**
- Expected high initial uncertainty (object just appeared)
- Exclude first 5 frames from correlation analysis
- Study: "How quickly does uncertainty stabilize after appearance?"

**Track Death:**
- Often coincides with high uncertainty (object leaving/occluded)
- Analyze: "Does uncertainty predict track loss?"
- Potential bonus finding for paper

**Tracks That Come and Go:**
- Common in MOT17 (people enter/exit frame)
- Solution: Analyze per-track lifetime, don't require full-sequence presence
- Aggregate statistics across all tracks for dataset-level insights

---

## Code Structure

### Directory Layout

```
/ssd_4TB/divake/temporal_uncertainty/
├── MOT17/
│   ├── train/                      # Image sequences (already exists)
│   │   ├── MOT17-02-FRCNN/
│   │   ├── MOT17-04-FRCNN/
│   │   ├── MOT17-05-FRCNN/
│   │   ├── MOT17-09-FRCNN/
│   │   ├── MOT17-10-FRCNN/
│   │   ├── MOT17-11-FRCNN/
│   │   └── MOT17-13-FRCNN/
│   └── video/                      # Pre-rendered MP4 videos (already exists)
├── models/                         # YOLOv8 weights (already exists)
│   ├── yolov8n.pt
│   ├── yolov8s.pt
│   ├── yolov8m.pt
│   ├── yolov8l.pt
│   └── yolov8x.pt
├── github_repos/                   # Cloned repositories (already exists)
│   ├── core_uncertainty/
│   ├── tracking_implementations/
│   ├── augmentation_libs/
│   ├── tta_specific/
│   ├── papers_with_code/
│   └── evaluation_metrics/
├── metadata/                       # Dataset analysis outputs (TO CREATE)
│   ├── seq02_gt_analysis.json
│   ├── seq04_gt_analysis.json
│   ├── seq05_gt_analysis.json
│   ├── seq09_gt_analysis.json
│   ├── seq10_gt_analysis.json
│   ├── seq11_gt_analysis.json
│   ├── seq13_gt_analysis.json
│   ├── seq11_hero_tracks.json      # Selected tracks for detailed analysis
│   ├── summary_all_sequences.json  # Cross-sequence statistics
│   └── visualizations/             # Dataset understanding plots
│       ├── seq11_gantt_chart.png
│       ├── seq11_visibility_heatmap.png
│       ├── seq11_crowding.png
│       └── ...
├── core/                           # Stable, reusable code (TO CREATE)
│   ├── data_loader.py              # Load MOT17 sequences (NEVER changes)
│   ├── model_loader.py             # Load YOLO models (NEVER changes)
│   ├── tracker.py                  # ByteTrack wrapper (from boxmot)
│   └── uncertainty/
│       ├── tta.py                  # TTA logic (from ttach)
│       ├── metrics.py              # Uncertainty metrics (from uncertainty-toolbox)
│       └── aleatoric.py            # Core σ² calculation (from Bayesian-Neural-Networks)
├── augmentations/                  # Augmentation implementations (TO CREATE)
│   ├── __init__.py
│   ├── blur.py                     # From albumentations
│   ├── noise.py                    # From albumentations
│   ├── brightness.py               # From albumentations
│   └── base.py                     # Base augmentation interface
├── scripts/                        # One-time analysis scripts (TO CREATE)
│   ├── analyze_dataset.py          # GT parsing and metadata generation
│   └── visualize_dataset.py        # Dataset visualization generation
├── experiments/                    # Experiment scripts (TO CREATE)
│   ├── seq11_yolov8n_clean.py      # First experiment: baseline
│   └── (future experiments)
├── utils/                          # Helper utilities (TO CREATE)
│   ├── visualization.py
│   └── io.py
├── results/                        # Experiment outputs (TO CREATE)
│   └── seq11_yolov8n_clean/
│       ├── detections.pkl
│       ├── tracks.pkl
│       ├── uncertainty.pkl
│       ├── track_summary.csv
│       ├── output.log
│       └── plots/
├── README.md                       # Quick start guide (already exists)
├── project_info.md                 # Complete implementation guide (already exists)
├── IMPLEMENTATION_DECISIONS.md     # This file
└── verify_setup.sh                 # Setup verification script (already exists)
```

### File Organization Principles

**Never-Changing Files** (`core/`):
- Data loading logic
- Model loading logic
- Core tracking algorithms
- Uncertainty computation functions
- These are infrastructure - write once, use forever

**Configuration Files** (add later when needed):
- Initially: Hardcode paths in scripts for simplicity
- After pipeline works: Refactor to YAML configs
- Location: `config/sequences.yaml`, `config/models.yaml`, `config/augmentations.yaml`

**Experiment Scripts** (`experiments/`):
- One script per experiment
- Clear naming: `{sequence}_{model}_{augmentation}.py`
- Self-contained: imports from `core/`, writes to `results/`

**Results Organization**:
- Flat structure initially: `results/seq11_yolov8n_clean/`
- Can reorganize later if needed
- Each experiment folder contains: data files + plots + logs

---

## First Phase: Dataset Analysis

### Decision: Analyze Data Before Writing Detection Code

**Rationale:**
1. Can't select "hero tracks" without knowing what exists in the data
2. Can't design occlusion detection without understanding GT format
3. Uncertainty analysis requires knowing normal behavior baselines
4. Avoids writing code for edge cases that don't exist
5. This is one-time work that informs all future decisions

### What to Extract from Ground Truth

MOT17 GT format (`gt/gt.txt`):
```
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>
```

**For Each Sequence, Extract:**

1. **Track Statistics:**
   - Total unique track IDs
   - Track lifetimes: min, max, median, mean, std, distribution
   - Tracks by category: pedestrian, rider, etc. (if applicable)
   - Tracks meeting minimum length threshold (≥30 frames)

2. **Occlusion Analysis:**
   - Tracks with occlusions (visibility < 1.0)
   - Occlusion events per track: start frame, end frame, duration
   - Occlusion frequency over time
   - Tracks with multiple occlusion events

3. **Spatial Patterns:**
   - Bounding box size distribution (width, height, area)
   - Position heatmap (where do objects appear in frame?)
   - Entry/exit zones (frame boundaries)
   - Interaction patterns (tracks that cross/overlap)

4. **Temporal Patterns:**
   - Track birth rate per frame (new tracks appearing)
   - Track death rate per frame (tracks disappearing)
   - Crowding statistics (number of active tracks per frame)
   - Crowding peaks (frames with maximum simultaneous tracks)

5. **Motion Analysis:**
   - Bbox displacement per frame (motion magnitude)
   - Static vs moving tracks
   - High-motion events (sudden acceleration)

### Hero Track Selection (Data-Driven)

**Automatically identify tracks for detailed analysis:**

```python
# Pseudocode for hero track selection
hero_tracks = {
    'occlusion_heavy': top_5_tracks_by_occlusion_count,
    'long_stable': top_5_tracks_by_lifetime_with_high_visibility,
    'high_motion': top_5_tracks_by_bbox_displacement,
    'edge_cases': tracks_with_interesting_patterns
}
```

**Output**: `metadata/seq11_hero_tracks.json`
```json
{
  "track_42": {
    "category": "occlusion_heavy",
    "lifetime": 245,
    "frames": [120, 121, ..., 365],
    "occlusion_events": [
      {"start": 120, "end": 135, "duration": 15},
      {"start": 200, "end": 210, "duration": 10}
    ],
    "reason": "Multiple occlusions with clean recovery",
    "avg_visibility": 0.72
  },
  "track_15": {
    "category": "long_stable",
    "lifetime": 680,
    "frames": [1, 2, ..., 680],
    "occlusion_events": [],
    "reason": "Longest unoccluded track - baseline comparison",
    "avg_visibility": 0.98
  }
}
```

### Dataset Visualizations

**Must-Generate Plots** (for understanding data):

1. **Track Lifetime Gantt Chart**: X-axis: frames, Y-axis: track IDs, shows when each track exists
2. **Visibility Heatmap**: X-axis: frames, Y-axis: track IDs, color: visibility (0-1)
3. **Crowding Over Time**: Line plot of active tracks per frame
4. **Occlusion Frequency Histogram**: Distribution of occlusion durations
5. **Track Length Distribution**: Histogram of track lifetimes
6. **Sample Frames with GT Overlays**: Visual verification (frames 1, 300, 600, 900)

**Output Location**: `metadata/visualizations/seq11_*.png`

### First Script: `scripts/analyze_dataset.py`

**Functionality:**
- Parse MOT17 GT file (`gt/gt.txt`)
- Compute all statistics listed above
- Identify hero tracks automatically
- Generate all visualization plots
- Save outputs:
  - `metadata/seq11_gt_analysis.json` (human-readable)
  - `metadata/seq11_gt_analysis.pkl` (Python-friendly)
  - `metadata/seq11_hero_tracks.json`
  - `metadata/visualizations/seq11_*.png`
- Print summary to console

**No Models, No Tracking, No Augmentation** - Just pure data understanding.

---

## Second Phase: Single Model Pipeline

### Target: MOT17-11 + YOLOv8n + Clean Data

**After dataset analysis is complete, build end-to-end pipeline.**

### Experiment Parameters

- **Sequence**: MOT17-11-FRCNN (900 frames, crowded scene)
- **Model**: YOLOv8n (fastest, 3.2M parameters)
- **Augmentation**: None (clean data baseline)
- **Frame Range**:
  - Initial testing: Frames 1-300 (faster iteration)
  - Final run: All 900 frames
- **Track Filter**: Tracks with ≥30 frames
- **Tracker**: ByteTrack with default hyperparameters

### What This Run Does

1. **Validate Pipeline**: Ensure all components work together
2. **Baseline Confidence**: Save YOLO detection confidence (not uncertainty yet)
3. **Track Verification**: Compare detected tracks to GT hero tracks
4. **Structure Preparation**: Create data structure for TTA analysis (next phase)

### Output Files

**Data Files:**
- `results/seq11_yolov8n_clean/detections.pkl`: Raw YOLO detections per frame
- `results/seq11_yolov8n_clean/tracks.pkl`: ByteTrack output with track IDs
- `results/seq11_yolov8n_clean/track_summary.csv`: Track metadata (ID, lifetime, mean confidence)
- `results/seq11_yolov8n_clean/output.log`: Console output saved

**Visualizations (minimal for first run):**
1. Track lifetime Gantt chart (detection-based, compare to GT)
2. Per-track confidence over time (top 10 longest tracks)
3. Track length distribution histogram

### Clean Data "Uncertainty"

**For clean data without augmentation:**
- No TTA variance yet (that's next phase)
- Save YOLO confidence scores as baseline
- Structure: `track_data['placeholder_uncertainty'] = None` (to be filled later)
- Purpose: Validate detection + tracking pipeline works

### Error Handling Example

```python
# Good: Explicit error with context
if not os.path.exists(sequence_path):
    raise FileNotFoundError(
        f"Sequence not found at {sequence_path}\n"
        f"Expected structure: {sequence_path}/img1/*.jpg\n"
        f"Available sequences: {os.listdir('/ssd_4TB/divake/temporal_uncertainty/data/MOT17/train')}"
    )

# Bad: Silent fallback (NEVER do this)
if not os.path.exists(sequence_path):
    sequence_path = default_path  # Hides the real problem!
```

### Progress Tracking

```python
# Simple print statements for first run
print(f"[INFO] Loading model from {model_path}")
print(f"[INFO] Processing {num_frames} frames")
for i, frame in enumerate(frames):
    if i % 100 == 0:
        print(f"[INFO] Processed {i}/{num_frames} frames")
```

Later: Convert to proper logging with levels.

---

## Future Phases

### Phase 3: Add Test-Time Augmentation (TTA)

**After clean pipeline works:**

- Keep same sequence (MOT17-11) and model (YOLOv8n)
- Add ONE augmentation: Gaussian Blur (simplest to start)
- Compute variance across augmented predictions
- This variance IS aleatoric uncertainty
- Compare to clean baseline

**Augmentation Parameters** (from albumentations):
```python
GaussianBlur(blur_limit=(3, 7), p=1.0)
```

**Expected Output:**
- `results/seq11_yolov8n_blur/uncertainty.pkl`: Variance-based uncertainty per track
- Plots: Uncertainty over time for hero tracks

### Phase 4: Multiple Augmentations

**Add remaining augmentations one by one:**

1. Gaussian Noise: `GaussNoise(var_limit=(10.0, 50.0), p=1.0)`
2. Brightness/Contrast: `RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0)`
3. Scale: `RandomScale(scale_limit=0.1, p=1.0)`
4. JPEG Compression: `ImageCompression(quality_lower=40, quality_upper=100, p=1.0)`

**Combined TTA:**
- Apply all 5 augmentations + clean = 6 predictions per frame
- Variance across these 6 predictions = aleatoric uncertainty
- Experiment name: `seq11_yolov8n_tta_combined`

### Phase 5: Multiple Models (Vertical Scaling)

**Once TTA works for YOLOv8n:**

- Run same experiment for: YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x
- Same sequence, same augmentations
- Compute cross-model correlation matrix (5×5)
- **Key Result**: Correlation > 0.85 proves aleatoric nature

**Code Change Required:**
- Minimal: Loop over model list instead of hardcoding
- `for model_name in ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']:`

### Phase 6: Multiple Sequences

**Run complete pipeline on all 7 sequences:**

- MOT17-02, 04, 05, 09, 10, 11, 13
- Same models, same augmentations
- Cross-sequence comparison
- Identify sequence-specific patterns

### Phase 7: Advanced Analysis

**Once all data is collected:**

- Temporal propagation model: `σ²_temporal(t) = α·σ²_current(t) + (1-α)·σ²_temporal(t-1)`
- Uncertainty prediction: Can uncertainty at frame t predict errors at t+k?
- Correlation with scene properties: Crowding, motion, occlusions
- Calibration analysis: Are high-uncertainty predictions actually less accurate?

---

## Decisions Chronology

### 2025-10-29: Initial Planning

**Decision 1: Horizontal-First Development Strategy**
- Rationale: Get one complete end-to-end pipeline working before scaling
- Impact: Reduces complexity, enables faster debugging, builds solid foundation
- Alternative rejected: Building multi-model framework from the start (premature)

**Decision 2: Track-Centric Analysis (Not Frame-Aggregate)**
- Rationale: Temporal uncertainty propagation requires following specific objects
- Impact: Enables per-track correlation analysis across models (key result)
- Alternative rejected: Frame-level aggregate (loses temporal structure)

**Decision 3: Dataset Analysis First, Code Second**
- Rationale: Can't make informed decisions without understanding the data
- Impact: Hero track selection, occlusion detection, baseline expectations all data-driven
- Alternative rejected: Writing detection code first, then discovering data issues

**Decision 4: Minimum Track Length = 30 Frames**
- Rationale: Need sufficient temporal context for uncertainty propagation
- Impact: Filters out noise tracks, focuses on meaningful temporal patterns
- Alternative rejected: Analyzing all tracks (too noisy) or only very long tracks (too restrictive)

**Decision 5: Use MOT17 Ground Truth for Occlusion Detection**
- Rationale: GT visibility annotations are reliable and already available
- Impact: Clean, accurate occlusion events without heuristic errors
- Alternative rejected: Heuristic-based detection (confidence drops, bbox shrink) - less accurate

**Decision 6: Sequence 11 for Initial Development**
- Rationale: 900 frames (medium length), crowded scene (interesting), manageable size
- Impact: Good balance between complexity and runtime for testing
- Alternative rejected: Seq 09 (shortest, but may lack diversity) or Seq 04 (longest, but slow testing)

**Decision 7: No Config Files in First Run**
- Rationale: One more potential failure point, adds complexity
- Impact: Faster initial development, clearer code flow
- Alternative rejected: YAML configs from start (premature abstraction)

**Decision 8: Borrow Code from Cloned Repos**
- Rationale: Production-quality implementations already exist
- Impact: Higher code quality, faster development, proven methods
- Alternative rejected: Writing from scratch (reinventing wheel, potential bugs)

**Decision 9: Fail Loudly (No Fallbacks)**
- Rationale: Silent failures hide real issues and cause downstream errors
- Impact: Easier debugging, clearer error messages, no mysterious behavior
- Alternative rejected: Try-except with fallbacks (masks problems)

**Decision 10: Clean Data Baseline First, Then Augmentations**
- Rationale: Validate detection + tracking pipeline works before adding TTA complexity
- Impact: Can isolate TTA issues from pipeline issues
- Alternative rejected: Adding augmentations immediately (too many variables)

---

## Notes for Future Updates

**This document should be updated when:**
- Major architectural decisions are made
- New phases begin
- Unexpected issues require approach changes
- Analysis reveals patterns that change strategy

**Update format:**
```markdown
### YYYY-MM-DD: Brief Title
**Decision**: What was decided
- Rationale: Why this was chosen
- Impact: What changes as a result
- Alternative rejected: What we didn't do and why
```

---

## Dataset Analysis Results (2025-10-29)

### Phase 1 Complete! ✅

**Analysis Completed**: All 7 MOT17 sequences analyzed
**Runtime**: ~2 minutes
**Files Generated**:
- 14 metadata files (JSON + PKL per sequence)
- 42 visualization plots (6 per sequence)
- Cross-sequence summary
- Hero tracks recommendations

### Key Findings

**MOT17-11-FRCNN Selected for Phase 2**:
- ✅ 900 frames (balanced length)
- ✅ 75 tracks, 61 meeting minimum length
- ✅ 135 occlusion events (perfect for uncertainty analysis)
- ✅ 81.3% occlusion rate (realistic)
- ✅ 10.5 avg tracks/frame (medium crowding)

**Hero Track Identified: Track 25**:
- ✅ Full sequence coverage (frames 1-900)
- ✅ 8 occlusion events with varied durations (7, 58, 62 frames)
- ✅ Avg visibility: 0.618 (moderate)
- ✅ Static motion: 3.8 px avg displacement
- ✅ **Perfect for temporal uncertainty propagation analysis**

**Cross-Sequence Statistics**:
- Total tracks analyzed: 546
- Occlusion rate range: 72-100%
- Crowding range: 8.3-45.3 tracks/frame
- Avg track length range: 52-573 frames

**See detailed findings**: `metadata/ANALYSIS_FINDINGS.md`

### Decision 11: Focus on Track 25 for Initial Experiments

**Rationale**: Track 25 provides ideal characteristics for proving temporal aleatoric uncertainty:
1. Full sequence visibility (can study entire uncertainty evolution)
2. Multiple occlusion events (repeated validation of uncertainty spikes)
3. Clean recovery periods (study uncertainty decay dynamics)
4. Static motion (removes motion blur confound)

**Impact**: First experiment will run on full seq 11 but analyze Track 25 in detail, with aggregate statistics on all tracks ≥30 frames.

---

## Open Questions (To Be Resolved)

*None currently - ready to proceed to Phase 2: Single Model Pipeline.*

---

**End of Document**

*Last updated: 2025-10-29*
