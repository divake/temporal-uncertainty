# Temporal Aleatoric Uncertainty in Video Object Tracking

## Quick Start

This project analyzes aleatoric (data-inherent) uncertainty in video object tracking using test-time augmentation on MOT17 dataset with multiple YOLOv8 models.

### Project Structure
- **MOT17/train/** - 7 sequences with FRCNN detections (525-1050 frames each)
- **MOT17/video/** - Pre-rendered MP4 videos for quick visualization
- **github_repos/** - 18 cloned repos with production-ready uncertainty code
- **models/** - YOLOv8 model weights (will be auto-downloaded)
- **project_info.md** - Complete implementation guide

### Key Statistics
- **Sequences**: 7 MOT17 sequences (5,316 total frames)
- **Models**: 5 YOLOv8 variants (3.2M to 68.2M parameters)
- **Detections**: FRCNN (Faster R-CNN) - more accurate than DPM
- **Code Resources**: 18 GitHub repositories with uncertainty implementations

### Why FRCNN over DPM?
FRCNN (Faster R-CNN) detections are used instead of DPM (Deformable Parts Model) because:
1. Higher accuracy (~5-10% mAP improvement)
2. Better localization (tighter bounding boxes)
3. More reliable for uncertainty quantification
4. Widely used in modern tracking benchmarks

See **project_info.md** for complete implementation details.
