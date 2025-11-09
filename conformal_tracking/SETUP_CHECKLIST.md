# Setup Checklist for V1 Implementation

## ✅ What You Already Have

### Data
- [x] MOT17 dataset at `/ssd_4TB/divake/temporal_uncertainty/data/MOT17/train/`
- [x] Comprehensive metadata at `/ssd_4TB/divake/temporal_uncertainty/metadata/`
- [x] YOLO models (n, s, m, l, x) at `/ssd_4TB/divake/temporal_uncertainty/models/`

### Prior Work
- [x] Mahalanobis KNN implementation (from UCI work)
- [x] Multi-source epistemic ensemble (from UCI work)
- [x] Code at `/ssd_4TB/divake/temporal_uncertainty/cacd/enhanced_cacd/src/`

---

## ❓ What You Need to Download/Install

### 1. Python Packages (CRITICAL)

Run this to install all dependencies:

```bash
cd /ssd_4TB/divake/temporal_uncertainty/conformal_tracking

# Create requirements.txt first (see below)
pip install -r requirements.txt
```

**requirements.txt**:
```
# Core scientific computing
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
pandas>=2.0.0

# Deep learning (for YOLO)
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0

# Kalman filtering
filterpy>=1.4.5

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0

# Optional: for advanced tracking
# lap>=0.4.0  # Linear assignment problem solver
```

**Check if you already have these**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import ultralytics; print('Ultralytics: OK')"
python -c "import filterpy; print('FilterPy: OK')"
```

---

### 2. ByteTrack (OPTIONAL - for tracker integration)

**Option A: Use existing tracking code**
- If you already have a tracker working, we can wrap it

**Option B: Use simple nearest-neighbor tracker**
- We can implement a minimal tracker ourselves for V1

**Option C: Install ByteTrack** (recommended for best results):
```bash
cd /ssd_4TB/divake/temporal_uncertainty
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack
pip install -r requirements.txt
python setup.py develop
```

**Recommendation**: Start without ByteTrack in Week 1, add in Week 2 if needed.

---

### 3. SAM Weights (NOT NEEDED for V1!)

**V1 uses YOLO features only** - no SAM required.

**For V3 (optional, Week 3+)**:
```bash
# Only download if you get to V3
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P /ssd_4TB/divake/temporal_uncertainty/models/

# Install SAM
pip install segment-anything
```

**Skip this for now!**

---

## 🔧 Environment Setup

### Check CUDA (for YOLO feature extraction)

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA Device: {torch.cuda.get_device_name(0)}')"
```

**Expected**: Should show your GPU (A100, RTX, etc.)

If CUDA not available, V1 will be slower but still works on CPU.

---

### Verify YOLO Works

```bash
python << 'EOF'
from ultralytics import YOLO
import torch

# Load YOLOv8n
model = YOLO('/ssd_4TB/divake/temporal_uncertainty/models/yolov8n.pt')

print(f"Model loaded: {model.model}")
print(f"Device: {next(model.model.parameters()).device}")

# Check if we can extract features
print("\nBackbone layers:")
for i, layer in enumerate(model.model.model):
    print(f"Layer {i}: {type(layer).__name__}")
EOF
```

**Expected output**: Should show model layers, no errors.

---

## 📁 Directory Structure Check

Run this to verify everything is in place:

```bash
cd /ssd_4TB/divake/temporal_uncertainty

echo "=== Checking Data ==="
ls -lh data/MOT17/train/ | head -5

echo -e "\n=== Checking Models ==="
ls -lh models/*.pt

echo -e "\n=== Checking Metadata ==="
ls -lh metadata/raw_outputs/*.json | head -5

echo -e "\n=== Checking Project Structure ==="
tree conformal_tracking/ -L 2
```

**Expected**: All directories exist, no "not found" errors.

---

## 🎯 What You DON'T Need

### NOT NEEDED for V1:
- ❌ SAM model weights (only for V3)
- ❌ KITTI dataset (only if doing multi-modal)
- ❌ DanceTrack dataset (optional extension)
- ❌ Cloud computing / GPUs (V1 runs on your local A100)

### NOT NEEDED at all:
- ❌ Custom YOLO training (we use pretrained)
- ❌ Additional annotations (MOT17 ground truth is sufficient)

---

## 🚀 Quick Start Validation

Once everything is installed, run this test:

```python
# Save as: conformal_tracking/test_setup.py

import numpy as np
import torch
from ultralytics import YOLO
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt

print("✓ All core imports successful!")

# Test YOLO loading
model = YOLO('/ssd_4TB/divake/temporal_uncertainty/models/yolov8n.pt')
print(f"✓ YOLO model loaded on device: {next(model.model.parameters()).device}")

# Test metadata loading
import json
metadata_path = '../metadata/raw_outputs/summary_all_sequences.json'
with open(metadata_path) as f:
    metadata = json.load(f)
print(f"✓ Metadata loaded: {len(metadata['all_sequences'])} sequences")

# Test KNN
X = np.random.randn(100, 10)
knn = NearestNeighbors(n_neighbors=5)
knn.fit(X)
print("✓ KNN works")

# Test KDE
kde = gaussian_kde(X.T)
print("✓ KDE works")

# Test Kalman
kf = KalmanFilter(dim_x=4, dim_z=2)
print("✓ Kalman filter works")

print("\n🎉 All systems ready! You can start V1 implementation.")
```

Run it:
```bash
cd /ssd_4TB/divake/temporal_uncertainty/conformal_tracking
python test_setup.py
```

---

## 📋 Final Checklist Before Starting V1

- [ ] PyTorch + Ultralytics installed and working
- [ ] YOLO models accessible
- [ ] MOT17 dataset accessible
- [ ] Metadata accessible
- [ ] FilterPy installed (for Kalman filter)
- [ ] Scikit-learn + Scipy working
- [ ] GPU accessible (check `nvidia-smi`)
- [ ] Test script runs without errors

**Once all checked → Ready to start Day 1!**

---

## 🆘 Common Issues

### Issue: "ModuleNotFoundError: No module named 'ultralytics'"
**Fix**: `pip install ultralytics`

### Issue: "CUDA out of memory"
**Fix**: Use smaller batch size or switch to CPU (slower but works)

### Issue: "filterpy not found"
**Fix**: `pip install filterpy`

### Issue: "Can't load YOLO model"
**Fix**: Check file path, ensure `.pt` file is not corrupted:
```bash
ls -lh /ssd_4TB/divake/temporal_uncertainty/models/yolov8n.pt
# Should be ~6.3M
```

---

**Next Step**: Once setup complete, create the implementation files and start Day 1!
