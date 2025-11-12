"""
DINO Model Wrapper for Integration with Uncertainty Pipeline

This wrapper provides a YOLO-like interface for the DINO transformer detector,
allowing it to be used in the same pipeline as YOLO models.
"""

import sys
import torch
import numpy as np
from pathlib import Path
import cv2
from PIL import Image

# Add DINO repo to path
DINO_REPO = Path("/ssd_4TB/divake/temporal_uncertainty/models/dino_repo")
sys.path.insert(0, str(DINO_REPO))

from models import build_model
from util.slconfig import SLConfig
from util import box_ops


class DINOWrapper:
    """Wrapper to make DINO work like YOLO for our pipeline"""

    def __init__(self, checkpoint_path, config_path=None, device='cuda'):
        self.device = device

        # Default config for 4scale model
        if config_path is None:
            config_path = DINO_REPO / "config" / "DINO" / "DINO_4scale.py"

        # Load config
        self.cfg = SLConfig.fromfile(str(config_path))
        self.cfg.device = device

        # Build model
        model, criterion, postprocessors = build_model(self.cfg)
        self.model = model.to(device)
        self.postprocessor = postprocessors['bbox']

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.model.eval()

        print(f"✓ DINO loaded from {checkpoint_path}")

    def preprocess_image(self, image_path):
        """Preprocess image for DINO"""
        # Load image
        img = Image.open(image_path).convert('RGB')
        w, h = img.size

        # Resize to standard size (keep aspect ratio)
        target_size = 800
        scale = target_size / min(h, w)
        if max(h, w) * scale > 1333:
            scale = 1333 / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        img_resized = img.resize((new_w, new_h), Image.BILINEAR)

        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(np.array(img_resized)).float()
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
        img_tensor /= 255.0

        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        return img_tensor.unsqueeze(0).to(self.device), (h, w), (new_h, new_w)

    @torch.no_grad()
    def __call__(self, image_path, conf_threshold=0.3, verbose=False, **kwargs):
        """
        Run DINO inference on an image

        Args:
            image_path: Path to image
            conf_threshold: Confidence threshold
            verbose: Ignored (for YOLO compatibility)
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            List containing Results object (YOLO-compatible format)
        """
        # Preprocess
        img_tensor, orig_size, new_size = self.preprocess_image(image_path)

        # Run model
        outputs = self.model(img_tensor)

        # Post-process
        orig_target_sizes = torch.tensor([orig_size], device=self.device)
        results = self.postprocessor(outputs, orig_target_sizes)[0]

        # Extract predictions
        scores = results['scores'].cpu().numpy()
        labels = results['labels'].cpu().numpy()
        boxes = results['boxes'].cpu().numpy()  # [x1, y1, x2, y2]

        # Filter by confidence
        keep = scores >= conf_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # Create YOLO-compatible result object
        result = DINOResult(boxes, scores, labels, orig_size)

        return [result]  # Return as list for YOLO compatibility

    def __repr__(self):
        return f"DINOWrapper(device={self.device})"


class DINOResult:
    """YOLO-compatible result object for DINO predictions"""

    def __init__(self, boxes, scores, labels, orig_size):
        """
        Args:
            boxes: numpy array of shape [N, 4] in [x1, y1, x2, y2] format
            scores: numpy array of shape [N]
            labels: numpy array of shape [N]
            orig_size: tuple (H, W)
        """
        self.boxes_xyxy = boxes
        self.scores = scores
        self.labels = labels
        self.orig_size = orig_size

        # Create boxes object (YOLO-compatible)
        self.boxes = self._create_boxes_object()

    def _create_boxes_object(self):
        """Create a boxes object similar to YOLO"""
        class Boxes:
            def __init__(self, boxes, scores):
                self.xyxy = boxes
                self.conf = scores
                self.data = np.concatenate([boxes, scores.reshape(-1, 1)], axis=1)

            def __len__(self):
                return len(self.xyxy)

        return Boxes(self.boxes_xyxy, self.scores)

    def __len__(self):
        return len(self.boxes_xyxy)


def load_dino_model(checkpoint_path, config_path=None, device='cuda'):
    """
    Load DINO model with YOLO-compatible interface

    Args:
        checkpoint_path: Path to DINO checkpoint (.pth)
        config_path: Path to config file (optional)
        device: Device to load on

    Returns:
        DINOWrapper instance
    """
    return DINOWrapper(checkpoint_path, config_path, device)


# Test function
if __name__ == "__main__":
    # Test loading
    checkpoint = "/ssd_4TB/divake/temporal_uncertainty/models/dino/DINO_models/checkpoint0011_4scale.pth"
    model = load_dino_model(checkpoint)
    print(f"✓ DINO model loaded successfully")
    print(f"Model: {model}")
