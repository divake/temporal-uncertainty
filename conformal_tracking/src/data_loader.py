"""
Simple YOLO Cache Data Loader
==============================

Loads pre-computed YOLO detections and features from .npz cache files.
"""

import numpy as np
from pathlib import Path


class YOLOCacheLoader:
    """Load data from YOLO cache .npz files."""

    def __init__(self, cache_path):
        """
        Args:
            cache_path: Path to .npz cache file
        """
        self.cache_path = Path(cache_path)
        self.cache = np.load(str(cache_path))
        self.sequence_name = self.cache_path.stem

    def get_features(self, layer_id=9):
        """
        Get features from specified layer.

        Args:
            layer_id: Layer number (4, 9, 15, or 21)

        Returns:
            features: [N, D] array
        """
        key = f'features/layer_{layer_id}'
        if key not in self.cache:
            raise ValueError(f"Layer {layer_id} not found in cache. Available: {list(self.cache.keys())}")
        return self.cache[key]

    def get_center_errors(self):
        """Get center errors (distance from detection to GT)."""
        return self.cache['gt_matching/center_error']

    def get_confidences(self):
        """Get detection confidences."""
        return self.cache['detections/confidences']

    def get_matched_mask(self):
        """Get mask of detections matched to ground truth."""
        # Create matched mask from det_indices
        n_detections = len(self.cache['detections/confidences'])
        matched = np.zeros(n_detections, dtype=bool)
        det_indices = self.cache['gt_matching/det_indices']
        matched[det_indices] = True
        return matched

    def get_ious(self):
        """Get IoU scores with ground truth."""
        return self.cache['gt_matching/iou']

    def get_frame_ids(self):
        """Get frame IDs for each detection."""
        return self.cache['detections/frame_id']

    def get_bboxes(self):
        """Get bounding boxes [N, 4] in xyxy format."""
        return self.cache['detections/bboxes']

    def __repr__(self):
        n_detections = len(self.get_confidences())
        n_matched = self.get_matched_mask().sum()
        return f"YOLOCacheLoader('{self.sequence_name}', {n_detections} dets, {n_matched} matched)"
