"""
YOLO feature extraction.
Model-agnostic interface for extracting features and detections.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
from ultralytics import YOLO


class YOLOFeatureExtractor:
    """
    Extract features and detections from YOLO model.
    Uses ROI pooling for efficiency (one forward pass per image).
    """

    def __init__(self, model_path: str, feature_layers: List[int], device: str = 'cuda'):
        """
        Args:
            model_path: Path to YOLO .pt file
            feature_layers: List of layer indices to extract features from
            device: 'cuda' or 'cpu'
        """
        self.model_path = model_path
        self.feature_layers = sorted(feature_layers)
        self.device = device if torch.cuda.is_available() else 'cpu'

        # Load YOLO model
        print(f"Loading YOLO model from {model_path}")
        self.model = YOLO(model_path)
        self.model.model.to(self.device)
        self.model.model.eval()

        # Register hooks to capture intermediate features
        self.feature_maps = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture features from specified layers."""

        def get_hook(layer_idx):
            def hook(module, input, output):
                self.feature_maps[layer_idx] = output
            return hook

        for layer_idx in self.feature_layers:
            if layer_idx < len(self.model.model.model):
                self.model.model.model[layer_idx].register_forward_hook(
                    get_hook(layer_idx)
                )

    def extract_detections_and_features(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.01
    ) -> Dict:
        """
        Run YOLO on image and extract detections + features.

        Args:
            image: RGB image [H, W, 3]
            conf_threshold: Minimum confidence threshold

        Returns:
            Dictionary with:
                - 'bboxes': [N, 4] bounding boxes [x, y, w, h]
                - 'confidences': [N] confidence scores
                - 'class_ids': [N] class IDs
                - 'features': Dict[layer_idx -> [N, D] features]
        """
        # Clear previous feature maps
        self.feature_maps = {}

        # Run YOLO (this triggers hooks)
        with torch.no_grad():
            results = self.model(image, conf=conf_threshold, verbose=False)

        # Extract detections
        boxes = results[0].boxes

        if len(boxes) == 0:
            # No detections
            return {
                'bboxes': np.zeros((0, 4), dtype=np.float64),
                'confidences': np.zeros((0,), dtype=np.float32),
                'class_ids': np.zeros((0,), dtype=np.int8),
                'features': {layer: np.zeros((0, 0), dtype=np.float32) for layer in self.feature_layers},
            }

        # Get bboxes in [x, y, w, h] format
        bboxes_xyxy = boxes.xyxy.cpu().numpy()  # [N, 4] in [x1, y1, x2, y2]
        bboxes_xywh = self._xyxy_to_xywh(bboxes_xyxy)

        confidences = boxes.conf.cpu().numpy().astype(np.float32)
        class_ids = boxes.cls.cpu().numpy().astype(np.int8)

        # Extract features for each bbox using ROI pooling
        features_dict = {}
        for layer_idx in self.feature_layers:
            if layer_idx in self.feature_maps:
                layer_features = self._roi_pool_features(
                    self.feature_maps[layer_idx],
                    bboxes_xyxy,
                    image.shape
                )
                features_dict[layer_idx] = layer_features

        return {
            'bboxes': bboxes_xywh.astype(np.float64),
            'confidences': confidences,
            'class_ids': class_ids,
            'features': features_dict,
        }

    def _xyxy_to_xywh(self, bboxes_xyxy: np.ndarray) -> np.ndarray:
        """Convert [x1, y1, x2, y2] to [x, y, w, h]."""
        bboxes_xywh = np.zeros_like(bboxes_xyxy)
        bboxes_xywh[:, 0] = bboxes_xyxy[:, 0]  # x = x1
        bboxes_xywh[:, 1] = bboxes_xyxy[:, 1]  # y = y1
        bboxes_xywh[:, 2] = bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]  # w = x2 - x1
        bboxes_xywh[:, 3] = bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]  # h = y2 - y1
        return bboxes_xywh

    def _roi_pool_features(
        self,
        feature_map: torch.Tensor,
        bboxes_xyxy: np.ndarray,
        image_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        ROI pooling: Extract features for each bbox region.

        Args:
            feature_map: [1, C, H, W] feature map from a layer
            bboxes_xyxy: [N, 4] bboxes in [x1, y1, x2, y2] format
            image_shape: (H, W, C) original image shape

        Returns:
            [N, C] features (global average pooled)
        """
        if feature_map is None or len(bboxes_xyxy) == 0:
            return np.zeros((0, 0), dtype=np.float32)

        # Get feature map dimensions
        feat_h, feat_w = feature_map.shape[2], feature_map.shape[3]
        img_h, img_w = image_shape[0], image_shape[1]

        # Scale bboxes to feature map coordinates
        scale_x = feat_w / img_w
        scale_y = feat_h / img_h

        features = []
        for bbox in bboxes_xyxy:
            x1, y1, x2, y2 = bbox

            # Map to feature coordinates
            fx1 = int(np.clip(x1 * scale_x, 0, feat_w - 1))
            fy1 = int(np.clip(y1 * scale_y, 0, feat_h - 1))
            fx2 = int(np.clip(x2 * scale_x, 1, feat_w))
            fy2 = int(np.clip(y2 * scale_y, 1, feat_h))

            # Ensure valid region
            if fx2 <= fx1:
                fx2 = fx1 + 1
            if fy2 <= fy1:
                fy2 = fy1 + 1

            # Extract region and global average pool
            region = feature_map[0, :, fy1:fy2, fx1:fx2]
            pooled = torch.nn.functional.adaptive_avg_pool2d(region, (1, 1))
            features.append(pooled.squeeze().cpu().numpy())

        return np.array(features, dtype=np.float32)
