"""
YOLOv8 Wrapper with MC Dropout
Loads YOLOv8 models and enables MC Dropout for uncertainty estimation
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from ultralytics import YOLO
import logging
import warnings

# Suppress YOLO warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class YOLOv8WithUncertainty:
    """YOLOv8 wrapper with MC Dropout uncertainty estimation"""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        enable_mc_dropout: bool = True,
        dropout_rate: float = 0.2
    ):
        """
        Initialize YOLOv8 with uncertainty capabilities

        Args:
            model_path: Path to YOLOv8 weights
            device: Device to run on
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            enable_mc_dropout: Whether to enable MC Dropout
            dropout_rate: Dropout rate for MC Dropout
        """
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.enable_mc_dropout = enable_mc_dropout
        self.dropout_rate = dropout_rate

        # Load model
        self.model = self._load_model(model_path)

        # Enable MC Dropout if requested
        if self.enable_mc_dropout:
            self._enable_dropout_layers()

        logger.info(f"Loaded YOLOv8 model from {model_path}")
        logger.info(f"MC Dropout: {self.enable_mc_dropout}, rate: {self.dropout_rate}")

    def _load_model(self, model_path: str) -> YOLO:
        """Load YOLOv8 model"""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = YOLO(model_path)
        model.to(self.device)

        return model

    def _enable_dropout_layers(self):
        """
        Enable dropout at test time for MC Dropout
        Since YOLOv8 doesn't have dropout layers, we inject them
        Borrowed concept from Bayesian-Neural-Networks/src/MC_dropout/model.py
        """
        # YOLOv8 doesn't have native dropout, so we add dropout after Conv layers
        if hasattr(self.model, 'model'):
            pytorch_model = self.model.model

            # Count existing dropout (should be 0)
            dropout_count = sum(1 for m in pytorch_model.modules()
                              if isinstance(m, (nn.Dropout, nn.Dropout2d)))

            if dropout_count == 0:
                logger.warning("YOLOv8 has no native dropout layers!")
                logger.info("Adding dropout layers after Conv2d layers for MC Dropout...")

                # Inject dropout after specific Conv2d layers in the backbone
                self._inject_dropout_layers(pytorch_model)
            else:
                # If model has dropout, enable them
                for module in pytorch_model.modules():
                    if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
                        module.p = self.dropout_rate
                        module.train()

    def _inject_dropout_layers(self, model):
        """
        Inject Dropout2d layers into YOLOv8 architecture
        Target: Add dropout after Conv layers in C2f blocks
        """
        dropout_added = 0

        # Add dropout to C2f modules (backbone feature extraction)
        for name, module in model.named_modules():
            if 'C2f' in type(module).__name__:
                # C2f has multiple Bottleneck blocks
                for bottleneck in module.m:  # m contains the bottleneck modules
                    if hasattr(bottleneck, 'cv2'):  # cv2 is the second conv in bottleneck
                        # Store original conv
                        original_conv = bottleneck.cv2

                        # Create sequential with conv + dropout
                        bottleneck.cv2 = nn.Sequential(
                            original_conv,
                            nn.Dropout2d(p=self.dropout_rate)
                        )
                        dropout_added += 1

        logger.info(f"Injected {dropout_added} Dropout2d layers into YOLOv8 model")

    def predict_single(
        self,
        image: np.ndarray,
        apply_dropout: bool = True
    ) -> Dict:
        """
        Single forward pass prediction

        Args:
            image: Input image (H, W, C)
            apply_dropout: Whether to apply dropout (for MC passes)

        Returns:
            Dictionary with predictions
        """
        # Set dropout mode
        if self.enable_mc_dropout and apply_dropout:
            self._set_dropout_mode(True)
        else:
            self._set_dropout_mode(False)

        # Run prediction
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )

        # Parse results
        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return {
                'boxes': np.array([]),
                'confidences': np.array([]),
                'classes': np.array([])
            }

        # Extract detections
        xyxy = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy()

        # Convert to [x, y, w, h] format
        xywh = np.zeros_like(xyxy)
        xywh[:, 0] = xyxy[:, 0]  # x
        xywh[:, 1] = xyxy[:, 1]  # y
        xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]  # w
        xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]  # h

        return {
            'boxes': xywh,
            'confidences': conf,
            'classes': cls.astype(int)
        }

    def predict_with_uncertainty(
        self,
        image: np.ndarray,
        num_forward_passes: int = 30,
        target_bbox: Optional[np.ndarray] = None,
        iou_threshold: float = 0.3
    ) -> Dict:
        """
        Multiple forward passes for uncertainty estimation
        Based on MC_dropout.sample_predict() from Bayesian-Neural-Networks

        Args:
            image: Input image (H, W, C)
            num_forward_passes: Number of MC Dropout passes
            target_bbox: Optional target bbox [x, y, w, h] to focus on

        Returns:
            Dictionary with predictions and uncertainty metrics
        """
        if not self.enable_mc_dropout:
            logger.warning("MC Dropout not enabled, returning single prediction")
            return self.predict_single(image, apply_dropout=False)

        # Collect predictions from multiple forward passes
        all_predictions = []

        for i in range(num_forward_passes):
            pred = self.predict_single(image, apply_dropout=True)
            all_predictions.append(pred)

            if (i + 1) % 10 == 0:
                logger.debug(f"Completed {i + 1}/{num_forward_passes} MC passes")

        # If target bbox provided, extract predictions for that region
        if target_bbox is not None:
            return self._compute_uncertainty_for_target(
                all_predictions, target_bbox, iou_threshold
            )
        else:
            return self._compute_uncertainty_global(all_predictions)

    def _compute_uncertainty_for_target(
        self,
        predictions: List[Dict],
        target_bbox: np.ndarray,
        iou_threshold: float = 0.3
    ) -> Dict:
        """
        Compute uncertainty for a specific target region

        Args:
            predictions: List of predictions from MC passes
            target_bbox: Target bbox [x, y, w, h]
            iou_threshold: Minimum IoU to consider a match

        Returns:
            Uncertainty metrics for target
        """
        matched_boxes = []
        matched_confs = []

        for pred in predictions:
            if len(pred['boxes']) == 0:
                continue

            # Find best matching box
            ious = self._compute_iou(target_bbox, pred['boxes'])
            if len(ious) > 0 and np.max(ious) > iou_threshold:
                best_idx = np.argmax(ious)
                matched_boxes.append(pred['boxes'][best_idx])
                matched_confs.append(pred['confidences'][best_idx])

        if len(matched_boxes) == 0:
            return {
                'found': False,
                'num_detections': 0,
                'target_bbox': target_bbox
            }

        matched_boxes = np.array(matched_boxes)
        matched_confs = np.array(matched_confs)

        # Compute variances
        bbox_mean = np.mean(matched_boxes, axis=0)
        bbox_var = np.var(matched_boxes, axis=0)
        conf_mean = np.mean(matched_confs)
        conf_var = np.var(matched_confs)

        # Combined uncertainty (weighted sum)
        bbox_uncertainty = np.mean(bbox_var)  # Average variance across x, y, w, h
        combined_uncertainty = 0.7 * bbox_uncertainty + 0.3 * conf_var

        return {
            'found': True,
            'num_detections': len(matched_boxes),
            'detection_rate': len(matched_boxes) / len(predictions),
            'bbox_mean': bbox_mean,
            'bbox_variance': bbox_var,
            'bbox_std': np.sqrt(bbox_var),
            'confidence_mean': conf_mean,
            'confidence_variance': conf_var,
            'confidence_std': np.sqrt(conf_var),
            'bbox_uncertainty': bbox_uncertainty,
            'combined_uncertainty': combined_uncertainty,
            'target_bbox': target_bbox,
            'all_boxes': matched_boxes,
            'all_confidences': matched_confs
        }

    def _compute_uncertainty_global(self, predictions: List[Dict]) -> Dict:
        """
        Compute global uncertainty metrics

        Args:
            predictions: List of predictions from MC passes

        Returns:
            Global uncertainty metrics
        """
        # Collect all detections
        all_boxes = []
        all_confs = []
        num_detections = []

        for pred in predictions:
            if len(pred['boxes']) > 0:
                all_boxes.append(pred['boxes'])
                all_confs.append(pred['confidences'])
            num_detections.append(len(pred['boxes']))

        if len(all_boxes) == 0:
            return {
                'num_detections_mean': 0,
                'num_detections_std': 0,
                'no_detections': True
            }

        # Statistics on number of detections
        num_det_mean = np.mean(num_detections)
        num_det_std = np.std(num_detections)

        # Aggregate confidence statistics
        all_confs_flat = np.concatenate(all_confs)
        conf_mean = np.mean(all_confs_flat)
        conf_std = np.std(all_confs_flat)

        return {
            'num_detections_mean': num_det_mean,
            'num_detections_std': num_det_std,
            'confidence_mean': conf_mean,
            'confidence_std': conf_std,
            'total_predictions': len(predictions),
            'predictions_with_detections': len(all_boxes)
        }

    def _compute_iou(self, box1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """
        Compute IoU between one box and multiple boxes

        Args:
            box1: Single box [x, y, w, h]
            boxes2: Multiple boxes [N, 4]

        Returns:
            IoU values [N]
        """
        # Convert to [x1, y1, x2, y2]
        b1_x1, b1_y1 = box1[0], box1[1]
        b1_x2, b1_y2 = box1[0] + box1[2], box1[1] + box1[3]

        b2_x1 = boxes2[:, 0]
        b2_y1 = boxes2[:, 1]
        b2_x2 = boxes2[:, 0] + boxes2[:, 2]
        b2_y2 = boxes2[:, 1] + boxes2[:, 3]

        # Intersection
        inter_x1 = np.maximum(b1_x1, b2_x1)
        inter_y1 = np.maximum(b1_y1, b2_y1)
        inter_x2 = np.minimum(b1_x2, b2_x2)
        inter_y2 = np.minimum(b1_y2, b2_y2)

        inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)

        # Union
        b1_area = box1[2] * box1[3]
        b2_area = boxes2[:, 2] * boxes2[:, 3]
        union_area = b1_area + b2_area - inter_area

        # IoU
        iou = inter_area / (union_area + 1e-6)

        return iou

    def _set_dropout_mode(self, enable: bool):
        """Set dropout mode for MC passes"""
        if not self.enable_mc_dropout:
            return

        if hasattr(self.model, 'model'):
            for module in self.model.model.modules():
                if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
                    if enable:
                        module.train()
                    else:
                        module.eval()


if __name__ == "__main__":
    # Test the wrapper
    import sys
    sys.path.append('/ssd_4TB/divake/temporal_uncertainty')

    logging.basicConfig(level=logging.INFO)

    # Load model
    model_path = "/ssd_4TB/divake/temporal_uncertainty/models/yolov8n.pt"
    yolo = YOLOv8WithUncertainty(
        model_path,
        device="cuda:0",
        enable_mc_dropout=True,
        dropout_rate=0.2
    )

    # Test on dummy image
    dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
    dummy_image[100:200, 100:200] = 255  # White box

    # Single prediction
    result = yolo.predict_single(dummy_image)
    print(f"Single prediction: {len(result['boxes'])} detections")

    # MC Dropout prediction
    uncertainty_result = yolo.predict_with_uncertainty(
        dummy_image,
        num_forward_passes=10,
        target_bbox=np.array([100, 100, 100, 100])
    )

    if uncertainty_result.get('found'):
        print(f"Target found in {uncertainty_result['num_detections']}/{10} passes")
        print(f"Bbox variance: {uncertainty_result.get('bbox_variance')}")
        print(f"Combined uncertainty: {uncertainty_result.get('combined_uncertainty'):.4f}")
    else:
        print("Target not found")