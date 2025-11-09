"""
Ground truth matching using Hungarian algorithm.
Dataset-agnostic IoU-based matching.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, Tuple


def compute_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU between two sets of boxes.

    Args:
        boxes1: [N, 4] boxes in [x, y, w, h] format
        boxes2: [M, 4] boxes in [x, y, w, h] format

    Returns:
        [N, M] IoU matrix
    """
    # Convert to [x1, y1, x2, y2]
    boxes1_x2y2 = np.zeros_like(boxes1)
    boxes1_x2y2[:, 0] = boxes1[:, 0]
    boxes1_x2y2[:, 1] = boxes1[:, 1]
    boxes1_x2y2[:, 2] = boxes1[:, 0] + boxes1[:, 2]
    boxes1_x2y2[:, 3] = boxes1[:, 1] + boxes1[:, 3]

    boxes2_x2y2 = np.zeros_like(boxes2)
    boxes2_x2y2[:, 0] = boxes2[:, 0]
    boxes2_x2y2[:, 1] = boxes2[:, 1]
    boxes2_x2y2[:, 2] = boxes2[:, 0] + boxes2[:, 2]
    boxes2_x2y2[:, 3] = boxes2[:, 1] + boxes2[:, 3]

    # Compute IoU
    iou_matrix = np.zeros((len(boxes1), len(boxes2)))

    for i, box1 in enumerate(boxes1_x2y2):
        for j, box2 in enumerate(boxes2_x2y2):
            # Intersection
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])

            if x2 < x1 or y2 < y1:
                continue

            intersection = (x2 - x1) * (y2 - y1)

            # Union
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection

            iou_matrix[i, j] = intersection / (union + 1e-10)

    return iou_matrix


def compute_center_error(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distance between box centers.

    Args:
        boxes1: [N, 4] boxes in [x, y, w, h]
        boxes2: [M, 4] boxes in [x, y, w, h]

    Returns:
        [N, M] center distance matrix
    """
    # Compute centers
    centers1 = boxes1[:, :2] + boxes1[:, 2:] / 2  # [N, 2]
    centers2 = boxes2[:, :2] + boxes2[:, 2:] / 2  # [M, 2]

    # Pairwise distances
    distances = np.zeros((len(centers1), len(centers2)))
    for i in range(len(centers1)):
        for j in range(len(centers2)):
            distances[i, j] = np.linalg.norm(centers1[i] - centers2[j])

    return distances


def match_detections_to_gt(
    det_bboxes: np.ndarray,
    gt_bboxes: np.ndarray,
    gt_data: np.ndarray,
    iou_threshold: float = 0.5
) -> Dict:
    """
    Match detections to ground truth using Hungarian algorithm.

    Args:
        det_bboxes: [N_det, 4] detection boxes [x, y, w, h]
        gt_bboxes: [N_gt, 4] ground truth boxes [x, y, w, h]
        gt_data: [N_gt, 9] full GT data (frame, id, bbox, conf, class, vis)
        iou_threshold: Minimum IoU for a valid match

    Returns:
        Dictionary with:
            - 'matched': Dict with matched detection info
            - 'unmatched_dets': Indices of unmatched detections (FP)
            - 'unmatched_gts': Indices of unmatched GT (FN)
    """
    if len(det_bboxes) == 0 or len(gt_bboxes) == 0:
        return {
            'matched': {
                'det_indices': np.array([], dtype=np.int32),
                'gt_indices': np.array([], dtype=np.int32),
                'iou': np.array([], dtype=np.float32),
                'center_error': np.array([], dtype=np.float64),
                'gt_bboxes': np.zeros((0, 4), dtype=np.float64),
                'gt_track_ids': np.array([], dtype=np.int32),
                'visibility': np.array([], dtype=np.float32),
            },
            'unmatched_dets': np.arange(len(det_bboxes), dtype=np.int32),
            'unmatched_gts': np.arange(len(gt_bboxes), dtype=np.int32),
        }

    # Compute IoU matrix
    iou_matrix = compute_iou(det_bboxes, gt_bboxes)

    # Hungarian matching (maximize IoU = minimize -IoU)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    # Filter matches by IoU threshold
    valid_matches = iou_matrix[row_ind, col_ind] >= iou_threshold
    det_matched = row_ind[valid_matches]
    gt_matched = col_ind[valid_matches]

    # Compute center errors for matched pairs
    center_errors = compute_center_error(
        det_bboxes[det_matched],
        gt_bboxes[gt_matched]
    )
    center_errors = np.diag(center_errors)  # Get diagonal (matched pairs)

    # Extract GT info
    matched_ious = iou_matrix[det_matched, gt_matched]
    matched_gt_bboxes = gt_bboxes[gt_matched]
    matched_gt_track_ids = gt_data[gt_matched, 1].astype(np.int32)  # Column 1 = track ID
    matched_visibility = gt_data[gt_matched, 8].astype(np.float32)  # Column 8 = visibility

    # Find unmatched
    all_det_indices = set(range(len(det_bboxes)))
    all_gt_indices = set(range(len(gt_bboxes)))
    matched_det_set = set(det_matched)
    matched_gt_set = set(gt_matched)

    unmatched_dets = np.array(list(all_det_indices - matched_det_set), dtype=np.int32)
    unmatched_gts = np.array(list(all_gt_indices - matched_gt_set), dtype=np.int32)

    return {
        'matched': {
            'det_indices': det_matched.astype(np.int32),
            'gt_indices': gt_matched.astype(np.int32),
            'iou': matched_ious.astype(np.float32),
            'center_error': center_errors.astype(np.float64),
            'gt_bboxes': matched_gt_bboxes.astype(np.float64),
            'gt_track_ids': matched_gt_track_ids,
            'visibility': matched_visibility,
        },
        'unmatched_dets': unmatched_dets,
        'unmatched_gts': unmatched_gts,
    }
