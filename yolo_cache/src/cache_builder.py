"""
Main cache building logic.
Orchestrates dataset loading, YOLO extraction, GT matching, and saving.
"""

import numpy as np
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import json
from datetime import datetime

try:
    from .dataset_loader import DatasetLoaderFactory
    from .yolo_extractor import YOLOFeatureExtractor
    from .gt_matcher import match_detections_to_gt
    from .config import CACHE_CONFIG
except ImportError:
    from dataset_loader import DatasetLoaderFactory
    from yolo_extractor import YOLOFeatureExtractor
    from gt_matcher import match_detections_to_gt
    from config import CACHE_CONFIG


class CacheBuilder:
    """
    Build cache for a single sequence.
    Dataset-agnostic and model-agnostic.
    """

    def __init__(
        self,
        dataset_type: str,
        sequence_path: Path,
        dataset_config: Dict,
        yolo_extractor: YOLOFeatureExtractor,
        yolo_config: Dict
    ):
        """
        Args:
            dataset_type: 'mot17', 'kitti', etc.
            sequence_path: Path to sequence directory
            dataset_config: Dataset configuration
            yolo_extractor: Initialized YOLO feature extractor
            yolo_config: YOLO model configuration
        """
        self.dataset_type = dataset_type
        self.dataset_config = dataset_config
        self.yolo_extractor = yolo_extractor
        self.yolo_config = yolo_config

        # Create dataset loader
        self.loader = DatasetLoaderFactory.create_loader(
            dataset_type,
            sequence_path,
            dataset_config
        )

        print(f"\nInitialized CacheBuilder for {self.loader.sequence_name}")
        print(f"  Frames: {self.loader.seqinfo['seq_length']}")
        print(f"  Resolution: {self.loader.seqinfo['im_width']}x{self.loader.seqinfo['im_height']}")

    def build(self) -> Dict:
        """
        Build complete cache for this sequence.

        Returns:
            Dictionary with all cached data
        """
        print(f"\n{'='*60}")
        print(f"Building cache for {self.loader.sequence_name}")
        print(f"{'='*60}")

        # Load ground truth
        print("\n[1/4] Loading ground truth...")
        gt_data = self.loader.load_ground_truth()
        print(f"  Loaded {len(gt_data)} ground truth boxes (valid pedestrians)")

        # Extract YOLO detections and features
        print("\n[2/4] Running YOLO and extracting features...")
        detections_data = self._extract_all_detections()

        # Match to ground truth
        print("\n[3/4] Matching detections to ground truth...")
        matching_data = self._match_to_ground_truth(detections_data, gt_data)

        # Build final cache structure
        print("\n[4/4] Building final cache structure...")
        cache = self._build_cache_structure(detections_data, matching_data, gt_data)

        print(f"\n✓ Cache built successfully!")
        print(f"  Total detections: {len(cache['detections']['frame_ids'])}")
        print(f"  Matched to GT: {len(cache['gt_matching']['det_indices'])}")
        print(f"  False positives: {len(cache['unmatched']['fp_det_indices'])}")
        print(f"  False negatives: {len(cache['unmatched']['fn_gt_indices'])}")

        return cache

    def _extract_all_detections(self) -> Dict:
        """Extract YOLO detections and features for all frames."""
        image_paths = self.loader.get_image_paths()

        all_bboxes = []
        all_confidences = []
        all_class_ids = []
        all_frame_ids = []
        all_features = {layer: [] for layer in self.yolo_config['feature_layers']}

        # Process each frame
        for frame_num, img_path in enumerate(tqdm(image_paths, desc="  Processing frames"), start=1):
            # Load image
            image = self.loader.load_image(frame_num)

            # Run YOLO with lowest threshold
            min_conf = min(CACHE_CONFIG['confidence_thresholds'])
            result = self.yolo_extractor.extract_detections_and_features(
                image,
                conf_threshold=min_conf
            )

            # Filter to person class only (YOLO class 0)
            person_mask = (result['class_ids'] == self.yolo_config['coco_person_class'])

            if person_mask.sum() == 0:
                continue  # No person detections in this frame

            # Store detections
            n_dets = person_mask.sum()
            all_bboxes.append(result['bboxes'][person_mask])
            all_confidences.append(result['confidences'][person_mask])
            all_class_ids.append(result['class_ids'][person_mask])
            all_frame_ids.extend([frame_num] * n_dets)

            # Store features
            for layer in self.yolo_config['feature_layers']:
                if layer in result['features']:
                    all_features[layer].append(result['features'][layer][person_mask])

        # Concatenate all
        all_bboxes = np.vstack(all_bboxes) if all_bboxes else np.zeros((0, 4))
        all_confidences = np.concatenate(all_confidences) if all_confidences else np.array([])
        all_class_ids = np.concatenate(all_class_ids) if all_class_ids else np.array([])
        all_frame_ids = np.array(all_frame_ids, dtype=np.int32)

        for layer in self.yolo_config['feature_layers']:
            if all_features[layer]:
                all_features[layer] = np.vstack(all_features[layer])
            else:
                all_features[layer] = np.zeros((0, self.yolo_config['feature_dims'][layer]))

        return {
            'bboxes': all_bboxes,
            'confidences': all_confidences,
            'class_ids': all_class_ids,
            'frame_ids': all_frame_ids,
            'features': all_features,
        }

    def _match_to_ground_truth(self, detections_data: Dict, gt_data: np.ndarray) -> Dict:
        """Match detections to ground truth per frame."""
        # Group detections and GT by frame
        unique_frames = np.unique(detections_data['frame_ids'])

        all_matched = {
            'det_indices': [],
            'gt_indices': [],
            'iou': [],
            'center_error': [],
            'gt_bboxes': [],
            'gt_track_ids': [],
            'visibility': [],
        }
        all_fp_indices = []
        all_fn_gt_indices = []

        current_det_offset = 0
        current_gt_offset = 0

        for frame_num in tqdm(unique_frames, desc="  Matching frames"):
            # Get detections for this frame
            det_mask = (detections_data['frame_ids'] == frame_num)
            det_indices_in_frame = np.where(det_mask)[0]
            frame_det_bboxes = detections_data['bboxes'][det_mask]

            # Get GT for this frame
            gt_mask = (gt_data[:, 0] == frame_num)
            gt_indices_in_frame = np.where(gt_mask)[0]
            frame_gt_data = gt_data[gt_mask]
            frame_gt_bboxes = frame_gt_data[:, 2:6]  # Columns 2-5 are bbox

            # Match
            match_result = match_detections_to_gt(
                frame_det_bboxes,
                frame_gt_bboxes,
                frame_gt_data,
                iou_threshold=CACHE_CONFIG['iou_threshold']
            )

            # Convert frame-local indices to global indices
            matched = match_result['matched']
            if len(matched['det_indices']) > 0:
                all_matched['det_indices'].append(det_indices_in_frame[matched['det_indices']])
                all_matched['gt_indices'].append(matched['gt_indices'] + current_gt_offset)
                all_matched['iou'].append(matched['iou'])
                all_matched['center_error'].append(matched['center_error'])
                all_matched['gt_bboxes'].append(matched['gt_bboxes'])
                all_matched['gt_track_ids'].append(matched['gt_track_ids'])
                all_matched['visibility'].append(matched['visibility'])

            # False positives (global detection indices)
            if len(match_result['unmatched_dets']) > 0:
                all_fp_indices.append(det_indices_in_frame[match_result['unmatched_dets']])

            # False negatives (global GT indices)
            if len(match_result['unmatched_gts']) > 0:
                # Convert to array for consistency
                fn_gt_global = match_result['unmatched_gts'] + current_gt_offset
                all_fn_gt_indices.append(fn_gt_global if isinstance(fn_gt_global, np.ndarray) else np.array(fn_gt_global, dtype=np.int32))

            current_gt_offset += len(frame_gt_bboxes)

        # Concatenate all matched data
        for key in all_matched:
            if len(all_matched[key]) > 0:
                all_matched[key] = np.concatenate(all_matched[key])
            else:
                # Empty arrays with correct dtype
                if key in ['det_indices', 'gt_indices', 'gt_track_ids']:
                    all_matched[key] = np.array([], dtype=np.int32)
                elif key in ['center_error']:
                    all_matched[key] = np.array([], dtype=np.float64)
                elif key in ['iou', 'visibility']:
                    all_matched[key] = np.array([], dtype=np.float32)
                elif key == 'gt_bboxes':
                    all_matched[key] = np.zeros((0, 4), dtype=np.float64)

        # Concatenate unmatched
        fp_indices = np.concatenate(all_fp_indices) if all_fp_indices else np.array([], dtype=np.int32)
        fn_indices = np.concatenate(all_fn_gt_indices) if all_fn_gt_indices else np.array([], dtype=np.int32)

        return {
            'matched': all_matched,
            'fp_indices': fp_indices,
            'fn_indices': fn_indices,
        }

    def _build_cache_structure(
        self,
        detections_data: Dict,
        matching_data: Dict,
        gt_data: np.ndarray
    ) -> Dict:
        """Build final extensible cache structure."""

        # Compute threshold indices
        threshold_indices = {}
        for threshold in CACHE_CONFIG['confidence_thresholds']:
            indices = np.where(detections_data['confidences'] >= threshold)[0]
            threshold_indices[f'conf_{threshold}'] = indices.astype(np.int32)

        # Build cache
        cache = {
            # Core detections
            'detections': {
                'frame_ids': detections_data['frame_ids'].astype(np.int32),
                'bboxes': detections_data['bboxes'].astype(np.float64),
                'confidences': detections_data['confidences'].astype(np.float32),
                'class_ids': detections_data['class_ids'].astype(np.int8),
            },

            # Multi-layer features
            'features': {
                f'layer_{layer}': detections_data['features'][layer].astype(np.float32)
                for layer in self.yolo_config['feature_layers']
            },

            # Threshold indices
            'thresholds': threshold_indices,

            # GT matching
            'gt_matching': matching_data['matched'],

            # Unmatched
            'unmatched': {
                'fp_det_indices': matching_data['fp_indices'],
                'fn_gt_indices': matching_data['fn_indices'],
                'fn_gt_bboxes': gt_data[matching_data['fn_indices'], 2:6].astype(np.float64) if len(matching_data['fn_indices']) > 0 else np.zeros((0, 4), dtype=np.float64),
            },

            # Frame info
            'frames': {
                'frame_ids': np.arange(1, self.loader.seqinfo['seq_length'] + 1, dtype=np.int32),
                'num_detections': np.array([
                    (detections_data['frame_ids'] == f).sum()
                    for f in range(1, self.loader.seqinfo['seq_length'] + 1)
                ], dtype=np.int32),
            },

            # Link to existing metadata
            'existing_metadata': {
                'metadata_path': str(self.loader.get_metadata_path()) if self.loader.get_metadata_path() else '',
            },

            # Extension placeholders
            'v2_extensions': {},
            'v3_extensions': {},

            # Metadata
            'meta': {
                'version': CACHE_CONFIG['cache_version'],
                'sequence': self.loader.sequence_name,
                'dataset_type': self.dataset_type,
                'yolo_model': Path(self.yolo_extractor.model_path).stem,
                'cached_layers': self.yolo_config['feature_layers'],
                'cached_thresholds': CACHE_CONFIG['confidence_thresholds'],
                'iou_threshold': CACHE_CONFIG['iou_threshold'],
                'date_cached': datetime.now().isoformat(),
                'total_frames': int(self.loader.seqinfo['seq_length']),
                'total_detections': int(len(detections_data['frame_ids'])),
                'total_matched': int(len(matching_data['matched']['det_indices'])),
                'class_mapping': {
                    'yolo_person': 0,
                    'mot17_pedestrian': 1,
                },
            }
        }

        return cache

    def save(self, cache: Dict, output_path: Path):
        """Save cache to compressed NPZ file."""
        print(f"\nSaving cache to {output_path}")

        # Flatten nested dictionaries for NPZ
        flat_cache = {}
        for key, value in cache.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict):
                        # Triple nested (e.g., features/layer_X)
                        for subsubkey, subsubvalue in subvalue.items():
                            flat_cache[f'{key}/{subkey}/{subsubkey}'] = subsubvalue
                    else:
                        flat_cache[f'{key}/{subkey}'] = subvalue
            else:
                flat_cache[key] = value

        # Save with compression
        np.savez_compressed(output_path, **flat_cache)

        # Save summary JSON
        summary_path = output_path.with_suffix('.json')
        summary = {
            'sequence': cache['meta']['sequence'],
            'total_detections': cache['meta']['total_detections'],
            'total_matched': cache['meta']['total_matched'],
            'total_frames': cache['meta']['total_frames'],
            'cached_layers': cache['meta']['cached_layers'],
            'date_cached': cache['meta']['date_cached'],
            'file_size_mb': output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0,
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"✓ Saved cache ({summary['file_size_mb']:.1f} MB)")
        print(f"✓ Saved summary to {summary_path}")
