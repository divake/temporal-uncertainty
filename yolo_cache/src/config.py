"""
Configuration for YOLO caching.
Centralized config makes it easy to adapt to new datasets/models.
"""

from pathlib import Path
from typing import Dict, List

# Paths
PROJECT_ROOT = Path("/ssd_4TB/divake/temporal_uncertainty")
YOLO_CACHE_ROOT = PROJECT_ROOT / "yolo_cache"
MOT17_DATA_ROOT = PROJECT_ROOT / "data" / "MOT17" / "train"
MOT20_DATA_ROOT = PROJECT_ROOT / "data" / "MOT20" / "train"
DANCETRACK_DATA_ROOT = PROJECT_ROOT / "data" / "DanceTrack" / "train1"
METADATA_ROOT = PROJECT_ROOT / "metadata"
YOLO_MODELS_ROOT = PROJECT_ROOT / "models"

# YOLO Configuration
YOLO_CONFIG = {
    'yolov8n': {
        'model_path': YOLO_MODELS_ROOT / 'yolov8n.pt',
        'feature_layers': [4, 9, 15, 21],  # Layers to cache
        'feature_dims': {
            4: 128,
            9: 256,
            15: 512,
            21: 1024,
        },
        'coco_person_class': 0,  # COCO class ID for person
    },
    'yolov8s': {
        'model_path': YOLO_MODELS_ROOT / 'yolov8s.pt',
        'feature_layers': [4, 9, 15, 21],
        'feature_dims': {
            4: 128,
            9: 256,
            15: 512,
            21: 1024,
        },
        'coco_person_class': 0,
    },
    'yolov8m': {
        'model_path': YOLO_MODELS_ROOT / 'yolov8m.pt',
        'feature_layers': [4, 9, 15, 21],
        'feature_dims': {
            4: 128,
            9: 256,
            15: 512,
            21: 1024,
        },
        'coco_person_class': 0,
    },
    'yolov8l': {
        'model_path': YOLO_MODELS_ROOT / 'yolov8l.pt',
        'feature_layers': [4, 9, 15, 21],
        'feature_dims': {
            4: 128,
            9: 256,
            15: 512,
            21: 1024,
        },
        'coco_person_class': 0,
    },
    'yolov8x': {
        'model_path': YOLO_MODELS_ROOT / 'yolov8x.pt',
        'feature_layers': [4, 9, 15, 21],
        'feature_dims': {
            4: 128,
            9: 256,
            15: 512,
            21: 1024,
        },
        'coco_person_class': 0,
    }
}

# Dataset Configuration
DATASET_CONFIG = {
    'mot17': {
        'data_root': MOT17_DATA_ROOT,
        'sequences': [
            'MOT17-02-FRCNN',
            'MOT17-04-FRCNN',
            'MOT17-11-FRCNN',  # Only the 3 sequences used in paper
        ],
        'gt_format': {
            'frame': 0,
            'track_id': 1,
            'bb_left': 2,
            'bb_top': 3,
            'bb_width': 4,
            'bb_height': 5,
            'conf': 6,
            'class': 7,
            'visibility': 8,
        },
        'valid_conf': 1,  # conf==1 means consider
        'pedestrian_class': 1,  # class==1 is pedestrian
        'image_dir': 'img1',
        'gt_file': 'gt/gt.txt',
        'seqinfo_file': 'seqinfo.ini',
    },
    'mot20': {
        'data_root': MOT20_DATA_ROOT,
        'sequences': [
            'MOT20-05',  # Only sequence used in paper
        ],
        'gt_format': {
            'frame': 0,
            'track_id': 1,
            'bb_left': 2,
            'bb_top': 3,
            'bb_width': 4,
            'bb_height': 5,
            'conf': 6,
            'class': 7,
            'visibility': 8,
        },
        'valid_conf': 1,
        'pedestrian_class': 1,
        'image_dir': 'img1',
        'gt_file': 'gt/gt.txt',
        'seqinfo_file': 'seqinfo.ini',
    },
    'dancetrack': {
        'data_root': DANCETRACK_DATA_ROOT,
        'sequences': [
            'dancetrack0019',  # Only sequence used in paper
        ],
        'gt_format': {
            'frame': 0,
            'track_id': 1,
            'bb_left': 2,
            'bb_top': 3,
            'bb_width': 4,
            'bb_height': 5,
            'conf': 6,
            'class': 7,
            'visibility': 8,
        },
        'valid_conf': 1,
        'pedestrian_class': 1,
        'image_dir': 'img1',
        'gt_file': 'gt/gt.txt',
        'seqinfo_file': 'seqinfo.ini',
    }
}

# Caching Configuration
CACHE_CONFIG = {
    'confidence_thresholds': [0.01, 0.3, 0.5, 0.7],
    'iou_threshold': 0.3,  # For GT matching (changed from 0.5 to include IoU 0.3-0.5 range)
    'dtypes': {
        'bboxes': 'float64',  # High precision
        'features': 'float32',  # Standard
        'center_error': 'float64',  # High precision (our target y!)
        'confidences': 'float32',
        'frame_ids': 'int32',
        'track_ids': 'int32',
        'class_ids': 'int8',
    },
    'cache_version': '1.0',
}

# Output paths
def get_cache_output_path(dataset: str, model: str) -> Path:
    """Get output directory for cache files."""
    output_dir = YOLO_CACHE_ROOT / "data" / dataset / model
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def get_validation_output_path() -> Path:
    """Get output directory for validation reports."""
    output_dir = YOLO_CACHE_ROOT / "validation_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
