"""
Dataset-agnostic data loading.
Each dataset needs a simple adapter class.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import configparser


class MOT17Loader:
    """Load MOT17 sequence data."""

    def __init__(self, sequence_path: Path, config: Dict):
        self.sequence_path = Path(sequence_path)
        self.config = config
        self.sequence_name = self.sequence_path.name

        # Load sequence info
        self.seqinfo = self._load_seqinfo()

    def _load_seqinfo(self) -> Dict:
        """Parse seqinfo.ini file."""
        seqinfo_path = self.sequence_path / self.config['seqinfo_file']
        config = configparser.ConfigParser()
        config.read(seqinfo_path)

        return {
            'name': config.get('Sequence', 'name'),
            'seq_length': int(config.get('Sequence', 'seqLength')),
            'im_width': int(config.get('Sequence', 'imWidth')),
            'im_height': int(config.get('Sequence', 'imHeight')),
            'im_ext': config.get('Sequence', 'imExt'),
            'frame_rate': int(config.get('Sequence', 'frameRate')),
        }

    def load_ground_truth(self) -> np.ndarray:
        """
        Load ground truth annotations.

        Returns:
            Array of shape [N, 9] with columns:
            [frame, track_id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility]
        """
        gt_path = self.sequence_path / self.config['gt_file']
        gt_data = np.loadtxt(gt_path, delimiter=',')

        # Filter to valid pedestrians only (conf==1, class==1)
        valid_mask = (
            (gt_data[:, self.config['gt_format']['conf']] == self.config['valid_conf']) &
            (gt_data[:, self.config['gt_format']['class']] == self.config['pedestrian_class'])
        )

        return gt_data[valid_mask]

    def get_image_paths(self) -> List[Path]:
        """Get list of image paths in order."""
        img_dir = self.sequence_path / self.config['image_dir']

        # MOT17 uses 6-digit frame numbers
        image_paths = []
        for frame_num in range(1, self.seqinfo['seq_length'] + 1):
            img_path = img_dir / f"{frame_num:06d}{self.seqinfo['im_ext']}"
            if img_path.exists():
                image_paths.append(img_path)

        return image_paths

    def load_image(self, frame_num: int) -> np.ndarray:
        """
        Load image for a specific frame.

        Args:
            frame_num: 1-indexed frame number

        Returns:
            RGB image as numpy array [H, W, 3]
        """
        img_dir = self.sequence_path / self.config['image_dir']
        img_path = img_dir / f"{frame_num:06d}{self.seqinfo['im_ext']}"

        image = Image.open(img_path)
        return np.array(image)

    def get_metadata_path(self) -> Path:
        """Get path to existing metadata JSON."""
        try:
            from .config import METADATA_ROOT
        except ImportError:
            from config import METADATA_ROOT

        # Extract sequence number (e.g., '02' from 'MOT17-02-FRCNN')
        seq_num = self.sequence_name.split('-')[1]
        metadata_path = METADATA_ROOT / "raw_outputs" / f"seq{seq_num}_metadata.json"

        if metadata_path.exists():
            return metadata_path
        return None


class DatasetLoaderFactory:
    """Factory to create dataset loaders based on dataset type."""

    @staticmethod
    def create_loader(dataset_type: str, sequence_path: Path, config: Dict):
        """
        Create appropriate loader for dataset type.

        Args:
            dataset_type: 'mot17', 'kitti', etc.
            sequence_path: Path to sequence directory
            config: Dataset configuration dict

        Returns:
            Dataset loader instance
        """
        if dataset_type == 'mot17':
            return MOT17Loader(sequence_path, config)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
