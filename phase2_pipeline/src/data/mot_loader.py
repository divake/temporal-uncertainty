"""
MOT17 Dataset Loader
Generic, reusable loader for MOT17 sequences
Write once, never touch again
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import logging

logger = logging.getLogger(__name__)


class MOT17Sequence:
    """Load and manage a MOT17 sequence"""

    def __init__(self, sequence_path: str, sequence_name: str = None):
        """
        Initialize MOT17 sequence loader

        Args:
            sequence_path: Path to sequence directory (e.g., /path/to/MOT17-11-FRCNN)
            sequence_name: Optional sequence name override
        """
        self.sequence_path = Path(sequence_path)

        # Validate path exists
        if not self.sequence_path.exists():
            raise FileNotFoundError(
                f"Sequence not found at {sequence_path}\n"
                f"Expected structure: {sequence_path}/img1/*.jpg\n"
                f"Please ensure MOT17 dataset is properly extracted"
            )

        self.sequence_name = sequence_name or self.sequence_path.name
        self.img_dir = self.sequence_path / "img1"
        self.gt_path = self.sequence_path / "gt" / "gt.txt"

        # Validate required directories
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        # Load sequence info
        self._load_sequence_info()

        # Load ground truth if available
        self.gt_data = None
        if self.gt_path.exists():
            self.gt_data = self._load_ground_truth()
            logger.info(f"Loaded ground truth with {len(self.gt_data)} annotations")

    def _load_sequence_info(self):
        """Load sequence metadata"""
        # Count frames
        self.frame_files = sorted(list(self.img_dir.glob("*.jpg")))
        self.num_frames = len(self.frame_files)

        if self.num_frames == 0:
            raise ValueError(f"No frames found in {self.img_dir}")

        # Get frame dimensions from first image
        first_frame = cv2.imread(str(self.frame_files[0]))
        if first_frame is None:
            raise ValueError(f"Could not read first frame: {self.frame_files[0]}")

        self.height, self.width = first_frame.shape[:2]
        self.channels = first_frame.shape[2] if len(first_frame.shape) > 2 else 1

        logger.info(f"Loaded sequence {self.sequence_name}: "
                   f"{self.num_frames} frames, {self.width}x{self.height}")

    def _load_ground_truth(self) -> pd.DataFrame:
        """Load ground truth annotations"""
        # MOT17 GT format columns
        columns = ['frame', 'track_id', 'bb_left', 'bb_top',
                  'bb_width', 'bb_height', 'conf', 'class', 'visibility']

        gt_df = pd.read_csv(self.gt_path, header=None, names=columns)

        # Filter only pedestrian class (1) and person on vehicle (2)
        gt_df = gt_df[gt_df['class'].isin([1, 2, 7])]

        # Sort by frame and track_id
        gt_df = gt_df.sort_values(['frame', 'track_id'])

        return gt_df

    def get_frame(self, frame_idx: int) -> np.ndarray:
        """
        Get a single frame

        Args:
            frame_idx: Frame index (0-based)

        Returns:
            Frame as numpy array (H, W, C)
        """
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise ValueError(f"Frame index {frame_idx} out of range [0, {self.num_frames})")

        frame_path = self.frame_files[frame_idx]
        frame = cv2.imread(str(frame_path))

        if frame is None:
            raise ValueError(f"Could not read frame: {frame_path}")

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame

    def get_frame_by_number(self, frame_num: int) -> np.ndarray:
        """
        Get frame by MOT frame number (1-based)

        Args:
            frame_num: MOT frame number (1-based)

        Returns:
            Frame as numpy array
        """
        return self.get_frame(frame_num - 1)

    def get_gt_for_frame(self, frame_num: int) -> pd.DataFrame:
        """
        Get ground truth annotations for a specific frame

        Args:
            frame_num: MOT frame number (1-based)

        Returns:
            DataFrame with annotations for the frame
        """
        if self.gt_data is None:
            raise ValueError("No ground truth data loaded")

        return self.gt_data[self.gt_data['frame'] == frame_num].copy()

    def get_track_data(self, track_id: int) -> pd.DataFrame:
        """
        Get all annotations for a specific track

        Args:
            track_id: Track ID

        Returns:
            DataFrame with all annotations for the track
        """
        if self.gt_data is None:
            raise ValueError("No ground truth data loaded")

        track_data = self.gt_data[self.gt_data['track_id'] == track_id].copy()

        if len(track_data) == 0:
            raise ValueError(f"Track {track_id} not found in ground truth")

        return track_data.sort_values('frame')

    def get_track_bbox_for_frame(self, track_id: int, frame_num: int) -> Optional[np.ndarray]:
        """
        Get bounding box for a specific track in a specific frame

        Args:
            track_id: Track ID
            frame_num: MOT frame number (1-based)

        Returns:
            Bounding box as [x, y, w, h] or None if not present
        """
        if self.gt_data is None:
            return None

        track_frame = self.gt_data[
            (self.gt_data['track_id'] == track_id) &
            (self.gt_data['frame'] == frame_num)
        ]

        if len(track_frame) == 0:
            return None

        row = track_frame.iloc[0]
        return np.array([row['bb_left'], row['bb_top'],
                        row['bb_width'], row['bb_height']])

    def iterate_frames(self, start_frame: int = 1, end_frame: int = None):
        """
        Iterator over frames

        Args:
            start_frame: Starting frame number (1-based)
            end_frame: Ending frame number (1-based), None for all

        Yields:
            Tuple of (frame_num, frame_image)
        """
        if end_frame is None:
            end_frame = self.num_frames

        for frame_num in range(start_frame, min(end_frame + 1, self.num_frames + 1)):
            yield frame_num, self.get_frame_by_number(frame_num)

    def get_sequence_stats(self) -> Dict:
        """Get sequence statistics"""
        stats = {
            'name': self.sequence_name,
            'num_frames': self.num_frames,
            'resolution': (self.width, self.height),
            'channels': self.channels
        }

        if self.gt_data is not None:
            stats['num_tracks'] = self.gt_data['track_id'].nunique()
            stats['num_annotations'] = len(self.gt_data)
            stats['avg_tracks_per_frame'] = (
                self.gt_data.groupby('frame')['track_id'].nunique().mean()
            )

        return stats


class MOT17Dataset:
    """Manager for multiple MOT17 sequences"""

    def __init__(self, root_path: str):
        """
        Initialize MOT17 dataset

        Args:
            root_path: Root path to MOT17 train directory
        """
        self.root_path = Path(root_path)

        if not self.root_path.exists():
            raise FileNotFoundError(f"MOT17 root not found: {root_path}")

        self.sequences = {}
        self._discover_sequences()

    def _discover_sequences(self):
        """Discover available sequences"""
        # Look for MOT17-XX-YYYY directories
        sequence_dirs = sorted([
            d for d in self.root_path.iterdir()
            if d.is_dir() and d.name.startswith("MOT17-")
        ])

        for seq_dir in sequence_dirs:
            seq_name = seq_dir.name
            self.sequences[seq_name] = seq_dir
            logger.info(f"Discovered sequence: {seq_name}")

    def load_sequence(self, sequence_name: str) -> MOT17Sequence:
        """
        Load a specific sequence

        Args:
            sequence_name: Sequence name (e.g., "MOT17-11-FRCNN")

        Returns:
            MOT17Sequence object
        """
        if sequence_name not in self.sequences:
            available = list(self.sequences.keys())
            raise ValueError(
                f"Sequence {sequence_name} not found.\n"
                f"Available sequences: {available}"
            )

        return MOT17Sequence(self.sequences[sequence_name])

    def list_sequences(self) -> List[str]:
        """List available sequences"""
        return list(self.sequences.keys())


if __name__ == "__main__":
    # Test the loader
    import sys
    sys.path.append('/ssd_4TB/divake/temporal_uncertainty')

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Load MOT17 dataset
    mot17 = MOT17Dataset("/ssd_4TB/divake/temporal_uncertainty/data/MOT17/train")
    print(f"Available sequences: {mot17.list_sequences()}")

    # Load sequence 11
    seq11 = mot17.load_sequence("MOT17-11-FRCNN")
    print(f"Sequence stats: {seq11.get_sequence_stats()}")

    # Get Track 25 data
    track25 = seq11.get_track_data(25)
    print(f"Track 25: {len(track25)} frames")
    print(f"Track 25 visibility: {track25['visibility'].mean():.3f}")

    # Get first frame with Track 25
    frame1 = seq11.get_frame_by_number(1)
    bbox1 = seq11.get_track_bbox_for_frame(25, 1)
    print(f"Frame 1 shape: {frame1.shape}")
    print(f"Track 25 bbox at frame 1: {bbox1}")