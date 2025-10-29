"""
Track Extractor
Extract specific tracks from MOT17 sequences using Phase 1 metadata
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TrackExtractor:
    """Extract and analyze specific tracks using Phase 1 metadata"""

    def __init__(self, metadata_path: str, sequence_name: str = "seq11"):
        """
        Initialize track extractor with Phase 1 metadata

        Args:
            metadata_path: Path to metadata directory
            sequence_name: Sequence name (e.g., "seq11")
        """
        self.metadata_path = Path(metadata_path)
        self.sequence_name = sequence_name

        # Load metadata
        self.metadata = self._load_metadata()
        self.tracks = self.metadata.get('tracks', {})

        logger.info(f"Loaded metadata for {sequence_name}: {len(self.tracks)} tracks")

    def _load_metadata(self) -> Dict:
        """Load Phase 1 metadata"""
        # Try pickle first (more complete), then JSON
        pkl_path = self.metadata_path / f"{self.sequence_name}_metadata.pkl"
        json_path = self.metadata_path / f"{self.sequence_name}_metadata.json"

        if pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                return pickle.load(f)
        elif json_path.exists():
            with open(json_path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(
                f"No metadata found for {self.sequence_name}\n"
                f"Looked for: {pkl_path} or {json_path}"
            )

    def get_track_info(self, track_id: int) -> Dict:
        """
        Get comprehensive info for a track

        Args:
            track_id: Track ID

        Returns:
            Track metadata dictionary
        """
        track_id_str = str(track_id)

        if track_id_str not in self.tracks:
            raise ValueError(f"Track {track_id} not found in metadata")

        return self.tracks[track_id_str]

    def get_track_occlusions(self, track_id: int) -> List[Dict]:
        """
        Get occlusion events for a track

        Args:
            track_id: Track ID

        Returns:
            List of occlusion events
        """
        track_info = self.get_track_info(track_id)
        return track_info.get('occlusion_events', [])

    def get_track_lifetime(self, track_id: int) -> Tuple[int, int]:
        """
        Get track lifetime (start and end frames)

        Args:
            track_id: Track ID

        Returns:
            Tuple of (start_frame, end_frame)
        """
        track_info = self.get_track_info(track_id)
        lifetime = track_info.get('lifetime_frames', [])

        if len(lifetime) != 2:
            raise ValueError(f"Invalid lifetime for track {track_id}")

        return lifetime[0], lifetime[1]

    def is_track_occluded(self, track_id: int, frame_num: int) -> bool:
        """
        Check if track is occluded at a specific frame

        Args:
            track_id: Track ID
            frame_num: Frame number (1-based)

        Returns:
            True if occluded, False otherwise
        """
        occlusions = self.get_track_occlusions(track_id)

        for occ in occlusions:
            if occ['start_frame'] <= frame_num <= occ['end_frame']:
                return True

        return False

    def get_occlusion_frames(self, track_id: int) -> List[int]:
        """
        Get all frames where track is occluded

        Args:
            track_id: Track ID

        Returns:
            List of frame numbers where track is occluded
        """
        occlusions = self.get_track_occlusions(track_id)
        occluded_frames = []

        for occ in occlusions:
            frames = list(range(occ['start_frame'], occ['end_frame'] + 1))
            occluded_frames.extend(frames)

        return sorted(occluded_frames)

    def get_clean_frames(self, track_id: int) -> List[int]:
        """
        Get frames where track is NOT occluded

        Args:
            track_id: Track ID

        Returns:
            List of clean (non-occluded) frame numbers
        """
        start_frame, end_frame = self.get_track_lifetime(track_id)
        all_frames = set(range(start_frame, end_frame + 1))
        occluded_frames = set(self.get_occlusion_frames(track_id))

        return sorted(list(all_frames - occluded_frames))

    def get_track_statistics(self, track_id: int) -> Dict:
        """
        Get comprehensive statistics for a track

        Args:
            track_id: Track ID

        Returns:
            Dictionary with track statistics
        """
        track_info = self.get_track_info(track_id)
        occlusions = self.get_track_occlusions(track_id)
        occluded_frames = self.get_occlusion_frames(track_id)
        clean_frames = self.get_clean_frames(track_id)

        stats = {
            'track_id': track_id,
            'duration': track_info.get('duration', 0),
            'lifetime_frames': track_info.get('lifetime_frames', []),
            'num_detections': track_info.get('num_detections', 0),
            'avg_visibility': track_info.get('avg_visibility', 0),
            'num_occlusion_events': len(occlusions),
            'total_occluded_frames': len(occluded_frames),
            'total_clean_frames': len(clean_frames),
            'occlusion_rate': len(occluded_frames) / track_info.get('duration', 1),
            'max_displacement': track_info.get('max_displacement', 0),
            'avg_displacement': track_info.get('avg_displacement', 0)
        }

        # Add detailed occlusion info
        if occlusions:
            stats['occlusion_details'] = occlusions
            stats['longest_occlusion'] = max(occ['duration'] for occ in occlusions)
            stats['shortest_occlusion'] = min(occ['duration'] for occ in occlusions)

        return stats

    def get_hero_tracks(self) -> Dict[str, List[int]]:
        """
        Get hero tracks identified in Phase 1

        Returns:
            Dictionary with hero track categories and IDs
        """
        hero_path = self.metadata_path / "hero_tracks_all_sequences.json"

        if not hero_path.exists():
            logger.warning("Hero tracks file not found, using default Track 25")
            return {
                'primary': [25],
                'long_stable': [1, 25, 14],
                'occlusion_heavy': [25, 30, 29]
            }

        with open(hero_path, 'r') as f:
            hero_data = json.load(f)

        # Extract track IDs for this sequence
        seq_key = self.sequence_name.replace("seq", "MOT17-")
        if not seq_key.endswith("-FRCNN"):
            seq_key += "-FRCNN"

        if seq_key not in hero_data:
            logger.warning(f"No hero tracks for {seq_key}")
            return {}

        seq_heroes = hero_data[seq_key]
        result = {}

        for category, tracks in seq_heroes.items():
            result[category] = list(tracks.keys()) if isinstance(tracks, dict) else tracks

        return result


class Track25Analyzer:
    """Specialized analyzer for Track 25"""

    def __init__(self, metadata_path: str):
        """Initialize Track 25 analyzer"""
        self.extractor = TrackExtractor(metadata_path, "seq11")
        self.track_id = 25

        # Validate Track 25 exists
        self.track_info = self.extractor.get_track_info(self.track_id)
        self.stats = self.extractor.get_track_statistics(self.track_id)

        logger.info(f"Track 25 initialized: {self.stats['duration']} frames, "
                   f"{self.stats['num_occlusion_events']} occlusions")

    def get_occlusion_periods(self) -> List[Tuple[int, int]]:
        """Get occlusion periods as (start, end) tuples"""
        occlusions = self.extractor.get_track_occlusions(self.track_id)
        return [(occ['start_frame'], occ['end_frame']) for occ in occlusions]

    def get_recovery_periods(self, min_length: int = 10) -> List[Tuple[int, int]]:
        """
        Get recovery periods (clean frames after occlusions)

        Args:
            min_length: Minimum length for a recovery period

        Returns:
            List of (start, end) tuples for recovery periods
        """
        occlusion_periods = self.get_occlusion_periods()
        recovery_periods = []

        for i, (occ_start, occ_end) in enumerate(occlusion_periods):
            # Look for clean period after this occlusion
            next_occ_start = (occlusion_periods[i + 1][0]
                            if i + 1 < len(occlusion_periods)
                            else self.stats['lifetime_frames'][1] + 1)

            recovery_start = occ_end + 1
            recovery_end = next_occ_start - 1

            if recovery_end - recovery_start + 1 >= min_length:
                recovery_periods.append((recovery_start, recovery_end))

        return recovery_periods

    def classify_frame(self, frame_num: int) -> str:
        """
        Classify a frame as occluded, recovery, or clean

        Args:
            frame_num: Frame number (1-based)

        Returns:
            Classification string
        """
        if self.extractor.is_track_occluded(self.track_id, frame_num):
            return "occluded"

        # Check if in recovery period
        for start, end in self.get_recovery_periods():
            if start <= frame_num <= end:
                # Calculate recovery progress
                progress = (frame_num - start) / (end - start + 1)
                if progress < 0.3:
                    return "early_recovery"
                elif progress < 0.7:
                    return "mid_recovery"
                else:
                    return "late_recovery"

        return "clean"

    def get_analysis_frames(self) -> Dict[str, List[int]]:
        """Get frames categorized for analysis"""
        frames = {
            'occluded': self.extractor.get_occlusion_frames(self.track_id),
            'clean': [],
            'early_recovery': [],
            'mid_recovery': [],
            'late_recovery': []
        }

        # Classify all frames
        start, end = self.stats['lifetime_frames']
        for frame_num in range(start, end + 1):
            classification = self.classify_frame(frame_num)
            if classification != 'occluded':
                frames[classification].append(frame_num)

        return frames


if __name__ == "__main__":
    # Test the extractor
    import sys
    sys.path.append('/ssd_4TB/divake/temporal_uncertainty')

    logging.basicConfig(level=logging.INFO)

    # Load Track 25 analyzer
    analyzer = Track25Analyzer("/ssd_4TB/divake/temporal_uncertainty/metadata")

    # Get statistics
    print(f"Track 25 Statistics:")
    for key, value in analyzer.stats.items():
        if key != 'occlusion_details':
            print(f"  {key}: {value}")

    # Get periods
    print(f"\nOcclusion Periods: {analyzer.get_occlusion_periods()}")
    print(f"Recovery Periods: {analyzer.get_recovery_periods()}")

    # Analyze specific frames
    test_frames = [1, 10, 67, 100, 200, 700]
    print(f"\nFrame Classifications:")
    for frame in test_frames:
        print(f"  Frame {frame}: {analyzer.classify_frame(frame)}")