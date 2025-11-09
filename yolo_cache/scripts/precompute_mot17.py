"""
Precompute YOLO cache for all MOT17 sequences.
Main entry point for cache generation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import (
    YOLO_CONFIG,
    DATASET_CONFIG,
    get_cache_output_path
)
from yolo_extractor import YOLOFeatureExtractor
from cache_builder import CacheBuilder


def main():
    print("="*80)
    print("MOT17 YOLO Cache Precomputation")
    print("="*80)

    # Configuration
    model_name = 'yolov8n'
    dataset_name = 'mot17'

    yolo_config = YOLO_CONFIG[model_name]
    dataset_config = DATASET_CONFIG[dataset_name]

    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Feature layers: {yolo_config['feature_layers']}")
    print(f"  Sequences: {len(dataset_config['sequences'])}")

    # Initialize YOLO extractor (only once!)
    print(f"\nInitializing YOLO extractor...")
    yolo_extractor = YOLOFeatureExtractor(
        model_path=str(yolo_config['model_path']),
        feature_layers=yolo_config['feature_layers'],
        device='cuda'
    )

    # Output directory
    output_dir = get_cache_output_path(dataset_name, model_name)
    print(f"Output directory: {output_dir}")

    # Process each sequence
    for i, sequence_name in enumerate(dataset_config['sequences'], 1):
        print(f"\n{'='*80}")
        print(f"Sequence {i}/{len(dataset_config['sequences'])}: {sequence_name}")
        print(f"{'='*80}")

        sequence_path = dataset_config['data_root'] / sequence_name

        if not sequence_path.exists():
            print(f"⚠ Sequence not found: {sequence_path}")
            continue

        # Build cache
        try:
            builder = CacheBuilder(
                dataset_type=dataset_name,
                sequence_path=sequence_path,
                dataset_config=dataset_config,
                yolo_extractor=yolo_extractor,
                yolo_config=yolo_config
            )

            cache = builder.build()

            # Save
            output_path = output_dir / f"{sequence_name}.npz"
            builder.save(cache, output_path)

        except Exception as e:
            print(f"✗ Error processing {sequence_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print("✓ All sequences processed!")
    print(f"{'='*80}")
    print(f"\nCache files saved to: {output_dir}")
    print(f"\nNext step: Run validation script to verify cache integrity")


if __name__ == '__main__':
    main()
