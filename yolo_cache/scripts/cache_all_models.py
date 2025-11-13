"""
Cache YOLO features for all models across all datasets.
This script processes:
- 5 YOLO models: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
- 3 datasets: mot17 (3 seqs), mot20 (1 seq), dancetrack (1 seq)
- Total: 25 cache files (5 models × 5 sequences)

Usage:
    # Cache all models for all datasets
    python cache_all_models.py

    # Cache specific model
    python cache_all_models.py --model yolov8s

    # Cache specific dataset
    python cache_all_models.py --dataset mot17

    # Cache specific model and dataset
    python cache_all_models.py --model yolov8s --dataset mot17
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import (
    YOLO_CONFIG,
    DATASET_CONFIG,
    get_cache_output_path
)
from yolo_extractor import YOLOFeatureExtractor
from cache_builder import CacheBuilder


def cache_model_dataset(model_name: str, dataset_name: str):
    """Cache features for a specific model on a specific dataset."""

    print("="*80)
    print(f"Processing: {model_name} on {dataset_name}")
    print("="*80)

    # Get configurations
    yolo_config = YOLO_CONFIG[model_name]
    dataset_config = DATASET_CONFIG[dataset_name]

    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Model path: {yolo_config['model_path']}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Sequences: {len(dataset_config['sequences'])}")
    print(f"  Feature layers: {yolo_config['feature_layers']}")

    # Check model file exists
    if not yolo_config['model_path'].exists():
        print(f"✗ Model file not found: {yolo_config['model_path']}")
        return False

    # Initialize YOLO extractor (only once per model!)
    print(f"\nInitializing YOLO extractor...")
    start_time = datetime.now()

    try:
        yolo_extractor = YOLOFeatureExtractor(
            model_path=str(yolo_config['model_path']),
            feature_layers=yolo_config['feature_layers'],
            device='cuda'
        )
        print(f"✓ Extractor initialized in {(datetime.now() - start_time).total_seconds():.1f}s")
    except Exception as e:
        print(f"✗ Failed to initialize extractor: {e}")
        return False

    # Output directory
    output_dir = get_cache_output_path(dataset_name, model_name)
    print(f"Output directory: {output_dir}")

    # Process each sequence
    success_count = 0
    fail_count = 0

    for i, sequence_name in enumerate(dataset_config['sequences'], 1):
        print(f"\n{'='*80}")
        print(f"Sequence {i}/{len(dataset_config['sequences'])}: {sequence_name}")
        print(f"{'='*80}")

        sequence_path = dataset_config['data_root'] / sequence_name

        if not sequence_path.exists():
            print(f"⚠ Sequence not found: {sequence_path}")
            fail_count += 1
            continue

        # Check if cache already exists
        output_path = output_dir / f"{sequence_name}.npz"
        if output_path.exists():
            print(f"⚠ Cache already exists, skipping: {output_path}")
            success_count += 1
            continue

        # Build cache
        try:
            seq_start_time = datetime.now()

            builder = CacheBuilder(
                dataset_type=dataset_name,
                sequence_path=sequence_path,
                dataset_config=dataset_config,
                yolo_extractor=yolo_extractor,
                yolo_config=yolo_config
            )

            cache = builder.build()

            # Save
            builder.save(cache, output_path)

            elapsed = (datetime.now() - seq_start_time).total_seconds()
            print(f"✓ Sequence completed in {elapsed:.1f}s")
            success_count += 1

        except Exception as e:
            print(f"✗ Error processing {sequence_name}: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1
            continue

    # Summary
    print(f"\n{'='*80}")
    print(f"Model {model_name} on {dataset_name} completed!")
    print(f"  Success: {success_count}/{len(dataset_config['sequences'])}")
    print(f"  Failed: {fail_count}/{len(dataset_config['sequences'])}")
    print(f"{'='*80}\n")

    return fail_count == 0


def main():
    parser = argparse.ArgumentParser(description='Cache YOLO features for all models and datasets')
    parser.add_argument('--model', type=str, default=None,
                       choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
                       help='Specific model to cache (default: all)')
    parser.add_argument('--dataset', type=str, default=None,
                       choices=['mot17', 'mot20', 'dancetrack'],
                       help='Specific dataset to cache (default: all)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip models that already have cached files')

    args = parser.parse_args()

    # Determine which models to process
    if args.model:
        models = [args.model]
    else:
        models = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']

    # Determine which datasets to process
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = ['mot17', 'mot20', 'dancetrack']

    # Calculate total work
    total_sequences = sum(len(DATASET_CONFIG[d]['sequences']) for d in datasets)
    total_caches = len(models) * total_sequences

    print("\n" + "="*80)
    print("YOLO Feature Cache Generation")
    print("="*80)
    print(f"\nModels to process: {len(models)}")
    for m in models:
        print(f"  - {m}")
    print(f"\nDatasets to process: {len(datasets)}")
    for d in datasets:
        print(f"  - {d}: {len(DATASET_CONFIG[d]['sequences'])} sequences")
    print(f"\nTotal cache files to generate: {total_caches}")
    print(f"Estimated time: ~{total_caches * 2:.0f} minutes (varies by model size)")
    print("="*80 + "\n")

    # Process each model-dataset combination
    overall_start = datetime.now()
    completed = 0
    failed = 0

    for model_idx, model_name in enumerate(models, 1):
        print(f"\n{'#'*80}")
        print(f"# MODEL {model_idx}/{len(models)}: {model_name.upper()}")
        print(f"{'#'*80}\n")

        for dataset_idx, dataset_name in enumerate(datasets, 1):
            success = cache_model_dataset(model_name, dataset_name)

            if success:
                completed += len(DATASET_CONFIG[dataset_name]['sequences'])
            else:
                failed += 1

            # Progress update
            done_sequences = completed + failed
            pct_complete = (done_sequences / total_caches) * 100
            elapsed = (datetime.now() - overall_start).total_seconds() / 60

            print(f"\nOverall Progress: {done_sequences}/{total_caches} cache files ({pct_complete:.1f}%)")
            print(f"Time elapsed: {elapsed:.1f} minutes")
            if done_sequences > 0:
                avg_time = elapsed / done_sequences
                remaining = (total_caches - done_sequences) * avg_time
                print(f"Estimated time remaining: {remaining:.1f} minutes\n")

    # Final summary
    total_elapsed = (datetime.now() - overall_start).total_seconds() / 60

    print("\n" + "="*80)
    print("CACHE GENERATION COMPLETE!")
    print("="*80)
    print(f"Total cache files generated: {completed}/{total_caches}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_elapsed:.1f} minutes ({total_elapsed/60:.1f} hours)")
    print(f"\nCache files location: {Path('/ssd_4TB/divake/temporal_uncertainty/yolo_cache/data')}")
    print("="*80 + "\n")

    print("Next steps:")
    print("1. Run validation script to verify cache integrity")
    print("2. Run proper experiments with real Triple-S framework")
    print("3. Fill tables with experimental results\n")


if __name__ == '__main__':
    main()
