#!/bin/bash

echo "=== Verifying Temporal Uncertainty Project Setup ==="
echo

# Check MOT17 sequences
echo "✓ Checking MOT17 sequences..."
for seq in 02 04 05 09 10 11 13; do
    dir="/ssd_4TB/divake/temporal_uncertainty/MOT17/train/MOT17-${seq}-FRCNN"
    if [ -d "$dir" ]; then
        frames=$(find "$dir/img1" -name "*.jpg" | wc -l)
        echo "  ✓ MOT17-${seq}-FRCNN: $frames frames"
    else
        echo "  ✗ MOT17-${seq}-FRCNN: NOT FOUND"
    fi
done
echo

# Check videos
echo "✓ Checking pre-rendered videos..."
video_count=$(find /ssd_4TB/divake/temporal_uncertainty/MOT17/video -name "*.mp4" | wc -l)
echo "  ✓ Found $video_count MP4 videos"
echo

# Check GitHub repos
echo "✓ Checking GitHub repositories..."
for category in core_uncertainty tracking_implementations augmentation_libs tta_specific papers_with_code evaluation_metrics; do
    count=$(find /ssd_4TB/divake/temporal_uncertainty/github_repos/$category -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "  ✓ $category: $count repos"
done
echo

# Summary
echo "=== Summary ==="
total_frames=$(find /ssd_4TB/divake/temporal_uncertainty/MOT17/train/*/img1 -name "*.jpg" | wc -l)
total_repos=$(find /ssd_4TB/divake/temporal_uncertainty/github_repos -mindepth 2 -maxdepth 2 -type d -name ".git" | wc -l)
echo "Total frames: $total_frames"
echo "Total repositories: $total_repos"
echo "Project ready for uncertainty analysis!"
