#!/usr/bin/env python3
"""
Memory-efficient ALOHA data conversion script.

Usage example:
    python run_memory_efficient_conversion.py \
        --raw-dir /path/to/aloha/data \
        --repo-id your-org/your-dataset-name \
        --use-memory-efficient \
        --dataset-config memory_optimized
"""

from pathlib import Path
import tyro
from convert_aloha_data_to_lerobot import port_aloha, MEMORY_OPTIMIZED_CONFIG, DEFAULT_DATASET_CONFIG


def main(
    raw_dir: Path,
    repo_id: str,
    task: str = "DEBUG",
    use_memory_efficient: bool = True,
    dataset_config: str = "memory_optimized",  # "default" or "memory_optimized"
    push_to_hub: bool = True,
    is_mobile: bool = True,
    episodes: list[int] | None = None,
) -> None:
    """
    Run ALOHA data conversion with memory-efficient settings.
    
    Args:
        raw_dir: Path to directory containing episode_*.hdf5 files
        repo_id: Repository ID for the dataset (org/dataset-name)
        task: Task name for the episodes
        use_memory_efficient: Use memory-efficient frame-by-frame processing
        dataset_config: "default" or "memory_optimized" configuration
        push_to_hub: Whether to push the dataset to Hugging Face Hub
        is_mobile: Whether this is a mobile ALOHA dataset
        episodes: Specific episode indices to convert (None for all)
    """
    
    # Select dataset configuration
    if dataset_config == "memory_optimized":
        config = MEMORY_OPTIMIZED_CONFIG
        print("Using memory-optimized configuration:")
        print(f"  - Image writer processes: {config.image_writer_processes}")
        print(f"  - Image writer threads: {config.image_writer_threads}")
    else:
        config = DEFAULT_DATASET_CONFIG
        print("Using default configuration")
    
    print(f"Memory-efficient processing: {'Enabled' if use_memory_efficient else 'Disabled'}")
    
    if use_memory_efficient:
        print("Memory optimizations:")
        print("  - Loading images frame-by-frame instead of entire episodes")
        print("  - Garbage collection every 100 frames")
        print("  - Explicit memory cleanup after each frame")
    
    # Run the conversion
    port_aloha(
        raw_dir=raw_dir,
        repo_id=repo_id,
        task=task,
        episodes=episodes,
        push_to_hub=push_to_hub,
        is_mobile=is_mobile,
        mode="image",  # Using image mode for better memory efficiency
        dataset_config=config,
        use_memory_efficient=use_memory_efficient,
    )
    
    print("Conversion completed successfully!")


if __name__ == "__main__":
    tyro.cli(main)
