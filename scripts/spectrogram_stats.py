#!/usr/bin/env python3
"""
Compute spectrogram normalization stats (mean and std) on the training set.
This should be run after updating your frontend to get accurate normalization values.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from birdset_waveform_dataloader import BirdSetDataLoader
from mae_module import AudioFrontend


def compute_spec_stats(
    dataloader: BirdSetDataLoader,
    frontend: AudioFrontend,
    device: torch.device,
    max_batches: int = None,
    log_interval: int = 10,
) -> tuple:
    """
    Compute mean and std of log-mel spectrograms across the dataset.
    
    Args:
        dataloader: BirdSetDataLoader instance (should have mixup disabled)
        frontend: AudioFrontend instance
        device: torch device
        max_batches: Maximum number of batches to process (None for all)
        log_interval: Print progress every N batches
    
    Returns:
        (mean, std, total_frames): Tuple of (mean scalar, std scalar, total frames processed)
    """
    frontend.eval()
    loader = dataloader.get_loader()
    
    # Initialize accumulators
    sum_spec = 0.0
    sum_spec_sq = 0.0
    total_frames = 0
    batch_count = 0
    num_samples = 0
    
    print("Computing spectrogram statistics...")
    print(f"Total batches: {len(loader) if max_batches is None else min(max_batches, len(loader))}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Processing batches")):
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            # Move batch to device
            waveforms = batch["waveforms"].to(device)
            waveform_lengths = batch["waveform_lengths"].to(device)
            mix_weights = batch["mix_weights"].to(device)
            constituent_labels = batch["constituent_labels"].to(device)
            mix_counts = batch["mix_counts"].to(device)
            
            # Forward through frontend
            specs, _ = frontend(
                waveforms,
                waveform_lengths,
                mix_weights,
                constituent_labels,
                mix_counts,
                training=False,
            )
            
            # specs shape: [B, 1, F, T] -> [B, F, T]
            specs = specs.squeeze(1)  # [B, F, T]
            
            # Update counts
            batch_frames = specs.numel()  # B * F * T
            batch_samples = specs.size(0)
            
            # Update accumulators (using running statistics for memory efficiency)
            batch_sum = specs.sum().item()
            batch_sum_sq = (specs ** 2).sum().item()
            
            # Online mean/variance update (Welford's algorithm)
            new_total_frames = total_frames + batch_frames
            sum_spec += batch_sum
            sum_spec_sq += batch_sum_sq
            
            total_frames = new_total_frames
            num_samples += batch_samples
            batch_count += 1
            
            # Log progress
            if batch_count % log_interval == 0:
                current_mean = sum_spec / total_frames
                current_var = (sum_spec_sq / total_frames) - (current_mean ** 2)
                current_std = np.sqrt(max(0, current_var))
                print(f"  Batch {batch_count}: {total_frames:,} frames processed, "
                      f"mean={current_mean:.4f}, std={current_std:.4f}")
    
    # Compute final statistics
    mean = sum_spec / total_frames
    var = (sum_spec_sq / total_frames) - (mean ** 2)
    std = np.sqrt(max(0, var))
    
    return mean, std, total_frames, num_samples


def main():
    parser = argparse.ArgumentParser(description="Compute spectrogram normalization stats")
    parser.add_argument("--dataset-path", type=str, 
                        default="/scratch/Projects/CFP-04/CFP04-CF-029/birdset",
                        help="Path to BirdSet dataset")
    parser.add_argument("--subset", type=str, default="XCL",
                        help="Dataset subset (POW, XCL, etc.)")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split (train, test_5s, etc.)")
    parser.add_argument("--output-path", type=str, default="./spec_norm_stats.json",
                        help="Path to save computed statistics (JSON format)")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Maximum number of batches to process (None for all)")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Print progress every N batches")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for dataloader")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of dataloader workers")
    parser.add_argument("--n-fft", type=int, default=1024,
                        help="FFT size for spectrogram computation")
    parser.add_argument("--normalize-audio", action="store_true", default=False,
                        help="Whether audio normalization is applied in frontend")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize dataloader WITHOUT mixup
    print("\nInitializing dataloader...")
    print(f"Dataset: {args.subset}, Split: {args.split}")
    print(f"Batch size: {args.batch_size}, Workers: {args.num_workers}")
    
    dataloader = BirdSetDataLoader(
        dataset_path=args.dataset_path,
        subset=args.subset,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,  # No need to shuffle for stats
        use_mixup=False,  # IMPORTANT: Disable mixup
        use_geo_mixup=False,
        window_duration=5.0,
        sample_rate=32000,
    )
    
    # Initialize frontend WITHOUT spectrogram normalization
    # (we're computing the stats that will be used for normalization)
    print("\nInitializing frontend...")
    frontend = AudioFrontend(
        sample_rate=32000,
        window_duration=5.0,
        n_mels=128,
        n_time_frames=512,
        n_fft=args.n_fft,
        spec_norm_path=None,  # No normalization applied
        apply_spec_norm=False,  # Disable normalization
        normalize_audio=args.normalize_audio,
        preemphasis_coeff=0.97,
    ).to(device)
    frontend.eval()
    
    # Compute statistics
    mean, std, total_frames, num_samples = compute_spec_stats(
        dataloader=dataloader,
        frontend=frontend,
        device=device,
        max_batches=args.max_batches,
        log_interval=args.log_interval,
    )
    
    # Create stats dictionary in your exact format
    stats = {
        "mean": mean,
        "std": std,
        "count": total_frames,  # Total time-frequency bins processed
        "num_samples": num_samples,  # Number of spectrogram samples (batch * batch)
        "subset": args.subset,
        "split": args.split,
        "sample_rate": 32000,
        "window_duration": 5.0,
        "n_mels": 128,
        "n_time_frames": 512,
        "n_fft": args.n_fft,
        "normalize_audio": args.normalize_audio,
    }
    
    # Save statistics as JSON
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n? Statistics saved to: {output_path}")
    print("\n" + "=" * 60)
    print("SPECTROGRAM STATISTICS RESULTS")
    print("=" * 60)
    print(f"Processed {num_samples:,} spectrograms")
    print(f"Total time-frequency bins: {total_frames:,}")
    print(f"Mean: {mean:.6f}")
    print(f"Std: {std:.6f}")
    print(f"n_fft: {args.n_fft}")
    print(f"Audio normalization: {args.normalize_audio}")
    
    print(f"\nTo use these stats in your frontend, pass:")
    print(f"  --spec-norm-path {output_path}")
    
    # Also print for easy copying
    print(f"\nYou can also hardcode these values:")
    print(f'mean={mean:.6f}, std={std:.6f}')


if __name__ == "__main__":
    main()