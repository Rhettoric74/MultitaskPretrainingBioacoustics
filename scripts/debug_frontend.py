#!/usr/bin/env python3
"""
Debug script for BirdSet dataloader and audio frontend.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from birdset_waveform_dataloader import BirdSetDataLoader
from mae_module import AudioFrontend  # Your actual frontend
import torchaudio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="/scratch/Projects/CFP-04/CFP04-CF-029/birdset")
    parser.add_argument("--subset", type=str, default="POW")
    parser.add_argument("--split", type=str, default="test_5s")
    parser.add_argument("--output-dir", type=str, default="./debug_output")
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--spec-norm-path", type=str, default="/home/svu/e1583377/MultitaskPretrainingBioacoustics/scripts/power_spec_norm_stats_log_10_non_peak_normalized_XCL.json")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataloader with mixup disabled
    print("Loading dataloader...")
    dataloader = BirdSetDataLoader(
        dataset_path=args.dataset_path,
        subset=args.subset,
        split=args.split,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        use_mixup=False,
        use_geo_mixup=False,
    )
    
    
    # Get raw samples (no mixup applied yet)
    print(f"Loading {args.num_samples} samples...")
    num_targets_found = 0
    target_samples = []
    ds = dataloader.dataset
    target_species_idx = 4
    print(ds.label_int2str)

    labels_col = ds.dataset[ds.label_field_name]  # list of variable-length label lists
    target_samples = [
        i for i, labs in enumerate(labels_col)
        if target_species_idx in labs
    ][:args.num_samples]
    ds.dataset = ds.dataset.select(target_samples)
    loader = dataloader.get_loader()
    
    # Initialize frontend
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frontend = AudioFrontend(
        sample_rate=32000,
        window_duration=5.0,
        n_mels=128,
        n_time_frames=512,
        n_fft=1024,
        spec_norm_path=args.spec_norm_path,
        apply_spec_norm=args.spec_norm_path is not None,
    ).to(device)
    frontend.eval()
    
    # Process each sample through frontend (no mixup)
    
    print("\nProcessing individual samples through frontend...")
    for i, batch in enumerate(loader):
        print(f"\n--- Sample {i} ---")
    
        
        # Move batch to device
        waveforms = batch["waveforms"].to(device)  # [1, M, T]
        waveform_lengths = batch["waveform_lengths"].to(device)
        mix_weights = batch["mix_weights"].to(device)
        constituent_labels = batch["constituent_labels"].to(device)
        mix_counts = batch["mix_counts"].to(device)
        
        # Get labels for printing
        labels = batch["labels"][0].cpu().numpy()
        active_labels = np.where(labels > 0.5)[0]
        print(active_labels)
        
        
        # Get label names
        idx_to_label = {v: k for k, v in dataloader.dataset.label_to_idx.items()}
        label_names = [idx_to_label.get(idx, f"class_{idx}") for idx in active_labels[:10]]
        
        print(f"  Active labels: {len(active_labels)}")
        print(f"  Label names: {label_names}")
        print(f"  Mix count: {mix_counts[0].item()}")
        print(f"  Waveform shape: {waveforms.shape}")
        print(f"  Waveform stats: mean={waveforms.mean().item():.4f}, std={waveforms.std().item():.4f}")
        torchaudio.save(
            str(output_dir / f"sample_{i}.wav"),
            waveforms[0, 0].cpu().unsqueeze(0),
            sample_rate=32000,
        )
        
        # Forward through frontend
        with torch.no_grad():
            specs, mixed_labels = frontend(
                waveforms,
                waveform_lengths,
                mix_weights,
                constituent_labels,
                mix_counts,
                training=False,
            )
        
        print(f"  Spectrogram shape: {specs.shape}")
        print(f"  Spectrogram stats: mean={specs.mean().item():.4f}, std={specs.std().item():.4f}")
        
        # Plot spectrogram
        spec_np = specs[0, 0].cpu().numpy().squeeze()  # [F, T]
        mixed_labels_np = mixed_labels[0].cpu().numpy()
        active_mixed = np.where(mixed_labels_np > 0.5)[0]
        mixed_names = [idx_to_label.get(idx, f"class_{idx}") for idx in active_mixed]
        print(f"  Mixed labels from frontend: {mixed_names}")
        print(f"  Mixed labels array (first 10 classes): {mixed_labels_np[:10]}")
        plt.figure(figsize=(10, 4))
        plt.imshow(spec_np, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='Log-mel magnitude')
        plt.xlabel('Time frame')
        plt.ylabel('Mel bin')
        plt.title(f'Sample {i} - Active labels: {", ".join(label_names[:5])}')
        plt.tight_layout()
        plt.savefig(output_dir / f"sample_{i}_spectrogram.png", dpi=150)
        plt.close()
        
        # Also plot waveform
        waveform_np = waveforms[0, 0].cpu().numpy()  # First batch, first constituent
        plt.figure(figsize=(10, 3))
        time = np.arange(len(waveform_np)) / 32000
        plt.plot(time, waveform_np)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(f'Sample {i} Waveform')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"sample_{i}_waveform.png", dpi=150)
        plt.close()
        if i >= args.num_samples:
            break
    
    # Test manual mixup by creating a batch with multiple constituents
    print("\n" + "="*50)
    print("Testing manual mixup with 2 samples...")
    
    if len(loader) >= 2:
        # Create a batch with 2 constituents
        batch1 = loader[0]
        batch2 = loader[1]
        
        # Combine into single batch with 2 constituents
        waveforms = torch.cat([batch1["waveforms"], batch2["waveforms"]], dim=1)  # [1, 2, T]
        waveform_lengths = torch.cat([batch1["waveform_lengths"], batch2["waveform_lengths"]], dim=1)
        
        # Create mix weights (e.g., 0.7 and 0.3)
        mix_weights = torch.tensor([0.7, 0.3], dtype=torch.float32, device=device)  # [1, 2]
        
        # Combine constituent labels
        constituent_labels = torch.cat([batch1["constituent_labels"], batch2["constituent_labels"]], dim=1)
        
        mix_counts = torch.tensor([2], dtype=torch.long, device=device)
        
        # Move to device
        waveforms = waveforms.to(device)
        waveform_lengths = waveform_lengths.to(device)
        constituent_labels = constituent_labels.to(device)
        
        # Forward through frontend
        with torch.no_grad():
            specs, mixed_labels = frontend(
                waveforms,
                waveform_lengths,
                mix_weights,
                constituent_labels,
                mix_counts,
                training=False,
            )
        
        print(f"  Mixed spectrogram shape: {specs.shape}")
        print(f"  Mixed spectrogram stats: mean={specs.mean().item():.4f}, std={specs.std().item():.4f}")
        
        # Get mixed labels
        mixed_labels_np = mixed_labels[0].cpu().numpy()
        active_mixed = np.where(mixed_labels_np > 0.5)[0]
        mixed_label_names = [idx_to_label.get(idx, f"class_{idx}") for idx in active_mixed[:10]]
        
        print(f"  Active labels in mix: {len(active_mixed)}")
        print(f"  Mixed label indices: {active_mixed}")
        
        # Plot mixed spectrogram
        spec_np = specs[0, 0].cpu().numpy().squeeze()
        plt.figure(figsize=(10, 4))
        plt.imshow(spec_np, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='Log-mel magnitude')
        plt.xlabel('Time frame')
        plt.ylabel('Mel bin')
        plt.title(f'Mixed Sample (0.7/0.3) - Labels: {", ".join(mixed_label_names[:5])}')
        plt.tight_layout()
        plt.savefig(output_dir / "mixed_spectrogram.png", dpi=150)
        plt.close()
    
    # Save label mapping
    #with open(output_dir / "label_mapping.json", "w") as f:
        #json.dump(idx_to_label, f, indent=2)
    
    print(f"\n? Debug complete! Check {output_dir} for visualizations")


if __name__ == "__main__":
    main()