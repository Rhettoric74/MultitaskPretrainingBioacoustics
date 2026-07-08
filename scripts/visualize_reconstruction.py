#!/usr/bin/env python3
"""
Visualize MAE reconstruction quality by comparing original, masked input, and reconstructed spectrograms.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

from birdset_waveform_dataloader import BirdSetDataLoader
from build_mae_model import build_model


def patchify_for_display(specs, patch_size=16):
    """Convert spectrogram to patch grid for visualization."""
    B, C, H, W = specs.shape
    patches = specs.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # patches shape: [B, C, H//p, W//p, p, p]
    patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
    return patches


def apply_masking_to_spectrogram(specs, context_indices, patch_size=16):
    """Create a masked spectrogram visualization from context indices."""
    B, C, H, W = specs.shape
    masked_spec = specs.clone()
    
    # Calculate patch grid dimensions
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    
    # Create mask of kept patches
    mask = torch.zeros(B, num_patches_h * num_patches_w, dtype=torch.bool, device=specs.device)
    for b in range(B):
        mask[b, context_indices[0][b]] = True
    
    # Reshape to grid and apply to spectrogram
    mask_grid = mask.view(B, num_patches_h, num_patches_w)
    
    for b in range(B):
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                if not mask_grid[b, i, j]:
                    # Mask out this patch (set to mean value for visualization)
                    y_start = i * patch_size
                    y_end = y_start + patch_size
                    x_start = j * patch_size
                    x_end = x_start + patch_size
                    masked_spec[b, :, y_start:y_end, x_start:x_end] = specs[b, :, y_start:y_end, x_start:x_end].min()
    
    return masked_spec


def visualize_reconstruction(model, batch, device, save_path, patch_size=16, idx=0):
    """Visualize original, masked input, and reconstructed spectrograms."""
    model.eval()
    
    with torch.no_grad():
        # Get spectrograms
        specs, labels = model.forward(batch, training=False)
        print(labels.sum())
        specs = specs.to(device)
        
        B, _, H, W = specs.shape
        num_patches = (H // patch_size) * (W // patch_size)
        
        # Generate mask (using same logic as training)
        context_ratio = 1.0 - model.mask_ratio
        context_indices, prediction_indices = generate_random_keep_indices(
            batch_size=B,
            num_patches=num_patches,
            context_ratio=context_ratio,
            device=specs.device,
        )
        
        # Forward through encoder with masking
        h_all, cls_token, h_patches = model.encoder(specs, context_indices)
        
        # Decode masked patches
        pred_patches = model.decoder(h_patches, context_indices, prediction_indices)
        
        # Get target patches
        target_patches = model._patchify(specs)
        target_masked = apply_keep_indices(target_patches, prediction_indices)
        
        # Reconstruct full spectrogram from predictions
        reconstructed_spec = reconstruct_spectrogram_from_patches(
            specs, pred_patches, context_indices, prediction_indices, patch_size
        )
        
        # Create masked spectrogram for visualization
        masked_spec = apply_masking_to_spectrogram(specs, context_indices, patch_size)
    
    # Plot for a single sample (idx)
    spec_np = specs[idx, 0].cpu().numpy()
    masked_np = masked_spec[idx, 0].cpu().numpy()
    reconstructed_np = reconstructed_spec[idx, 0].cpu().numpy()
    
    # Calculate reconstruction error
    error_np = np.abs(spec_np - reconstructed_np)
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Original spectrogram
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(spec_np, aspect='auto', origin='lower', cmap='viridis')
    ax1.set_title(f'Original Spectrogram\nSample {idx}', fontsize=12)
    ax1.set_xlabel('Time Frame')
    ax1.set_ylabel('Mel Bin')
    plt.colorbar(im1, ax=ax1, label='Log-mel magnitude')
    
    # Masked input (what encoder sees)
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(masked_np, aspect='auto', origin='lower', cmap='viridis')
    ax2.set_title(f'Masked Input (75% masked)\nKept: {context_indices[0][idx].shape[0]} patches', fontsize=12)
    ax2.set_xlabel('Time Frame')
    ax2.set_ylabel('Mel Bin')
    plt.colorbar(im2, ax=ax2, label='Log-mel magnitude')
    
    # Reconstructed spectrogram
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(reconstructed_np, aspect='auto', origin='lower', cmap='viridis')
    ax3.set_title(f'Reconstructed Spectrogram\nMSE: {np.mean(error_np**2):.4f}', fontsize=12)
    ax3.set_xlabel('Time Frame')
    ax3.set_ylabel('Mel Bin')
    plt.colorbar(im3, ax=ax3, label='Log-mel magnitude')
    
    # Reconstruction error heatmap
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(error_np, aspect='auto', origin='lower', cmap='hot')
    ax4.set_title(f'Reconstruction Error (Absolute)\nMean: {np.mean(error_np):.4f}', fontsize=12)
    ax4.set_xlabel('Time Frame')
    ax4.set_ylabel('Mel Bin')
    plt.colorbar(im4, ax=ax4, label='Absolute Error')
    
    # Histogram of errors
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(error_np.flatten(), bins=50, alpha=0.7, color='blue')
    ax5.set_xlabel('Absolute Error')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Error Distribution')
    ax5.axvline(np.mean(error_np), color='red', linestyle='--', label=f'Mean: {np.mean(error_np):.4f}')
    ax5.legend()
    
    # Patch-level error (which patches are hardest to reconstruct)
    ax6 = fig.add_subplot(gs[1, 2])
    patch_errors = compute_patch_errors(spec_np, reconstructed_np, patch_size)
    im6 = ax6.imshow(patch_errors, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=np.percentile(patch_errors, 95))
    ax6.set_title(f'Patch-level Reconstruction Error\n(Each cell = {patch_size}x{patch_size} mel-time region)', fontsize=12)
    ax6.set_xlabel('Patch Column')
    ax6.set_ylabel('Patch Row')
    plt.colorbar(im6, ax=ax6, label='Mean Patch Error')
    
    plt.suptitle(f'MAE Reconstruction Visualization\nMask Ratio: {model.mask_ratio:.2%}', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to {save_path}")
    
    # Print statistics
    print(f"\nReconstruction Statistics:")
    print(f"  MSE: {np.mean(error_np**2):.6f}")
    print(f"  MAE: {np.mean(error_np):.6f}")
    print(f"  Max Error: {np.max(error_np):.6f}")
    print(f"  P95 Error: {np.percentile(error_np, 95):.6f}")


def reconstruct_spectrogram_from_patches(original_specs, pred_patches, context_indices, prediction_indices, patch_size):
    """Reconstruct full spectrogram from predicted patches."""
    B, C, H, W = original_specs.shape
    reconstructed = original_specs.clone()
    
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    
    # Unpatchify predictions
    pred_patches_unflat = pred_patches.view(B, -1, patch_size, patch_size)
    
    # Create full patch grid
    all_patches = torch.zeros(B, num_patches_h * num_patches_w, patch_size, patch_size, device=original_specs.device)
    
    # Fill in predictions for masked patches
    for b in range(B):
        pred_indices = prediction_indices[0][b]
        for i, patch_idx in enumerate(pred_indices):
            all_patches[b, patch_idx] = pred_patches_unflat[b, i]
        
        # Fill in original patches for context patches
        context_indices_batch = context_indices[0][b]
        original_patches = patchify_for_display(original_specs[b:b+1], patch_size)
        for patch_idx in context_indices_batch:
            patch_row = patch_idx // num_patches_w
            patch_col = patch_idx % num_patches_w
            all_patches[b, patch_idx] = original_patches[0, 0, patch_idx]
    
    # Reconstruct spectrogram from patches
    for b in range(B):
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                patch_idx = i * num_patches_w + j
                y_start = i * patch_size
                y_end = y_start + patch_size
                x_start = j * patch_size
                x_end = x_start + patch_size
                reconstructed[b, 0, y_start:y_end, x_start:x_end] = all_patches[b, patch_idx]
    
    return reconstructed


def compute_patch_errors(original, reconstructed, patch_size):
    """Compute mean error per patch for visualization."""
    H, W = original.shape
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    
    patch_errors = np.zeros((num_patches_h, num_patches_w))
    
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            y_start = i * patch_size
            y_end = y_start + patch_size
            x_start = j * patch_size
            x_end = x_start + patch_size
            
            patch_orig = original[y_start:y_end, x_start:x_end]
            patch_recon = reconstructed[y_start:y_end, x_start:x_end]
            patch_errors[i, j] = np.mean(np.abs(patch_orig - patch_recon))
    
    return patch_errors


def main():
    parser = argparse.ArgumentParser(description="Visualize MAE reconstruction quality")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset-path", type=str, default="/scratch/Projects/CFP-04/CFP04-CF-029/birdset")
    parser.add_argument("--subset", type=str, default="POW")
    parser.add_argument("--split", type=str, default="test_5s")
    parser.add_argument("--output-dir", type=str, default="./recon_viz")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of samples to visualize")
    parser.add_argument("--spec-norm-path", type=str, default="/home/svu/e1583377/MultitaskPretrainingBioacoustics/scripts/power_spec_norm_stats_XCL.json")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = build_model(
        num_classes=9734,  # Placeholder, not used for visualization
        classifier_type="linear",
        use_cls_token=True,
        spec_norm_path=args.spec_norm_path,
    )
    
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only = False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Load data
    print(f"Loading data from {args.subset}/{args.split}")
    dataloader = BirdSetDataLoader(
        dataset_path=args.dataset_path,
        subset=args.subset,
        split=args.split,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        use_mixup=True,
        use_geo_mixup=True,
        mix_omega = 10.0,
    )
    loader = dataloader.get_loader()
    
    # Visualize samples
    for i, batch in enumerate(loader):
        if i >= args.num_samples:
            break
        
        print(f"\nProcessing sample {i+1}/{args.num_samples}")
        
        # Move batch to device
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        # ---------- FIXED LABEL EXTRACTION ----------
        # Get the first sample's mix data
        constituent_labels = batch["constituent_labels"][0]  # Shape: [num_constituents, num_classes]
        mix_weights = batch["mix_weights"][0]                # Shape: [num_constituents]
        print("Waveform shape:", batch["waveforms"].shape)
        
        # Compute the ACTUAL mixed labels (weighted sum)
        # Ensure shapes match: [num_constituents, 1] * [num_constituents, num_classes]
        mixed_labels = (mix_weights.unsqueeze(1) * constituent_labels).sum(dim=0)
        
        # For visualization, get labels with significant weight
        threshold = 0.1  # Adjust as needed
        active_indices = torch.where(mixed_labels > threshold)[0]
        
        # Get label names with their weights
        idx_to_label = {v: k for k, v in dataloader.dataset.label_to_idx.items()}
        label_names = [
            f"{idx_to_label.get(idx.item(), f'class_{idx.item()}')} ({mixed_labels[idx].item():.3f})"
            for idx in active_indices
        ]
        
        print(f"  Mixed labels (weighted): {label_names}")
        print(f"  Mix recipe: {len(constituent_labels)} constituents with weights {mix_weights.tolist()}")
        
        # Optional: Also show the constituent labels for debugging
        if len(constituent_labels) > 1:
            print(f"  Constituents:")
            for j, (const_labels, weight) in enumerate(zip(constituent_labels, mix_weights)):
                const_active = torch.where(const_labels > 0.5)[0]
                const_names = [idx_to_label.get(idx.item(), f"class_{idx.item()}") for idx in const_active]
                print(f"    {j+1}: weight={weight:.3f}, labels={const_names}")
        
        # Visualize reconstruction
        save_path = output_dir / f"reconstruction_sample_{i}.png"
        visualize_reconstruction(model, batch, device, save_path, patch_size=16, idx=0)


if __name__ == "__main__":
    # Import helper functions from your model
    from mae_module import generate_random_keep_indices, apply_keep_indices
    
    main()