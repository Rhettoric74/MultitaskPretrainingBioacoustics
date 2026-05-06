import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from birdset_dataloader import BirdSetDataLoader
from utils import generate_random_masks

# -----------------------------
# CONFIG
# -----------------------------
DATASET_PATH = "/scratch/Projects/CFP-04/CFP04-CF-029/birdset"
OUT_DIR = "debug_viz"
NUM_SAMPLES = 8
PATCH_SIZE = 16

os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------------
# Helper: plot spectrogram (raw)
# -----------------------------
def plot_spectrogram(spec, title, save_path):
    plt.figure(figsize=(10, 4))
    plt.imshow(spec, aspect='auto', origin='lower', cmap='magma')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Mel bins")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# -----------------------------
# Helper: overlay mask (COLOR VERSION)
# -----------------------------
def overlay_mask_rgb(spec, mask, grid_size, color="blue"):
    """
    spec: (H, W)
    mask: (N,)
    grid_size: (H_p, W_p)
    """
    H, W = spec.shape
    H_p, W_p = grid_size

    patch_h = H // H_p
    patch_w = W // W_p

    mask_2d = mask.reshape(H_p, W_p)

    # normalize spectrogram to [0,1]
    spec_norm = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)

    # convert to RGB
    spec_rgb = np.stack([spec_norm]*3, axis=-1)

    for i in range(H_p):
        for j in range(W_p):
            if mask_2d[i, j]:
                h0 = i * patch_h
                w0 = j * patch_w

                if color == "blue":
                    spec_rgb[h0:h0+patch_h, w0:w0+patch_w] = [0.2, 0.2, 1.0]
                elif color == "red":
                    spec_rgb[h0:h0+patch_h, w0:w0+patch_w] = [1.0, 0.2, 0.2]
                elif color == "white":
                    spec_rgb[h0:h0+patch_h, w0:w0+patch_w] = [1.0, 1.0, 1.0]

    return spec_rgb


# -----------------------------
# Helper: extract mask safely
# -----------------------------
def extract_mask(mask):
    """
    Handles both tensor and list formats
    """
    if isinstance(mask, list):
        mask = mask[0]
    return mask[0].cpu().numpy()


# -----------------------------
# Main
# -----------------------------
def main():

    data = BirdSetDataLoader(
        dataset_path=DATASET_PATH,
        batch_size=1,
        num_workers=0,  # debugging
        use_mixup=True,
        mixup_prob=0.5
    )

    loader = data.get_loader()

    for i, batch in enumerate(loader):
        if i >= NUM_SAMPLES:
            break

        spec = batch["spectrograms"][0, 0].cpu().numpy()
        labels = batch["labels"][0]

        # -----------------------------
        # Save raw spectrogram
        # -----------------------------
        plot_spectrogram(
            spec,
            title=f"Sample {i} | labels={labels.sum().item()}",
            save_path=os.path.join(OUT_DIR, f"sample_{i}.png")
        )

        # -----------------------------
        # Generate RANDOM masks (new)
        # -----------------------------
        H, W = spec.shape
        grid_size = (H // PATCH_SIZE, W // PATCH_SIZE)
        num_patches = grid_size[0] * grid_size[1]

        context_masks, pred_masks = generate_random_masks(
            batch_size=1,
            num_patches=num_patches,
            mask_ratio=0.50,
            device="cpu"
        )

        context_mask = extract_mask(context_masks)
        pred_mask = extract_mask(pred_masks)

        # -----------------------------
        # Visualizations
        # -----------------------------
        spec_context = overlay_mask_rgb(spec, context_mask, grid_size, "blue")
        spec_pred = overlay_mask_rgb(spec, pred_mask, grid_size, "red")

        # Combined overlay (optional)
        spec_combined = overlay_mask_rgb(spec, context_mask, grid_size, "blue")
        spec_combined = overlay_mask_rgb(
            spec_combined[..., 0], pred_mask, grid_size, "red"
        )
        # flip upside down to handle how imsave doesn't have "lower" option
        spec_context = np.flipud(spec_context)
        spec_pred = np.flipud(spec_pred)
        spec_combined = np.flipud(spec_combined)
        # Save images
        plt.imsave(os.path.join(OUT_DIR, f"sample_{i}_context.png"), spec_context)
        plt.imsave(os.path.join(OUT_DIR, f"sample_{i}_pred.png"), spec_pred)
        plt.imsave(os.path.join(OUT_DIR, f"sample_{i}_combined.png"), spec_combined)

        # -----------------------------
        # Mixup detection
        # -----------------------------
        num_active_labels = labels.sum().item()

        if num_active_labels > 1:
            print(f"[Mixup candidate] sample {i} has {num_active_labels} labels")


if __name__ == "__main__":
    main()