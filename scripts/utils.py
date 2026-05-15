import torch
import torch.nn as nn
from typing import List, Tuple


# =========================================================
# EMA (Exponential Moving Average) for target encoder
# =========================================================
class EMA:
    def __init__(self, momentum: float = 0.996):
        """
        Args:
            momentum: EMA decay factor (higher = slower updates)
        """
        self.momentum = momentum

    @torch.no_grad()
    def update(self, online_model: nn.Module, target_model: nn.Module):
        """
        Update target model parameters using EMA of online model.
        """
        for p_online, p_target in zip(online_model.parameters(), target_model.parameters()):
            p_target.data.mul_(self.momentum).add_(p_online.data * (1.0 - self.momentum))


# =========================================================
# Apply masks to patch embeddings
# =========================================================
def apply_masks(h, masks):
    """
    h: (B, N, D)
    masks: list of (B, N)

    Returns:
        (B * num_masks, num_masked, D)
    """
    B, N, D = h.shape
    outputs = []

    for mask in masks:
        masked = []

        for b in range(B):
            masked.append(h[b][mask[b]])

        masked = torch.stack(masked)  # (B, num_masked, D)
        outputs.append(masked)

    return torch.cat(outputs, dim=0)


# =========================================================
# Random mask generator (baseline)
# =========================================================
def generate_random_masks(
    batch_size: int,
    num_patches: int,
    mask_ratio: float = 0.5,
    num_masks: int = 4,
    device: torch.device = "cpu"
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Generate random JEPA-style masks.

    Args:
        batch_size: batch size
        num_patches: number of patches per sample
        mask_ratio: fraction of patches to mask
        num_masks: number of prediction masks
        device: torch device

    Returns:
        context_mask: (B, N)
        prediction_masks: list of (B, N)
    """
    prediction_masks = []

    num_masked = int(mask_ratio * num_patches)
    
    for _ in range(num_masks):
        masks = []
        for _ in range(batch_size):
            idx = torch.randperm(num_patches, device=device)[:num_masked]
    
            mask = torch.zeros(num_patches, dtype=torch.bool, device=device)
            mask[idx] = True
            masks.append(mask)
    
        masks = torch.stack(masks, dim=0)
        prediction_masks.append(masks)

    # Context mask = complement of first mask
    context_mask = ~prediction_masks[0]
    
    return context_mask, prediction_masks


# =========================================================
# Block masking (better for general audio with larger blocks per sound event)
# =========================================================
def generate_block_masks(
    batch_size: int,
    grid_size: Tuple[int, int],
    mask_ratio: float = 0.5,
    num_masks: int = 4,
    device: torch.device = "cpu"
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Generate block masks for time-frequency patches.

    Args:
        grid_size: (num_time_patches, num_freq_patches)
    """
    T, F = grid_size
    N = T * F

    def random_block():
        mask = torch.zeros(T, F, dtype=torch.bool, device=device)

        block_t = max(1, int(T * mask_ratio))
        block_f = max(1, int(F * mask_ratio))

        t0 = torch.randint(0, T - block_t + 1, (1,))
        f0 = torch.randint(0, F - block_f + 1, (1,))

        mask[t0:t0 + block_t, f0:f0 + block_f] = True
        return mask.flatten()

    prediction_masks = []

    for _ in range(num_masks):
        masks = torch.stack([random_block() for _ in range(batch_size)], dim=0)
        prediction_masks.append(masks)

    context_mask = ~prediction_masks[0]

    return context_mask, prediction_masks






def apply_keep_indices(x: torch.Tensor, masks: List[torch.Tensor]) -> torch.Tensor:
    """
    Audio-JEPA-style keep-index gather.

    Args:
        x: [B, N, D]
        masks: list of [B, K] LongTensors containing patch indices to KEEP

    Returns:
        [B * len(masks), K, D]
    """
    if x.dim() != 3:
        raise ValueError(f"x must have shape [B, N, D], got {tuple(x.shape)}")

    all_x = []
    for m in masks:
        if m.dtype != torch.long:
            raise TypeError(f"Mask indices must be torch.long, got {m.dtype}")
        if m.dim() != 2:
            raise ValueError(f"Each mask must have shape [B, K], got {tuple(m.shape)}")

        idx = m.unsqueeze(-1).expand(-1, -1, x.size(-1))  # [B, K, D]
        all_x.append(torch.gather(x, dim=1, index=idx))

    return torch.cat(all_x, dim=0)


def generate_random_keep_indices(
    batch_size: int,
    num_patches: int,
    context_ratio: float = 0.5,
    device: torch.device | str = "cpu",
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Returns:
        context_indices: list containing one [B, Kc] LongTensor
        prediction_indices: list containing one [B, Kp] LongTensor

    Semantics:
        - context_indices = patches to keep for the context encoder
        - prediction_indices = patches to keep for the target loss
    """
    if not (0.0 < context_ratio < 1.0):
        raise ValueError("context_ratio must be between 0 and 1")

    num_context = int(round(context_ratio * num_patches))
    num_context = max(1, min(num_context, num_patches - 1))
    num_pred = num_patches - num_context

    context_list = []
    pred_list = []

    for _ in range(batch_size):
        perm = torch.randperm(num_patches, device=device)
        ctx = torch.sort(perm[:num_context]).values
        pred = torch.sort(perm[num_context:]).values
        context_list.append(ctx)
        pred_list.append(pred)

    context_indices = [torch.stack(context_list, dim=0)]      # [B, Kc]
    prediction_indices = [torch.stack(pred_list, dim=0)]      # [B, Kp]
    return context_indices, prediction_indices