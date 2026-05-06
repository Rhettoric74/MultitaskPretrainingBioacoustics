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
# Block masking (better for spectrograms)
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


# =========================================================
# Simple attention pooling (better than mean pooling)
# =========================================================
class AttentionPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D)
        returns: (B, D)
        """
        weights = torch.softmax(self.attn(x), dim=1)  # (B, N, 1)
        return (x * weights).sum(dim=1)


# =========================================================
# Gradient cosine similarity (for debugging multi-objective)
# =========================================================
def grad_cosine_similarity(model: nn.Module, loss1: torch.Tensor, loss2: torch.Tensor) -> float:
    """
    Compute cosine similarity between gradients of two losses.

    Useful for detecting gradient conflict.
    """
    grads1 = torch.autograd.grad(loss1, model.parameters(), retain_graph=True, allow_unused=True)
    grads2 = torch.autograd.grad(loss2, model.parameters(), retain_graph=True, allow_unused=True)

    g1 = torch.cat([g.flatten() for g in grads1 if g is not None])
    g2 = torch.cat([g.flatten() for g in grads2 if g is not None])

    return torch.nn.functional.cosine_similarity(g1, g2, dim=0).item()


# =========================================================
# Simple classifier head
# =========================================================
class ClassifierHead(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)