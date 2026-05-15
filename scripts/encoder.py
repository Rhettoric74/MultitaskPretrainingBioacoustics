import torch
import torch
import torch.nn as nn
import numpy as np
import timm
from utils import apply_keep_indices


# =========================================================
# Correct 2D Sin/Cos Positional Embedding
# =========================================================
def get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w):
    """
    Returns:
        pos_embed: (grid_h * grid_w, embed_dim)
    """
    grid_h_vec = np.arange(grid_h, dtype=np.float32)
    grid_w_vec = np.arange(grid_w, dtype=np.float32)

    grid = np.meshgrid(grid_w_vec, grid_h_vec)  # (W, H)
    grid = np.stack(grid, axis=0)               # (2, H, W)
    grid = grid.reshape(2, -1)                 # (2, N)

    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, grid[1])

    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed(embed_dim, pos):
    assert embed_dim % 2 == 0

    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega = 1. / (10000 ** (omega / (embed_dim / 2)))

    out = np.einsum('n,d->nd', pos, omega)

    sin = np.sin(out)
    cos = np.cos(out)

    return np.concatenate([sin, cos], axis=1)


# =========================================================
# Encoder
# =========================================================
class JEPATimmViT(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224", img_size=(128, 512), pretrained=False):
        super().__init__()
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=img_size,
            in_chans=1,
            num_classes=0,
        )

        self.embed_dim = self.vit.embed_dim
        H_p, W_p = self.vit.patch_embed.grid_size
        num_patches = H_p * W_p

        pos_embed = get_2d_sincos_pos_embed(self.embed_dim, H_p, W_p)
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0)  # [1, N, D]
        self.register_buffer("pos_embed", pos_embed)

        print(f"[Encoder] Patch grid: {H_p} x {W_p} = {num_patches}")
        print(f"[Encoder] Pos embed shape: {self.pos_embed.shape}")

    def forward(self, x, context_indices=None):
        # Patch embedding
        x = self.vit.patch_embed(x)
        if x.dim() == 4:
            x = x.flatten(2).transpose(1, 2)  # [B, N, D]

        x = x + self.pos_embed

        # Keep only the context patches
        if context_indices is not None:
            x = apply_keep_indices(x, context_indices)  # [B, Kc, D] if one mask list item

        for blk in self.vit.blocks:
            x = blk(x)

        x = self.vit.norm(x)
        return x