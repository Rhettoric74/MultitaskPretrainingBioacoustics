# predictor.py
# based on code from https://github.com/LudovicTuncay/Audio-JEPA.git
"""@inproceedings{tuncay2025audio,
  title = {{Audio-JEPA: Joint-Embedding Predictive Architecture for Audio Representation Learning}},
  author = {Tuncay, Ludovic and Labb{\'e}, Etienne and Benetos, Emmanouil and Pellegrini, Thomas},
  booktitle = {ICME 2025},
  address = {Nantes, France},
  year = {2025},
  url = {https://hal.science/hal-05128180}
}
"""
# predictor.py
# simplified single-mask JEPA predictor

import torch
import torch.nn as nn


# =========================================================
# Transformer Block
# =========================================================
class PredictorBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


# =========================================================
# JEPA Predictor
# hide_mask: True means this patch is hidden / to be predicted
# =========================================================
class JEPAPredictor(nn.Module):
    def __init__(self, num_patches: int, embed_dim=768, predictor_embed_dim=384, depth=2, num_heads=8):
        super().__init__()
        self.num_patches = num_patches

        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, predictor_embed_dim))

        self.blocks = nn.ModuleList(
            [PredictorBlock(predictor_embed_dim, num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(predictor_embed_dim)
        self.proj = nn.Linear(predictor_embed_dim, embed_dim)

    def forward(self, h, prediction_indices):
        """
        h: [B, Kc, D] context tokens from encoder
        prediction_indices: list containing one [B, Kp] LongTensor
        """
        if len(prediction_indices) != 1:
            raise ValueError("For now this version expects exactly one prediction mask per sample.")

        pred_idx = prediction_indices[0]
        if pred_idx.dtype != torch.long:
            raise TypeError(f"prediction_indices must be torch.long, got {pred_idx.dtype}")

        B, Kc, _ = h.shape
        x_ctx = self.predictor_embed(h)  # [B, Kc, d]

        # Prediction tokens get mask token + position of the hidden patches
        pos = self.pos_embed.expand(B, -1, -1)  # [B, N, d]
        pred_pos = torch.gather(
            pos,
            dim=1,
            index=pred_idx.unsqueeze(-1).expand(-1, -1, pos.size(-1)),
        )  # [B, Kp, d]

        pred_tokens = self.mask_token.expand(B, pred_idx.size(1), -1) + pred_pos

        # Concatenate context tokens and prediction tokens
        x_full = torch.cat([x_ctx, pred_tokens], dim=1)

        for blk in self.blocks:
            x_full = blk(x_full)

        x_full = self.norm(x_full)

        pred = x_full[:, Kc:, :]
        pred = self.proj(pred)
        return pred