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
    def __init__(
        self,
        num_patches: int,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=2,
        num_heads=12,
    ):
        super().__init__()

        self.num_patches = num_patches

        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, predictor_embed_dim)
        )

        self.blocks = nn.ModuleList(
            [PredictorBlock(predictor_embed_dim, num_heads) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(predictor_embed_dim)
        self.proj = nn.Linear(predictor_embed_dim, embed_dim)

    def forward(self, h, hide_mask):
        """
        h: (B, N, D)
        hide_mask: (B, N) bool tensor, True = hidden / predict this token
        """
        if hide_mask is None:
            raise ValueError("JEPAPredictor.forward expects hide_mask, got None")
        if hide_mask.dtype != torch.bool:
            raise TypeError(f"hide_mask must be bool, got {hide_mask.dtype}")

        B, N, _ = h.shape
        if N > self.num_patches:
            raise ValueError(
                f"Input has N={N} patches, but predictor was built for {self.num_patches}"
            )
        x = self.predictor_embed(h)

        pos = self.pos_embed[:, :N, :].expand(B, -1, -1)

        visible_mask = ~hide_mask

        # Add positional embeddings only to visible tokens
        x = x + pos * visible_mask.unsqueeze(-1)

        # Gather visible context tokens
        context_tokens = x[visible_mask].view(B, -1, x.shape[-1])

        # Gather positional embeddings for hidden tokens
        pos_hidden = pos[hide_mask].view(B, -1, x.shape[-1])
        hidden_idx = torch.where(hide_mask[0])[0]

        # Create hidden tokens from mask token + positional encoding
        pred_tokens = self.mask_token.expand(B, pos_hidden.shape[1], -1)
        pred_tokens = pred_tokens + pos_hidden

        # Concatenate visible context tokens and prediction tokens
        x_full = torch.cat([context_tokens, pred_tokens], dim=1)

        for blk in self.blocks:
            x_full = blk(x_full)

        x_full = self.norm(x_full)

        # Final predicted embeddings correspond to the hidden part
        pred = x_full[:, context_tokens.shape[1]:]
        pred = self.proj(pred)

        return pred
if __name__ == '__main__':

    # Tiny synthetic setup
    B, N, D = 2, 8, 4
    
    h = torch.zeros(B, N, D)
    for b in range(B):
        for n in range(N):
            h[b, n] = n  # patch index encoded in features
    
    mask = torch.tensor([
        [False, True, False, True, False, False, True, False],
        [True, False, False, True, False, True, False, False],
    ], dtype=torch.bool)
    
    # Tiny predictor instance
    predictor = JEPAPredictor(
        num_patches=N,
        embed_dim=D,
        predictor_embed_dim=8,
        depth=1,
        num_heads=2,
    )
    
    # Call forward
    with torch.no_grad():
        pred = predictor(h, mask)
    
    print("pred shape:", pred.shape)