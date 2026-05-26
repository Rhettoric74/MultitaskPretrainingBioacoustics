from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import lightning as L
import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
import dcgd


# =========================================================
# Mask helpers
# =========================================================
def apply_keep_indices(x: torch.Tensor, masks: List[torch.Tensor]) -> torch.Tensor:
    """Gather patch tokens by index list.

    Args:
        x: [B, N, D]
        masks: list of [B, K] LongTensors containing patch indices to KEEP

    Returns:
        [B * len(masks), K, D]
    """
    if x.dim() != 3:
        raise ValueError(f"x must have shape [B, N, D], got {tuple(x.shape)}")

    out = []
    for m in masks:
        if m.dtype != torch.long:
            raise TypeError(f"Mask indices must be torch.long, got {m.dtype}")
        if m.dim() != 2:
            raise ValueError(f"Each mask must have shape [B, K], got {tuple(m.shape)}")
        idx = m.unsqueeze(-1).expand(-1, -1, x.size(-1))
        out.append(torch.gather(x, dim=1, index=idx))
    return torch.cat(out, dim=0)


def generate_random_keep_indices(
    batch_size: int,
    num_patches: int,
    context_ratio: float = 0.25,
    device: torch.device | str = "cpu",
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Generate complementary context/prediction index sets.

    Returns:
        context_indices: list containing one [B, Kc] LongTensor
        prediction_indices: list containing one [B, Kp] LongTensor
    """
    if not (0.0 < context_ratio < 1.0):
        raise ValueError("context_ratio must be between 0 and 1")

    num_context = int(round(context_ratio * num_patches))
    num_context = max(1, min(num_context, num_patches - 1))

    context_list = []
    pred_list = []
    for _ in range(batch_size):
        perm = torch.randperm(num_patches, device=device)
        ctx = torch.sort(perm[:num_context]).values
        pred = torch.sort(perm[num_context:]).values
        context_list.append(ctx)
        pred_list.append(pred)

    return [torch.stack(context_list, dim=0)], [torch.stack(pred_list, dim=0)]


# =========================================================
# Small utilities
# =========================================================
def compute_grad_metrics(model, loss_recon, loss_cls):
    """Cosine similarity between encoder gradients induced by reconstruction and classification."""
    named_params = [(name, p) for name, p in model.encoder.named_parameters() if p.requires_grad]
    if not named_params:
        z = torch.tensor(0.0, device=loss_recon.device)
        return z, z, z

    names = [n for n, _ in named_params]
    params = [p for _, p in named_params]

    grads_recon = torch.autograd.grad(loss_recon, params, retain_graph=True, allow_unused=True)
    grads_cls = torch.autograd.grad(loss_cls, params, retain_graph=True, allow_unused=True)

    g1_list = []
    g2_list = []
    for name, g1, g2 in zip(names, grads_recon, grads_cls):
        if g1 is not None and g2 is not None:
            g1_list.append(g1.detach().float().reshape(-1))
            g2_list.append(g2.detach().float().reshape(-1))

    if len(g1_list) == 0:
        z = torch.tensor(0.0, device=loss_recon.device)
        return z, z, z

    g1 = torch.cat(g1_list)
    g2 = torch.cat(g2_list)
    eps = 1e-12
    cos = torch.dot(g1, g2) / (g1.norm() * g2.norm() + eps)
    return cos, g1.norm(), g2.norm()


def _load_spec_norm(path: str | Path) -> Tuple[float, float]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict) or "mean" not in obj or "std" not in obj:
        raise ValueError(f"Spectrogram norm file must contain mean/std keys: {path}")
    return float(obj["mean"]), float(obj["std"])


class SpectrogramNormalize(nn.Module):
    def __init__(self, mean: float, std: float, eps: float = 1e-8):
        super().__init__()
        self.register_buffer("mean", torch.tensor(float(mean), dtype=torch.float32))
        self.register_buffer("std", torch.tensor(float(std), dtype=torch.float32))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)


# =========================================================
# Transformer blocks / positional embeddings
# =========================================================
def get_1d_sincos_pos_embed(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2)))
    out = torch.einsum("n,d->nd", pos.float(), omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w):
    assert embed_dim % 2 == 0
    grid_h_vec = torch.arange(grid_h, dtype=torch.float32)
    grid_w_vec = torch.arange(grid_w, dtype=torch.float32)
    grid = torch.meshgrid(grid_w_vec, grid_h_vec, indexing="ij")
    grid = torch.stack(grid, dim=0).reshape(2, -1)
    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, grid[1])
    return torch.cat([emb_h, emb_w], dim=1)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        y, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + self.drop(y)
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


# =========================================================
# Patch embedding / encoder / decoder
# =========================================================
class PatchEmbed(nn.Module):
    def __init__(self, input_size=(128, 512), patch_size=(16, 16), in_chans=1, embed_dim=768):
        super().__init__()
        img_height, img_width = input_size
        patch_height, patch_width = patch_size
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_patches_h = img_height // patch_height
        self.num_patches_w = img_width // patch_width
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

"""
class SpectrogramEncoder(nn.Module):
    def __init__(
        self,
        input_size=(128, 512),
        patch_size=(16, 16),
        in_chans=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(input_size=input_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        pos_embed = get_2d_sincos_pos_embed(embed_dim, self.patch_embed.num_patches_h, self.patch_embed.num_patches_w)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0), persistent=False)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, context_indices=None):
        x = self.patch_embed(x)
        x = x + self.pos_embed.to(device=x.device, dtype=x.dtype)
        if context_indices is not None:
            x = apply_keep_indices(x, context_indices)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)
"""
        
class SpectrogramEncoder(nn.Module):
    def __init__(
        self,
        input_size=(128, 512),
        patch_size=(16, 16),
        in_chans=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        
        self.patch_embed = PatchEmbed(
            input_size=input_size, 
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=embed_dim
        )
        
        # Number of patches
        num_patches = self.patch_embed.num_patches
        
        # Position embeddings for patches (sinusoidal)
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim, 
            self.patch_embed.num_patches_h, 
            self.patch_embed.num_patches_w
        )
        
        if use_cls_token:
            # Learnable CLS token (only this, no extra positional embedding)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            
            # Position embeddings for patches only (CLS doesn't need position)
            self.register_buffer(
                "pos_embed", 
                pos_embed.unsqueeze(0),
                persistent=False
            )
        else:
            self.register_buffer("pos_embed", pos_embed.unsqueeze(0), persistent=False)
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=mlp_ratio) 
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize CLS token
        if use_cls_token:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x, context_indices=None):
        # x: [B, C, H, W]
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, N, D]
        
        # Add position embeddings to patches
        x = x + self.pos_embed.expand(B, -1, -1)
        
        # Add CLS token if used
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, D]
        
        # Apply masking if needed
        if context_indices is not None:
            if self.use_cls_token:
                # Keep CLS token (index 0) plus the selected patch indices
                # Shift patch indices by 1 to account for CLS at front
                cls_mask = torch.zeros(B, 1, dtype=torch.long, device=x.device)
                shifted_indices = context_indices[0] + 1
                keep_indices = torch.cat([cls_mask, shifted_indices], dim=1)
                x = apply_keep_indices(x, [keep_indices])
            else:
                x = apply_keep_indices(x, context_indices)
        
        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        
        # Return all tokens and CLS token separately
        if self.use_cls_token:
            cls_token = x[:, 0:1, :]  # [B, 1, D]
            patch_tokens = x[:, 1:, :]  # [B, N, D] (or [B, Kc, D] if masked)
            return x, cls_token, patch_tokens
        else:
            return x, None, None


class MAEDecoder(nn.Module):
    def __init__(
        self,
        num_patches_h: int,
        num_patches_w: int,
        patch_dim: int,
        encoder_dim=768,
        decoder_dim=384,
        depth=4,
        num_heads=6,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.num_patches = num_patches_h * num_patches_w
        self.patch_dim = patch_dim
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.register_buffer(
            "pos_embed",
            get_2d_sincos_pos_embed(decoder_dim, num_patches_h, num_patches_w).unsqueeze(0),
            persistent=False,
        )
        self.blocks = nn.ModuleList([Block(decoder_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred_head = nn.Linear(decoder_dim, patch_dim)

    def forward(self, h_ctx, context_indices, prediction_indices):
        B, Kc, _ = h_ctx.shape
        x = self.decoder_embed(h_ctx)
        full = self.mask_token.expand(B, self.num_patches, -1).clone()

        ctx_idx = context_indices[0].unsqueeze(-1).expand(-1, -1, x.size(-1))
        full.scatter_(dim=1, index=ctx_idx, src=x)
        full = full + self.pos_embed.to(device=full.device, dtype=full.dtype)

        for blk in self.blocks:
            full = blk(full)
        full = self.norm(full)
        pred_full = self.pred_head(full)
        return apply_keep_indices(pred_full, prediction_indices)


# =========================================================
# On-device audio frontend
# =========================================================
class AudioFrontend(nn.Module):
    def __init__(
        self,
        sample_rate: int = 32000,
        window_duration: float = 5.0,
        n_mels: int = 128,
        n_time_frames: int = 512,
        n_fft: int = 1024,
        normalize_audio: bool = True,
        spec_norm_path: Optional[str] = None,
        apply_spec_norm: bool = True,
        preemphasis_coeff = 0.97,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.window_samples = int(window_duration * sample_rate)
        self.n_mels = n_mels
        self.n_time_frames = n_time_frames
        self.n_fft = n_fft
        self.normalize_audio = normalize_audio
        self.apply_spec_norm = apply_spec_norm

        hop_length = (self.window_samples - n_fft) // (n_time_frames - 1)
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0,
            n_mels=n_mels,
            center=False,
        )

        self.spec_norm: Optional[SpectrogramNormalize] = None
        if spec_norm_path is not None:
            mean, std = _load_spec_norm(spec_norm_path)
            self.spec_norm = SpectrogramNormalize(mean, std)
            print(f"[INFO] Loaded spectrogram normalization from {spec_norm_path}")
            print(f"[INFO] Spectrogram norm mean={mean:.6f}, std={std:.6f}")
        if preemphasis_coeff:
            self.preemphasis = torchaudio.transforms.Preemphasis(coeff=0.97)
        else:
            self.preemphasis = None


    def _crop_or_pad_constituents(self, waveforms: torch.Tensor, lengths: torch.Tensor, training: bool) -> torch.Tensor:
        """waveforms: [B, M, Tmax] -> [B, M, window_samples]"""
        B, M, T = waveforms.shape
        out = waveforms.new_zeros((B, M, self.window_samples))
        for b in range(B):
            for m in range(M):
                L = int(lengths[b, m].item())
                if L <= 0:
                    continue
                x = waveforms[b, m, :L]
                if L < self.window_samples:
                    out[b, m, :L] = x
                elif L == self.window_samples:
                    out[b, m] = x
                else:
                    if training:
                        start = torch.randint(0, L - self.window_samples + 1, (1,), device=waveforms.device).item()
                    else:
                        start = max(0, (L - self.window_samples) // 2)
                    out[b, m] = x[start : start + self.window_samples]
        return out

    def _normalize_waveforms(self, waveforms: torch.Tensor) -> torch.Tensor:
        if not self.normalize_audio:
            return waveforms
        max_val = waveforms.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        return waveforms * (0.25 / max_val)

    def forward(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        mix_weights: torch.Tensor,
        constituent_labels: Optional[torch.Tensor] = None,
        mix_counts: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return spectrograms and mixed labels.

        Args:
            waveforms: [B, M, T]
            waveform_lengths: [B, M]
            mix_weights: [B, M]
            constituent_labels: [B, M, C]
            mix_counts: [B]
        """
        # Moved to dataloader to save GPU memory and avoid extra long waveforms on the GPU
        #waveforms = self._crop_or_pad_constituents(waveforms, waveform_lengths, training=training)
    
        if self.preemphasis:
            waveforms = self.preemphasis(waveforms)
        
        waveforms = self._normalize_waveforms(waveforms)
        valid = torch.arange(waveforms.size(1), device=waveforms.device).unsqueeze(0) < mix_counts.unsqueeze(1)
        weights = mix_weights.to(device=waveforms.device, dtype=waveforms.dtype) * valid.to(dtype=waveforms.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        
        mixed_waveforms = (waveforms * weights.unsqueeze(-1)).sum(dim=1)
        mixed_labels = None
        if constituent_labels is not None:
            # weighted labels is bad for multi-label objective, should be true multi-hot.
            #mixed_labels = (constituent_labels.to(device=waveforms.device, dtype=waveforms.dtype) * weights.unsqueeze(-1)).sum(dim=1)
            mixed_labels = constituent_labels.max(dim=1)[0]  # [B, C]

        # Instead of peak normalization after mixing:
        gain = 1.0 / torch.sqrt((weights ** 2).sum(dim=1, keepdim=True))  # [B, 1]
        mixed_waveforms = mixed_waveforms * gain  # [B, 1] -> [B, 1, 1] to broadcast over time
        specs = self.mel_spectrogram(mixed_waveforms)
        specs = torch.clamp(specs, min=1e-10).log()
        if self.apply_spec_norm and self.spec_norm is not None:
            specs = self.spec_norm(specs)

        return specs.unsqueeze(1), mixed_labels
        
class AttentionPoolClassifier(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        scores = self.attn(x).squeeze(-1)              # [B, N]
        weights = torch.softmax(scores, dim=1)         # [B, N]
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)  # [B, D]
        return self.classifier(pooled)
        
class PatchProtoPNetClassifier(nn.Module):
    """
    ProtoPNet-style classifier over patch/token embeddings.

    Input:
        x: [B, N, D] token embeddings from the encoder
    Output:
        logits: [B, C]
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        num_prototypes_per_class: int = 5,
        activation: str = "cosine",   # "cosine" or "euclidean"
        temperature: float = 0.1,
        learnable_readout: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        self.num_prototypes = num_classes * num_prototypes_per_class
        self.activation = activation
        self.temperature = temperature
        self.learnable_readout = learnable_readout

        # Class-specific prototypes: [C, K, D]
        self.prototypes = nn.Parameter(
            torch.randn(num_classes, num_prototypes_per_class, embed_dim) * 0.1
        )
        self.use_learnable_aggregation = bool(learnable_readout)

        if learnable_readout:
            # Class-specific linear weights over prototype activations: [C, K]
            self.class_weights = nn.Parameter(
                torch.randn(num_classes, num_prototypes_per_class) * 0.01
            )
            self.class_bias = nn.Parameter(torch.zeros(num_classes))

    def _similarities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute token-prototype similarities.

        Args:
            x: [B, N, D]
        Returns:
            sim: [B, C, K, N]
        """
        if self.activation == "cosine":
            x_n = F.normalize(x, dim=-1)
            p_n = F.normalize(self.prototypes, dim=-1)
            sim = torch.einsum("bnd,ckd->bckn", x_n, p_n)
            return sim / self.temperature

        if self.activation == "euclidean":
            # Negative squared Euclidean distance as similarity
            dot = torch.einsum("bnd,ckd->bckn", x, self.prototypes)
            x_sq = (x ** 2).sum(dim=-1)[:, None, None, :]              # [B, 1, 1, N]
            p_sq = (self.prototypes ** 2).sum(dim=-1)[None, :, :, None] # [1, C, K, 1]
            dist_sq = x_sq + p_sq - 2.0 * dot
            return -dist_sq / self.temperature

        raise ValueError(f"Unknown activation: {self.activation}")

    def forward(self, x: torch.Tensor, chunk_size: int = 256) -> torch.Tensor:
        # x: [B, N, D]
        B, N, D = x.shape
        x = F.normalize(x, dim=-1)
        protos = F.normalize(self.prototypes.view(-1, D), dim=-1)  # [P, D]
    
        proto_acts_chunks = []
        # chunking similarity computations to avoid OOMs
        for start in range(0, protos.shape[0], chunk_size):
            p = protos[start:start + chunk_size]  # [Pc, D]
            sim = torch.einsum("bnd,pd->bpn", x, p) / self.temperature  # [B, Pc, N]
            acts = sim.max(dim=-1).values  # [B, Pc]
            proto_acts_chunks.append(acts)
    
        proto_acts = torch.cat(proto_acts_chunks, dim=1)  # [B, P]
        proto_acts = proto_acts.view(B, self.num_classes, self.num_prototypes_per_class)
    
        if self.use_learnable_aggregation:
            # proto_acts: [B, C, K]
            logits = (proto_acts * self.class_weights.unsqueeze(0)).sum(dim=-1)
            logits = logits + self.class_bias
        else:
            logits = proto_acts.max(dim=-1).values
    
        return logits


# =========================================================
# MAE Module
# =========================================================
class BioacousticMAEModule(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        sample_rate: int = 32000,
        window_duration: float = 5.0,
        n_mels: int = 128,
        n_time_frames: int = 512,
        n_fft: int = 1024,
        patch_size: int = 16,
        encoder_embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_heads: int = 12,
        decoder_embed_dim: int = 384,
        decoder_depth: int = 4,
        decoder_heads: int = 6,
        classifier_type: str = "attention",
        classifier_kwargs: Optional[Dict] = {},
        mask_ratio: float = 0.75,
        lambda_recon: float = 1.0,
        lambda_cls: float = 1.0,
        objective_mode: str = "joint",
        optimizer_type: str = "adam",
        learning_rate: float = 3e-4,
        weight_decay: float = 0.05,
        normalize_audio: bool = True,
        spec_norm_path: Optional[str] = "home/svu/e1583377/MultitaskPretrainingBioacoustics/scripts/power_spec_norm_stats_XCL.json",
        apply_spec_norm: bool = True,
        criterion_recon: Optional[nn.Module] = None,
        criterion_cls: Optional[nn.Module] = None,
        debug_every_n_batches: int = 100,
        use_cls_token = True,
    ):
        super().__init__()

        self.automatic_optimization = False
        self.save_hyperparameters(logger=False)

        self.num_classes = num_classes
        self.mask_ratio = mask_ratio
        self.lambda_recon = lambda_recon
        self.lambda_cls = lambda_cls
        self.objective_mode = objective_mode
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_cls_token = use_cls_token

        self.frontend = AudioFrontend(
            sample_rate=sample_rate,
            window_duration=window_duration,
            n_mels=n_mels,
            n_time_frames=n_time_frames,
            n_fft=n_fft,
            normalize_audio=normalize_audio,
            spec_norm_path=spec_norm_path,
            apply_spec_norm=apply_spec_norm,
        )

        self.grid_h = n_mels // patch_size
        self.grid_w = n_time_frames // patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.patch_size = patch_size
        self.patch_dim = patch_size * patch_size

        self.encoder = SpectrogramEncoder(
            input_size=(n_mels, n_time_frames),
            patch_size=(patch_size, patch_size),
            in_chans=1,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            use_cls_token = self.use_cls_token,
        )
        self.decoder = MAEDecoder(
            num_patches_h=self.grid_h,
            num_patches_w=self.grid_w,
            patch_dim=self.patch_dim,
            encoder_dim=encoder_embed_dim,
            decoder_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_heads,
        )
        if classifier_type == "attention":
            self.classifier = AttentionPoolClassifier(
                embed_dim=encoder_embed_dim,
                num_classes=num_classes,
                **classifier_kwargs,
            )
        elif classifier_type == "proto":
            self.classifier = PatchProtoPNetClassifier(
                embed_dim=encoder_embed_dim,
                num_classes=num_classes,
                **classifier_kwargs,
            )
        elif classifier_type == "linear":
            self.use_cls_token = True
            self.classifier = nn.Linear(encoder_embed_dim, num_classes)
        else:
            raise ValueError(f"Unknown classifier_type: {classifier_type}")

        self.criterion_recon = criterion_recon if criterion_recon is not None else nn.MSELoss()
        self.criterion_cls = criterion_cls if criterion_cls is not None else nn.BCEWithLogitsLoss()

        self.debug_every_n_batches = debug_every_n_batches

    # -------------------------------------------------
    # Forward helpers
    # -------------------------------------------------
    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, H, W] -> [B, N, patch_dim]"""
        B, C, H, W = x.shape
        p = self.patch_size
        if H % p != 0 or W % p != 0:
            raise ValueError(f"Spectrogram size {(H, W)} must be divisible by patch size {p}")
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.reshape(B, -1, C * p * p)

    def _pool_context(self, h: torch.Tensor) -> torch.Tensor:
        return h.mean(dim=1)

    def forward(self, batch: Dict[str, torch.Tensor], training: bool = True):
        waveforms = batch["waveforms"]
        waveform_lengths = batch["waveform_lengths"]
        mix_weights = batch["mix_weights"]
        constituent_labels = batch["constituent_labels"]
        mix_counts = batch.get("mix_counts", None)
        with torch.no_grad():
            specs, labels = self.frontend(
                waveforms,
                waveform_lengths,
                mix_weights,
                constituent_labels=constituent_labels,
                mix_counts=mix_counts,
                training=self.training,
            )
            return specs, labels

    def model_step(self, batch):
        specs, labels = self.forward(batch, training=self.training)
        if labels is None:
            labels = batch["labels"].float().to(specs.device)
    
        B, _, H, W = specs.shape
        num_patches = (H // self.patch_size) * (W // self.patch_size)
        
        # Use mask_ratio=0 for validation (no masking)
        effective_mask_ratio = 0.0 if not self.training else self.mask_ratio
        
        if effective_mask_ratio > 0:
            context_ratio = 1.0 - effective_mask_ratio
            context_indices, prediction_indices = generate_random_keep_indices(
                batch_size=B,
                num_patches=num_patches,
                context_ratio=context_ratio,
                device=specs.device,
            )
        else:
            # No masking - use all patches
            context_indices = None
            prediction_indices = None
        
        loss_recon = torch.tensor(0.0, device=specs.device)
        loss_cls = torch.tensor(0.0, device=specs.device)
        logits = None
        
        # Reconstruction branch: uses masked patches (for MAE objective)
        if self.objective_mode in ("mae", "joint") and effective_mask_ratio > 0:
            # Encode with masking for reconstruction
            _, _, h_patches_masked = self.encoder(specs, context_indices)
            pred_patches = self.decoder(h_patches_masked, context_indices, prediction_indices)
            target_patches = self._patchify(specs)
            target_patches = apply_keep_indices(target_patches, prediction_indices)
            loss_recon = self.criterion_recon(pred_patches, target_patches)
        
        # Classification branch: ALWAYS uses full patches (no masking)
        if self.objective_mode in ("class", "joint"):
            # Encode with full spectrogram (context_indices=None)
            _, cls_token, h_patches_full = self.encoder(specs, None)
            
            if self.use_cls_token:
                logits = self.classifier(cls_token.squeeze(1))  # [B, D] -> [B, C]
            else:
                logits = self.classifier(h_patches_full) if h_patches_full is not None else self.classifier(h_all)
            loss_cls = self.criterion_cls(logits, labels)
        
        # Combine losses
        loss = self.lambda_recon * loss_recon + self.lambda_cls * loss_cls
        
        # Metrics
        prefix = "train" if self.training else "val"
        metrics = {
            f"{prefix}/recon_loss": loss_recon.detach(),
            f"{prefix}/cls_loss": loss_cls.detach(),
            f"{prefix}/total_loss": loss.detach(),
        }
        
        return loss, loss_recon, loss_cls, logits, metrics



    # -------------------------------------------------
    # Lightning hooks
    # -------------------------------------------------
    def training_step(self, batch, batch_idx):
        loss, loss_recon, loss_cls, logits, metrics = self.model_step(batch)
        self.log_dict(metrics, on_step=True, on_epoch=False, prog_bar=False, logger=True)
    
        if self.objective_mode == "joint" and batch_idx % 100 == 0:
            cos_sim, norm_recon, norm_cls = compute_grad_metrics(self, loss_recon, loss_cls)
            self.log("train/grad_cosine", cos_sim, prog_bar=True)
            self.log("train/grad_norm_recon", norm_recon)
            self.log("train/grad_norm_cls", norm_cls)
    
        opt = self.optimizers()
    
        if self.optimizer_type == "adam":
            opt.zero_grad(set_to_none=True)
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            opt.step()
    
            sched = self.lr_schedulers()
            if sched is not None:
                sched.step()
    
        elif self.optimizer_type == "dcgd":
            if hasattr(opt, "optimizer"):
                dcgd_opt = opt.optimizer
            else:
                dcgd_opt = opt
    
            dcgd_opt.zero_grad(set_to_none=True)
            dcgd_opt.step([loss_recon, loss_cls])
    
            sched = self.lr_schedulers()
            if sched is not None:
                sched.step()
    
        else:
            raise ValueError(f"Unknown optimizer_type: {self.optimizer_type}")
        """
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/recon_loss", loss_recon, prog_bar=True)
        self.log("train/cls_loss", loss_cls, prog_bar=True)
        """
    
        if self.optimizer_type == "dcgd" and hasattr(opt, "optimizer"):
            lr = opt.optimizer.param_groups[0]["lr"]
        else:
            lr = opt.param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True)
        
        if torch.cuda.is_available() and batch_idx % 1000 == 0:
            device = torch.cuda.current_device()
        
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
            max_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
        
            print(
                f"[GPU MEM] "
                f"alloc={allocated:.2f} GB | "
                f"reserved={reserved:.2f} GB | "
                f"max_alloc={max_allocated:.2f} GB | "
                f"max_reserved={max_reserved:.2f} GB"
            )
    
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, loss_recon, loss_cls, logits, metrics = self.model_step(batch)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/recon_loss", loss_recon, sync_dist=True)
        self.log("val/cls_loss", loss_cls, sync_dist=True)
        if logits is not None:
            self.log("val/variance", torch.var(logits), sync_dist=True)
    
    def infer_logits(self, batch: Dict[str, torch.Tensor]):
        specs, _ = self.forward(batch, training=False)
        h_full, cls_token, patches = self.encoder(specs, None)
        
        # Check what type of classifier we're using
        if self.use_cls_token:
            # Linear classifier expects CLS token
            return self.classifier(cls_token.squeeze(1))  # [B, D] -> [B, C]
        else:
            # Attention pooler or other classifiers expect patch tokens
            return self.classifier(patches if patches is not None else h_full)

    # -------------------------------------------------
    # Optimizers
    # -------------------------------------------------
    def _parameter_groups(self):
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 1 or name.endswith("bias") or "norm" in name.lower():
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {"params": decay, "weight_decay": self.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def configure_optimizers(self):
        all_params = []
    
        for p in self.encoder.parameters():
            all_params.append(p)
        decoder_start_idx = len(all_params)
    
        for p in self.decoder.parameters():
            all_params.append(p)
        classifier_start_idx = len(all_params)
    
        for p in self.classifier.parameters():
            all_params.append(p)
    
        classifier_param_indices = list(range(classifier_start_idx, len(all_params)))
    
        if self.optimizer_type == "dcgd":
            print("Initializing DCGD optimizer")
            base_opt = torch.optim.Adam(
                all_params,
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.0,
            )
    
            from dcgd import DCGD
            optimizer = DCGD(
                base_opt,
                num_pde=1,
                type="center",
                classifier_param_indices=classifier_param_indices,
                predictor_start_idx=decoder_start_idx,
                classifier_start_idx=classifier_start_idx,
            )
            return optimizer
    
        elif self.optimizer_type == "adam":
            optimizer = torch.optim.AdamW(
                all_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
    
            total_steps = self.trainer.estimated_stepping_batches
            warmup_steps = max(1, int(0.1 * total_steps))
    
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(
                        optimizer,
                        start_factor=0.05,
                        end_factor=1.0,
                        total_iters=warmup_steps,
                    ),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=max(1, total_steps - warmup_steps),
                        eta_min=3e-5,
                    ),
                ],
                milestones=[warmup_steps],
            )
    
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
    
        else:
            raise ValueError(f"Unknown optimizer_type: {self.optimizer_type}")

# =========================================================
# Factory helper
# =========================================================
def build_bioacoustic_mae_module(num_classes: int, **kwargs) -> BioacousticMAEModule:
    return BioacousticMAEModule(num_classes=num_classes, **kwargs)
