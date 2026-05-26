import torch
import torch.nn as nn
from timm.loss import AsymmetricLossMultiLabel

from mae_module import build_bioacoustic_mae_module



class MeanReducedLoss(nn.Module):
    """
    Wraps timm's AsymmetricLossMultiLabel and mean-reduces the summed loss
    by batch size (or by number of samples in the target tensor).

    This keeps the underlying ASL behavior unchanged, while making the
    magnitude less dependent on batch size.
    """

    def __init__(self, base_loss: nn.Module):
        super().__init__()
        self.base_loss = base_loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.base_loss(logits, targets)

        # If the base loss already returns a scalar, divide by batch size.
        # If it returns per-sample loss, mean it.
        if loss.ndim == 0:
            return loss / max(logits.shape[0], 1)

        return loss.mean()


def build_model(num_classes: int, **overrides):
    """
    Build the joint MAE + classifier model.

    Default settings are chosen to be close to the Audio-JEPA / MAE-style
    recipe you were already experimenting with, but now using the MAE module.
    """
    defaults = dict(
        sample_rate=32000,
        window_duration=5.0,
        n_mels=128,
        n_time_frames=512,
        n_fft=1024,
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_heads=16,
        mask_ratio=0.75,
        lambda_recon=1.0,
        lambda_cls=0.01,
        objective_mode="joint",
        classifier_type= "attention",
        classifier_kwargs= None,
        use_cls_token= False,
        optimizer_type="adam",
        learning_rate=3e-4,
        weight_decay=0.05,
        normalize_audio=True,
        spec_norm_path="/home/svu/e1583377/MultitaskPretrainingBioacoustics/scripts/power_spec_norm_stats_XCL.json",
        apply_spec_norm=True,
        criterion_recon=nn.MSELoss(),
        criterion_cls=MeanReducedLoss(
            AsymmetricLossMultiLabel(
                gamma_pos=0.0,
                gamma_neg=4.0,
                clip=0.05,
            )
        ),
        debug_every_n_batches=10,
    )

    defaults.update(overrides)
    return build_bioacoustic_mae_module(num_classes=num_classes, **defaults)