import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        loss = sigmoid_focal_loss(
            logits,
            targets,
            gamma=self.gamma,
            reduction="none"
        )

        # per-sample normalization
        pos_count = targets.sum(dim=1).clamp(min=1)

        loss_per_sample = loss.sum(dim=1)

        return (loss_per_sample / pos_count).mean()


class NormalizedMSELoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.eps = epsilon

    def forward(self, pred, target):
        # ---- Step 1: normalize target (AudioJEPA-style) ----

        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / torch.sqrt(var + self.eps)

        # ---- Step 2: normalize both embeddings (cosine geometry) ----
        pred = F.normalize(pred, dim=-1, p=2)
        target = F.normalize(target, dim=-1, p=2)

        # ---- Step 3: cosine distance ----
        cos_sim = (pred * target).sum(dim=-1)

        loss = 2 - 2 * cos_sim

        return loss.mean()




class NormalizedMSEWithVarianceLoss(nn.Module):
    def __init__(self, epsilon=1e-6, var_weight=0.1, var_target=1.0):
        super().__init__()
        self.eps = epsilon
        self.var_weight = var_weight
        self.var_target = var_target

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            return x.reshape(-1, x.shape[-1])
        return x

    def _std_loss(self, z: torch.Tensor):
        z = z - z.mean(dim=0, keepdim=True)
        std = torch.sqrt(z.var(dim=0, unbiased=False) + self.eps)
        loss = torch.mean(F.relu(self.var_target - std))
        return loss, std

    def forward(self, pred, target):
        pred = self._flatten(pred)
        target = self._flatten(target)

        pred_n = F.normalize(pred, dim=-1, p=2)
        target_n = F.normalize(target, dim=-1, p=2)

        cos_sim = (pred_n * target_n).sum(dim=-1)
        mse_loss = (2.0 - 2.0 * cos_sim).mean()

        var_loss, pred_std = self._std_loss(pred)

        total = mse_loss + self.var_weight * var_loss
        return total