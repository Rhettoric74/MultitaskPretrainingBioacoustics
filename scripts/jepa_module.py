import torch
import torch.nn as nn
import lightning as L
from copy import deepcopy

from utils import EMA, apply_masks, generate_random_masks


# =========================================================
# Gradient utilities (robust)
# =========================================================
def compute_grad_metrics(model, loss_jepa, loss_cls):
    named_params = [
        (name, p) for name, p in model.encoder.named_parameters() if p.requires_grad
    ]
    if not named_params:
        device = loss_jepa.device
        z = torch.tensor(0.0, device=device)
        return z, z, z

    names = [n for n, _ in named_params]
    params = [p for _, p in named_params]

    grads_jepa = torch.autograd.grad(
        loss_jepa, params, retain_graph=True, allow_unused=True
    )
    grads_cls = torch.autograd.grad(
        loss_cls, params, retain_graph=True, allow_unused=True
    )

    common = []
    g1_list = []
    g2_list = []

    for name, g1, g2 in zip(names, grads_jepa, grads_cls):
        if g1 is not None and g2 is not None:
            common.append(name)
            g1_list.append(g1.detach().float().reshape(-1))
            g2_list.append(g2.detach().float().reshape(-1))

    if len(common) == 0:
        device = loss_jepa.device
        z = torch.tensor(0.0, device=device)
        return z, z, z

    g_jepa = torch.cat(g1_list)
    g_cls = torch.cat(g2_list)

    eps = 1e-12
    norm_jepa = g_jepa.norm()
    norm_cls = g_cls.norm()
    cos = torch.dot(g_jepa, g_cls) / (norm_jepa * norm_cls + eps)

    return cos, norm_jepa, norm_cls


# =========================================================
# Reconstruction helper
# =========================================================
def reconstruct_full_patches(h, z_pred, pred_mask):
    """
    h: (B, N, D) encoder outputs
    z_pred: (B, N_masked, D) predictor outputs
    pred_mask: (B, N) bool tensor, True = hidden / predicted positions
    """
    if pred_mask.dtype != torch.bool:
        raise TypeError(f"pred_mask must be bool, got {pred_mask.dtype}")

    full = torch.zeros_like(h)

    # Visible patches come from encoder output
    full[~pred_mask] = h[~pred_mask]

    # Hidden patches come from predictor output
    full[pred_mask] = z_pred.reshape(-1, h.shape[-1])

    return full


# =========================================================
# Module
# =========================================================
class BioacousticJEPAModule(L.LightningModule):
    def __init__(
        self,
        encoder: torch.nn.Module,
        predictor: torch.nn.Module,
        classifier: torch.nn.Module,
        criterion_jepa: torch.nn.Module,
        criterion_cls: torch.nn.Module,
        lambda_cls: float = 1.0,
        optimizer_type: str = "adam",
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        ema_momentum: float = 0.996,
        classify_with_preds: bool = False,
        objective_mode="joint",
    ):
        super().__init__()

        self.automatic_optimization = False
        self.save_hyperparameters(
            logger=False,
            ignore=["encoder", "predictor", "classifier"],
        )

        self.encoder = encoder
        self.predictor = predictor
        self.classifier = classifier
        self.criterion_jepa = criterion_jepa
        self.criterion_cls = criterion_cls
        self.lambda_cls = lambda_cls
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.classify_with_preds = classify_with_preds
        self.objective_mode = objective_mode

        self.ema = EMA(ema_momentum)
        self.target_encoder = deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    # -------------------------------------------------
    def forward(self, x):
        return self.encoder(x, None)

    # -------------------------------------------------
    def model_step(self, batch):
        spectrograms = batch["spectrograms"]
        labels = batch["labels"].float()
    
        B, _, H, W = spectrograms.shape
        grid_size = (H // 16, W // 16)
        num_patches = grid_size[0] * grid_size[1]
    
        # Generate masks (only needed for modes that require them)
        context_masks, prediction_masks = None, None
        pred_mask = None
        
        if self.objective_mode in ["joint", "jepa"]:
            context_masks, prediction_masks = generate_random_masks(
                batch_size=B,
                num_patches=num_patches,
                mask_ratio=0.5,
                device=spectrograms.device,
                num_masks=1,
            )
            pred_mask = prediction_masks[0]
    
        loss_jepa = 0
        loss_cls = 0
        logits = None
    
        # ---- Mode: JEPA only ----
        if self.objective_mode == "jepa":
            h = self.encoder(spectrograms, pred_mask)
            z_pred = self.predictor(h, pred_mask)
    
            with torch.no_grad():
                h_target = self.target_encoder(spectrograms, None)
                h_target = apply_masks(h_target, prediction_masks)
    
            loss_jepa = self.criterion_jepa(z_pred, h_target)
    
        # ---- Mode: Classification only ----
        elif self.objective_mode == "class":
            h_full = self.encoder(spectrograms, None)
            logits = self.classifier(h_full)
            loss_cls = self.criterion_cls(logits, labels)
    
        # ---- Mode: Joint (JEPA + Classification with reconstruction) ----
        elif self.objective_mode == "joint":
            # JEPA branch
            h = self.encoder(spectrograms, pred_mask)
            z_pred = self.predictor(h, pred_mask)
    
            with torch.no_grad():
                h_target = self.target_encoder(spectrograms, None)
                h_target = apply_masks(h_target, prediction_masks)
    
            loss_jepa = self.criterion_jepa(z_pred, h_target)
    
            # Classification branch with reconstruction
            if self.classify_with_preds:
                full_rep = reconstruct_full_patches(h, z_pred, pred_mask)
                logits = self.classifier(full_rep)
            else:
                h_full = self.encoder(spectrograms, None)
                logits = self.classifier(h_full)
            
            loss_cls = self.criterion_cls(logits, labels)
    
        else:
            raise ValueError(f"Unknown objective_mode: {self.objective_mode}")
    
        loss = loss_jepa + self.lambda_cls * loss_cls
        return loss, loss_jepa, loss_cls, logits

    # -------------------------------------------------
    def training_step(self, batch, batch_idx):
        loss, loss_jepa, loss_cls, logits = self.model_step(batch)

        if self.objective_mode == "joint" and batch_idx % 10 == 0:
            cos_sim, norm_jepa, norm_cls = compute_grad_metrics(
                self, loss_jepa, loss_cls
            )
            self.log("train/grad_cosine", cos_sim, prog_bar=True)
            self.log("train/grad_norm_jepa", norm_jepa)
            self.log("train/grad_norm_cls", norm_cls)

        opt = self.optimizers()

        if self.optimizer_type == "adam":
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()

        elif self.optimizer_type == "dcgd":
            if hasattr(opt, "optimizer"):
                dcgd_opt = opt.optimizer
            else:
                dcgd_opt = opt

            dcgd_opt.zero_grad()
            dcgd_opt.step([loss_jepa, loss_cls])

        else:
            raise ValueError(f"Unknown optimizer_type: {self.optimizer_type}")

        self.log("train_loss", loss, prog_bar=True)
        self.log("train/jepa_loss", loss_jepa, prog_bar=True)
        self.log("train/cls_loss", loss_cls, prog_bar=True)

        if self.optimizer_type == "dcgd" and hasattr(opt, "optimizer"):
            lr = opt.optimizer.param_groups[0]["lr"]
        else:
            lr = opt.param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True)

        return loss

    # -------------------------------------------------
    def validation_step(self, batch, batch_idx):
        loss, loss_jepa, loss_cls, logits = self.model_step(batch)

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/jepa_loss", loss_jepa, sync_dist=True)
        self.log("val/cls_loss", loss_cls, sync_dist=True)
        self.log("val/variance", torch.var(logits), sync_dist=True)

    # -------------------------------------------------
    def infer_logits(self, x):
        h_full = self.encoder(x, None)
        return self.classifier(h_full)

    # -------------------------------------------------
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.ema.update(self.encoder, self.target_encoder)

    # -------------------------------------------------
    def configure_optimizers(self):
        all_params = []
    
        for p in self.encoder.parameters():
            all_params.append(p)
        predictor_start_idx = len(all_params)
    
        for p in self.predictor.parameters():
            all_params.append(p)
        classifier_start_idx = len(all_params)
    
        for p in self.classifier.parameters():
            all_params.append(p)
    
        classifier_param_indices = list(range(classifier_start_idx, len(all_params)))
    
        if self.optimizer_type == "dcgd":
            # base sgd to avoid integration issues with wrapping DCGD around AdamW
            print("Initializing DCGD optimizer")
            base_opt = torch.optim.Adam(
                all_params,
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.0,
            )
    
            from dcgd import DCGD
            return DCGD(
                base_opt,
                num_pde=1,
                type="center",
                classifier_param_indices=classifier_param_indices,
                predictor_start_idx=predictor_start_idx,
                classifier_start_idx=classifier_start_idx,
            )
    
        elif self.optimizer_type == "adam":
            return torch.optim.AdamW(
                all_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
    
        else:
            raise ValueError(f"Unknown optimizer_type: {self.optimizer_type}")