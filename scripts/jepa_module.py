import torch
import torch.nn as nn
import lightning as L
from copy import deepcopy
import torch.nn.functional as F

from utils import EMA, apply_keep_indices, generate_random_keep_indices


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
def reconstruct_full_patches(
    num_patches: int,
    context_indices: torch.Tensor,      # [B, Kc]
    prediction_indices: torch.Tensor,   # [B, Kp]
    context_tokens: torch.Tensor,       # [B, Kc, D]
    pred_tokens: torch.Tensor,          # [B, Kp, D]
) -> torch.Tensor:
    B, Kc, D = context_tokens.shape
    _, Kp, _ = pred_tokens.shape

    full = torch.zeros(B, num_patches, D, device=context_tokens.device, dtype=context_tokens.dtype)

    # scatter context tokens
    ctx_idx = context_indices.unsqueeze(-1).expand(-1, -1, D)
    full.scatter_(dim=1, index=ctx_idx, src=context_tokens)

    # scatter predicted tokens
    pred_idx = prediction_indices.unsqueeze(-1).expand(-1, -1, D)
    full.scatter_(dim=1, index=pred_idx, src=pred_tokens)

    return full


# =========================================================
# Module
# =========================================================
class BioacousticJEPAModule(L.LightningModule):
    def __init__(
        self,
        encoder: torch.nn.Module,
        target_encoder: torch.nn.Module,
        predictor: torch.nn.Module,
        classifier: torch.nn.Module,
        criterion_jepa: torch.nn.Module,
        criterion_cls: torch.nn.Module,
        lambda_cls: float = 1.0,
        optimizer_type: str = "adam",
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        ema_momentum: float = 0.999,
        classify_with_preds: bool = False,
        objective_mode="joint",
        mask_ratio=0.3,
        debug_every_n_batches: int = 100,
        debug_num_params: int = 0,
    ):
        super().__init__()

        self.automatic_optimization = False
        self.save_hyperparameters(
            logger=False,
            ignore=["encoder", "predictor", "classifier"],
        )
        self.mask_ratio = mask_ratio
        self.encoder = encoder
        self.target_encoder = target_encoder
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

        self.debug_every_n_batches = debug_every_n_batches
        self.debug_num_params = debug_num_params
        self._debug_param_names = None
        self._encoder_before = None
        self._target_before = None

    def on_fit_start(self):
        # Pick a few representative encoder parameters to track
        self._debug_param_names = [
            n for n, p in self.encoder.named_parameters() if p.requires_grad
        ][: self.debug_num_params]
        self.print(f"Debugging params: {self._debug_param_names}")

    def _snapshot_selected_params(self, module):
        snap = {}
        for name, p in module.named_parameters():
            if name in self._debug_param_names:
                snap[name] = p.detach().float().cpu().clone()
        return snap

    def _mean_abs_delta(self, module, before_snap):
        vals = []
        for name, p in module.named_parameters():
            if name in before_snap:
                cur = p.detach().float().cpu()
                prev = before_snap[name]
                vals.append((cur - prev).abs().mean())
        if not vals:
            return 0.0
        return torch.stack(vals).mean().item()

    def _mean_abs_gap(self, module_a, module_b):
        vals = []
        params_b = dict(module_b.named_parameters())
        for name, p in module_a.named_parameters():
            if name in params_b:
                vals.append(
                    (p.detach().float().cpu() - params_b[name].detach().float().cpu())
                    .abs()
                    .mean()
                )
        if not vals:
            return 0.0
        return torch.stack(vals).mean().item()

    def _target_has_grad(self):
        return any(p.grad is not None for p in self.target_encoder.parameters())

    def on_train_batch_start(self, batch, batch_idx):
        self.target_encoder.eval()
        if self._debug_param_names is None:
            return

        if batch_idx % self.debug_every_n_batches == 0:
            self._encoder_before = self._snapshot_selected_params(self.encoder)
            self._target_before = self._snapshot_selected_params(self.target_encoder)

    def on_after_backward(self):
        # Helpful to confirm the encoder is actually receiving gradient
        if self._debug_param_names is None:
            return

        if self.trainer.global_step % self.debug_every_n_batches != 0:
            return

        for name, p in self.encoder.named_parameters():
            if name in self._debug_param_names:
                gnorm = 0.0 if p.grad is None else p.grad.detach().norm().item()
                print(f"debug/grad_norm/{name}", gnorm)

        print("debug/target_has_grad", float(self._target_has_grad()))

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # EMA update happens here
        self.ema.update(self.encoder, self.target_encoder)
        
        if self._debug_param_names is None:
            return

        if batch_idx % self.debug_every_n_batches != 0:
            return

        encoder_delta = self._mean_abs_delta(self.encoder, self._encoder_before)
        target_delta = self._mean_abs_delta(self.target_encoder, self._target_before)
        gap_after = self._mean_abs_gap(self.encoder, self.target_encoder)

        print("debug/encoder_step_delta", encoder_delta)
        print("debug/target_ema_delta", target_delta)
        print("debug/encoder_target_gap", gap_after)
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
            context_indices, prediction_indices = generate_random_keep_indices(
                batch_size=B,
                num_patches=num_patches,
                context_ratio=self.mask_ratio,
                device=spectrograms.device,
            )
    
        loss_jepa = 0
        loss_cls = 0
        logits = None
        metrics = None
    
        # ---- Mode: JEPA only ----
        if self.objective_mode == "jepa":
            
            # Context encoder sees only the visible patches
            h = self.encoder(spectrograms, context_indices)
            
            # Predictor predicts the hidden patches
            z_pred = self.predictor(h, prediction_indices)
            
            with torch.no_grad():
                # Target encoder sees the full spectrogram
                h_target = self.target_encoder(spectrograms, None)
            
                # Keep only the prediction patches from the target embeddings
                h_target = apply_keep_indices(h_target, prediction_indices)
                
            print("N total:", num_patches)
            print("context keep count:", context_indices[0].shape[1])
            print("prediction keep count:", prediction_indices[0].shape[1])
            print("h_context:", h.shape)
            print("z_pred:", z_pred.shape)
            print("h_target:", h_target.shape)
            
            loss_jepa, metrics = self.criterion_jepa(z_pred, h_target, h)
    
        # ---- Mode: Classification only ----
        elif self.objective_mode == "class":
            h_full = self.encoder(spectrograms, None)
            logits = self.classifier(h_full)
            loss_cls = self.criterion_cls(logits, labels)
    
        # ---- Mode: Joint (JEPA + Classification with reconstruction) ----
        elif self.objective_mode == "joint":
            # context_indices: list with one [B, Kc] LongTensor
            # prediction_indices: list with one [B, Kp] LongTensor
        
            context_indices, prediction_indices = generate_random_keep_indices(
                batch_size=spectrograms.size(0),
                num_patches=num_patches,
                context_ratio=self.mask_ratio,
                device=spectrograms.device,
            )
        
            # JEPA branch
            h_ctx = self.encoder(spectrograms, context_indices)          # [B, Kc, D]
            z_pred = self.predictor(h_ctx, prediction_indices)           # [B, Kp, D]
        
            with torch.no_grad():
                h_target_full = self.target_encoder(spectrograms, None)   # [B, N, D]
                h_target = apply_keep_indices(h_target_full, prediction_indices)  # [B, Kp, D]
        
            # Optional full reconstruction for the classifier head
            full_rep = reconstruct_full_patches(
                num_patches=num_patches,
                context_indices=context_indices[0],
                prediction_indices=prediction_indices[0],
                context_tokens=h_ctx,
                pred_tokens=z_pred,
            )
        
            loss_jepa, metrics = self.criterion_jepa(z_pred, h_target, h_ctx)
        
            # Classification branch
            if self.classify_with_preds:
                logits = self.classifier(full_rep)
            else:
                h_full = self.encoder(spectrograms, None)
                logits = self.classifier(h_full)
        
            loss_cls = self.criterion_cls(logits, labels)
            
        else:
            raise ValueError(f"Unknown objective_mode: {self.objective_mode}")
            
        loss = loss_jepa + self.lambda_cls * loss_cls
        return loss, loss_jepa, loss_cls, logits, metrics

    # -------------------------------------------------
    def training_step(self, batch, batch_idx):
        loss, loss_jepa, loss_cls, logits, metrics = self.model_step(batch)
        if metrics:
            self.log_dict(metrics, on_step=True, on_epoch=False, prog_bar=False, logger=True)

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
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            opt.step()
            sched = self.lr_schedulers()
            sched.step()

        elif self.optimizer_type == "dcgd":
            if hasattr(opt, "optimizer"):
                dcgd_opt = opt.optimizer
            else:
                dcgd_opt = opt

            dcgd_opt.zero_grad()
            dcgd_opt.step([loss_jepa, loss_cls])
            sched = self.lr_schedulers()
            if sched:
                sched.step()

        else:
            raise ValueError(f"Unknown optimizer_type: {self.optimizer_type}")

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
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
                predictor_start_idx=predictor_start_idx,
                classifier_start_idx=classifier_start_idx,
            )
    
            # If DCGD behaves like a standard optimizer, you can attach a scheduler here too.
            # If it does not, skip the scheduler for this branch until you confirm compatibility.
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