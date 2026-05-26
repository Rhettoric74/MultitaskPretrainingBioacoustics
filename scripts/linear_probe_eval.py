import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from birdset_waveform_dataloader import BirdSetDataLoader
from build_mae_model import build_model


# ==================================================
# Checkpoint loading
# ==================================================
def load_checkpoint_state(model: torch.nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    print("Loaded model weights successfully!")
    if missing:
        print(f"[WARN] Missing keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [move_batch_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(v, device) for v in batch)
    return batch


# ==================================================
# Metrics
# ==================================================
@torch.no_grad()
def compute_metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(logits)

    y_score = probs.detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy().astype(np.int32)

    ap_vals: List[float] = []
    auc_vals: List[float] = []

    for c in range(y_true.shape[1]):
        yt = y_true[:, c]
        ys = y_score[:, c]
        pos = yt.sum()
        neg = yt.shape[0] - pos
        if pos == 0 or neg == 0:
            continue
        ap_vals.append(average_precision_score(yt, ys))
        auc_vals.append(roc_auc_score(yt, ys))

    pred_idx = probs.argmax(dim=1)
    row_idx = torch.arange(labels.shape[0], device=labels.device)
    top1_correct = (labels[row_idx, pred_idx] > 0.5).sum().item()
    top1_total = labels.shape[0]

    return {
        "cmAP": float(np.mean(ap_vals)) if ap_vals else float("nan"),
        "AUROC": float(np.mean(auc_vals)) if auc_vals else float("nan"),
        "top1_acc": float(top1_correct / max(top1_total, 1)),
        "num_classes_used_for_map": int(len(ap_vals)),
        "num_classes_used_for_auroc": int(len(auc_vals)),
        "num_samples": int(top1_total),
    }


# ==================================================
# Label mapping debug
# ==================================================
def verify_label_mapping(args):
    data = BirdSetDataLoader(
        dataset_path=args.dataset_path,
        subset=args.subset,
        split=args.train_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        use_mixup=False,
        use_geo_mixup=False,
        label_vocab_path=args.label_vocab_path,
    )
    loader = data.get_loader()
    dataset = loader.dataset

    print(f"Dataset: {args.subset} - {args.train_split}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Label field: {dataset.label_field_name}")
    print(f"Label vocab size: {len(dataset.label_to_idx)}")

    batch = next(iter(loader))
    labels = batch["labels"].numpy()

    print(f"\nBatch shape: {labels.shape}")
    print(f"Labels are binary: {np.all(np.isin(labels, [0, 1]))}")
    print(f"Labels sum per sample (mean): {labels.sum(axis=1).mean():.2f}")
    print(f"Labels sum per class (mean): {labels.sum(axis=0).mean():.2f}")

    class_counts = labels.sum(axis=0)
    active_classes = np.where(class_counts > 0)[0]
    print(f"\nActive classes in first batch: {len(active_classes)} out of {dataset.num_classes}")

    idx_to_label = {v: k for k, v in dataset.label_to_idx.items()}
    print("\nActive classes with their original labels:")
    for class_idx in active_classes[:50]:
        print(f"  Index {class_idx} -> {idx_to_label.get(class_idx, 'UNKNOWN')}")


# ==================================================
# Embedding extraction
# ==================================================
@torch.no_grad()
def extract_attention_pooled_embeddings(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    use_head_layernorm: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract clip-level attention pooled embeddings using the pretrained classifier attention."""
    model.eval()

    feats: List[torch.Tensor] = []
    labels_all: List[torch.Tensor] = []

    for batch_idx, batch in enumerate(loader, start=1):
        if batch_idx == 1:
            print(f"[INFO] Extracting embeddings from {len(loader)} batches")

        batch = move_batch_to_device(batch, device)

        specs, _ = model.frontend(
            batch["waveforms"],
            batch["waveform_lengths"],
            batch["mix_weights"],
            constituent_labels=batch.get("constituent_labels", None),
            mix_counts=batch.get("mix_counts", None),
            training=False,
        )
        h_full = model.encoder(specs, None)  # [B, N, D]

        scores = model.classifier.attn(h_full).squeeze(-1)         # [B, N]
        weights = torch.softmax(scores, dim=1)                     # [B, N]
        pooled = torch.sum(h_full * weights.unsqueeze(-1), dim=1)  # [B, D]

        if use_head_layernorm and hasattr(model.classifier, "classifier"):
            first_layer = model.classifier.classifier[0]
            if isinstance(first_layer, nn.LayerNorm):
                pooled = first_layer(pooled)

        feats.append(pooled.float().cpu())
        labels_all.append(batch["labels"].float().cpu())

        if batch_idx % 5 == 0:
            print("Extracted features from", batch_idx, "batches.")

    features = torch.cat(feats, dim=0)
    labels = torch.cat(labels_all, dim=0)

    print(f"[INFO] Final embedding tensor shape: {tuple(features.shape)}")
    print(f"[INFO] Final label tensor shape:     {tuple(labels.shape)}")

    return features, labels


@torch.no_grad()
def extract_patch_tokens(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
    drop_cls_token: bool = False,
) -> torch.Tensor:
    """Return patch/token embeddings from the frozen MAE encoder."""
    batch = move_batch_to_device(batch, device)

    specs, _ = model.frontend(
        batch["waveforms"],
        batch["waveform_lengths"],
        batch["mix_weights"],
        constituent_labels=batch.get("constituent_labels", None),
        mix_counts=batch.get("mix_counts", None),
        training=False,
    )

    tokens = model.encoder(specs, None)
    if isinstance(tokens, (tuple, list)):
        tokens = tokens[0]

    if tokens.dim() == 4:
        # [B, H, W, D] -> [B, N, D]
        b, h, w, d = tokens.shape
        tokens = tokens.reshape(b, h * w, d)
    elif tokens.dim() != 3:
        raise ValueError(f"Unexpected encoder output shape: {tuple(tokens.shape)}")

    if drop_cls_token and tokens.shape[1] > 1:
        tokens = tokens[:, 1:, :]

    return tokens


# ==================================================
# Probe models
# ==================================================
class LinearProbe(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class PatchProtoPNetProbe(nn.Module):
    """ProtoPNet-style probe over patch/token embeddings.

    If use_learnable_aggregation=False, the head uses per-class max over prototypes.
    If True, it uses a learnable class-specific linear readout of prototype activations.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        num_prototypes_per_class: int = 5,
        prototype_activation: str = "cosine",
        temperature: float = 0.1,
        use_learnable_aggregation: bool = False,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_classes = int(num_classes)
        self.num_prototypes_per_class = int(num_prototypes_per_class)
        self.num_prototypes = self.num_classes * self.num_prototypes_per_class
        self.prototype_activation = prototype_activation
        self.temperature = float(temperature)
        self.use_learnable_aggregation = bool(use_learnable_aggregation)

        self.prototypes = nn.Parameter(
            torch.randn(self.num_classes, self.num_prototypes_per_class, self.embed_dim) * 0.1
        )

        if self.use_learnable_aggregation:
            self.aggregation = nn.Linear(self.num_prototypes, self.num_classes)
        else:
            self.register_buffer(
                "prototype_class_identity",
                torch.zeros(self.num_prototypes, self.num_classes),
            )
            for c in range(self.num_classes):
                for k in range(self.num_prototypes_per_class):
                    self.prototype_class_identity[c * self.num_prototypes_per_class + k, c] = 1.0

    def compute_similarities(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, N, D] -> similarities: [B, C, K, N]"""
        if self.prototype_activation == "cosine":
            tokens_n = F.normalize(tokens, dim=-1)
            protos_n = F.normalize(self.prototypes, dim=-1)
            sim = torch.einsum("bnd,ckd->bckn", tokens_n, protos_n)
            return sim / self.temperature

        if self.prototype_activation == "euclidean":
            dot = torch.einsum("bnd,ckd->bckn", tokens, self.prototypes)
            token_sq = (tokens ** 2).sum(dim=-1)[:, None, None, :]
            proto_sq = (self.prototypes ** 2).sum(dim=-1)[None, :, :, None]
            dist_sq = token_sq + proto_sq - 2.0 * dot
            return -dist_sq / self.temperature

        raise ValueError(f"Unknown prototype_activation={self.prototype_activation}")

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        sim = self.compute_similarities(tokens)      # [B, C, K, N]
        proto_acts = sim.max(dim=-1).values          # [B, C, K]

        if self.use_learnable_aggregation:
            flat = proto_acts.reshape(tokens.shape[0], -1)
            logits = self.aggregation(flat)
        else:
            logits = proto_acts.max(dim=-1).values    # [B, C]

        return logits


# ==================================================
# Training utilities
# ==================================================
def _compute_pos_weight(loader, device: torch.device) -> torch.Tensor:
    pos = None
    n_samples = 0
    for batch in loader:
        labels = batch["labels"].to(device, non_blocking=True)
        batch_pos = labels.sum(dim=0)
        pos = batch_pos if pos is None else pos + batch_pos
        n_samples += labels.shape[0]

    if pos is None:
        raise RuntimeError("Could not compute pos_weight: empty loader")

    neg = float(n_samples) - pos
    pos_weight = neg / (pos + 1e-6)
    return pos_weight.clamp(max=50.0)


def train_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
) -> Tuple[LinearProbe, Dict[str, float]]:
    """Fit a linear probe on the full train split with train-set normalization."""
    feat_mean = train_features.mean(dim=0, keepdim=True)
    feat_std = train_features.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_features = (train_features - feat_mean) / feat_std

    train_ds = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    embed_dim = train_features.shape[1]
    num_classes = train_labels.shape[1]
    probe = LinearProbe(embed_dim=embed_dim, num_classes=num_classes).to(device)

    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    pos_weight = _compute_pos_weight(train_loader, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    last_loss = float("nan")
    for epoch in range(1, epochs + 1):
        probe.train()
        running_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = probe(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        last_loss = running_loss / max(n_batches, 1)
        print(f"[Linear Probe] epoch={epoch:03d} | train_loss={last_loss:.5f}")

    return probe, {"train_loss": last_loss, "feat_mean": feat_mean, "feat_std": feat_std}


def train_patch_proto_probe(
    backbone: torch.nn.Module,
    probe: PatchProtoPNetProbe,
    train_loader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    drop_cls_token: bool = False,
) -> Dict[str, float]:
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    pos_weight = _compute_pos_weight(train_loader, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

    last_loss = float("nan")
    for epoch in range(1, epochs + 1):
        probe.train()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            labels = batch["labels"].to(device, non_blocking=True).float()

            with torch.no_grad():
                tokens = extract_patch_tokens(
                    backbone,
                    batch,
                    device=device,
                    drop_cls_token=drop_cls_token,
                )

            optimizer.zero_grad(set_to_none=True)
            logits = probe(tokens)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        last_loss = running_loss / max(n_batches, 1)
        print(f"[Proto Probe] epoch={epoch:03d} | train_loss={last_loss:.5f}")

    return {"train_loss": last_loss}


@torch.no_grad()
def evaluate_linear_probe(
    probe: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    feat_mean: Optional[torch.Tensor] = None,
    feat_std: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    probe.eval()
    if feat_mean is not None and feat_std is not None:
        features = (features - feat_mean) / feat_std
    logits = probe(features.to(device)).cpu()
    return compute_metrics_from_logits(logits, labels)


@torch.no_grad()
def evaluate_patch_proto_probe(
    backbone: torch.nn.Module,
    probe: PatchProtoPNetProbe,
    loader,
    device: torch.device,
    drop_cls_token: bool = False,
) -> Dict[str, float]:
    backbone.eval()
    probe.eval()

    all_logits: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    for batch in loader:
        labels = batch["labels"].to(device, non_blocking=True).float()
        tokens = extract_patch_tokens(
            backbone,
            batch,
            device=device,
            drop_cls_token=drop_cls_token,
        )
        logits = probe(tokens)
        all_logits.append(logits.detach().cpu())
        all_targets.append(labels.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_targets, dim=0)
    return compute_metrics_from_logits(logits, labels)


# ==================================================
# Utilities
# ==================================================
def quick_checkpoint_summary(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    print(f"\nCheckpoint: {ckpt_path}")
    print(f"Size: {Path(ckpt_path).stat().st_size / 1e6:.2f} MB")

    if isinstance(ckpt, dict):
        print(f"Top-level keys: {list(ckpt.keys())}")
        if "epoch" in ckpt:
            print(f"  - Epoch: {ckpt['epoch']}")
        if "global_step" in ckpt:
            print(f"  - Global step: {ckpt['global_step']}")
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            print(f"  - State dict keys: {len(state_dict)}")
            print(f"  - Sample keys: {list(state_dict.keys())[:10]}")


# ==================================================
# Main
# ==================================================
def main():
    parser = argparse.ArgumentParser(
        description="Linear or prototypical probe on BirdSet using pretrained MAE embeddings"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--label-vocab-path", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, default="/scratch/Projects/CFP-04/CFP04-CF-029/birdset")
    parser.add_argument("--subset", type=str, default="POW")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--test-split", type=str, default="test_5s")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=16)

    parser.add_argument("--pretrain-num-classes", type=int, default=9734)
    parser.add_argument("--probe-type", type=str, choices=["linear", "prototypical"], default="linear")

    parser.add_argument("--probe-epochs", type=int, default=5)
    parser.add_argument("--probe-lr", type=float, default=1e-2)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--probe-batch-size", type=int, default=256)

    parser.add_argument("--num-prototypes-per-class", type=int, default=5)
    parser.add_argument("--prototype-activation", type=str, choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--prototype-temperature", type=float, default=0.1)
    parser.add_argument("--prototype-learnable-aggregation", action="store_true")
    parser.add_argument("--drop-cls-token", action="store_true")

    parser.add_argument("--use-head-layernorm", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--debug-labels", action="store_true")
    args = parser.parse_args()

    quick_checkpoint_summary(args.checkpoint)
    if args.debug_labels:
        verify_label_mapping(args)
        sys.exit(0)

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Building datasets...")
    train_data = BirdSetDataLoader(
        dataset_path=args.dataset_path,
        subset=args.subset,
        split=args.train_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        use_mixup=False,
        use_geo_mixup=False,
        label_vocab_path=args.label_vocab_path,
    )
    train_loader = train_data.get_loader()

    test_data = BirdSetDataLoader(
        dataset_path=args.dataset_path,
        subset=args.subset,
        split=args.test_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        use_mixup=False,
        use_geo_mixup=False,
        label_vocab_path=args.label_vocab_path,
    )
    test_loader = test_data.get_loader()

    num_classes_probe = train_data.dataset.num_classes
    print(f"[INFO] Probe classes: {num_classes_probe}")

    print("Building model and loading checkpoint...")
    backbone = build_model(
        num_classes=args.pretrain_num_classes,
        objective_mode="joint",
        optimizer_type="dcgd",
        mask_ratio=0.75,
        lambda_recon=1.0,
        lambda_cls=0.01,
        learning_rate=3e-4,
        weight_decay=0.05,
    )
    load_checkpoint_state(backbone, args.checkpoint)
    backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    print(f"[INFO] Probe type: {args.probe_type}")

    if args.probe_type == "linear":
        print("Extracting train embeddings...")
        train_features, train_labels = extract_attention_pooled_embeddings(
            backbone,
            train_loader,
            device,
            use_head_layernorm=args.use_head_layernorm,
        )
        print("Extracting test embeddings...")
        test_features, test_labels = extract_attention_pooled_embeddings(
            backbone,
            test_loader,
            device,
            use_head_layernorm=args.use_head_layernorm,
        )

        print(f"[INFO] Train samples: {train_features.shape[0]}")
        print(f"[INFO] Test samples:  {test_features.shape[0]}")
        print(f"[INFO] Embedding dim: {train_features.shape[1]}")

        probe, probe_info = train_linear_probe(
            train_features=train_features,
            train_labels=train_labels,
            device=device,
            epochs=args.probe_epochs,
            lr=args.probe_lr,
            weight_decay=args.probe_weight_decay,
            batch_size=args.probe_batch_size,
        )

        test_metrics = evaluate_linear_probe(
            probe,
            test_features,
            test_labels,
            device,
            feat_mean=probe_info.get("feat_mean", None),
            feat_std=probe_info.get("feat_std", None),
        )

    else:
        sample_batch = next(iter(train_loader))
        with torch.no_grad():
            sample_tokens = extract_patch_tokens(
                backbone,
                sample_batch,
                device=device,
                drop_cls_token=args.drop_cls_token,
            )
        embed_dim = sample_tokens.shape[-1]
        print(f"[INFO] Patch/token embedding dim: {embed_dim}")
        print(f"[INFO] Token count per sample: {sample_tokens.shape[1]}")

        probe = PatchProtoPNetProbe(
            embed_dim=embed_dim,
            num_classes=num_classes_probe,
            num_prototypes_per_class=args.num_prototypes_per_class,
            prototype_activation=args.prototype_activation,
            temperature=args.prototype_temperature,
            use_learnable_aggregation=args.prototype_learnable_aggregation,
        ).to(device)

        probe_info = train_patch_proto_probe(
            backbone=backbone,
            probe=probe,
            train_loader=train_loader,
            device=device,
            epochs=args.probe_epochs,
            lr=args.probe_lr,
            weight_decay=args.probe_weight_decay,
            drop_cls_token=args.drop_cls_token,
        )
        test_metrics = evaluate_patch_proto_probe(
            backbone,
            probe,
            test_loader,
            device,
            drop_cls_token=args.drop_cls_token,
        )

    print(f"\n=== {args.probe_type.upper()} PROBE RESULTS ===")
    print(f"Subset: {args.subset}")
    print(f"Train split: {args.train_split}")
    print(f"Test split:  {args.test_split}")
    print(f"Train loss:   {probe_info['train_loss']:.6f}")
    if args.probe_type == "prototypical":
        print(f"Num prototypes: {probe.num_prototypes}")
        print(f"Prototype activation: {args.prototype_activation}")
        print(f"Learnable aggregation: {args.prototype_learnable_aggregation}")
    print(f"Test cmAP:    {test_metrics['cmAP']:.6f}")
    print(f"Test AUROC:   {test_metrics['AUROC']:.6f}")
    print(f"Test top-1:   {test_metrics['top1_acc']:.6f}")
    print(f"classes used for cmAP:  {test_metrics['num_classes_used_for_map']}")
    print(f"classes used for AUROC: {test_metrics['num_classes_used_for_auroc']}")
    print(f"samples: {test_metrics['num_samples']}")

    if args.output_json is not None:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "probe_type": args.probe_type,
                    "train_loss": probe_info["train_loss"],
                    "test_metrics": test_metrics,
                    "probe_config": {
                        "epochs": args.probe_epochs,
                        "lr": args.probe_lr,
                        "weight_decay": args.probe_weight_decay,
                        "batch_size": args.probe_batch_size,
                        "use_head_layernorm": args.use_head_layernorm,
                    }
                    if args.probe_type == "linear"
                    else {
                        "epochs": args.probe_epochs,
                        "lr": args.probe_lr,
                        "weight_decay": args.probe_weight_decay,
                        "batch_size": args.probe_batch_size,
                        "num_prototypes_per_class": args.num_prototypes_per_class,
                        "prototype_activation": args.prototype_activation,
                        "temperature": args.prototype_temperature,
                        "learnable_aggregation": args.prototype_learnable_aggregation,
                        "drop_cls_token": args.drop_cls_token,
                    },
                },
                indent=2,
            )
        )
        print(f"Saved metrics to: {out_path}")


if __name__ == "__main__":
    main()

