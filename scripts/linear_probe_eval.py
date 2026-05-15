import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

from birdset_dataloader_v2 import BirdSetDataLoader
from build_model import build_model


# -------------------------------------------------
# Checkpoint loading
# -------------------------------------------------
def load_checkpoint_state(
    model: torch.nn.Module,
    ckpt_path: str,
    strict: bool = False,
    skip_prefixes: Sequence[str] = ("classifier.",),
) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    if skip_prefixes:
        filtered_state_dict = {
            k: v
            for k, v in state_dict.items()
            if not any(k.startswith(prefix) for prefix in skip_prefixes)
        }
        skipped = sorted([k for k in state_dict.keys() if k not in filtered_state_dict])
        print(f"[INFO] Skipping {len(skipped)} checkpoint keys from load (e.g. pretrained head).")
        if skipped:
            print(f"[INFO] First skipped keys: {skipped[:5]}")
        state_dict = filtered_state_dict

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        print(f"[WARN] Missing keys (expected for new probe head): {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")


# -------------------------------------------------
# Dataset / vocab helpers
# -------------------------------------------------
def _dataset_from_loader(loader):
    ds = loader.dataset
    if hasattr(ds, "dataset"):
        return ds.dataset
    return ds


def _get_label_vocab_in_feature_order(raw_dataset, label_field_name: str) -> List[str]:
    """Return label names in the raw dataset's canonical order."""
    feat = raw_dataset.features[label_field_name]
    candidates = [feat, getattr(feat, "feature", None)]

    for candidate in candidates:
        if candidate is None:
            continue

        names = getattr(candidate, "names", None)
        if names:
            return [str(x) for x in names]

        num_classes = getattr(candidate, "num_classes", None)
        int2str = getattr(candidate, "int2str", None)
        if num_classes is not None and int2str is not None:
            try:
                return [str(int2str(i)) for i in range(int(num_classes))]
            except Exception:
                pass

    raise ValueError(f"Could not infer canonical label order from field {label_field_name}")


@torch.no_grad()
def build_label_remapper(xcl_vocab: Sequence[str], pow_vocab: Sequence[str], device: torch.device):
    """Map labels from XCL one-hot order -> POW canonical one-hot order."""
    xcl_to_idx = {label: i for i, label in enumerate(xcl_vocab)}
    pow_to_xcl = []
    missing = []

    for code in pow_vocab:
        xcl_idx = xcl_to_idx.get(code, -1)
        pow_to_xcl.append(xcl_idx)
        if xcl_idx < 0:
            missing.append(code)

    if missing:
        print(f"[WARN] {len(missing)} POW labels were not found in the XCL vocab.")
        print(f"[WARN] First missing POW labels: {missing[:10]}")

    pow_to_xcl_idx = torch.tensor(pow_to_xcl, dtype=torch.long, device=device)

    def remap(labels: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(labels.shape[0], len(pow_vocab), dtype=labels.dtype, device=labels.device)
        valid = pow_to_xcl_idx >= 0
        if valid.any():
            out[:, valid] = labels.index_select(dim=1, index=pow_to_xcl_idx[valid])
        return out

    return remap


# -------------------------------------------------
# Mean-pooling probe head
# -------------------------------------------------
class MeanPoolLinearHead(nn.Module):
    """Mean-pool token embeddings, then apply a single linear layer."""

    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, N, D]
        pooled = h.mean(dim=1)
        return self.classifier(pooled)

    @property
    def out_features(self) -> int:
        return int(self.classifier.out_features)


# -------------------------------------------------
# Metrics
# -------------------------------------------------
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


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    remap_labels,
) -> Dict[str, float]:
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    for batch_idx, batch in enumerate(loader, start=1):
        if batch_idx == 1:
            ds = _dataset_from_loader(loader)
            label_field = loader.dataset.label_field_name
            int2str = loader.dataset.label_int2str

            print("=== LABEL DEBUG ===")
            sample = None
            for i in range(len(ds)):
                raw_labels = ds[i].get(label_field)
                if raw_labels:
                    sample = ds[i]
                    break

            if sample is not None:
                raw_labels = sample.get(label_field)
                print("Raw labels:", raw_labels)
                for l in raw_labels:
                    if isinstance(l, (int, np.integer)) and int2str is not None:
                        code = int2str(int(l))
                    else:
                        code = str(l)
                    print(f"raw: {l} -> code: {code}")
            print("===================\n")

        spectrograms = batch["spectrograms"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True).float()
        labels = remap_labels(labels)

        logits = model.infer_logits(spectrograms)
        all_logits.append(logits.detach().cpu())
        all_targets.append(labels.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_targets, dim=0)
    return compute_metrics_from_logits(logits, labels)


# -------------------------------------------------
# Utilities
# -------------------------------------------------
def set_requires_grad(module: torch.nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad


def build_dataloader(
    dataset_path: str,
    subset: str,
    split: str,
    batch_size: int,
    num_workers: int,
    label_vocab_path: str,
    spec_norm_path: str,
    apply_spec_norm: bool,
    shuffle: bool,
    use_mixup: bool = False,
    use_geo_mixup: bool = False,
):
    data = BirdSetDataLoader(
        dataset_path=dataset_path,
        subset=subset,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        use_mixup=use_mixup,
        use_geo_mixup=use_geo_mixup,
        label_vocab_path=label_vocab_path,
        spec_norm_path=spec_norm_path,
        apply_spec_norm=apply_spec_norm,
    )
    return data, data.get_loader()


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Frozen-backbone mean-pool probe for BirdSet POW")
    parser.add_argument("--checkpoint", type=str, required=True, help="JEPA checkpoint to load")
    parser.add_argument("--output-dir", type=str, default="./pow_probe_runs")
    parser.add_argument("--label-vocab-path", type=str, default="/home/svu/e1583377/MultitaskPretrainingBioacoustics/scripts/xcl_label_vocab.json")
    parser.add_argument("--dataset-path", type=str, default="/scratch/Projects/CFP-04/CFP04-CF-029/birdset")
    parser.add_argument("--subset", type=str, default="POW")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--test-split", type=str, default="test_5s")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--spec-norm-path", type=str, default="/home/svu/e1583377/MultitaskPretrainingBioacoustics/scripts/xcl_spec_stats_true_log.json")
    parser.add_argument("--apply-spec-norm", action="store_true", help="Apply spectrogram normalization")
    parser.add_argument("--save-best-name", type=str, default="best_probe.ckpt")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building datasets...")
    train_data, train_loader = build_dataloader(
        dataset_path=args.dataset_path,
        subset=args.subset,
        split=args.train_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        label_vocab_path=args.label_vocab_path,
        spec_norm_path=args.spec_norm_path,
        apply_spec_norm=args.apply_spec_norm,
        shuffle=True,
        use_mixup=False,
        use_geo_mixup=False,
    )
    test_data, test_loader = build_dataloader(
        dataset_path=args.dataset_path,
        subset=args.subset,
        split=args.test_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        label_vocab_path=args.label_vocab_path,
        spec_norm_path=args.spec_norm_path,
        apply_spec_norm=args.apply_spec_norm,
        shuffle=False,
        use_mixup=False,
        use_geo_mixup=False,
    )

    # POW canonical vocab comes from the raw POW dataset feature order.
    train_raw_ds = _dataset_from_loader(train_loader)
    test_raw_ds = _dataset_from_loader(test_loader)
    label_field = train_loader.dataset.label_field_name

    pow_vocab = _get_label_vocab_in_feature_order(train_raw_ds, label_field)
    test_pow_vocab = _get_label_vocab_in_feature_order(test_raw_ds, label_field)

    print(f"[INFO] Raw dataset class count (loader label vocab): {train_data.dataset.num_classes}")
    print(f"[INFO] POW canonical vocab size (train): {len(pow_vocab)}")
    print(f"[INFO] POW canonical vocab size (test):  {len(test_pow_vocab)}")

    if pow_vocab != test_pow_vocab:
        print("[WARN] POW train/test canonical label order differs. Using train order for the probe head.")

    xcl_vocab = train_loader.dataset.label_list
    xcl_to_idx = train_loader.dataset.label_to_idx

    print("[INFO] First 10 POW labels in canonical POW order and their XCL indices:")
    for i, code in enumerate(pow_vocab[:10]):
        print(f"  POW idx {i:2d} -> {code:8s} -> XCL idx {xcl_to_idx.get(code, None)}")

    remap_labels = build_label_remapper(xcl_vocab, pow_vocab, device)

    num_classes = len(pow_vocab)
    num_patches = (128 // args.patch_size) * (512 // args.patch_size)
    print(f"[INFO] Probe head classes: {num_classes}")
    print(f"[INFO] num_patches={num_patches}")

    print("Building model and loading checkpoint...")
    model = build_model(num_classes, num_patches)
    load_checkpoint_state(model, args.checkpoint, strict=False, skip_prefixes=("classifier.",))
    model.to(device)

    # Replace the existing classifier head with a mean-pooling probe head.
    embed_dim = None
    if hasattr(model, "encoder"):
        if hasattr(model.encoder, "embed_dim"):
            embed_dim = int(model.encoder.embed_dim)
        elif hasattr(model.encoder, "vit") and hasattr(model.encoder.vit, "embed_dim"):
            embed_dim = int(model.encoder.vit.embed_dim)

    if embed_dim is None:
        raise AttributeError("Could not infer encoder embedding dimension for mean-pool head")

    model.classifier = MeanPoolLinearHead(embed_dim=embed_dim, num_classes=num_classes).to(device)
    print(f"[INFO] Replaced classifier with MeanPoolLinearHead(embed_dim={embed_dim}, num_classes={num_classes})")

    if hasattr(model, "encoder"):
        set_requires_grad(model.encoder, False)
        model.encoder.eval()
    if hasattr(model, "predictor"):
        set_requires_grad(model.predictor, False)
        model.predictor.eval()
    set_requires_grad(model.classifier, True)
    model.classifier.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[INFO] Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # Sanity check one batch before training.
    sanity_batch = next(iter(train_loader))
    sanity_raw = sanity_batch["labels"].to(device, non_blocking=True).float()
    sanity_pow = remap_labels(sanity_raw)
    print(f"[CHECK] Raw batch labels shape:  {tuple(sanity_raw.shape)}")
    print(f"[CHECK] POW batch labels shape:  {tuple(sanity_pow.shape)}")
    print(f"[CHECK] POW positives/sample mean: {sanity_pow.sum(dim=1).float().mean().item():.4f}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_test_cmap = -float("inf")
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        if hasattr(model, "encoder"):
            model.encoder.eval()
        if hasattr(model, "predictor"):
            model.predictor.eval()
        model.classifier.train()

        running_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader, start=1):
            if batch_idx == 1:
                print(f"Starting epoch {epoch} training loop...")

            spectrograms = batch["spectrograms"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True).float()
            labels = remap_labels(labels)

            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits = model.infer_logits(spectrograms)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                if args.grad_clip_norm and args.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model.infer_logits(spectrograms)
                loss = criterion(logits, labels)
                loss.backward()
                if args.grad_clip_norm and args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip_norm)
                optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

            if batch_idx % 10 == 0:
                current_loss = float(loss.item())
                avg_loss = running_loss / num_batches
                print(
                    f"Epoch {epoch} | batch {batch_idx}/{len(train_loader)} | "
                    f"current loss {current_loss:.5f} | avg loss {avg_loss:.5f}"
                )

        train_loss = running_loss / max(num_batches, 1)
        test_metrics = evaluate(model, test_loader, device, remap_labels)

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            **{f"test_{k}": float(v) for k, v in test_metrics.items()},
        }
        history.append(row)

        print(
            f"Epoch {epoch}: train_loss={train_loss:.6f} | "
            f"test_cmAP={test_metrics['cmAP']:.6f} | test_AUROC={test_metrics['AUROC']:.6f} | test_top1={test_metrics['top1_acc']:.6f}"
        )

        if test_metrics["cmAP"] > best_test_cmap:
            best_test_cmap = test_metrics["cmAP"]
            best_path = out_dir / args.save_best_name
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "test_metrics": test_metrics,
                    "args": vars(args),
                    "pow_vocab": pow_vocab,
                },
                best_path,
            )
            print(f"Saved new best checkpoint to {best_path}")

        latest_path = out_dir / "last_probe.ckpt"
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "test_metrics": test_metrics,
                "args": vars(args),
                "pow_vocab": pow_vocab,
            },
            latest_path,
        )

    final_metrics = history[-1] if history else {}
    print("\nFinal metrics:")
    for k, v in final_metrics.items():
        print(f"{k}: {v}")

    if args.output_json is not None:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"history": history, "best_test_cmAP": best_test_cmap}, indent=2))
        print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
