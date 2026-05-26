import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

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
        print(f"[WARN] Missing keys: {missing}")
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
# Top-k pooling
# -------------------------------------------------
def topk_pool_tokens(h: torch.Tensor, top_k: int = 8, score: str = "norm") -> torch.Tensor:
    """Pool token embeddings by selecting the top-k tokens per sample.

    Args:
        h: [B, N, D] token embeddings.
        top_k: number of tokens to keep per sample.
        score: token scoring strategy. Currently supports:
            - "norm": token L2 norm
    Returns:
        [B, D] pooled embeddings
    """
    if h.dim() != 3:
        raise ValueError(f"Expected [B, N, D], got {tuple(h.shape)}")

    B, N, D = h.shape
    k = max(1, min(int(top_k), N))

    if score != "norm":
        raise ValueError(f"Unsupported score mode: {score}")

    token_scores = h.norm(dim=-1)  # [B, N]
    top_idx = torch.topk(token_scores, k=k, dim=1, largest=True, sorted=False).indices  # [B, K]
    pooled_tokens = torch.gather(h, dim=1, index=top_idx.unsqueeze(-1).expand(-1, -1, D))
    pooled = pooled_tokens.mean(dim=1)
    return pooled


# -------------------------------------------------
# Feature extraction
# -------------------------------------------------
@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    top_k: int,
    remap_labels,
    use_amp: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract top-k pooled embeddings and POW labels from a loader."""
    model.eval()

    all_feats: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    autocast_enabled = bool(use_amp and device.type == "cuda")

    for batch_idx, batch in enumerate(loader, start=1):
        spectrograms = batch["spectrograms"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True).float()
        labels = remap_labels(labels)

        # Use the frozen encoder directly; no masks for the kNN probe.
        if autocast_enabled:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                h = model.encoder(spectrograms, None)
        else:
            h = model.encoder(spectrograms, None)

        pooled = topk_pool_tokens(h, top_k=top_k)
        pooled = F.normalize(pooled, dim=-1)

        all_feats.append(pooled.detach().cpu())
        all_targets.append(labels.detach().cpu())

        if batch_idx == 1:
            print(f"[INFO] Example encoder output shape: {tuple(h.shape)}")
            print(f"[INFO] Example pooled embedding shape: {tuple(pooled.shape)}")
            print(f"[INFO] Example labels shape: {tuple(labels.shape)}")
        if batch_idx % 5 == 0:
            print("Processed batch:", batch_idx)

    feats = torch.cat(all_feats, dim=0).numpy().astype(np.float32)
    targets = torch.cat(all_targets, dim=0).numpy().astype(np.int32)
    return feats, targets


# -------------------------------------------------
# kNN evaluation
# -------------------------------------------------
@torch.no_grad()
def knn_predict(
    train_feats: np.ndarray,
    train_targets: np.ndarray,
    test_feats: np.ndarray,
    k: int = 5,
    metric: str = "cosine",
) -> np.ndarray:
    """Return predicted class scores for multilabel / one-label evaluation.

    For each test point, nearest neighbors vote with inverse-distance weighting.
    """
    if metric != "cosine":
        raise ValueError("This script currently supports cosine kNN only.")

    nn_model = NearestNeighbors(n_neighbors=min(k, len(train_feats)), metric="cosine")
    nn_model.fit(train_feats)
    distances, indices = nn_model.kneighbors(test_feats, return_distance=True)

    # Convert cosine distance to similarity weights.
    sims = 1.0 - distances
    sims = np.clip(sims, 0.0, None)
    sims_sum = sims.sum(axis=1, keepdims=True)
    sims_sum[sims_sum == 0] = 1.0
    weights = sims / sims_sum

    num_classes = train_targets.shape[1]
    y_score = np.zeros((len(test_feats), num_classes), dtype=np.float32)

    for i in range(len(test_feats)):
        neigh_labels = train_targets[indices[i]]  # [k, C]
        # Weighted vote over class dimension.
        y_score[i] = (weights[i][:, None] * neigh_labels).sum(axis=0)

    return y_score


@torch.no_grad()
def compute_metrics_from_scores(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    ap_vals: List[float] = []
    auc_vals: List[float] = []

    for c in range(labels.shape[1]):
        yt = labels[:, c]
        ys = scores[:, c]
        pos = yt.sum()
        neg = yt.shape[0] - pos
        if pos == 0 or neg == 0:
            continue
        ap_vals.append(average_precision_score(yt, ys))
        auc_vals.append(roc_auc_score(yt, ys))

    pred_idx = scores.argmax(axis=1)
    row_idx = np.arange(labels.shape[0])
    top1_correct = (labels[row_idx, pred_idx] > 0.5).sum()
    top1_total = labels.shape[0]

    return {
        "cmAP": float(np.mean(ap_vals)) if ap_vals else float("nan"),
        "AUROC": float(np.mean(auc_vals)) if auc_vals else float("nan"),
        "top1_acc": float(top1_correct / max(top1_total, 1)),
        "num_classes_used_for_map": int(len(ap_vals)),
        "num_classes_used_for_auroc": int(len(auc_vals)),
        "num_samples": int(top1_total),
    }


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Frozen-backbone top-k pooling kNN probe for BirdSet POW")
    parser.add_argument("--checkpoint", type=str, required=True, help="JEPA checkpoint to load")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, default="/scratch/Projects/CFP-04/CFP04-CF-029/birdset")
    parser.add_argument("--label-vocab-path", type=str, default="/home/svu/e1583377/MultitaskPretrainingBioacoustics/scripts/xcl_label_vocab.json")
    parser.add_argument("--spec-norm-path", type=str, default="/home/svu/e1583377/MultitaskPretrainingBioacoustics/scripts/xcl_spec_stats_true_log.json")
    parser.add_argument("--subset", type=str, default="POW")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--test-split", type=str, default="test_5s")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--apply-spec-norm", action="store_true")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.use_amp and device.type == "cuda")

    print("Building datasets...")
    train_data, train_loader = BirdSetDataLoader(
        dataset_path=args.dataset_path,
        subset=args.subset,
        split=args.train_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        use_mixup=False,
        use_geo_mixup=False,
        label_vocab_path=args.label_vocab_path,
        spec_norm_path=args.spec_norm_path,
        apply_spec_norm=args.apply_spec_norm,
    ), None
    train_loader = train_data.get_loader()

    test_data, test_loader = BirdSetDataLoader(
        dataset_path=args.dataset_path,
        subset=args.subset,
        split=args.test_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        use_mixup=False,
        use_geo_mixup=False,
        label_vocab_path=args.label_vocab_path,
        spec_norm_path=args.spec_norm_path,
        apply_spec_norm=args.apply_spec_norm,
    ), None
    test_loader = test_data.get_loader()

    train_raw_ds = _dataset_from_loader(train_loader)
    test_raw_ds = _dataset_from_loader(test_loader)
    label_field = train_loader.dataset.label_field_name

    pow_vocab = _get_label_vocab_in_feature_order(train_raw_ds, label_field)
    test_pow_vocab = _get_label_vocab_in_feature_order(test_raw_ds, label_field)

    print(f"[INFO] POW canonical vocab size (train): {len(pow_vocab)}")
    print(f"[INFO] POW canonical vocab size (test):  {len(test_pow_vocab)}")
    if pow_vocab != test_pow_vocab:
        print("[WARN] POW train/test canonical vocab differs; using train order for probe.")

    xcl_vocab = train_loader.dataset.label_list
    remap_labels = build_label_remapper(xcl_vocab, pow_vocab, device)

    num_classes = len(pow_vocab)
    num_patches = (128 // args.patch_size) * (512 // args.patch_size)
    print(f"[INFO] Probe classes: {num_classes}")
    print(f"[INFO] num_patches={num_patches}")
    print(f"[INFO] top-k pooling: k={args.top_k}")
    print(f"[INFO] kNN neighbors: k={args.knn_k}")
    model = build_model(num_classes, num_patches)
    load_checkpoint_state(model, args.checkpoint, strict=False, skip_prefixes=("classifier.",))
    model.to(device)

    if hasattr(model, "encoder"):
        model.encoder.eval()
        for p in model.encoder.parameters():
            p.requires_grad = False
    if hasattr(model, "predictor"):
        model.predictor.eval()
        for p in model.predictor.parameters():
            p.requires_grad = False
    if hasattr(model, "classifier"):
        # Not used for kNN probing.
        pass

    print("Extracting train embeddings...")
    train_feats, train_targets = extract_embeddings(
        model=model,
        loader=train_loader,
        device=device,
        top_k=args.top_k,
        remap_labels=remap_labels,
        use_amp=use_amp,
    )
    print("Extracting test embeddings...")
    test_feats, test_targets = extract_embeddings(
        model=model,
        loader=test_loader,
        device=device,
        top_k=args.top_k,
        remap_labels=remap_labels,
        use_amp=use_amp,
    )

    print("Running kNN...")
    test_scores = knn_predict(
        train_feats=train_feats,
        train_targets=train_targets,
        test_feats=test_feats,
        k=args.knn_k,
        metric="cosine",
    )

    metrics = compute_metrics_from_scores(test_scores, test_targets)
    print("\nFinal metrics:")
    print(f"cmAP:   {metrics['cmAP']:.6f}")
    print(f"AUROC:  {metrics['AUROC']:.6f}")
    print(f"top-1:  {metrics['top1_acc']:.6f}")
    print(f"classes used for cmAP:  {metrics['num_classes_used_for_map']}")
    print(f"classes used for AUROC: {metrics['num_classes_used_for_auroc']}")
    print(f"samples: {metrics['num_samples']}")

    if args.output_json is not None:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2))
        print(f"Saved metrics to: {out_path}")


if __name__ == "__main__":
    main()
