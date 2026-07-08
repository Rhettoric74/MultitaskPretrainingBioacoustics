import argparse
from pathlib import Path
from typing import Dict, List, Any
import sys
import json

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from birdset_waveform_dataloader import BirdSetDataLoader, _raw_labels_to_codes
from build_mae_model import build_model


def load_checkpoint_state(model: torch.nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
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


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device, open_label = True) -> Dict[str, float]:
    # Save original state
    was_training = model.training
    model.eval()

    all_probs: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    top1_correct = 0
    top1_total = 0
    pow_class_indices = loader.dataset.get_active_class_indices()
    print(pow_class_indices, len(pow_class_indices))

    batch_num = 1
    for batch in loader:
        print(f"{batch_num} batches processed")

        # --- DEBUG LABEL MAPPING ---
        if batch_num == 1:
            ds = loader.dataset
            print("\n=== LABEL DEBUG ===")
            print(f"Dataset class: {type(ds).__name__}")
            print(f"Number of classes: {getattr(ds, 'num_classes', 'UNKNOWN')}")
            print(f"Label field: {getattr(ds, 'label_field_name', 'UNKNOWN')}")
            print(f"Label vocab size: {len(getattr(ds, 'label_to_idx', {}))}")

            # Check a batch directly
            batch_labels = batch["labels"].cpu().numpy()
            print("Batch labels shape:", batch_labels.shape)
            print("Batch labels are binary:", np.all(np.isin(batch_labels, [0, 1])))
            print("Labels sum per sample (mean):", batch_labels.sum(axis=1).mean())
            print("Labels sum per class (mean):", batch_labels.sum(axis=0).mean())
            batch_labels_pow = batch_labels[:, pow_class_indices]
            num_samples = min(20, batch_labels.shape[0])
            for i in range(num_samples):
                positive_indices = np.where(batch_labels_pow[i] >= 1)[0]
                print(f"Sample {i}: Positive class indices = {positive_indices.tolist()}")

            # Try to inspect a raw sample if available
            try:
                if hasattr(ds, "samples") and len(ds.samples) > 0:
                    sample = ds.samples[0]
                    raw_labels = sample.get(ds.label_field_name)
                    print("Raw labels from first sample:", raw_labels)
                    print("First sample keys:", list(sample.keys())[:10])
            except Exception as e:
                print(f"[WARN] Could not inspect raw sample labels: {e}")

            print("===================\n")

        batch_num += 1

        batch = move_batch_to_device(batch, device)

        # MAE model expects the full batch dict, not just spectrogram tensors
        logits = model.infer_logits(batch)
        print(batch["coordinates"][0])
        probs = torch.sigmoid(logits)
        probs_pow = probs[:, pow_class_indices]  # Shape: (N, 48)
        labels = batch["labels"].float()
        labels_pow = labels[:, pow_class_indices]    # Shape: (N, 48)
        all_probs.append(probs_pow.float().cpu())
        all_targets.append(labels_pow.float().cpu())

        # Top-1 for multilabel: is the highest-scoring class among the true labels?
        if open_label:
            pred_idx = probs.argmax(dim=1)
            row_idx = torch.arange(labels.shape[0], device=labels.device)
            
            has_label = labels.sum(dim=1) > 0
            correct = (labels[row_idx, pred_idx] > 0.5) & has_label
            
            top1_correct += correct.sum().item()
            top1_total += has_label.sum().item()
            print("Batch samples with at least one label:", has_label.sum().item())
        else:
            pred_idx = probs_pow.argmax(dim=1)
            row_idx = torch.arange(labels_pow.shape[0], device=labels_pow.device)
            
            has_label = labels_pow.sum(dim=1) > 0
            correct = (labels_pow[row_idx, pred_idx] > 0.5) & has_label
            
            top1_correct += correct.sum().item()
            top1_total += has_label.sum().item()
            print("Batch samples with at least one label:", has_label.sum().item())

    y_score = torch.cat(all_probs, dim=0).numpy()
    y_true = torch.cat(all_targets, dim=0).numpy().astype(np.int32)

    ap_vals = []
    auc_vals = []
    pos_labels = []

    for c in range(y_true.shape[1]):
        yt = y_true[:, c]
        ys = y_score[:, c]
        pos = yt.sum()
        neg = yt.shape[0] - pos
        if pos == 0 or neg == 0:
            continue
        pos_labels.append(pos)
        ap_vals.append(average_precision_score(yt, ys))
        auc_vals.append(roc_auc_score(yt, ys))

    metrics = {
        "cmAP": float(np.mean(ap_vals)) if ap_vals else float("nan"),
        "AUROC": float(np.mean(auc_vals)) if auc_vals else float("nan"),
        "top1_acc": float(top1_correct / max(top1_total, 1)),
        "num_classes_used_for_map": int(len(ap_vals)),
        "num_classes_used_for_auroc": int(len(auc_vals)),
        "ap_vals": ap_vals,
        "auc_vals": auc_vals,
        "pos_labels_per_class": pos_labels,
        "num_samples": int(top1_total),
    }
    
    # Restore original training state
    if was_training:
        model.train()
    
    return metrics


def verify_label_mapping(args):
    """Standalone function to verify label mapping without running full evaluation."""
    data = BirdSetDataLoader(
        dataset_path=args.dataset_path,
        subset=args.subset,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        use_mixup=False,
        use_geo_mixup=False,
        label_vocab_path=args.label_vocab_path,
    )

    loader = data.get_loader()
    dataset = loader.dataset
    print("\n=== FULL SPLIT LABEL STATS ===")

    raw_counts = []
    unique_counts = []
    
    for i in range(len(dataset.dataset)):
        sample = dataset.dataset[i]
        raw = sample.get(dataset.label_field_name)
        codes = _raw_labels_to_codes(raw, dataset.label_int2str)
    
        raw_counts.append(len(codes))
        unique_counts.append(len(set(codes)))
    
    print(f"Num samples: {len(raw_counts)}")
    print(f"Mean raw labels/sample: {np.mean(raw_counts):.6f}")
    print(f"Mean unique labels/sample: {np.mean(unique_counts):.6f}")
    print(f"Median raw labels/sample: {np.median(raw_counts):.6f}")
    print(f"Median unique labels/sample: {np.median(unique_counts):.6f}")
    print(f"Min raw labels/sample: {min(raw_counts)}")
    print(f"Max raw labels/sample: {max(raw_counts)}")
    print(f"Min unique labels/sample: {min(unique_counts)}")
    print(f"Max unique labels/sample: {max(unique_counts)}")
    print("==============================\n")

    print(f"Dataset: {args.subset} - {args.split}")
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

    return dataset


def quick_checkpoint_summary(ckpt_path):
    """Quick summary of checkpoint contents without loading model."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    print(f"\nCheckpoint: {ckpt_path}")
    print(f"Size: {Path(ckpt_path).stat().st_size / 1e6:.2f} MB")

    if isinstance(ckpt, dict):
        print(f"Top-level keys: {list(ckpt.keys())}")

        if "epoch" in ckpt:
            print(f"  - Epoch: {ckpt['epoch']}")
        if "global_step" in ckpt:
            print(f"  - Global step: {ckpt['global_step']}")
        if "optimizer_states" in ckpt:
            print("  - Has optimizer states")

        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            print(f"  - State dict keys: {len(state_dict)}")
            keys_sample = list(state_dict.keys())[:10]
            print(f"  - Sample keys: {keys_sample}")
    else:
        print(f"Checkpoint is direct state_dict with {len(ckpt)} keys")
        keys_sample = list(ckpt.keys())
        print(f"Sample keys: {keys_sample}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a pretrained BirdSet MAE model on POW test_5s")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--label-vocab-path",
        type=str,
        default="/home/svu/e1583377/MultitaskPretrainingBioacoustics/scripts/xcl_label_vocab.json",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/scratch/Projects/CFP-04/CFP04-CF-029/birdset",
    )
    parser.add_argument("--subset", type=str, default="POW")
    parser.add_argument("--split", type=str, default="test_5s")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--debug-labels", action="store_true", help="Debug label mapping without running evaluation")

    # If your waveform loader expects these, keep them available.
    parser.add_argument("--spec-norm-path", type=str, default="/home/svu/e1583377/MultitaskPretrainingBioacoustics/scripts/power_spec_norm_stats_XCL.json")
    parser.add_argument("--apply-spec-norm", action="store_true", default=True)

    args = parser.parse_args()

    quick_checkpoint_summary(args.checkpoint)
    if args.debug_labels:
        verify_label_mapping(args)
        sys.exit(0)

    torch.set_float32_matmul_precision("high")

    # Waveform loader: no spectrogram input here.
    data = BirdSetDataLoader(
        dataset_path=args.dataset_path,
        subset=args.subset,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        use_mixup=False,
        use_geo_mixup=False,
        label_vocab_path=args.label_vocab_path,
    )
    loader = data.get_loader()

    num_classes = data.dataset.num_classes

    # MAE model: use the same build_model signature you trained with.
    model = build_model(
        num_classes=num_classes,
        classifier_type="proto",
        use_cls_token=False,
        objective_mode="joint",
        optimizer_type="dcgd",
        mask_ratio=0.75,
        lambda_recon=1.0,
        lambda_cls=1.0,
        learning_rate=2e-4,
        weight_decay=0.0,
    )

    load_checkpoint_state(model, args.checkpoint)
    class_idx = 75
    pow_indices = [75, 351, 1118, 1707, 2453, 3231, 3277, 3297, 3325, 3346, 4792, 5806, 5820, 6054, 6303, 6371, 6373, 6376, 6377, 6386, 6396, 6402, 6406, 6407, 6414, 6418, 6879, 6933, 6951, 6956, 7103, 7109, 7111, 7116, 7171, 7179, 7206, 7761, 7774, 7807, 7931, 8016, 8041, 8048, 8095, 9550, 9573, 9675]
    print("Model loaded!")
    print("Class bias for POW indices:", model.classifier.class_bias[pow_indices].detach().cpu())
    print("Class weights for POW indices:", model.classifier.class_weights[pow_indices].detach().cpu())
    orthogonality_loss = model.classifier.orthogonality_loss().detach().cpu()
    print("Orthogonality loss:", orthogonality_loss)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    metrics = evaluate(model, loader, device)

    print(f"Subset: {args.subset}")
    print(f"Split:  {args.split}")
    print(f"cmAP:   {metrics['cmAP']:.6f}")
    print(f"AUROC:  {metrics['AUROC']:.6f}")
    print(f"top-1:  {metrics['top1_acc']:.6f}")
    print(f"classes used for cmAP:  {metrics['num_classes_used_for_map']}")
    print(f"classes used for AUROC: {metrics['num_classes_used_for_auroc']}")
    print(f"cmAP values: {metrics['ap_vals']}")
    print(f"AUROC values: {metrics['auc_vals']}")
    print(f"samples: {metrics['num_samples']}")
    print(f"positive labels per_class: {metrics['pos_labels_per_class']}")

    if args.output_json is not None:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2))
        print(f"Saved metrics to: {out_path}")


if __name__ == "__main__":
    main()