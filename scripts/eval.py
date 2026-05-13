import argparse
from pathlib import Path
from typing import Dict, List
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from birdset_dataloader_v2 import BirdSetDataLoader
from build_model import build_model


def load_checkpoint_state(model: torch.nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # Some checkpoints may include Lightning prefixing; strict=False makes this
    # usable across minor refactors while still warning about mismatches.
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        print(f"[WARN] Missing keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")



@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device) -> Dict[str, float]:
    model.eval()

    all_probs: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    top1_correct = 0
    top1_total = 0
    batch_num = 1
    for batch in loader:
        print(str(batch_num) + " batches processed")
        # --- DEBUG LABEL MAPPING ---
        if batch_num == 1:
            ds = loader.dataset.dataset
            label_field = loader.dataset.label_field_name
            int2str = loader.dataset.label_int2str
            label_to_idx = loader.dataset.label_to_idx
        
            print("\n=== LABEL DEBUG ===")
        
            # find a sample with labels
            for i in range(len(ds)):
                raw_labels = ds[i].get(label_field)
                if raw_labels:  # non-empty
                    sample = ds[i]
                    break
        
            raw_labels = sample.get(label_field)
            print("Raw labels:", raw_labels)
        
            for l in raw_labels:
                if isinstance(l, (int, np.integer)) and int2str is not None:
                    code = int2str(int(l))
                else:
                    code = str(l)
        
                mapped_idx = label_to_idx.get(code, None)
        
                print(f"raw: {l} -> code: {code} -> mapped idx: {mapped_idx}")
        
            print("===================\n")
        batch_num += 1
        spectrograms = batch["spectrograms"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True).float()

        logits = model.infer_logits(spectrograms)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.float().cpu())
        all_targets.append(labels.float().cpu())

        # Top-1 for multilabel: is the highest-scoring class among the true labels?
        pred_idx = probs.argmax(dim=1)
        row_idx = torch.arange(labels.shape[0], device=labels.device)
        top1_correct += (labels[row_idx, pred_idx] > 0.5).sum().item()
        top1_total += labels.shape[0]

    y_score = torch.cat(all_probs, dim=0).numpy()
    y_true = torch.cat(all_targets, dim=0).numpy().astype(np.int32)

    ap_vals = []
    auc_vals = []

    for c in range(y_true.shape[1]):
        yt = y_true[:, c]
        ys = y_score[:, c]
        pos = yt.sum()
        neg = yt.shape[0] - pos
        if pos == 0 or neg == 0:
            continue

        ap_vals.append(average_precision_score(yt, ys))
        auc_vals.append(roc_auc_score(yt, ys))

    metrics = {
        "cmAP": float(np.mean(ap_vals)) if ap_vals else float("nan"),
        "AUROC": float(np.mean(auc_vals)) if auc_vals else float("nan"),
        "top1_acc": float(top1_correct / max(top1_total, 1)),
        "num_classes_used_for_map": int(len(ap_vals)),
        "num_classes_used_for_auroc": int(len(auc_vals)),
        "num_samples": int(top1_total),
    }
    return metrics

def verify_label_mapping(args):
    """Standalone function to verify label mapping without running full evaluation"""
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
    
    print(f"Dataset: {args.subset} - {args.split}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Label field: {dataset.label_field_name}")
    print(f"Label vocab size: {len(dataset.label_to_idx)}")
    
    # Check first batch
    batch = next(iter(loader))
    labels = batch["labels"].numpy()
    
    print(f"\nBatch shape: {labels.shape}")
    print(f"Labels are binary: {np.all(np.isin(labels, [0, 1]))}")
    print(f"Labels sum per sample (mean): {labels.sum(axis=1).mean():.2f}")
    print(f"Labels sum per class (mean): {labels.sum(axis=0).mean():.2f}")
    
    # Check if any class appears
    class_counts = labels.sum(axis=0)
    active_classes = np.where(class_counts > 0)[0]
    print(f"\nActive classes in first batch: {len(active_classes)} out of {dataset.num_classes}")
    
    # Map back to original labels
    idx_to_label = {v: k for k, v in dataset.label_to_idx.items()}
    print("\n Active classes with their original labels:")
    for class_idx in active_classes:
        print(f"  Index {class_idx} -> {idx_to_label.get(class_idx, 'UNKNOWN')}")
    
    return dataset


def quick_checkpoint_summary(ckpt_path):
    """Quick summary of checkpoint contents without loading model"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    print(f"\n?? Checkpoint: {ckpt_path}")
    print(f"Size: {Path(ckpt_path).stat().st_size / 1e6:.2f} MB")
    
    if isinstance(ckpt, dict):
        print(f"Top-level keys: {list(ckpt.keys())}")
        
        # Check for common training artifacts
        if 'epoch' in ckpt:
            print(f"  - Epoch: {ckpt['epoch']}")
        if 'global_step' in ckpt:
            print(f"  - Global step: {ckpt['global_step']}")
        if 'optimizer_states' in ckpt:
            print(f"  - Has optimizer states")
        
        # Check state_dict size
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
            print(f"  - State dict keys: {len(state_dict)}")
            
            # Show what's in the state dict
            keys_sample = list(state_dict.keys())[:10]
            print(f"  - Sample keys: {keys_sample}")
    else:
        print(f"Checkpoint is direct state_dict with {len(ckpt)} keys")
        keys_sample = list(ckpt.keys())
        print(f"Sample keys: {keys_sample}")



def main():
    parser = argparse.ArgumentParser(description="Evaluate a pretrained BirdSet model on POW test_5s")
    parser.add_argument("--checkpoint", type=str, required=True)
    # /scratch/Projects/CFP-04/CFP04-CF-029/checkpoints/jepa_audio/initial_run/last.ckpt
    parser.add_argument("--label-vocab-path", type=str, default = "/home/svu/e1583377/MultitaskPretrainingBioacoustics/scripts/xcl_label_vocab.json")
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
    parser.add_argument("--debug-labels", action="store_true", 
                       help="Debug label mapping without running evaluation")
    args = parser.parse_args()
    
    quick_checkpoint_summary(args.checkpoint)
    if args.debug_labels:
        verify_label_mapping(args)
        sys.exit(0)

    torch.set_float32_matmul_precision("high")

    # IMPORTANT: turn off training augmentations for evaluation.
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
        spec_norm_path="/home/svu/e1583377/MultitaskPretrainingBioacoustics/scripts/xcl_spec_stats_true_log.json",
        apply_spec_norm=True,
    )
    loader = data.get_loader()
    

    num_classes = data.dataset.num_classes
    num_patches = (128 // args.patch_size) * (512 // args.patch_size)

    model = build_model(num_classes, num_patches)
    load_checkpoint_state(model, args.checkpoint)

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
    print(f"samples: {metrics['num_samples']}")

    if args.output_json is not None:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(__import__("json").dumps(metrics, indent=2))
        print(f"Saved metrics to: {out_path}")


if __name__ == "__main__":
    main()
