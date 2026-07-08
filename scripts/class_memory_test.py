#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

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


def collect_target_indices(dataset, target_species_idx: int, max_samples: int) -> List[int]:
    """
    Scan raw label metadata only; do not decode audio.
    """
    target_indices: List[int] = []
    raw_label_column = dataset.dataset[dataset.label_field_name]

    for i, raw_labels in enumerate(raw_label_column):
        if target_species_idx in raw_labels:
            print(raw_labels)
            target_indices.append(i)
            if len(target_indices) >= max_samples:
                break
    # add some negative examples for comparison
    for i, raw_labels in enumerate(raw_label_column):
        if target_species_idx not in raw_labels:
            print(raw_labels)
            target_indices.append(i)
            if len(target_indices) >= 2 * max_samples:
                break
    return target_indices


@torch.no_grad()
def evaluate_subset(
    model: torch.nn.Module,
    loader: DataLoader,
    target_species_idx: int,
    device: torch.device,
    top_k: int = 5,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()

    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    total_loss = 0.0
    total_samples = 0
    target_correct = 0
    target_species_idx = 2453

    idx_to_label = None
    if hasattr(loader.dataset, "dataset") and hasattr(loader.dataset, "label_to_idx"):
        idx_to_label = {v: k for k, v in loader.dataset.label_to_idx.items()}

    for batch_num, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)

        logits = model.infer_logits(batch)
        probs = torch.sigmoid(logits)
        labels = batch["labels"].float()

        loss = criterion(logits, labels)
        bs = labels.shape[0]
        total_loss += loss.item() * bs
        total_samples += bs

        target_probs = probs[:, target_species_idx]
        target_true = labels[:, target_species_idx]

        # "Accuracy" for the target species = thresholded binary correctness
        target_pred = (target_probs >= 0.5).float()
        target_correct += (target_pred == target_true).sum().item()

        # Print sample-level diagnostics
        source_indices = batch.get("_source_index", None)
        if source_indices is None:
            source_indices = list(range(batch_num * bs, batch_num * bs + bs))

        for i in range(bs):
            sample_idx = int(source_indices[i]) if not torch.is_tensor(source_indices) else int(source_indices[i].item())
            tp = float(target_probs[i].item())
            tt = float(target_true[i].item())

            top_probs, top_idx = torch.topk(probs[i], k=min(top_k, probs.shape[1]))
            top_pairs = []
            for p, c in zip(top_probs.tolist(), top_idx.tolist()):
                if idx_to_label is not None:
                    name = idx_to_label.get(c, f"class_{c}")
                    top_pairs.append(f"{c}:{name}={p:.4f}")
                else:
                    top_pairs.append(f"{c}={p:.4f}")

            true_top_rank = torch.argsort(probs[i], descending=True).tolist().index(target_species_idx) + 1

            print("\n--- Sample ---")
            print(f"source_index: {sample_idx}")
            print(f"target_species_idx: {target_species_idx}")
            print(f"true target label: {tt:.0f}")
            print(f"target probability: {tp:.6f}")
            print(f"target rank among all classes: {true_top_rank}")
            print(f"top-{top_k} predictions:")
            for s in top_pairs:
                print(f"  {s}")

    metrics = {
        "loss": float(total_loss / max(total_samples, 1)),
        "target_accuracy": float(target_correct / max(total_samples, 1)),
        "num_samples": int(total_samples),
    }

    if was_training:
        model.train()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Species memory test for BirdSet MAE/Proto model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--target-species-idx", type=int, required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
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
    parser.add_argument("--subset", type=str, default="XCL")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output-json", type=str, default=None)

    args = parser.parse_args()

    print("Loading dataset...")
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

    print("Collecting target indices from labels only...")
    target_indices = collect_target_indices(dataset, args.target_species_idx, args.num_samples)
    print(f"Found {len(target_indices)} samples for target_species_idx={args.target_species_idx}")

    if len(target_indices) == 0:
        raise RuntimeError("No samples found for that species.")

    class IndexedSubset(Dataset):
        def __init__(self, base_dataset, indices):
            self.base_dataset = base_dataset
            self.indices = list(indices)

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            src_idx = self.indices[i]
            sample = self.base_dataset[src_idx]
            if isinstance(sample, dict):
                sample = dict(sample)
                sample["_source_index"] = src_idx
            return sample

    subset_ds = IndexedSubset(dataset, target_indices)

    subset_loader = DataLoader(
        subset_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # keep this simple for a small probe
        collate_fn=loader.collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    num_classes = data.dataset.num_classes

    print("Loading model...")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Running probe on {len(target_indices)} samples...")
    metrics = evaluate_subset(
        model=model,
        loader=subset_loader,
        target_species_idx=args.target_species_idx,
        device=device,
        top_k=5,
    )

    print("\n=== Probe summary ===")
    print(f"Subset: {args.subset}")
    print(f"Split:  {args.split}")
    print(f"Target species idx: {args.target_species_idx}")
    print(f"Loss:   {metrics['loss']:.6f}")
    print(f"Acc:    {metrics['target_accuracy']:.6f}")
    print(f"Samples: {metrics['num_samples']}")

    if args.output_json is not None:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2))
        print(f"Saved metrics to: {out_path}")


if __name__ == "__main__":
    main()