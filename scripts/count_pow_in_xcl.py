#!/usr/bin/env python3
"""
Count XCL training samples for each POW species using the dataloader's label mappings.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from birdset_waveform_dataloader import BirdSetDataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="/scratch/Projects/CFP-04/CFP04-CF-029/birdset")
    parser.add_argument("--xcl-split", type=str, default="train")
    parser.add_argument("--pow-split", type=str, default="train")
    parser.add_argument("--label-vocab-path", type=str, 
                        default="/home/svu/e1583377/MultitaskPretrainingBioacoustics/scripts/xcl_label_vocab.json")
    args = parser.parse_args()
    
    # Load POW dataset to get species list and label mapping
    print("Loading POW dataset...")
    pow_data = BirdSetDataLoader(
        dataset_path=args.dataset_path,
        subset="POW",
        split=args.pow_split,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        use_mixup=False,
        use_geo_mixup=False,
        label_vocab_path=None,  # Use same vocab to align indices
    )
    
    # Get POW label mapping (ebird code -> index)
    pow_label_to_idx = pow_data.dataset.label_to_idx
    pow_codes = pow_data.dataset.label_list
    print(f"POW species: {pow_codes[:10]}...")
    print(f"Total: {len(pow_codes)}")
    #pow_codes = list(pow_label_to_idx.keys())
    
    print(f"POW has {len(pow_codes)} species")
    print(f"First few POW species: {pow_codes[:10]}")
    
    # Load XCL dataset
    print("\nLoading XCL dataset...")
    xcl_data = BirdSetDataLoader(
        dataset_path=args.dataset_path,
        subset="XCL",
        split=args.xcl_split,
        batch_size=64,
        num_workers=8,
        shuffle=False,
        use_mixup=False,
        use_geo_mixup=False,
        label_vocab_path=args.label_vocab_path,
    )
    xcl_loader = xcl_data.get_loader()
    
    # Get XCL label mapping
    xcl_label_to_idx = xcl_data.dataset.label_to_idx
    xcl_idx_to_label = {v: k for k, v in xcl_label_to_idx.items()}
    
    # Create mapping from POW code to XCL index (since both use same ebird codes)
    pow_code_to_xcl_idx = {}
    for code in pow_codes:
        if code in xcl_label_to_idx:
            pow_code_to_xcl_idx[code] = xcl_label_to_idx[code]
    
    print(f"\nFound {len(pow_code_to_xcl_idx)}/{len(pow_codes)} POW species in XCL vocabulary")
    
    missing_species = [code for code in pow_codes if code not in xcl_label_to_idx]
    if missing_species:
        print(f"Missing species: {missing_species[:10]}...")
    
    # Count samples in XCL for each POW species
    pow_counts = defaultdict(int)
    total_batches = len(xcl_loader)
    
    print("\nCounting samples in XCL...")
    for batch in tqdm(xcl_loader, total=total_batches):
        labels = batch["labels"]  # [B, num_xcl_classes]
        
        for b in range(labels.shape[0]):
            active_indices = torch.where(labels[b] > 0.5)[0].tolist()
            for xcl_idx in active_indices:
                code = xcl_idx_to_label[xcl_idx]
                if code in pow_label_to_idx:
                    pow_counts[code] += 1
    
    # Print results
    print("\n" + "=" * 50)
    print("POW SPECIES COUNTS IN XCL")
    print("=" * 50)
    print(f"{'Code':<15} {'Count':<10}")
    print("-" * 30)
    
    # Sort by count
    for code, count in sorted(pow_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{code:<15} {count:<10}")
    
    # Species with zero counts
    zero_counts = [code for code in pow_codes if code not in pow_counts]
    if zero_counts:
        print("\n" + "-" * 30)
        print("SPECIES WITH ZERO SAMPLES:")
        for code in zero_counts:
            print(f"  {code}")
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total POW samples in XCL: {sum(pow_counts.values())}")
    print(f"Species with >=1 sample: {len(pow_counts)}/{len(pow_codes)}")
    print(f"Species with 0 samples: {len(zero_counts)}/{len(pow_codes)}")


if __name__ == "__main__":
    main()