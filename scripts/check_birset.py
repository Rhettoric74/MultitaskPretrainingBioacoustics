from datasets import load_dataset
import numpy as np

DATASET_PATH = "/scratch/Projects/CFP-04/CFP04-CF-029/birdset"

# Load POW test_5s directly from BirdSet
ds = load_dataset(
    "DBD-research-group/BirdSet",
    name="POW",
    split="test_5s",
    cache_dir=DATASET_PATH,
)

print(ds)

# Prefer multilabel field if available
label_field = None
for candidate in ["ebird_code_multilabel", "ebird_code", "labels", "label"]:
    if candidate in ds.features:
        label_field = candidate
        break

print(f"\nUsing label field: {label_field}")

counts = []

for sample in ds:
    raw = sample.get(label_field)

    if raw is None:
        counts.append(0)
    elif isinstance(raw, (list, tuple, np.ndarray)):
        counts.append(len(raw))
    else:
        # scalar label
        counts.append(1)

counts = np.asarray(counts)

print("\n=== LABEL STATS ===")
print(f"Num samples: {len(counts)}")
print(f"Mean labels/sample: {counts.mean():.6f}")
print(f"Median labels/sample: {np.median(counts):.6f}")
print(f"Min labels/sample: {counts.min()}")
print(f"Max labels/sample: {counts.max()}")

# Optional: inspect first few examples
print("\n=== FIRST 10 SAMPLES ===")
for i in range(min(10, len(ds))):
    print(i, ds[i][label_field])