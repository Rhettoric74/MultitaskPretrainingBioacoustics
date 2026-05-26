#!/usr/bin/env python3
"""
Pack a Hugging Face audio dataset into tar shards without decoding audio.

What this script does:
- loads the requested HF dataset split
- rewrites migrated audio paths when needed
- copies the original compressed audio bytes (.ogg) into tar shards
- writes a JSON metadata sidecar for each sample inside the tar
- writes a per-shard JSONL manifest on disk for easier indexing/debugging

What this script does NOT do:
- it does not decode audio into waveforms
- it does not resample audio
- it does not create Arrow shards

This is intended for preserving the original compressed storage footprint
while reducing the many-small-files problem.

Example:
python build_audio_shards.py \
  --dataset-name DBD-research-group/BirdSet \
  --subset XCL \
  --split train \
  --cache-dir /scratch/Projects/CFP-04/CFP04-CF-029/birdset \
  --output-dir /scratch/Projects/CFP-04/CFP04-CF-029/birdset_tar_shards \
  --audio-column audio \
  --old-audio-root /scratch/projects/CFP04/CFP04-CF-029/birdset \
  --new-audio-root /scratch/Projects/CFP-04/CFP04-CF-029/birdset \
  --num-shards 64
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from datasets import load_dataset


# =========================================================
# PATH REWRITE
# =========================================================
def rewrite_audio_path(
    path: str,
    old_root: str | None,
    new_root: str | None,
) -> str:
    """Rewrite dataset paths from an old cluster layout to a new one."""
    path = str(path)

    if old_root and new_root and path.startswith(old_root):
        return new_root + path[len(old_root) :]

    # If the stored path is relative, make it relative to the new root if given.
    if new_root and not os.path.isabs(path):
        return os.path.join(new_root, path)

    return path


# =========================================================
# JSON HELPERS
# =========================================================
def _jsonable(obj: Any) -> Any:
    """Convert common numpy / pathlib objects into JSON-safe types."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (bytes, bytearray)):
        return None
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    try:
        return obj.item()  # type: ignore[attr-defined]
    except Exception:
        return str(obj)


# =========================================================
# AUDIO FILE RESOLUTION / COPYING
# =========================================================
def _resolve_audio_path(
    audio_obj: Any,
    old_audio_root: str | None,
    new_audio_root: str | None,
) -> Tuple[str, Optional[bytes]]:
    """
    Return (resolved_path, inline_bytes).

    If the HF audio object already contains raw bytes, return them directly.
    Otherwise return the resolved file path and None.
    """
    if isinstance(audio_obj, dict):
        raw_path = audio_obj.get("path", None)
        raw_bytes = audio_obj.get("bytes", None)

        if raw_bytes is not None:
            return str(raw_path) if raw_path is not None else "", bytes(raw_bytes)

        if raw_path is None:
            raise KeyError("Audio object missing 'path' field")

        resolved = rewrite_audio_path(str(raw_path), old_audio_root, new_audio_root)
        return resolved, None

    # Best effort for Audio-like objects
    raw_path = getattr(audio_obj, "path", None)
    raw_bytes = getattr(audio_obj, "bytes", None)
    if raw_bytes is not None:
        return str(raw_path) if raw_path is not None else "", bytes(raw_bytes)
    if raw_path is None:
        raise TypeError(f"Unsupported audio object type: {type(audio_obj)!r}")
    resolved = rewrite_audio_path(str(raw_path), old_audio_root, new_audio_root)
    return resolved, None


def _read_audio_bytes_from_path(path: str) -> bytes:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        return f.read()


# =========================================================
# NAME / METADATA HELPERS
# =========================================================
def _safe_stem(text: Any) -> str:
    s = str(text)
    s = s.strip().replace(os.sep, "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._")
    return s or "sample"


def _pick_sample_id(sample: Dict[str, Any], idx: int) -> str:
    for key in ("id", "sample_id", "uuid", "audio_id", "recording_id"):
        if key in sample and sample[key] is not None:
            return _safe_stem(sample[key])
    return f"sample_{idx:09d}"


def _sample_metadata(
    sample: Dict[str, Any],
    audio_column: str,
    source_audio_path: str,
    resolved_audio_path: str,
    sample_id: str,
    shard_name: str,
    audio_member_name: str,
) -> Dict[str, Any]:
    meta = {k: _jsonable(v) for k, v in sample.items() if k != audio_column}
    meta["sample_id"] = sample_id
    meta["source_audio_path"] = source_audio_path
    meta["resolved_audio_path"] = resolved_audio_path
    meta["shard_name"] = shard_name
    meta["audio_member_name"] = audio_member_name
    return meta


# =========================================================
# SHARDING
# =========================================================
def _iter_shard_bounds(n_items: int, num_shards: int) -> Iterable[Tuple[int, int]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be > 0")
    base = n_items // num_shards
    rem = n_items % num_shards
    start = 0
    for shard_idx in range(num_shards):
        size = base + (1 if shard_idx < rem else 0)
        end = start + size
        if start < end:
            yield start, end
        start = end


def build_tar_shards(
    dataset_name: str,
    subset: str,
    split: str,
    cache_dir: str,
    audio_column: str,
    output_dir: str,
    old_audio_root: str | None,
    new_audio_root: str | None,
    num_shards: int,
) -> None:
    """Pack compressed audio bytes into tar shards."""
    ds = load_dataset(
        dataset_name,
        name=subset,
        split=split,
        cache_dir=cache_dir,
    )

    if audio_column not in ds.column_names:
        raise KeyError(
            f"Audio column {audio_column!r} not found. Available columns: {ds.column_names}"
        )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    n = len(ds)
    if n == 0:
        raise ValueError("Dataset is empty")

    print(f"[INFO] Loaded subset={subset} split={split} with {n} samples")
    print(f"[INFO] Writing {num_shards} tar shards to {output_dir}")

    for shard_idx, (start, end) in enumerate(_iter_shard_bounds(n, num_shards)):
        shard_name = f"bird_set-{split}-{shard_idx:05d}-of-{num_shards:05d}"
        tar_path = output_root / f"{shard_name}.tar"
        manifest_path = output_root / f"{shard_name}.jsonl"

        print(f"[INFO] Shard {shard_idx + 1}/{num_shards}: samples {start}:{end} -> {tar_path.name}")

        with tarfile.open(tar_path, mode="w") as tar, manifest_path.open("w", encoding="utf-8") as mf:
            for idx in range(start, end):
                sample = ds[idx]
                audio_obj = sample[audio_column]

                source_audio_path, inline_bytes = _resolve_audio_path(
                    audio_obj,
                    old_audio_root=old_audio_root,
                    new_audio_root=new_audio_root,
                )

                sample_id = _pick_sample_id(sample, idx)

                if inline_bytes is not None:
                    audio_bytes = inline_bytes
                    audio_ext = Path(source_audio_path).suffix if source_audio_path else ".ogg"
                else:
                    resolved = source_audio_path
                    audio_bytes = _read_audio_bytes_from_path(resolved)
                    audio_ext = Path(resolved).suffix or ".ogg"

                audio_ext = audio_ext if audio_ext.startswith(".") else f".{audio_ext}"
                audio_member_name = f"{sample_id}{audio_ext}"
                meta_member_name = f"{sample_id}.json"

                # Write audio member
                audio_info = tarfile.TarInfo(name=audio_member_name)
                audio_info.size = len(audio_bytes)
                audio_info.mode = 0o644
                tar.addfile(audio_info, io.BytesIO(audio_bytes))

                # Write metadata member
                resolved_path = source_audio_path
                meta = _sample_metadata(
                    sample=sample,
                    audio_column=audio_column,
                    source_audio_path=str(sample[audio_column].get("path", "")) if isinstance(sample[audio_column], dict) else str(getattr(sample[audio_column], "path", "")),
                    resolved_audio_path=str(resolved_path),
                    sample_id=sample_id,
                    shard_name=shard_name,
                    audio_member_name=audio_member_name,
                )
                meta_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")
                meta_info = tarfile.TarInfo(name=meta_member_name)
                meta_info.size = len(meta_bytes)
                meta_info.mode = 0o644
                tar.addfile(meta_info, io.BytesIO(meta_bytes))

                # Sidecar manifest for fast indexing/debugging
                manifest_entry = {
                    "sample_index": idx,
                    "sample_id": sample_id,
                    "shard_name": shard_name,
                    "audio_member_name": audio_member_name,
                    "meta_member_name": meta_member_name,
                    "source_audio_path": meta["source_audio_path"],
                    "resolved_audio_path": meta["resolved_audio_path"],
                    "audio_suffix": audio_ext,
                    "metadata": {k: v for k, v in meta.items() if k not in {"source_audio_path", "resolved_audio_path"}},
                }
                mf.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")

        print(f"[INFO] Wrote {tar_path.name} and {manifest_path.name}")


# =========================================================
# CLI
# =========================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--dataset-name", type=str, default="DBD-research-group/BirdSet")
    p.add_argument("--subset", type=str, default="XCL")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--audio-column", type=str, default="audio")
    p.add_argument("--old-audio-root", type=str, default=None)
    p.add_argument("--new-audio-root", type=str, default=None)
    p.add_argument("--num-shards", type=int, default=64)

    return p.parse_args()


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    args = parse_args()

    print("[INFO] Starting tar sharding (compressed audio only)")
    print(f"[INFO] dataset_name={args.dataset_name}")
    print(f"[INFO] subset={args.subset}")
    print(f"[INFO] split={args.split}")
    print(f"[INFO] cache_dir={args.cache_dir}")
    print(f"[INFO] output_dir={args.output_dir}")
    print(f"[INFO] audio_column={args.audio_column}")
    print(f"[INFO] old_audio_root={args.old_audio_root}")
    print(f"[INFO] new_audio_root={args.new_audio_root}")
    print(f"[INFO] num_shards={args.num_shards}")

    build_tar_shards(
        dataset_name=args.dataset_name,
        subset=args.subset,
        split=args.split,
        cache_dir=args.cache_dir,
        audio_column=args.audio_column,
        output_dir=args.output_dir,
        old_audio_root=args.old_audio_root,
        new_audio_root=args.new_audio_root,
        num_shards=args.num_shards,
    )

    print("[INFO] Done.")


if __name__ == "__main__":
    main()

