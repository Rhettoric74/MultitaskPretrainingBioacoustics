from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
from datasets import load_dataset
from scipy.spatial import KDTree
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")


# =========================================================
# Helpers
# =========================================================
EARTH_RADIUS_KM = 6371.0088
EARTH_RADIUS_MI = 3958.7613

def haversine_distance(lat1, lon1, lat2, lon2, radius=EARTH_RADIUS_KM):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * radius * np.arcsin(np.sqrt(a))
def debug_geo_mix(anchor_sample, neighbor_sample):
        lat1, lon1 = anchor_sample["lat"], anchor_sample["long"]
        lat2, lon2 = neighbor_sample["lat"], neighbor_sample["long"]
    
        km = haversine_distance(lat1, lon1, lat2, lon2, radius=EARTH_RADIUS_KM)
        mi = haversine_distance(lat1, lon1, lat2, lon2, radius=EARTH_RADIUS_MI)
    
        print(f"anchor=({lat1:.4f}, {lon1:.4f})  neighbor=({lat2:.4f}, {lon2:.4f})", flush=True)
        print(f"great-circle distance: {km:.2f} km ({mi:.2f} mi)", flush=True)

def _is_finite_number(x: Any) -> bool:
    try:
        return x is not None and np.isfinite(x)
    except Exception:
        return False
        
        


def _load_json_list(path: str | Path) -> List[str]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return [str(x) for x in obj]

    if isinstance(obj, dict):
        if "id2label" in obj and isinstance(obj["id2label"], dict):
            items = obj["id2label"]
        else:
            items = obj

        try:
            return [str(items[str(i)]) for i in range(len(items))]
        except Exception:
            try:
                return [str(v) for _, v in sorted(items.items(), key=lambda kv: int(kv[0]))]
            except Exception:
                return [str(v) for v in items.values()]

    raise ValueError(f"Unsupported label vocab format in {path}")


def _save_json_list(path: str | Path, values: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(list(values), f, indent=2)


def _get_label_field_and_converter(dataset) -> Tuple[str, Optional[Any]]:
    """Return (label_field_name, int_to_str_converter)."""
    candidates = ("ebird_code_multilabel", "ebird_code", "label", "labels")
    for key in candidates:
        try:
            feat = dataset.features[key]
        except Exception:
            continue

        inner = getattr(feat, "feature", None)
        if inner is not None and hasattr(inner, "int2str"):
            return key, inner.int2str

        if hasattr(feat, "int2str"):
            return key, feat.int2str

    for key in candidates:
        if key in getattr(dataset, "features", {}):
            return key, None

    raise KeyError(
        "Could not find a label field among: ebird_code_multilabel, ebird_code, label, labels"
    )


def _as_list(raw_value: Any) -> List[Any]:
    if raw_value is None:
        return []
    if isinstance(raw_value, (str, int, np.integer)):
        return [raw_value]
    if isinstance(raw_value, (list, tuple, np.ndarray)):
        return list(raw_value)
    try:
        return list(raw_value)
    except Exception:
        return [raw_value]


def _raw_labels_to_codes(raw_value: Any, int2str_converter: Optional[Any]) -> List[str]:
    out: List[str] = []
    for item in _as_list(raw_value):
        if item is None:
            continue
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, (int, np.integer)):
            if int2str_converter is not None:
                out.append(str(int2str_converter(int(item))))
            else:
                out.append(str(int(item)))
        else:
            try:
                if np.issubdtype(type(item), np.integer):
                    if int2str_converter is not None:
                        out.append(str(int2str_converter(int(item))))
                    else:
                        out.append(str(int(item)))
                else:
                    out.append(str(item))
            except Exception:
                out.append(str(item))
    return out


def _build_vocab_from_dataset(dataset, label_field_name: str, int2str_converter: Optional[Any]) -> List[str]:
    feat = dataset.features[label_field_name]
    candidates = [feat, getattr(feat, "feature", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        names = getattr(candidate, "names", None)
        if names:
            return [str(x) for x in names]
        num_classes = getattr(candidate, "num_classes", None)
        if num_classes is not None and int2str_converter is not None:
            try:
                return [str(int2str_converter(i)) for i in range(int(num_classes))]
            except Exception:
                pass

    raw_items = set()
    saw_any_string = False
    saw_any_int = False
    for sample in dataset:
        for item in _as_list(sample.get(label_field_name)):
            if item is None:
                continue
            if isinstance(item, (int, np.integer)):
                raw_items.add(int(item))
                saw_any_int = True
            elif isinstance(item, str):
                raw_items.add(item)
                saw_any_string = True
            else:
                try:
                    if np.issubdtype(type(item), np.integer):
                        raw_items.add(int(item))
                        saw_any_int = True
                    else:
                        raw_items.add(str(item))
                        saw_any_string = True
                except Exception:
                    raw_items.add(str(item))
                    saw_any_string = True

    if len(raw_items) == 0:
        raise ValueError("No labels found while building the label vocabulary")

    if saw_any_int and not saw_any_string and int2str_converter is not None:
        sorted_ids = sorted(int(x) for x in raw_items)
        return [str(int2str_converter(i)) for i in sorted_ids]

    return sorted(str(x) for x in raw_items)


def _crop_or_pad_1d(audio: np.ndarray, window_samples: int, training: bool) -> Tuple[np.ndarray, int]:
    """Return a fixed-length 1D waveform and the effective pre-padding length.
    
    For shorter waveforms, uses looping (repetition) instead of zero-padding
    to preserve acoustic continuity.
    """
    audio = np.asarray(audio, dtype=np.float32)
    length = int(audio.shape[0])

    if length <= 0:
        return np.zeros(window_samples, dtype=np.float32), 0

    if length == window_samples:
        return audio, window_samples

    if length < window_samples:
        # Loop the audio to reach target length
        # Calculate how many times we need to repeat
        repeats = (window_samples + length - 1) // length
        # Tile the audio
        out = np.tile(audio, repeats)[:window_samples]
        return out, length

    # For longer samples: crop
    if training:
        start = np.random.randint(0, length - window_samples + 1)
    else:
        start = max(0, (length - window_samples) // 2)

    return audio[start:start + window_samples], window_samples


# =========================================================
# Dataset
# =========================================================
class BirdSetDataset(Dataset):
    """BirdSet dataset that returns fixed-length windows on CPU.

    The model/frontend should no longer do waveform crop/pad.
    It can assume waveforms arrive already windowed to `window_duration`.
    """

    def __init__(
        self,
        dataset_path: str,
        subset: str = "XCL",
        split: str = "train",
        sample_rate: int = 32000,
        window_duration: float = 5.0,
        use_mixup: bool = True,
        mixup_prob: float = 1.0,
        mix_alpha: float = 91.3,
        mix_beta: float = 100.0,
        mix_omega: float = 1.0,
        mix_n: int = 2,
        use_geo_mixup: bool = True,
        geo_k: int = 100,
        label_vocab_path: Optional[str] = None,
        save_label_vocab_path: Optional[str] = None,
    ):
        self.dataset_path = dataset_path
        self.subset = subset
        self.split = split
        self.sample_rate = sample_rate
        self.window_samples = int(window_duration * sample_rate)
        self.use_mixup = use_mixup
        self.mixup_prob = mixup_prob
        self.mix_alpha = mix_alpha
        self.mix_beta = mix_beta
        self.mix_omega = mix_omega
        print(self.mix_omega)
        self.mix_n = mix_n
        self.use_geo_mixup = use_geo_mixup
        self.geo_k = geo_k
        self.training = split == "train"

        ds = load_dataset(
            "DBD-research-group/BirdSet",
            name=subset,
            cache_dir=dataset_path,
            trust_remote_code=True,
        )
        if split not in ds:
            available = list(ds.keys())
            raise KeyError(
                f"Split '{split}' not found for subset '{subset}'. Available splits: {available}"
            )

        self.dataset = ds[split]
        print(f"[INFO] Loaded subset={subset} split={split} with {len(self.dataset)} samples")

        self.label_field_name, self.label_int2str = _get_label_field_and_converter(self.dataset)
        print(f"[INFO] Label field: {self.label_field_name}")

        if label_vocab_path is not None:
            self.label_list = _load_json_list(label_vocab_path)
            print(f"[INFO] Loaded label vocabulary from {label_vocab_path}")
        else:
            self.label_list = _build_vocab_from_dataset(
                self.dataset,
                self.label_field_name,
                self.label_int2str,
            )
            print("[INFO] Built label vocabulary from XCL/train sample scan")
            if save_label_vocab_path is not None:
                _save_json_list(save_label_vocab_path, self.label_list)
                print(f"[INFO] Saved label vocabulary to {save_label_vocab_path}")

        self.label_to_idx = {label: i for i, label in enumerate(self.label_list)}
        self.num_classes = len(self.label_list)
        print(f"[INFO] Using {self.num_classes} classes")

        self.resamplers: Dict[int, torchaudio.transforms.Resample] = {}

        if self.use_geo_mixup:
            self._build_kdtree()
        else:
            self.kdtree = None
            self.kdtree_indices = []

    # -------------------------------------------------
    # GEO
    # -------------------------------------------------
    @staticmethod
    def _latlon_to_cartesian(lat, lon):
        lat = np.radians(lat)
        lon = np.radians(lon)
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)
        return [x, y, z]

    def _build_kdtree(self):
        coords = []
        valid_indices = []

        for i, sample in enumerate(self.dataset):
            lat = sample.get("lat")
            lon = sample.get("long")
            if _is_finite_number(lat) and _is_finite_number(lon):
                coords.append(self._latlon_to_cartesian(lat, lon))
                valid_indices.append(i)

        if len(coords) == 0:
            self.kdtree = None
            self.kdtree_indices = []
            print("[WARN] KDTree not built (no valid coordinates)")
            return

        self.kdtree = KDTree(np.asarray(coords))
        self.kdtree_indices = valid_indices
        self.idx_to_kdtree_pos = {idx: pos for pos, idx in enumerate(self.kdtree_indices)}
        print(f"[INFO] KDTree built with {len(coords)} points")

    def _get_geo_neighbor(self, idx, exclude_labels=None):
        """
        Get a geographically close sample, optionally excluding samples with specific labels.
        
        Args:
            idx: Index of the anchor sample
            exclude_labels: Optional set of label indices to exclude (e.g., species already in mix)
        """
        if self.kdtree is None:
            return None
        
        sample = self.dataset[idx]
        lat = sample.get("lat")
        lon = sample.get("long")
        
        if not (_is_finite_number(lat) and _is_finite_number(lon)):
            return None
        
        query = self._latlon_to_cartesian(lat, lon)
        if not np.all(np.isfinite(query)):
            return None
        
        # Get k nearest neighbors (using geo_k, could be 100 as you said)
        distances, nn = self.kdtree.query(query, k=self.geo_k)
        nn = np.atleast_1d(nn)
        
        # Remove the anchor point itself
        anchor_idx = self.idx_to_kdtree_pos.get(idx, None)
        if anchor_idx is not None:
            nn = nn[nn != anchor_idx]
        
        if len(nn) == 0:
            return None
        
        # Filter by label exclusion if specified
        if exclude_labels is not None and len(exclude_labels) > 0:
            filtered_neighbors = []
            for neighbor_kdtree_pos in nn:
                neighbor_idx = self.kdtree_indices[neighbor_kdtree_pos]
                neighbor_sample = self.dataset[neighbor_idx]
                
                # Get neighbor's labels
                neighbor_labels = self._get_sample_labels(neighbor_sample)  # Returns set of label indices
                
                # Check if neighbor has any label we want to exclude
                if not any(label in exclude_labels for label in neighbor_labels):
                    filtered_neighbors.append(neighbor_kdtree_pos)
            
            # If we found good neighbors, use them
            if filtered_neighbors:
                return self.kdtree_indices[np.random.choice(filtered_neighbors)]
            
            # Fallback: No neighbor without excluded labels, try random sampling
            # Option C: Return None to trigger random sampling in caller
            return None  # Let caller fall back to uniform random
        
        # No exclusion requested, return random from all neighbors
        return self.kdtree_indices[np.random.choice(nn)]
    # -------------------------------------------------
    # AUDIO
    # -------------------------------------------------
    def _get_resampler(self, src_sr: int):
        src_sr = int(src_sr)
        if src_sr not in self.resamplers:
            self.resamplers[src_sr] = torchaudio.transforms.Resample(src_sr, self.sample_rate)
        return self.resamplers[src_sr]

    def _load_audio(self, sample) -> Tuple[np.ndarray, int]:
        path = sample["audio"]["path"]
    
        if "downloads/" in path:
            rel = path.split("downloads/", 1)[1]
            path = os.path.join(self.dataset_path, "downloads", rel)
        else:
            path = os.path.join(self.dataset_path, path)
    
        # Fast path: ask torchaudio for metadata, then decode only the selected window.
        try:
            info = torchaudio.info(path)
            src_sr = int(info.sample_rate)
            total_frames = int(info.num_frames)
    
            if total_frames <= 0:
                raise RuntimeError("audio duration unavailable")
    
            # Convert the desired 5s window into source-sample frames.
            target_frames = int(round(self.window_samples * (src_sr / self.sample_rate)))
            target_frames = max(1, target_frames)
    
            if total_frames > target_frames:
                if self.training:
                    start = np.random.randint(0, total_frames - target_frames + 1)
                else:
                    start = max(0, (total_frames - target_frames) // 2)
            else:
                start = 0
                target_frames = total_frames
    
            waveform, read_sr = torchaudio.load(
                path,
                frame_offset=start,
                num_frames=target_frames,
            )
    
            # mono
            if waveform.ndim == 2 and waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.squeeze(0)
    
            audio = waveform.cpu().numpy().astype(np.float32)
    
            # If the source rate differs, resample after slicing.
            if int(read_sr) != self.sample_rate:
                audio_t = torch.from_numpy(audio).float()
                resampler = self._get_resampler(int(read_sr))
                audio = resampler(audio_t).numpy().astype(np.float32)
    
            # Safety pad/crop to exact window length if resampling changed length slightly.
            if audio.shape[0] != self.window_samples:
                audio, effective_length = _crop_or_pad_1d(
                    audio,
                    window_samples=self.window_samples,
                    training=self.training,
                )
            else:
                effective_length = self.window_samples
    
            return audio, effective_length
    
        except Exception:
            # Fallback: old full-file decode path.
            print("Error decoding only target frames, falling back to decoding full file")
            try:
                audio, sr = sf.read(path)
            except Exception:
                import librosa
                audio, sr = librosa.load(path, sr=None)
    
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
    
            audio = np.asarray(audio, dtype=np.float32)
            if sr != self.sample_rate:
                audio_t = torch.from_numpy(audio).float()
                resampler = self._get_resampler(sr)
                audio = resampler(audio_t).numpy().astype(np.float32)
    
            audio, effective_length = _crop_or_pad_1d(
                audio,
                window_samples=self.window_samples,
                training=self.training,
            )
            return audio, effective_length

    def _labels_to_vec(self, raw_labels: Any) -> np.ndarray:
        label_vec = np.zeros(self.num_classes, dtype=np.float32)
        for code in _raw_labels_to_codes(raw_labels, self.label_int2str):
            if code in self.label_to_idx:
                label_vec[self.label_to_idx[code]] = 1.0
            else:
                print("Warning: code " + str(code) +  " doesn't have a valid mapping in XCL")
        return label_vec
        
    def _get_sample_labels(self, sample) -> set:
        """Extract label indices as a set for quick membership testing."""
        raw_labels = sample.get(self.label_field_name, [])
        if isinstance(raw_labels, str):
            raw_labels = [raw_labels]
        
        label_set = set()
        for code in _raw_labels_to_codes(raw_labels, self.label_int2str):
            if code in self.label_to_idx:
                label_set.add(self.label_to_idx[code])
            else:
                print("Label not found in indices!")
        return label_set
    def get_active_class_indices(self) -> List[int]:
        """Efficiently get active classes using column-wise access."""
        # Get the entire label column as a list
        label_column = self.dataset[self.label_field_name]
        
        active_classes = set()
        
        # Process labels in batches for memory efficiency
        for raw_labels in label_column:
            for code in _raw_labels_to_codes(raw_labels, self.label_int2str):
                if code in self.label_to_idx:
                    active_classes.add(self.label_to_idx[code])
        
        return sorted(list(active_classes))

    # -------------------------------------------------
    # MIXUP RECIPE
    # -------------------------------------------------
    def _sample_mix_recipe(self, idx: int) -> Tuple[List[int], np.ndarray]:
        if (not self.use_mixup) or (np.random.rand() >= self.mixup_prob):
            return [idx], np.array([1.0], dtype=np.float32)
        
        # Sample number of constituents
        p = np.random.beta(self.mix_alpha, self.mix_beta)
        k = np.random.binomial(self.mix_n, p)
        n_constituents = max(1, int(k) + 1)
        
        # Keep track of labels we've already included
        anchor_sample = self.dataset[idx]
        included_labels = self._get_sample_labels(anchor_sample)
        
        indices = [idx]
        for _ in range(n_constituents - 1):
            if self.use_geo_mixup:
                # Pass current label set to exclude
                j = self._get_geo_neighbor(idx, exclude_labels=included_labels)
                if j is None:
                    # Fallback to uniform random
                    #print("No valid samples for Geo Mixup!", flush = True)
                    return [idx], np.array([1.0], dtype=np.float32)
            else:
                j = np.random.randint(len(self.dataset))
            
            # Update included labels with this new sample
            new_labels = self._get_sample_labels(self.dataset[j])
            included_labels.update(new_labels)
            indices.append(int(j))
        
        weights = np.random.dirichlet([self.mix_omega] * n_constituents).astype(np.float32)
        return indices, weights

    # -------------------------------------------------
    def __len__(self):
        return len(self.dataset)

    # -------------------------------------------------
    def __getitem__(self, idx):
        primary = self.dataset[idx]
        mix_indices, mix_weights = self._sample_mix_recipe(idx)

        waveforms = []
        waveform_lengths = []
        constituent_labels = []

        for mix_idx in mix_indices:
            sample = self.dataset[mix_idx]
            audio, effective_length = self._load_audio(sample)

            waveforms.append(torch.tensor(audio, dtype=torch.float32))
            waveform_lengths.append(int(effective_length))
            constituent_labels.append(
                torch.tensor(self._labels_to_vec(sample.get(self.label_field_name)), dtype=torch.float32)
            )

        primary_labels = torch.tensor(self._labels_to_vec(primary.get(self.label_field_name)), dtype=torch.float32)

        return {
            "waveforms": waveforms,  # each tensor is already [window_samples]
            "waveform_lengths": waveform_lengths,
            "constituent_labels": constituent_labels,
            "labels": primary_labels,
            "mix_weights": torch.tensor(mix_weights, dtype=torch.float32),
            "mix_count": int(len(mix_indices)),
            "sample_rate": int(self.sample_rate),
            "mix_indices": mix_indices,
            "coordinates": torch.tensor(
                [primary.get("long", float("nan")), primary.get("lat", float("nan"))],
                dtype=torch.float32,
            ),
        }


# =========================================================
# COLLATE
# =========================================================
def collate_fn(batch):
    batch_size = len(batch)
    max_m = max(len(item["waveforms"]) for item in batch)
    fixed_t = batch[0]["waveforms"][0].shape[0]
    num_classes = batch[0]["labels"].shape[0]

    waveforms = torch.zeros(batch_size, max_m, fixed_t, dtype=torch.float32)
    waveform_lengths = torch.zeros(batch_size, max_m, dtype=torch.long)
    constituent_labels = torch.zeros(batch_size, max_m, num_classes, dtype=torch.float32)
    mix_weights = torch.zeros(batch_size, max_m, dtype=torch.float32)
    mix_counts = torch.zeros(batch_size, dtype=torch.long)
    sample_rates = torch.zeros(batch_size, dtype=torch.long)
    coordinates = torch.stack([item["coordinates"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    mix_indices = [item["mix_indices"] for item in batch]

    for b, item in enumerate(batch):
        m = len(item["waveforms"])
        mix_counts[b] = m
        sample_rates[b] = int(item["sample_rate"])
        mix_weights[b, :m] = item["mix_weights"]

        for i in range(m):
            w = item["waveforms"][i]
            if w.shape[0] != fixed_t:
                raise ValueError(
                    f"Expected fixed-length windows of {fixed_t}, got {w.shape[0]}. "
                    "Check that windowing is happening in __getitem__."
                )
            waveforms[b, i] = w
            waveform_lengths[b, i] = int(item["waveform_lengths"][i])
            constituent_labels[b, i] = item["constituent_labels"][i]

    return {
        "waveforms": waveforms,
        "waveform_lengths": waveform_lengths,
        "constituent_labels": constituent_labels,
        "labels": labels,
        "mix_weights": mix_weights,
        "mix_counts": mix_counts,
        "sample_rates": sample_rates,
        "coordinates": coordinates,
        "mix_indices": mix_indices,
    }


# =========================================================
# DATALOADER
# =========================================================
class BirdSetDataLoader:
    def __init__(
        self,
        dataset_path: str,
        subset: str = "XCL",
        split: str = "train",
        batch_size: int = 32,
        num_workers: int = 8,
        shuffle: Optional[bool] = None,
        **kwargs,
    ):
        if shuffle is None:
            shuffle = split == "train"

        self.dataset = BirdSetDataset(
            dataset_path=dataset_path,
            subset=subset,
            split=split,
            **kwargs,
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.subset = subset
        self.split = split

    def get_loader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )