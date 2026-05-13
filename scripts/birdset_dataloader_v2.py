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


def _load_spec_norm(path: str | Path) -> Tuple[float, float]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict) or "mean" not in obj or "std" not in obj:
        raise ValueError(f"Spectrogram norm file must contain mean/std keys: {path}")

    return float(obj["mean"]), float(obj["std"])


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
    """Convert raw sample label values into eBird code strings."""
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


def _build_vocab_from_xcl_train(dataset, label_field_name: str, int2str_converter: Optional[Any]) -> List[str]:
    """Build the canonical XCL vocab in the old checkpoint-facing order."""
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
        raise ValueError("No labels found while building the XCL training vocabulary")

    if saw_any_int and not saw_any_string and int2str_converter is not None:
        sorted_ids = sorted(int(x) for x in raw_items)
        return [str(int2str_converter(i)) for i in sorted_ids]

    return sorted(str(x) for x in raw_items)


# =========================================================
# Spectrogram normalization module
# =========================================================
class SpectrogramNormalize(torch.nn.Module):
    def __init__(self, mean: float, std: float, eps: float = 1e-8):
        super().__init__()
        self.register_buffer("mean", torch.tensor(float(mean), dtype=torch.float32))
        self.register_buffer("std", torch.tensor(float(std), dtype=torch.float32))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)


# =========================================================
# Dataset
# =========================================================
class BirdSetDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        subset: str = "XCL",
        split: str = "train",
        sample_rate: int = 32000,
        window_duration: float = 5.0,
        n_mels: int = 128,
        n_time_frames: int = 512,
        n_fft: int = 1024,
        normalize_audio: bool = True,
        # Mixup
        use_mixup: bool = True,
        mixup_prob: float = 0.3,
        mix_alpha: float = 91.3,
        mix_beta: float = 100,
        mix_omega: float = 1.0,
        mix_n: int = 2,
        # Geo mixup
        use_geo_mixup: bool = True,
        geo_k: int = 100,
        # Canonical label mapping
        label_vocab_path: Optional[str] = None,
        save_label_vocab_path: Optional[str] = None,
        # Spectrogram normalization
        spec_norm_path: Optional[str] = None,
        apply_spec_norm: bool = True,
    ):
        self.dataset_path = dataset_path
        self.subset = subset
        self.split = split
        self.sample_rate = sample_rate
        self.window_samples = int(window_duration * sample_rate)
        self.n_mels = n_mels
        self.n_time_frames = n_time_frames
        self.n_fft = n_fft
        self.normalize_audio = normalize_audio

        self.use_mixup = use_mixup
        self.mixup_prob = mixup_prob
        self.mix_alpha = mix_alpha
        self.mix_beta = mix_beta
        self.mix_omega = mix_omega
        self.mix_n = mix_n

        self.use_geo_mixup = use_geo_mixup
        self.geo_k = geo_k

        self.label_vocab_path = label_vocab_path
        self.save_label_vocab_path = save_label_vocab_path

        self.apply_spec_norm = apply_spec_norm
        self.spec_norm: Optional[SpectrogramNormalize] = None

        if spec_norm_path is not None:
            mean, std = _load_spec_norm(spec_norm_path)
            self.spec_norm = SpectrogramNormalize(mean, std)
            print(f"[INFO] Loaded spectrogram normalization from {spec_norm_path}")
            print(f"[INFO] Spectrogram norm mean={mean:.6f}, std={std:.6f}")

        ds = load_dataset(
            "DBD-research-group/BirdSet",
            name=subset,
            cache_dir=dataset_path,
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
            if not (subset == "XCL" and split == "train"):
                raise ValueError(
                    "When label_vocab_path is not provided, the label vocab can only be built "
                    "from subset='XCL' and split='train'."
                )

            self.label_list = _build_vocab_from_xcl_train(
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

        hop_length = (self.window_samples - n_fft) // (n_time_frames - 1)
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=False,
        )

        # Cache resamplers by source sample rate to avoid repeatedly constructing them.
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

        coords = np.asarray(coords)
        self.kdtree = KDTree(coords)
        self.kdtree_indices = valid_indices
        print(f"[INFO] KDTree built with {len(coords)} points")

    def _get_geo_neighbor(self, idx):
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

        _, nn = self.kdtree.query(query, k=self.geo_k)
        nn = np.atleast_1d(nn)

        anchor_idx = self.kdtree_indices.index(idx) if idx in self.kdtree_indices else None
        if anchor_idx is not None:
            nn = nn[nn != anchor_idx]

        if len(nn) == 0:
            return None

        chosen = np.random.choice(nn)
        return self.kdtree_indices[chosen]

    # -------------------------------------------------
    # AUDIO
    # -------------------------------------------------
    def _get_resampler(self, src_sr: int):
        src_sr = int(src_sr)
        if src_sr not in self.resamplers:
            self.resamplers[src_sr] = torchaudio.transforms.Resample(src_sr, self.sample_rate)
        return self.resamplers[src_sr]

    def _load_audio(self, sample):
        path = sample["audio"]["path"]

        if "downloads/" in path:
            rel = path.split("downloads/", 1)[1]
            path = os.path.join(self.dataset_path, "downloads", rel)
        else:
            path = os.path.join(self.dataset_path, path)

        try:
            audio, sr = sf.read(path)
        except Exception:
            import librosa
            audio, sr = librosa.load(path, sr=None)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != self.sample_rate:
            audio = torch.tensor(audio).float()
            resampler = self._get_resampler(sr)
            audio = resampler(audio).numpy()

        return audio

    def _extract_window(self, audio):
        L = len(audio)
        if L < self.window_samples:
            return np.pad(audio, (0, self.window_samples - L))
        if L == self.window_samples:
            return audio
        start = np.random.randint(0, L - self.window_samples)
        return audio[start : start + self.window_samples]

    def _normalize(self, audio):
        if not self.normalize_audio:
            return audio
        max_val = np.max(np.abs(audio))
        if max_val == 0:
            return audio
        scale = 0.25 / max_val
        return audio * scale

    # -------------------------------------------------
    # PERCH MIXUP
    # -------------------------------------------------
    def _multi_mix_audio(self, idx):
        p = np.random.beta(self.mix_alpha, self.mix_beta)
        k = np.random.binomial(self.mix_n, p)
        N = k + 1

        indices = [idx]
        for _ in range(N - 1):
            if self.use_geo_mixup:
                j = self._get_geo_neighbor(idx)
                if j is None:
                    j = np.random.randint(len(self.dataset))
            else:
                j = np.random.randint(len(self.dataset))
            indices.append(j)

        audios = []
        labels = []
        for i in indices:
            sample_i = self.dataset[i]
            audio_i = self._load_audio(sample_i)
            audio_i = self._extract_window(audio_i)
            audios.append(audio_i)
            labels.append(_raw_labels_to_codes(sample_i.get(self.label_field_name), self.label_int2str))

        audios = np.stack(audios)
        weights = np.random.dirichlet([self.mix_omega] * N)

        mixed = np.sum(weights[:, None] * audios, axis=0)
        mixed = mixed / np.sqrt(np.sum(weights**2))

        label_vec = np.zeros(self.num_classes, dtype=np.float32)
        for lbls in labels:
            for code in lbls:
                if code in self.label_to_idx:
                    label_vec[self.label_to_idx[code]] = 1.0

        return mixed, label_vec

    # -------------------------------------------------
    # SPEC AUGMENT
    # -------------------------------------------------
    def _specaugment(self, mel):
        if np.random.rand() < 0.5:
            t = mel.shape[1]
            w = np.random.randint(0, int(0.2 * t))
            t0 = np.random.randint(0, max(1, t - w))
            mel[:, t0 : t0 + w] = 0

        if np.random.rand() < 0.5:
            f = mel.shape[0]
            h = np.random.randint(0, int(0.2 * f))
            f0 = np.random.randint(0, max(1, f - h))
            mel[f0 : f0 + h, :] = 0

        return mel

    # -------------------------------------------------
    def __len__(self):
        return len(self.dataset)

    # -------------------------------------------------
    def __getitem__(self, idx):
        sample = self.dataset[idx]

        if self.use_mixup and np.random.rand() < self.mixup_prob:
            audio, label_vec = self._multi_mix_audio(idx)
        else:
            audio = self._load_audio(sample)
            audio = self._extract_window(audio)

            label_vec = np.zeros(self.num_classes, dtype=np.float32)
            for code in _raw_labels_to_codes(sample.get(self.label_field_name), self.label_int2str):
                if code in self.label_to_idx:
                    label_vec[self.label_to_idx[code]] = 1.0

        if self.normalize_audio:
            audio = self._normalize(audio)

        mel = torch.from_numpy(audio).float().unsqueeze(0)
        mel = self.mel_spectrogram(mel).squeeze(0)
        # Keep true log scaling.
        mel = torch.clamp(mel, min=1e-10).log()

        if self.apply_spec_norm and self.spec_norm is not None:
            mel = self.spec_norm(mel)

        return {
            "mel_spectrogram": mel,
            "labels": torch.tensor(label_vec, dtype=torch.float32),
            "coordinates": torch.tensor(
                [sample.get("lat", 0.0) or 0.0, sample.get("long", 0.0) or 0.0],
                dtype=torch.float32,
            ),
        }


# =========================================================
# COLLATE
# =========================================================
def collate_fn(batch):
    return {
        "spectrograms": torch.stack([b["mel_spectrogram"] for b in batch]).unsqueeze(1),
        "labels": torch.stack([b["labels"] for b in batch]),
        "coordinates": torch.stack([b["coordinates"] for b in batch]),
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
