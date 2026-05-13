# fast_spec_stats.py

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


def _load_audio(path: str, dataset_path: str, target_sr: int) -> np.ndarray:
    if "downloads/" in path:
        rel = path.split("downloads/", 1)[1]
        path = str(Path(dataset_path) / "downloads" / rel)
    else:
        path = str(Path(dataset_path) / path)

    # soundfile is usually faster than librosa
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        audio_t = torch.from_numpy(audio).float()
        audio_t = torchaudio.transforms.Resample(sr, target_sr)(audio_t)
        audio = audio_t.numpy()

    return audio


def _normalize_audio_peak(audio: np.ndarray, target_peak: float = 0.25) -> np.ndarray:
    max_val = float(np.max(np.abs(audio)))
    if max_val == 0.0:
        return audio
    return audio * (target_peak / max_val)


def _center_crop_or_pad(audio: np.ndarray, window_samples: int) -> np.ndarray:
    L = len(audio)
    if L < window_samples:
        return np.pad(audio, (0, window_samples - L))
    if L == window_samples:
        return audio
    start = max(0, (L - window_samples) // 2)
    return audio[start : start + window_samples]


class BirdSetWaveformDataset(Dataset):
    """
    Returns fixed-length normalized waveforms only.
    We compute spectrogram stats in batched form outside the workers.
    """

    def __init__(
        self,
        dataset_path: str,
        subset: str = "XCL",
        split: str = "train",
        sample_rate: int = 32000,
        window_duration: float = 5.0,
        normalize_audio: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.dataset_path = dataset_path
        self.sample_rate = sample_rate
        self.window_samples = int(window_duration * sample_rate)
        self.normalize_audio = normalize_audio

        ds = load_dataset("DBD-research-group/BirdSet", name=subset, cache_dir=dataset_path)
        if split not in ds:
            raise KeyError(f"Split '{split}' not found. Available: {list(ds.keys())}")

        split_ds = ds[split]
        n = len(split_ds) if max_samples is None else min(len(split_ds), max_samples)

        # Pull just the audio paths once.
        self.paths = [split_ds[i]["audio"]["path"] for i in range(n)]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.paths[idx]
        audio = _load_audio(path, self.dataset_path, self.sample_rate)
        audio = _center_crop_or_pad(audio, self.window_samples)

        if self.normalize_audio:
            audio = _normalize_audio_peak(audio)

        return torch.from_numpy(audio).float()


def compute_spectrogram_stats_fast(
    dataset_path: str,
    subset: str = "XCL",
    split: str = "train",
    sample_rate: int = 32000,
    window_duration: float = 5.0,
    n_mels: int = 128,
    n_time_frames: int = 512,
    n_fft: int = 1024,
    normalize_audio: bool = True,
    batch_size: int = 64,
    num_workers: int = 8,
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """
    Fast CPU stats computation:
    - parallel audio decode/resample via DataLoader workers
    - batched spectrogram computation in main process
    - deterministic center crop, no mixup, no augmentation
    """

    # Reduce oversubscription when many workers are used.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)

    dataset = BirdSetWaveformDataset(
        dataset_path=dataset_path,
        subset=subset,
        split=split,
        sample_rate=sample_rate,
        window_duration=window_duration,
        normalize_audio=normalize_audio,
        max_samples=max_samples,
    )

    window_samples = int(window_duration * sample_rate)
    hop_length = (window_samples - n_fft) // (n_time_frames - 1)

    mel_fn = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        center=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=False,
    )

    total_sum = 0.0
    total_sumsq = 0.0
    total_count = 0

    with torch.no_grad():
        for i, wavs in enumerate(loader):
            # wavs: [B, T]
            mel = mel_fn(wavs)          # [B, n_mels, time]
            #mel = torch.log1p(mel)
            mel = torch.clamp(mel, min=1e-10).log()

            x = mel.reshape(-1).double()
            total_sum += x.sum().item()
            total_sumsq += (x * x).sum().item()
            total_count += x.numel()

            if (i + 1) % 100 == 0:
                seen = min((i + 1) * batch_size, len(dataset))
                print(f"Processed ~{seen}/{len(dataset)} samples...")

    mean = total_sum / total_count
    var = max(total_sumsq / total_count - mean * mean, 1e-12)
    std = float(np.sqrt(var))

    return {
        "mean": float(mean),
        "std": std,
        "count": int(total_count),
        "num_samples": int(len(dataset)),
        "subset": subset,
        "split": split,
        "sample_rate": sample_rate,
        "window_duration": window_duration,
        "n_mels": n_mels,
        "n_time_frames": n_time_frames,
        "n_fft": n_fft,
        "normalize_audio": normalize_audio,
    }


def save_stats(stats: Dict[str, float], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    stats = compute_spectrogram_stats_fast(
        dataset_path="/scratch/Projects/CFP-04/CFP04-CF-029/birdset",
        subset="XCL",
        split="train",
        sample_rate=32000,
        window_duration=5.0,
        n_mels=128,
        n_time_frames=512,
        n_fft=1024,
        normalize_audio=True,
        batch_size=64,
        num_workers=16,
    )
    print(stats)
    save_stats(stats, "xcl_spec_stats_true_log.json")