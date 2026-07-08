import numpy as np
import torch
import torchaudio
from transformers import AutoFeatureExtractor
import matplotlib.pyplot as plt

YOUR_PARAMS = {
    "sample_rate": 32000,
    "window_duration": 5.0,
    "n_mels": 128,
    "n_time_frames": 512,
    "n_fft": 1024,
    "normalize_audio": False,
    "apply_spec_norm": True,
    "preemphasis_coeff": 0.97,
}

window_samples = int(YOUR_PARAMS["window_duration"] * YOUR_PARAMS["sample_rate"])
hop_length = (window_samples - YOUR_PARAMS["n_fft"]) // (YOUR_PARAMS["n_time_frames"] - 1)

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=YOUR_PARAMS["sample_rate"],
    n_fft=YOUR_PARAMS["n_fft"],
    hop_length=hop_length,
    power=2.0,
    n_mels=YOUR_PARAMS["n_mels"],
    center=False,
)

preemphasis = torchaudio.transforms.Preemphasis(coeff=YOUR_PARAMS["preemphasis_coeff"])

def your_frontend_like_model(audio_np, sr):
    # audio_np: 1D numpy array
    audio = torch.tensor(audio_np, dtype=torch.float32)

    # resample if needed
    if sr != YOUR_PARAMS["sample_rate"]:
        audio = torchaudio.functional.resample(audio, sr, YOUR_PARAMS["sample_rate"])

    # match your model input shape: [B, M, T]
    waveforms = audio.unsqueeze(0).unsqueeze(0)

    # pre-emphasis
    waveforms = preemphasis(waveforms)

    # optional amplitude normalization
    if YOUR_PARAMS["normalize_audio"]:
        max_val = waveforms.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        waveforms = waveforms * (0.25 / max_val)

    # for this fair comparison, use one constituent with weight 1
    weights = torch.tensor([[1.0]], dtype=waveforms.dtype, device=waveforms.device)

    # mix to [B, T]
    mixed_waveforms = (waveforms * weights.unsqueeze(-1)).sum(dim=1)

    # gain normalization
    gain = 1.0 / torch.sqrt((weights ** 2).sum(dim=1, keepdim=True))
    mixed_waveforms = mixed_waveforms * gain

    # mel spectrogram
    spec = mel_spectrogram(mixed_waveforms)

    # log10 scaling
    spec = torch.log10(spec + 1e-10)

    # optional spec normalization: apply the same mean/std you loaded in your model
    if YOUR_PARAMS["apply_spec_norm"]:
        spec = (spec + 3.17) / 2.29

    return spec.unsqueeze(1).cpu().numpy()  # [B, 1, n_mels, time]

# --- 2. Load Official Bird-MAE Frontend ---
model_name = "DBD-research-group/Bird-MAE-Base"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)

# --- 3. Load and Process Your Audio Sample ---
audio_path = "/home/svu/e1583377/asian_fairy_bluebird_audio.mp3"  # <--- CHANGE THIS
waveform, sr = torchaudio.load(audio_path)

if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0)

if sr != 32000:
    waveform = torchaudio.functional.resample(
        waveform, sr, 32000
    )
    sr = 32000
audio = waveform[:32000 * 5]

# A. Process with your frontend
your_spec = your_frontend_like_model(audio, sr)

# B. Process with the official frontend
# The feature extractor expects a list of waveforms and returns a PyTorch tensor
inputs = feature_extractor([audio], return_tensors="pt")
print(f"Type: {type(inputs)}, Keys: {inputs.keys() if isinstance(inputs, dict) else 'not a dict'}")
official_spec = inputs[0, 0].cpu().numpy()
print(official_spec.shape)
official_spec = official_spec.T

# --- 4. Generate Comparison Report ---
print("="*50)
print("SPECTROGRAM COMPARISON REPORT")
print("="*50)
print(f"Your Frontend   - Shape: {your_spec.shape}, Mean: {your_spec.mean():.3f}, Std: {your_spec.std():.3f}")
print(f"Official Frontend - Shape: {official_spec.shape}, Mean: {official_spec.mean():.3f}, Std: {official_spec.std():.3f}")
print(f"\nDifference in shape: {your_spec.shape[0] - official_spec.shape[0]} mel bins, {your_spec.shape[1] - official_spec.shape[1]} time frames")
print("The Bird-MAE frontend uses a spectrogram size of 512x128 [citation:1].")

# --- 5. Create a Side-by-Side Visualization ---
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
im1 = axes[0].imshow(your_spec.squeeze(), aspect='auto', origin='lower')
axes[0].set_title("Your Frontend")
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(official_spec, aspect='auto', origin='lower')
axes[1].set_title("Official Bird-MAE Frontend")
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.savefig("frontend_comparison.png", dpi=150)
print("\n? Comparison plot saved as 'frontend_comparison.png'")