from __future__ import annotations
from functools import lru_cache
import numpy as np
import soundfile as sf

TARGET_SR = 16000


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load audio file → mono float32 numpy array at 16 kHz."""
    import torch

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    # audio shape: (samples, channels) → convert to mono
    audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
    if sr != TARGET_SR:
        import torchaudio.transforms as T
        waveform = torch.from_numpy(audio).unsqueeze(0)
        waveform = _get_resampler(sr)(waveform)
        audio = waveform.squeeze(0).numpy()
    return audio.astype(np.float32), TARGET_SR


@lru_cache(maxsize=8)
def _get_resampler(orig_sr: int):
    import torchaudio.transforms as T
    return T.Resample(orig_sr, TARGET_SR)


def detect_clipping(audio: np.ndarray, threshold: float = 0.99) -> bool:
    return bool(np.any(np.abs(audio) >= threshold))


def estimate_snr(audio: np.ndarray, sr: int, frame_length: int = 2048) -> float:
    """Estimate spectral signal-to-noise ratio (tonal SNR).

    Measures how much energy is concentrated in dominant frequency bins versus
    the broadband noise floor. High values indicate clean/tonal signals; low
    values indicate broadband noise. Returns 60.0 (clamped max) for silence.
    """
    if np.max(np.abs(audio)) < 1e-10:
        return 60.0
    fft = np.fft.rfft(audio)
    mag_sq = np.abs(fft) ** 2
    total_power = np.sum(mag_sq)
    if total_power < 1e-10:
        return 60.0
    # Sum power in a narrow band around the peak (±3 bins)
    peak_idx = int(np.argmax(mag_sq))
    lo, hi = max(0, peak_idx - 3), min(len(mag_sq), peak_idx + 4)
    signal_power = np.sum(mag_sq[lo:hi])
    noise_power = max(total_power - signal_power, 1e-12)
    return float(np.clip(10.0 * np.log10(signal_power / noise_power), -20.0, 60.0))
