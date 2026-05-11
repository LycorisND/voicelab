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


def estimate_snr(audio: np.ndarray, sr: int, frame_ms: int = 20) -> float:
    """Estimate SNR using a hybrid method.

    Stationary/tonal signals (sine waves) → spectral peak-to-noise ratio.
    Speech-like signals (energy modulation across frames) → percentile method:
    bottom 20% frames = noise floor, top 80% = signal.

    The two paths are selected by the coefficient of variation of frame RMS:
    CV < 0.3 → stationary → spectral; CV >= 0.3 → speech → percentile.
    """
    if np.max(np.abs(audio)) < 1e-10:
        return 60.0
    frame_len = int(sr * frame_ms / 1000)
    n_frames = len(audio) // frame_len
    if n_frames < 5:
        return 60.0

    frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    rms_nonzero = rms[rms > 1e-10]
    if len(rms_nonzero) < 5:
        return 60.0

    cv = float(np.std(rms_nonzero) / (np.mean(rms_nonzero) + 1e-12))

    if cv < 0.3:
        # Stationary signal → spectral approach
        fft = np.fft.rfft(audio)
        mag_sq = np.abs(fft) ** 2
        total_power = np.sum(mag_sq)
        if total_power < 1e-10:
            return 60.0
        peak_idx = int(np.argmax(mag_sq))
        lo, hi = max(0, peak_idx - 3), min(len(mag_sq), peak_idx + 4)
        signal_power = np.sum(mag_sq[lo:hi])
        noise_power = max(total_power - signal_power, 1e-12)
        snr = 10.0 * np.log10(signal_power / noise_power)
        return float(np.clip(snr, 0.0, 60.0))
    else:
        # Speech-like signal → percentile approach
        rms_sorted = np.sort(rms_nonzero)
        noise_floor = np.mean(rms_sorted[: max(1, len(rms_sorted) // 5)])
        signal_level = np.mean(rms_sorted[len(rms_sorted) // 5 :])
        snr = 20.0 * np.log10(signal_level / (noise_floor + 1e-12))
        return float(np.clip(snr, 0.0, 60.0))
