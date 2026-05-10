import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T

TARGET_SR = 16000


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load audio file → mono float32 numpy array at 16 kHz."""
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    # audio shape: (samples, channels) — convert to mono
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]
    # Resample if needed using torchaudio
    if sr != TARGET_SR:
        waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, samples)
        waveform = T.Resample(sr, TARGET_SR)(waveform)
        audio = waveform.squeeze(0).numpy()
    return audio.astype(np.float32), TARGET_SR


def detect_clipping(audio: np.ndarray, threshold: float = 0.99) -> bool:
    return bool(np.any(np.abs(audio) >= threshold))


def estimate_snr(audio: np.ndarray, sr: int) -> float:
    """Estimate SNR using spectral peak-to-noise ratio.

    Returns 60.0 (clamped max) for silence and pure tones.
    Returns low values for broadband noise.
    """
    if np.max(np.abs(audio)) < 1e-10:
        return 60.0

    fft = np.fft.rfft(audio)
    mag_sq = np.abs(fft) ** 2
    total_power = np.sum(mag_sq)
    if total_power < 1e-10:
        return 60.0

    peak_idx = int(np.argmax(mag_sq))
    width = 3
    lo = max(0, peak_idx - width)
    hi = min(len(mag_sq), peak_idx + width + 1)
    signal_power = np.sum(mag_sq[lo:hi])
    noise_power = max(total_power - signal_power, 1e-12)

    snr = 10.0 * np.log10(signal_power / noise_power)
    return float(np.clip(snr, -20.0, 60.0))
