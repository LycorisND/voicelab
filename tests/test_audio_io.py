# tests/test_audio_io.py
import numpy as np
import pytest
from voicelab.core.audio_io import load_audio, detect_clipping, estimate_snr

TARGET_SR = 16000

def test_load_audio_mono(sine_440_wav):
    path, expected_audio, sr = sine_440_wav
    audio, out_sr = load_audio(path)
    assert out_sr == TARGET_SR
    assert audio.ndim == 1
    assert len(audio) == pytest.approx(TARGET_SR * 2, abs=TARGET_SR * 0.01)

def test_load_audio_resamples_stereo(stereo_wav):
    path = stereo_wav
    audio, sr = load_audio(path)
    assert sr == TARGET_SR
    assert audio.ndim == 1

def test_detect_clipping_clean(sine_440_wav):
    _, audio, _ = sine_440_wav
    assert detect_clipping(audio) is False

def test_detect_clipping_saturated():
    audio = np.ones(16000, dtype=np.float32)  # fully clipped
    assert detect_clipping(audio) is True

def test_estimate_snr_sine(sine_440_wav):
    _, audio, sr = sine_440_wav
    snr = estimate_snr(audio, sr)
    assert snr > 20.0  # clean sine wave → high SNR

def test_estimate_snr_silence(silence_wav):
    _, audio, sr = silence_wav
    snr = estimate_snr(audio, sr)
    assert snr == pytest.approx(60.0, abs=5.0)
