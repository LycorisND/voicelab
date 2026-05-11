# tests/test_spectral.py
import numpy as np
import pytest
from voicelab.dsp.spectral import extract_spectral_features

SR = 16000

def _make_sine(freq: float, duration: float = 1.0) -> np.ndarray:
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

def test_mfcc_shape():
    audio = _make_sine(440)
    result = extract_spectral_features(audio, SR, n_mfcc=13)
    assert result.mfcc.ndim == 2
    assert result.mfcc.shape[1] == 13
    assert result.mfcc.shape[0] > 0

def test_centroid_increases_with_frequency():
    low = _make_sine(200)
    high = _make_sine(3000)
    r_low = extract_spectral_features(low, SR)
    r_high = extract_spectral_features(high, SR)
    assert r_high.centroid_mean > r_low.centroid_mean

def test_rms_energy():
    audio = _make_sine(440)
    result = extract_spectral_features(audio, SR)
    assert 0.1 < result.rms_mean < 0.6

def test_zcr_noise_higher_than_sine():
    rng = np.random.default_rng(0)
    noise = (0.1 * rng.standard_normal(SR)).astype(np.float32)
    sine = _make_sine(440)
    r_noise = extract_spectral_features(noise, SR)
    r_sine = extract_spectral_features(sine, SR)
    assert r_noise.zcr_mean > r_sine.zcr_mean

def test_silence_rms_near_zero(silence_wav):
    _, audio, _ = silence_wav
    result = extract_spectral_features(audio, SR)
    assert result.rms_mean < 1e-4
