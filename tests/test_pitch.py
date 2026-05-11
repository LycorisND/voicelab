# tests/test_pitch.py
import numpy as np
import pytest
from voicelab.dsp.pitch import extract_pitch

SR = 16000

def _make_voiced(freq: float, duration: float = 2.0) -> np.ndarray:
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

def test_f0_mean_within_tolerance():
    audio = _make_voiced(220.0)
    result = extract_pitch(audio, SR)
    assert abs(result.f0_mean - 220.0) < 10.0

def test_voiced_fraction_sine():
    audio = _make_voiced(220.0)
    result = extract_pitch(audio, SR)
    assert result.voiced_fraction > 0.7

def test_silence_voiced_fraction_zero(silence_wav):
    _, audio, _ = silence_wav
    result = extract_pitch(audio, SR)
    assert result.voiced_fraction == pytest.approx(0.0, abs=0.05)

def test_f0_contour_is_array():
    audio = _make_voiced(300.0)
    result = extract_pitch(audio, SR)
    assert isinstance(result.f0_contour, np.ndarray)
    assert len(result.f0_contour) > 0

def test_jitter_shimmer_nonnegative():
    audio = _make_voiced(220.0)
    result = extract_pitch(audio, SR)
    assert result.jitter_local >= 0.0
    assert result.shimmer_local >= 0.0
