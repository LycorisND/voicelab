# tests/test_formants.py
import numpy as np
import pytest
from voicelab.dsp.formants import extract_formants

SR = 16000

def _make_voiced_sine(freq: float, duration: float = 2.0) -> np.ndarray:
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

def test_formants_returns_four_values():
    audio = _make_voiced_sine(150.0)
    result = extract_formants(audio, SR)
    assert "F1" in result and "F2" in result
    assert "F3" in result and "F4" in result

def test_formant_values_nonnegative():
    audio = _make_voiced_sine(150.0)
    result = extract_formants(audio, SR)
    for key in ("F1", "F2", "F3", "F4"):
        assert result[key] >= 0.0

def test_silence_formants_zero(silence_wav):
    _, audio, _ = silence_wav
    result = extract_formants(audio, SR)
    assert result["F1"] < 200.0
