# tests/test_prosody.py
import numpy as np
import pytest
from voicelab.dsp.prosody import extract_prosody

SR = 16000

def _make_speech_like(duration: float = 3.0) -> np.ndarray:
    rng = np.random.default_rng(1)
    audio = np.zeros(int(SR * duration), dtype=np.float32)
    t = 0
    while t < int(SR * duration):
        seg_len = int(SR * 0.5)
        end = min(t + seg_len, int(SR * duration))
        audio[t:end] = 0.3 * rng.standard_normal(end - t).astype(np.float32)
        t = end + int(SR * 0.2)
    return audio

def test_energy_profile_shape():
    audio = _make_speech_like()
    result = extract_prosody(audio, SR)
    assert len(result.energy_profile) > 0

def test_pause_ratio_range():
    audio = _make_speech_like()
    result = extract_prosody(audio, SR)
    assert 0.0 <= result.pause_ratio <= 1.0

def test_silence_pause_ratio_high(silence_wav):
    _, audio, _ = silence_wav
    result = extract_prosody(audio, SR)
    assert result.pause_ratio > 0.8

def test_tempo_positive():
    audio = _make_speech_like()
    result = extract_prosody(audio, SR)
    assert result.tempo_bpm > 0.0
