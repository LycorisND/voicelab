# tests/conftest.py
import numpy as np
import soundfile as sf
import pytest

SR = 16000

@pytest.fixture
def sine_440_wav(tmp_path):
    """440 Hz sine, 2 seconds, 16 kHz mono."""
    t = np.linspace(0, 2.0, int(SR * 2.0), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = str(tmp_path / "sine_440.wav")
    sf.write(path, audio, SR)
    return path, audio, SR

@pytest.fixture
def sine_220_wav(tmp_path):
    """220 Hz sine, 2 seconds — used for pitch detection ground truth."""
    t = np.linspace(0, 2.0, int(SR * 2.0), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    path = str(tmp_path / "sine_220.wav")
    sf.write(path, audio, SR)
    return path, audio, SR

@pytest.fixture
def silence_wav(tmp_path):
    """2 seconds of silence."""
    audio = np.zeros(int(SR * 2.0), dtype=np.float32)
    path = str(tmp_path / "silence.wav")
    sf.write(path, audio, SR)
    return path, audio, SR

@pytest.fixture
def stereo_wav(tmp_path):
    """Stereo WAV at 44100 Hz — tests resampling and mono conversion."""
    sr_orig = 44100
    t = np.linspace(0, 1.0, sr_orig, endpoint=False)
    left = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    right = (0.4 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    audio = np.stack([left, right], axis=1)
    path = str(tmp_path / "stereo_44k.wav")
    sf.write(path, audio, sr_orig)
    return str(path)

@pytest.fixture
def white_noise_wav(tmp_path):
    """White noise — voice activity should be near zero."""
    rng = np.random.default_rng(42)
    audio = (0.1 * rng.standard_normal(SR * 2)).astype(np.float32)
    path = str(tmp_path / "noise.wav")
    sf.write(path, audio, SR)
    return path, audio, SR
