import numpy as np
import pytest
from voicelab.analysis.voice_stream import VoiceStream
from voicelab.schema import Config, FrameResult

SR = 16000


def _synthetic_source(audio: np.ndarray, chunk: int = 256):
    """Simulate microphone by yielding chunks from a numpy array."""
    for i in range(0, len(audio), chunk):
        yield audio[i : i + chunk].astype(np.float32)


def test_stream_yields_frame_results(sine_440_wav):
    _, audio, _ = sine_440_wav
    cfg = Config(neural=False)
    vs = VoiceStream(cfg, sr=SR)
    frames = list(vs._process_source(_synthetic_source(audio)))
    assert len(frames) > 0
    assert all(isinstance(f, FrameResult) for f in frames)


def test_stream_timestamps_increase(sine_440_wav):
    _, audio, _ = sine_440_wav
    cfg = Config(neural=False)
    vs = VoiceStream(cfg, sr=SR)
    frames = list(vs._process_source(_synthetic_source(audio)))
    timestamps = [f.timestamp for f in frames]
    assert all(timestamps[i] <= timestamps[i + 1] for i in range(len(timestamps) - 1))


def test_stream_energy_nonzero_for_sine(sine_440_wav):
    _, audio, _ = sine_440_wav
    cfg = Config(neural=False)
    vs = VoiceStream(cfg, sr=SR)
    frames = list(vs._process_source(_synthetic_source(audio)))
    energies = [f.energy for f in frames]
    assert max(energies) > 0.01
