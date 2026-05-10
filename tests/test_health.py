# tests/test_health.py
import numpy as np
import pytest
from unittest.mock import MagicMock
from voicelab.neural.health import get_health_indicators
from voicelab.core.model_registry import ModelRegistry
from voicelab.schema import PitchFeatures


@pytest.fixture(autouse=True)
def clean_registry():
    ModelRegistry.instance().clear()
    yield
    ModelRegistry.instance().clear()


def _mock_health_model():
    """Returns a callable that returns score dict."""
    return lambda audio, sr: {"dysphonia": 0.2, "fatigue": 0.1, "hoarseness": 0.15}


def _make_pitch(hnr: float, jitter: float) -> PitchFeatures:
    return PitchFeatures(
        f0_mean=200.0, f0_std=10.0, f0_min=180.0, f0_max=220.0,
        f0_contour=np.array([200.0]), jitter_local=jitter,
        shimmer_local=0.05, hnr=hnr, voiced_fraction=0.8,
    )


def test_scores_in_range():
    ModelRegistry.instance().register("health", _mock_health_model)
    audio = np.zeros(16000, dtype=np.float32)
    pitch = _make_pitch(hnr=15.0, jitter=0.01)
    result = get_health_indicators(audio, pitch, sr=16000)
    assert 0.0 <= result.dysphonia_score <= 1.0
    assert 0.0 <= result.fatigue_index <= 1.0
    assert 0.0 <= result.hoarseness <= 1.0


def test_pathology_flags_is_list():
    ModelRegistry.instance().register("health", _mock_health_model)
    audio = np.zeros(16000, dtype=np.float32)
    pitch = _make_pitch(hnr=15.0, jitter=0.01)
    result = get_health_indicators(audio, pitch, sr=16000)
    assert isinstance(result.pathology_flags, list)


def test_high_jitter_raises_dysphonia_flag():
    ModelRegistry.instance().register("health", _mock_health_model)
    audio = np.zeros(16000, dtype=np.float32)
    pitch = _make_pitch(hnr=2.0, jitter=0.05)  # high jitter, low HNR
    result = get_health_indicators(audio, pitch, sr=16000)
    assert "dysphonia" in result.pathology_flags
