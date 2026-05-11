# tests/test_speaker_profile.py
import numpy as np
import pytest
from voicelab.neural.speaker_profile import get_speaker_profile
from voicelab.core.model_registry import ModelRegistry


@pytest.fixture(autouse=True)
def clean_registry():
    ModelRegistry.instance().clear()
    yield
    ModelRegistry.instance().clear()


def _mock_gender_age():
    """Returns a callable that returns ("M", "25-35")."""
    return lambda audio, sr: ("M", "25-35")


def _mock_lang_id():
    return lambda audio, sr: "en"


def test_speaker_profile_gender_valid():
    ModelRegistry.instance().register("gender_age", _mock_gender_age)
    ModelRegistry.instance().register("lang_id", _mock_lang_id)
    audio = np.zeros(16000, dtype=np.float32)
    emb = np.ones(192, dtype=np.float32) * 0.1
    profile = get_speaker_profile(audio, emb, sr=16000)
    assert profile.gender in ("M", "F", "unknown")


def test_speaker_profile_age_format():
    ModelRegistry.instance().register("gender_age", _mock_gender_age)
    ModelRegistry.instance().register("lang_id", _mock_lang_id)
    audio = np.zeros(16000, dtype=np.float32)
    emb = np.ones(192, dtype=np.float32) * 0.1
    profile = get_speaker_profile(audio, emb, sr=16000)
    assert isinstance(profile.age_range, str)


def test_speaker_profile_embedding_stored():
    ModelRegistry.instance().register("gender_age", _mock_gender_age)
    ModelRegistry.instance().register("lang_id", _mock_lang_id)
    audio = np.zeros(16000, dtype=np.float32)
    emb = np.ones(192, dtype=np.float32) * 0.42
    profile = get_speaker_profile(audio, emb, sr=16000)
    np.testing.assert_array_equal(profile.embedding, emb)
