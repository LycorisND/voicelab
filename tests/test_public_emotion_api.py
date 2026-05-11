# tests/test_public_emotion_api.py
import numpy as np
import pytest
from voicelab.core.model_registry import ModelRegistry
import voicelab.neural.embeddings       # noqa: F401
import voicelab.neural.speaker_profile  # noqa: F401
import voicelab.neural.health           # noqa: F401
import voicelab.emotion.backbone        # noqa: F401
import voicelab.emotion.fusion          # noqa: F401
from unittest.mock import MagicMock


@pytest.fixture(autouse=True)
def clean_registry():
    ModelRegistry.instance().clear()
    yield
    ModelRegistry.instance().clear()


def _register_all_mocks():
    m = MagicMock()
    m.encode_batch.return_value = (np.ones((1, 192), dtype=np.float32) * 0.1,)
    ModelRegistry.instance().register("ecapa", lambda: m)
    ModelRegistry.instance().register("gender_age", lambda: (lambda *a: ("M", "25-35")))
    ModelRegistry.instance().register("lang_id", lambda: (lambda audio, sr: "en"))
    ModelRegistry.instance().register("health", lambda: (lambda *a: {"dysphonia": 0.1, "fatigue": 0.1, "hoarseness": 0.1}))
    n = 7
    logits = np.zeros(n); logits[3] = 2.0
    va = np.array([0.4, 0.5])
    ModelRegistry.instance().register(
        "ser_backbone",
        lambda: (lambda audio, sr: (0.4, 0.6)),
    )
    ModelRegistry.instance().register(
        "ser_fusion",
        lambda: (lambda feat: (logits.copy(), va.copy())),
    )
    ModelRegistry.instance().register(
        "ser_fusion_lite",
        lambda: (lambda feat: (logits.copy(), va.copy())),
    )


def test_analyze_emotion_in_public_api():
    import voicelab as vl
    assert hasattr(vl, "analyze_emotion")


def test_emotion_stream_in_public_api():
    import voicelab as vl
    assert hasattr(vl, "emotion_stream")


def test_emotion_config_importable():
    import voicelab as vl
    cfg = vl.EmotionConfig()
    assert cfg.fast is False


def test_analyze_emotion_backbone(sine_440_wav):
    _register_all_mocks()
    import voicelab as vl
    path, _, _ = sine_440_wav
    result = vl.analyze_emotion(path)
    assert isinstance(result, vl.EmotionResult)
    assert result.path == "backbone"
    assert abs(sum(result.scores.values()) - 1.0) < 1e-5


def test_analyze_emotion_fast(sine_440_wav):
    _register_all_mocks()
    import voicelab as vl
    path, _, _ = sine_440_wav
    result = vl.analyze_emotion(path, vl.EmotionConfig(fast=True))
    assert result.path == "fusion"


def test_emotion_stream_is_context_manager():
    import voicelab as vl
    es = vl.emotion_stream()
    assert hasattr(es, "__enter__")
    assert hasattr(es, "__exit__")
