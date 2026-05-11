# tests/test_emotion_backbone.py
import numpy as np
import pytest
from voicelab.core.model_registry import ModelRegistry
from voicelab.schema import EmotionConfig
import voicelab.emotion.backbone  # noqa: F401 — triggers module-level register()


@pytest.fixture(autouse=True)
def clean_registry():
    ModelRegistry.instance().clear()
    yield
    ModelRegistry.instance().clear()


def _register_mock_backbone():
    # Mock returns (valence, arousal) in [-1, 1]
    ModelRegistry.instance().register(
        "ser_backbone",
        lambda: (lambda audio, sr: (0.3, 0.7)),
    )


def test_backbone_returns_emotion_result():
    _register_mock_backbone()
    from voicelab.emotion.backbone import _run_backbone
    audio = np.zeros(16000, dtype=np.float32)
    cfg = EmotionConfig()
    result = _run_backbone(audio, 16000, cfg)
    assert result.emotion in cfg.emotion_labels


def test_backbone_scores_sum_to_one():
    _register_mock_backbone()
    from voicelab.emotion.backbone import _run_backbone
    audio = np.zeros(16000, dtype=np.float32)
    result = _run_backbone(audio, 16000, EmotionConfig())
    assert abs(sum(result.scores.values()) - 1.0) < 1e-5


def test_backbone_confidence_matches_emotion():
    _register_mock_backbone()
    from voicelab.emotion.backbone import _run_backbone
    audio = np.zeros(16000, dtype=np.float32)
    result = _run_backbone(audio, 16000, EmotionConfig())
    assert result.confidence == result.scores[result.emotion]


def test_backbone_path_label():
    _register_mock_backbone()
    from voicelab.emotion.backbone import _run_backbone
    audio = np.zeros(16000, dtype=np.float32)
    result = _run_backbone(audio, 16000, EmotionConfig())
    assert result.path == "backbone"


def test_backbone_va_in_range():
    _register_mock_backbone()
    from voicelab.emotion.backbone import _run_backbone
    audio = np.zeros(16000, dtype=np.float32)
    result = _run_backbone(audio, 16000, EmotionConfig())
    assert -1.0 <= result.valence <= 1.0
    assert -1.0 <= result.arousal <= 1.0


def test_backbone_dominant_emotions_subset_of_scores():
    _register_mock_backbone()
    from voicelab.emotion.backbone import _run_backbone
    audio = np.zeros(16000, dtype=np.float32)
    result = _run_backbone(audio, 16000, EmotionConfig())
    assert all(e in result.scores for e in result.dominant_emotions)


def test_build_emotion_result_directly():
    from voicelab.emotion.backbone import _build_emotion_result
    logits = np.array([2.0, 1.0, 0.5, 0.3, 0.1, 0.0, 0.0])
    va = np.array([0.3, 0.7])
    cfg = EmotionConfig()
    result = _build_emotion_result(logits, va, cfg, "backbone")
    assert result.emotion == cfg.emotion_labels[0]
    assert abs(sum(result.scores.values()) - 1.0) < 1e-5
