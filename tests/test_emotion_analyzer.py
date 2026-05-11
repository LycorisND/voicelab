# tests/test_emotion_analyzer.py
import numpy as np
import pytest
from unittest.mock import MagicMock
from voicelab.core.model_registry import ModelRegistry
from voicelab.schema import (
    EmotionConfig, AnalysisResult, AudioMetadata,
    PitchFeatures, SpectralFeatures, ProsodyFeatures, SpeakerProfile, HealthIndicators,
)
import voicelab.neural.embeddings   # noqa: F401
import voicelab.neural.speaker_profile  # noqa: F401
import voicelab.neural.health       # noqa: F401
import voicelab.emotion.backbone    # noqa: F401
import voicelab.emotion.fusion      # noqa: F401


@pytest.fixture(autouse=True)
def clean_registry():
    ModelRegistry.instance().clear()
    yield
    ModelRegistry.instance().clear()


def _register_module1_mocks():
    ModelRegistry.instance().register("ecapa", lambda: _mock_ecapa())
    ModelRegistry.instance().register("gender_age", lambda: (lambda *a: ("M", "25-35")))
    ModelRegistry.instance().register("lang_id", lambda: (lambda audio, sr: "en"))
    ModelRegistry.instance().register("health", lambda: (lambda *a: {"dysphonia": 0.1, "fatigue": 0.1, "hoarseness": 0.1}))


def _register_emotion_mocks():
    n = 7
    logits = np.zeros(n); logits[3] = 2.0
    va = np.array([0.5, 0.4])
    ModelRegistry.instance().register(
        "ser_backbone",
        lambda: (lambda audio, sr: (0.4, 0.6)),
    )
    ModelRegistry.instance().register(
        "ser_fusion",
        lambda: (lambda feat: (logits.copy(), va.copy())),
    )


def _mock_ecapa():
    m = MagicMock()
    m.encode_batch.return_value = (np.ones((1, 192), dtype=np.float32) * 0.1,)
    return m


def test_analyze_backbone_returns_emotion_result(sine_440_wav):
    _register_module1_mocks()
    _register_emotion_mocks()
    from voicelab.emotion.analyzer import EmotionAnalyzer
    path, _, _ = sine_440_wav
    ea = EmotionAnalyzer(EmotionConfig(fast=False))
    result = ea.analyze(path)
    from voicelab.schema import EmotionResult
    assert isinstance(result, EmotionResult)
    assert result.path == "backbone"


def test_analyze_fusion_returns_emotion_result(sine_440_wav):
    _register_module1_mocks()
    _register_emotion_mocks()
    from voicelab.emotion.analyzer import EmotionAnalyzer
    path, _, _ = sine_440_wav
    ea = EmotionAnalyzer(EmotionConfig(fast=True))
    result = ea.analyze(path)
    assert result.path == "fusion"


def test_analyze_result_uses_fusion(sine_440_wav):
    _register_module1_mocks()
    _register_emotion_mocks()
    from voicelab.emotion.analyzer import EmotionAnalyzer
    from voicelab.analysis.voice_analyzer import VoiceAnalyzer
    from voicelab.schema import Config
    path, _, _ = sine_440_wav
    analysis = VoiceAnalyzer(Config(neural=True)).analyze(path)
    ea = EmotionAnalyzer()
    result = ea.analyze_result(analysis)
    assert result.path == "fusion"


def test_analyze_result_scores_sum_to_one(sine_440_wav):
    _register_module1_mocks()
    _register_emotion_mocks()
    from voicelab.emotion.analyzer import EmotionAnalyzer
    from voicelab.analysis.voice_analyzer import VoiceAnalyzer
    from voicelab.schema import Config
    path, _, _ = sine_440_wav
    analysis = VoiceAnalyzer(Config(neural=True)).analyze(path)
    result = EmotionAnalyzer().analyze_result(analysis)
    assert abs(sum(result.scores.values()) - 1.0) < 1e-5
