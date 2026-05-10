import numpy as np
import pytest
from unittest.mock import MagicMock
from voicelab.analysis.voice_analyzer import VoiceAnalyzer
from voicelab.schema import Config, AnalysisResult
from voicelab.core.model_registry import ModelRegistry
# Import neural modules so their module-level register() calls run before any test,
# ensuring _register_mocks() can safely override them per-test.
import voicelab.neural.embeddings  # noqa: F401
import voicelab.neural.speaker_profile  # noqa: F401
import voicelab.neural.health  # noqa: F401


@pytest.fixture(autouse=True)
def clean_registry():
    ModelRegistry.instance().clear()
    yield
    ModelRegistry.instance().clear()


def _register_mocks():
    ModelRegistry.instance().register("ecapa", lambda: _mock_ecapa())
    ModelRegistry.instance().register("gender_age", lambda: (lambda *a: ("M", "25-35")))
    ModelRegistry.instance().register("lang_id", lambda: _mock_lang_id())
    ModelRegistry.instance().register("health", lambda: (lambda *a: {"dysphonia": 0.1, "fatigue": 0.1, "hoarseness": 0.1}))


def _mock_ecapa():
    m = MagicMock()
    m.encode_batch.return_value = (np.ones((1, 192), dtype=np.float32) * 0.1,)
    return m


def _mock_lang_id():
    m = MagicMock()
    m.classify_batch.return_value = (None, None, None, ["en"])
    return m


def test_analyze_returns_analysis_result(sine_440_wav):
    _register_mocks()
    path, _, _ = sine_440_wav
    analyzer = VoiceAnalyzer(Config(neural=True))
    result = analyzer.analyze(path)
    assert isinstance(result, AnalysisResult)


def test_analyze_duration_matches_audio(sine_440_wav):
    _register_mocks()
    path, _, _ = sine_440_wav
    analyzer = VoiceAnalyzer(Config(neural=True))
    result = analyzer.analyze(path)
    assert abs(result.duration - 2.0) < 0.1


def test_analyze_cpu_mode_skips_neural(sine_440_wav):
    """With neural=False, speaker embedding is zeros and scores are 0."""
    path, _, _ = sine_440_wav
    analyzer = VoiceAnalyzer(Config(neural=False))
    result = analyzer.analyze(path)
    assert isinstance(result, AnalysisResult)
    np.testing.assert_array_equal(result.speaker.embedding, np.zeros(192))
