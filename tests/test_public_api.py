import numpy as np
import pytest
from unittest.mock import MagicMock
from voicelab.core.model_registry import ModelRegistry
# Pre-import neural modules so their module-level register() runs before tests
import voicelab.neural.embeddings  # noqa: F401
import voicelab.neural.speaker_profile  # noqa: F401
import voicelab.neural.health  # noqa: F401


@pytest.fixture(autouse=True)
def clean_registry():
    ModelRegistry.instance().clear()
    yield
    ModelRegistry.instance().clear()


def test_analyze_returns_result(sine_440_wav):
    import voicelab as vl
    path, _, _ = sine_440_wav
    result = vl.analyze(path, config=vl.Config(neural=False))
    from voicelab.schema import AnalysisResult
    assert isinstance(result, AnalysisResult)


def test_config_default_device():
    import voicelab as vl
    import torch
    cfg = vl.Config()
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert cfg.device == expected


def test_stream_is_context_manager():
    import voicelab as vl
    vs = vl.stream(config=vl.Config(neural=False))
    assert hasattr(vs, "__enter__") and hasattr(vs, "__exit__")
