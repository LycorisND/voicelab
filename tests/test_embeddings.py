# tests/test_embeddings.py
import numpy as np
import pytest
from unittest.mock import MagicMock
from voicelab.neural.embeddings import get_speaker_embedding
from voicelab.core.model_registry import ModelRegistry


@pytest.fixture(autouse=True)
def clean_registry():
    ModelRegistry.instance().clear()
    yield
    ModelRegistry.instance().clear()


def _mock_ecapa():
    model = MagicMock()
    model.encode_batch.return_value = (
        np.ones((1, 192), dtype=np.float32) * 0.1,
    )
    return model


def test_embedding_shape():
    ModelRegistry.instance().register("ecapa", _mock_ecapa)
    audio = np.zeros(16000, dtype=np.float32)
    emb = get_speaker_embedding(audio, sr=16000)
    assert emb.shape == (192,)


def test_embedding_dtype():
    ModelRegistry.instance().register("ecapa", _mock_ecapa)
    audio = np.zeros(16000, dtype=np.float32)
    emb = get_speaker_embedding(audio, sr=16000)
    assert emb.dtype == np.float32


def test_embedding_is_normalized():
    ModelRegistry.instance().register("ecapa", _mock_ecapa)
    audio = np.zeros(16000, dtype=np.float32)
    emb = get_speaker_embedding(audio, sr=16000)
    norm = float(np.linalg.norm(emb))
    assert norm > 0.0
