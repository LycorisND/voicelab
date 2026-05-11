# tests/test_emotion_stream.py
import numpy as np
import pytest
from voicelab.core.model_registry import ModelRegistry
from voicelab.schema import EmotionConfig, EmotionFrame
import voicelab.emotion.backbone   # noqa: F401
import voicelab.emotion.fusion     # noqa: F401


@pytest.fixture(autouse=True)
def clean_registry():
    ModelRegistry.instance().clear()
    yield
    ModelRegistry.instance().clear()


def _register_emotion_lite_mock():
    n = 7
    logits = np.zeros(n); logits[6] = 3.0  # "neutral" wins
    va = np.array([0.0, 0.1])
    ModelRegistry.instance().register(
        "ser_fusion_lite",
        lambda: (lambda feat: (logits.copy(), va.copy())),
    )


def _make_source(n_chunks: int = 5, chunk_size: int = 512, sr: int = 16000):
    for _ in range(n_chunks):
        yield np.zeros(chunk_size, dtype=np.float32)


def test_stream_yields_emotion_frames():
    _register_emotion_lite_mock()
    from voicelab.emotion.stream import EmotionStream
    es = EmotionStream(EmotionConfig())
    frames = list(es._process_source(_make_source()))
    assert len(frames) > 0
    assert all(isinstance(f, EmotionFrame) for f in frames)


def test_stream_timestamps_increase():
    _register_emotion_lite_mock()
    from voicelab.emotion.stream import EmotionStream
    es = EmotionStream(EmotionConfig())
    frames = list(es._process_source(_make_source()))
    timestamps = [f.timestamp for f in frames]
    assert timestamps == sorted(timestamps)


def test_stream_emotion_in_labels():
    _register_emotion_lite_mock()
    from voicelab.emotion.stream import EmotionStream
    cfg = EmotionConfig()
    es = EmotionStream(cfg)
    frames = list(es._process_source(_make_source()))
    assert all(f.emotion in cfg.emotion_labels for f in frames)


def test_stream_scores_sum_to_one():
    _register_emotion_lite_mock()
    from voicelab.emotion.stream import EmotionStream
    es = EmotionStream(EmotionConfig())
    frames = list(es._process_source(_make_source()))
    for f in frames:
        assert abs(sum(f.scores.values()) - 1.0) < 1e-5
