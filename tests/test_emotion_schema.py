# tests/test_emotion_schema.py
import pytest
from voicelab.schema import EmotionResult, EmotionFrame, EmotionConfig


def _make_result(**overrides):
    defaults = dict(
        emotion="joy",
        confidence=0.6,
        scores={"anger": 0.2, "joy": 0.6, "neutral": 0.2},
        valence=0.5,
        arousal=0.4,
        dominant_emotions=["joy"],
        path="backbone",
    )
    return EmotionResult(**{**defaults, **overrides})


def test_emotion_result_instantiates():
    r = _make_result()
    assert r.emotion == "joy"
    assert r.path == "backbone"


def test_scores_sum_to_one():
    r = _make_result()
    assert abs(sum(r.scores.values()) - 1.0) < 1e-5


def test_confidence_matches_scores():
    r = _make_result()
    assert r.confidence == r.scores[r.emotion]


def test_valence_arousal_in_range():
    r = _make_result()
    assert -1.0 <= r.valence <= 1.0
    assert -1.0 <= r.arousal <= 1.0


def test_dominant_emotions_in_scores():
    r = _make_result()
    assert all(e in r.scores for e in r.dominant_emotions)


def test_dominant_emotions_exceed_threshold():
    cfg = EmotionConfig()
    r = _make_result()
    assert all(r.scores[e] > cfg.dominant_threshold for e in r.dominant_emotions)


def test_emotion_frame_instantiates():
    f = EmotionFrame(
        timestamp=0.5,
        emotion="neutral",
        confidence=0.8,
        scores={"neutral": 0.8, "joy": 0.2},
        valence=0.0,
        arousal=0.1,
        dominant_emotions=["neutral"],
    )
    assert f.timestamp == 0.5
    assert f.emotion == "neutral"


def test_emotion_frame_confidence_matches_scores():
    f = EmotionFrame(
        timestamp=0.5,
        emotion="neutral",
        confidence=0.8,
        scores={"neutral": 0.8, "joy": 0.2},
        valence=0.0,
        arousal=0.1,
        dominant_emotions=["neutral"],
    )
    assert f.confidence == f.scores[f.emotion]


def test_emotion_frame_va_in_range():
    f = EmotionFrame(
        timestamp=0.5,
        emotion="neutral",
        confidence=0.8,
        scores={"neutral": 0.8, "joy": 0.2},
        valence=0.0,
        arousal=0.1,
        dominant_emotions=["neutral"],
    )
    assert -1.0 <= f.valence <= 1.0
    assert -1.0 <= f.arousal <= 1.0


def test_emotion_config_defaults():
    cfg = EmotionConfig()
    assert cfg.fast is False
    assert "anger" in cfg.emotion_labels
    assert len(cfg.emotion_labels) == 7
    assert cfg.dominant_threshold == 0.2
    assert cfg.device in ("cpu", "cuda")
