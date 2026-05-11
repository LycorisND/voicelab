# tests/test_emotion_fusion.py
import numpy as np
import pytest
from unittest.mock import MagicMock
from voicelab.core.model_registry import ModelRegistry
from voicelab.schema import (
    EmotionConfig, AnalysisResult, AudioMetadata,
    PitchFeatures, SpectralFeatures, ProsodyFeatures, SpeakerProfile, HealthIndicators,
    FrameResult,
)
import voicelab.emotion.backbone  # noqa: F401
import voicelab.emotion.fusion    # noqa: F401


@pytest.fixture(autouse=True)
def clean_registry():
    ModelRegistry.instance().clear()
    yield
    ModelRegistry.instance().clear()


def _make_analysis_result():
    return AnalysisResult(
        duration=2.0,
        pitch=PitchFeatures(
            f0_mean=150.0, f0_std=20.0, f0_min=100.0, f0_max=200.0,
            f0_contour=np.zeros(50), jitter_local=0.01, shimmer_local=0.05,
            hnr=15.0, voiced_fraction=0.8,
        ),
        spectral=SpectralFeatures(
            mfcc=np.zeros((50, 13)), centroid_mean=1200.0,
            rolloff_mean=2000.0, zcr_mean=0.05, rms_mean=0.1,
        ),
        prosody=ProsodyFeatures(
            tempo_bpm=120.0, pause_ratio=0.2,
            pause_count=3, energy_profile=np.zeros(50),
        ),
        speaker=SpeakerProfile(
            embedding=np.ones(192, dtype=np.float32) * 0.1,
            gender="M", age_range="25-35", language="en", accent="unknown",
        ),
        health=HealthIndicators(0.1, 0.1, 0.1),
        metadata=AudioMetadata(sample_rate=16000, duration=2.0, snr_db=30.0,
                               clipping_detected=False, mono=True),
    )


def _register_mock_fusion():
    n = 7
    logits = np.zeros(n); logits[3] = 2.0  # "joy" wins
    va = np.array([0.6, 0.5])
    ModelRegistry.instance().register(
        "ser_fusion",
        lambda: (lambda feat: (logits.copy(), va.copy())),
    )
    ModelRegistry.instance().register(
        "ser_fusion_lite",
        lambda: (lambda feat: (logits.copy(), va.copy())),
    )


def test_run_fusion_returns_emotion_result():
    _register_mock_fusion()
    from voicelab.emotion.fusion import _run_fusion
    result = _make_analysis_result()
    cfg = EmotionConfig()
    er = _run_fusion(result, cfg)
    assert er.emotion in cfg.emotion_labels


def test_fusion_scores_sum_to_one():
    _register_mock_fusion()
    from voicelab.emotion.fusion import _run_fusion
    er = _run_fusion(_make_analysis_result(), EmotionConfig())
    assert abs(sum(er.scores.values()) - 1.0) < 1e-5


def test_fusion_path_label():
    _register_mock_fusion()
    from voicelab.emotion.fusion import _run_fusion
    er = _run_fusion(_make_analysis_result(), EmotionConfig())
    assert er.path == "fusion"


def test_fusion_lite_returns_emotion_frame():
    _register_mock_fusion()
    from voicelab.emotion.fusion import _run_fusion_lite
    frame = FrameResult(
        timestamp=0.5,
        pitch=220.0,
        energy=0.05,
        is_voiced=True,
        mfcc=np.zeros(13, dtype=np.float32),
    )
    ef = _run_fusion_lite(frame, EmotionConfig())
    assert ef.emotion in EmotionConfig().emotion_labels
    assert ef.timestamp == 0.5


def test_fusion_lite_path_label():
    _register_mock_fusion()
    from voicelab.emotion.fusion import _run_fusion_lite
    frame = FrameResult(
        timestamp=1.0, pitch=float("nan"), energy=0.0,
        is_voiced=False, mfcc=np.zeros(13, dtype=np.float32),
    )
    ef = _run_fusion_lite(frame, EmotionConfig())
    assert ef.path == "fusion-lite"


def test_fusion_lite_nan_pitch_handled():
    _register_mock_fusion()
    from voicelab.emotion.fusion import _run_fusion_lite
    frame = FrameResult(
        timestamp=0.0, pitch=float("nan"), energy=0.0,
        is_voiced=False, mfcc=np.zeros(13, dtype=np.float32),
    )
    ef = _run_fusion_lite(frame, EmotionConfig())
    assert isinstance(ef.emotion, str)
