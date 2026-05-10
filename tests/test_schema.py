# tests/test_schema.py
import numpy as np
from voicelab.schema import (
    PitchFeatures, SpectralFeatures, ProsodyFeatures,
    SpeakerProfile, HealthIndicators, AudioMetadata,
    AnalysisResult, FrameResult, Config,
)

def test_pitch_features_fields():
    pf = PitchFeatures(
        f0_mean=185.0, f0_std=20.0, f0_min=130.0, f0_max=280.0,
        f0_contour=np.array([185.0, 190.0, 180.0]),
        jitter_local=0.01, shimmer_local=0.05, hnr=15.0, voiced_fraction=0.8,
    )
    assert pf.f0_mean == 185.0
    assert pf.voiced_fraction == 0.8

def test_analysis_result_fields():
    result = AnalysisResult(
        duration=2.0,
        pitch=PitchFeatures(0,0,0,0,np.array([]),0,0,0,0),
        spectral=SpectralFeatures(np.zeros((10,13)),0,0,0,0),
        prosody=ProsodyFeatures(0,0,0,np.array([])),
        speaker=SpeakerProfile(np.zeros(192),"M","25-35","en","standard"),
        health=HealthIndicators(0.0,0.0,0.0,[]),
        metadata=AudioMetadata(16000,2.0,30.0,False,True),
    )
    assert result.duration == 2.0
    assert result.speaker.gender == "M"

def test_frame_result_fields():
    fr = FrameResult(timestamp=0.032, pitch=185.0, energy=0.05, is_voiced=True, mfcc=np.zeros(13))
    assert fr.is_voiced is True

def test_config_defaults():
    cfg = Config()
    assert cfg.device in ("cuda", "cpu")
    assert cfg.neural is True
    assert cfg.n_mfcc == 13
