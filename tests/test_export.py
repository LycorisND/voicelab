import json
import numpy as np
import pytest
from voicelab.utils.export import to_dict, to_json, to_csv
from voicelab.schema import (
    AnalysisResult, PitchFeatures, SpectralFeatures,
    ProsodyFeatures, SpeakerProfile, HealthIndicators, AudioMetadata,
)


def _make_result() -> AnalysisResult:
    return AnalysisResult(
        duration=2.0,
        pitch=PitchFeatures(200.0, 10.0, 180.0, 220.0, np.array([200.0, 205.0]),
                            0.01, 0.05, 15.0, 0.8),
        spectral=SpectralFeatures(np.zeros((10, 13)), 1500.0, 4000.0, 0.05, 0.35),
        prosody=ProsodyFeatures(120.0, 0.2, 3, np.array([0.1, 0.2, 0.3])),
        speaker=SpeakerProfile(np.ones(192) * 0.1, "M", "25-35", "en", "standard"),
        health=HealthIndicators(0.1, 0.05, 0.08, []),
        metadata=AudioMetadata(16000, 2.0, 35.0, False, True),
    )


def test_to_dict_is_json_serialisable():
    d = to_dict(_make_result())
    s = json.dumps(d)
    assert '"duration"' in s


def test_to_json_writes_file(tmp_path):
    path = str(tmp_path / "result.json")
    to_json(_make_result(), path)
    data = json.loads(open(path).read())
    assert data["duration"] == 2.0


def test_to_csv_writes_scalars(tmp_path):
    path = str(tmp_path / "result.csv")
    to_csv(_make_result(), path)
    content = open(path).read()
    assert "f0_mean" in content
    assert "200" in content
