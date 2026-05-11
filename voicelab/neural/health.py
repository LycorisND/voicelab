from __future__ import annotations
import numpy as np
from voicelab.core.model_registry import ModelRegistry
from voicelab.schema import HealthIndicators, PitchFeatures


def _load_health_model():
    import torch
    import torch.nn as nn
    from transformers import HubertModel, Wav2Vec2FeatureExtractor

    class HealthHead(nn.Module):
        def __init__(self, hidden: int = 768) -> None:
            super().__init__()
            self.backbone = HubertModel.from_pretrained("facebook/hubert-base-ls960")
            self.head = nn.Sequential(
                nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, 3), nn.Sigmoid()
            )

        def forward(self, input_values):
            out = self.backbone(input_values=input_values).last_hidden_state
            return self.head(out.mean(dim=1))

    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    model = HealthHead()
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return processor, model


ModelRegistry.instance().register("health", _load_health_model)


def get_health_indicators(
    audio: np.ndarray, pitch: PitchFeatures, sr: int
) -> HealthIndicators:
    dysphonia, fatigue, hoarseness = _acoustic_scores(pitch)

    flags: list[str] = []
    if pitch.jitter_local > 0.03 or pitch.hnr < 5.0:
        flags.append("dysphonia")
        dysphonia = max(dysphonia, 0.6)
    if pitch.shimmer_local > 0.1:
        flags.append("hoarseness")
        hoarseness = max(hoarseness, 0.5)

    return HealthIndicators(
        dysphonia_score=float(np.clip(dysphonia, 0.0, 1.0)),
        fatigue_index=float(np.clip(fatigue, 0.0, 1.0)),
        hoarseness=float(np.clip(hoarseness, 0.0, 1.0)),
        pathology_flags=flags,
    )


def _acoustic_scores(pitch: PitchFeatures) -> tuple[float, float, float]:
    """Compute health scores from acoustic features.

    Formulas based on clinical voice literature (Baken & Orlikoff, 2000):
    - Dysphonia correlates with elevated jitter and low HNR
    - Hoarseness correlates with elevated shimmer
    - Fatigue correlates with reduced F0 variability and low voiced fraction
    """
    # Dysphonia: jitter contributes 60%, HNR deficit contributes 40%
    # Normal jitter < 0.01, pathological > 0.03; HNR normal > 15 dB
    jitter_score = np.clip(pitch.jitter_local / 0.05, 0.0, 1.0)
    hnr_score = np.clip(1.0 - (pitch.hnr + 20.0) / 35.0, 0.0, 1.0)
    dysphonia = 0.6 * jitter_score + 0.4 * hnr_score

    # Hoarseness: shimmer is primary indicator
    # Normal shimmer < 0.03, pathological > 0.1
    hoarseness = np.clip(pitch.shimmer_local / 0.15, 0.0, 1.0)

    # Fatigue: low voiced fraction + flat F0 contour = tired/monotone voice
    voiced_deficit = np.clip(1.0 - pitch.voiced_fraction / 0.7, 0.0, 1.0)
    # High F0 std relative to mean = expressive (not fatigued)
    variation = pitch.f0_std / (pitch.f0_mean + 1e-8)
    monotone = np.clip(1.0 - variation / 0.3, 0.0, 1.0)
    fatigue = 0.5 * voiced_deficit + 0.5 * monotone

    return float(dysphonia), float(fatigue), float(hoarseness)
