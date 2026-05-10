from __future__ import annotations
import numpy as np
from voicelab.core.model_registry import ModelRegistry
from voicelab.schema import HealthIndicators, PitchFeatures


def _load_health_model():
    import torch
    import torch.nn as nn
    from transformers import HubertModel, Wav2Vec2Processor

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

    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-base-ls960")
    model = HealthHead()
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return processor, model


ModelRegistry.instance().register("health", _load_health_model)


def get_health_indicators(
    audio: np.ndarray, pitch: PitchFeatures, sr: int
) -> HealthIndicators:
    dysphonia, fatigue, hoarseness = _neural_scores(audio, sr)

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


def _neural_scores(audio: np.ndarray, sr: int) -> tuple[float, float, float]:
    import torch
    model_data = ModelRegistry.instance().get("health")
    # Mock: callable returning dict
    if callable(model_data) and not isinstance(model_data, tuple):
        scores = model_data(audio, sr)
        return scores["dysphonia"], scores["fatigue"], scores["hoarseness"]
    processor, model = model_data
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        out = model(inputs["input_values"])
    scores = out.squeeze().cpu().numpy()
    return float(scores[0]), float(scores[1]), float(scores[2])
