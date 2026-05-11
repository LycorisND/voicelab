# voicelab/emotion/fusion.py
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from voicelab.core.model_registry import ModelRegistry
from voicelab.schema import AnalysisResult, EmotionConfig, EmotionFrame, EmotionResult, FrameResult
from voicelab.emotion.backbone import _build_emotion_result


class _FusionMLP(nn.Module):
    """Full fusion: ECAPA(192) + prosody scalars(6) = 198-dim input."""

    def __init__(self, in_dim: int, n_labels: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),    nn.ReLU(),
        )
        self.classifier = nn.Linear(64, n_labels)
        self.va_head = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        return self.classifier(h), torch.tanh(self.va_head(h))


class _FusionLiteMLP(nn.Module):
    """Lite fusion: MFCC(13) + energy(1) + pitch(1) = 15-dim input."""

    def __init__(self, n_labels: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(15, 32), nn.ReLU())
        self.classifier = nn.Linear(32, n_labels)
        self.va_head = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        return self.classifier(h), torch.tanh(self.va_head(h))


def _mlp_call(mlp: nn.Module, features: np.ndarray):
    x = torch.from_numpy(features.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        logits_t, va_t = mlp(x)
    return logits_t.squeeze(0).numpy(), va_t.squeeze(0).numpy()


def _load_fusion():
    mlp = _FusionMLP(198, 7)
    mlp.eval()
    return lambda feat: _mlp_call(mlp, feat)


def _load_fusion_lite():
    mlp = _FusionLiteMLP(7)
    mlp.eval()
    return lambda feat: _mlp_call(mlp, feat)


ModelRegistry.instance().register("ser_fusion", _load_fusion)
ModelRegistry.instance().register("ser_fusion_lite", _load_fusion_lite)


def _build_features_198(result: AnalysisResult) -> np.ndarray:
    emb = result.speaker.embedding.astype(np.float32)
    if emb.shape[0] != 192:
        raise ValueError(f"Speaker embedding must be 192-dim, got {emb.shape[0]}")

    def _safe(v: float) -> float:
        return 0.0 if not np.isfinite(v) else float(v)

    prosody = np.array([
        _safe(result.prosody.tempo_bpm),
        _safe(result.prosody.pause_ratio),
        _safe(result.pitch.f0_mean),
        _safe(result.pitch.f0_std),
        _safe(result.pitch.hnr),
        _safe(result.spectral.rms_mean),
    ], dtype=np.float32)
    return np.concatenate([emb, prosody])


def _build_features_15(frame: FrameResult) -> np.ndarray:
    pitch = 0.0 if (frame.pitch != frame.pitch) else float(frame.pitch)  # NaN guard
    energy = 0.0 if (frame.energy != frame.energy) else float(frame.energy)  # NaN guard
    return np.concatenate([
        frame.mfcc.astype(np.float32),
        np.array([energy, pitch], dtype=np.float32),
    ])


def _run_fusion(result: AnalysisResult, config: EmotionConfig) -> EmotionResult:
    features = _build_features_198(result)
    model = ModelRegistry.instance().get("ser_fusion")
    logits, va = model(features)
    return _build_emotion_result(logits, va, config, "fusion")


def _run_fusion_lite(frame: FrameResult, config: EmotionConfig) -> EmotionFrame:
    features = _build_features_15(frame)
    model = ModelRegistry.instance().get("ser_fusion_lite")
    logits, va = model(features)
    er = _build_emotion_result(logits, va, config, "fusion-lite")
    return EmotionFrame(
        timestamp=frame.timestamp,
        emotion=er.emotion,
        confidence=er.confidence,
        scores=er.scores,
        valence=er.valence,
        arousal=er.arousal,
        dominant_emotions=er.dominant_emotions,
        path=er.path,
    )
