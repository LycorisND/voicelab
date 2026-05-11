# voicelab/emotion/backbone.py
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from voicelab.core.model_registry import ModelRegistry
from voicelab.schema import EmotionConfig, EmotionResult

# Emotion centroids in Russell's circumplex (valence, arousal), range [-1, 1].
# Sources: Russell 1980, Warriner et al. 2013.
_CIRCUMPLEX: dict[str, tuple[float, float]] = {
    "anger":    (-0.6,  0.7),
    "disgust":  (-0.5,  0.1),
    "fear":     (-0.5,  0.8),
    "joy":      ( 0.8,  0.6),
    "sadness":  (-0.7, -0.5),
    "surprise": ( 0.3,  0.7),
    "neutral":  ( 0.0,  0.0),
}


class _ModelHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, 1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class _BackboneModel:
    """Wraps audeering/wav2vec2-large-robust-12-ft-emotion-age-gender.

    The model regresses arousal, dominance, valence in [0, 1].
    We expose (valence, arousal) in [-1, 1].
    """

    def __init__(self) -> None:
        from transformers import Wav2Vec2Processor
        from transformers.models.wav2vec2.modeling_wav2vec2 import (
            Wav2Vec2Model, Wav2Vec2PreTrainedModel,
        )

        class _Net(Wav2Vec2PreTrainedModel):
            _tied_weights_keys: list = []
            all_tied_weights_keys: dict = {}

            def __init__(self, config):
                super().__init__(config)
                self.wav2vec2 = Wav2Vec2Model(config)
                self.arousal = _ModelHead(config)
                self.dominance = _ModelHead(config)
                self.valence = _ModelHead(config)
                self.init_weights()

            def forward(self, input_values):
                hidden = self.wav2vec2(input_values)[0].mean(dim=1)
                return (
                    torch.sigmoid(self.arousal(hidden)),
                    torch.sigmoid(self.dominance(hidden)),
                    torch.sigmoid(self.valence(hidden)),
                )

        model_id = "audeering/wav2vec2-large-robust-12-ft-emotion-age-gender"
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = _Net.from_pretrained(model_id)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def __call__(self, audio: np.ndarray, sr: int) -> tuple[float, float]:
        """Returns (valence, arousal) in [-1.0, 1.0]."""
        y = self.processor(audio, sampling_rate=sr)["input_values"][0]
        y = torch.from_numpy(y.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            arousal_raw, _, valence_raw = self.model(y)
        valence = float(valence_raw.item()) * 2.0 - 1.0
        arousal = float(arousal_raw.item()) * 2.0 - 1.0
        return valence, arousal


def _load_backbone() -> _BackboneModel:
    # License: CC-BY-NC-4.0 — replace with own model before commercial release
    return _BackboneModel()


ModelRegistry.instance().register("ser_backbone", _load_backbone)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _va_to_logits(valence: float, arousal: float, labels: list[str]) -> np.ndarray:
    """Inverse-distance in VA space → logits for softmax.

    Unknown labels get centroid (0, 0) — neutral position.
    """
    logits = np.zeros(len(labels))
    for i, label in enumerate(labels):
        cv, ca = _CIRCUMPLEX.get(label, (0.0, 0.0))
        dist = np.sqrt((valence - cv) ** 2 + (arousal - ca) ** 2)
        logits[i] = -dist
    return logits


def _build_emotion_result(
    logits: np.ndarray,
    va: np.ndarray,
    config: EmotionConfig,
    path: str,
) -> EmotionResult:
    """Shared result builder — used by backbone and fusion."""
    probs = _softmax(logits)
    scores = dict(zip(config.emotion_labels, probs.tolist()))
    top = max(scores, key=scores.__getitem__)
    valence = float(np.clip(va[0], -1.0, 1.0))
    arousal = float(np.clip(va[1], -1.0, 1.0))
    dominant = [e for e, s in scores.items() if s > config.dominant_threshold]
    return EmotionResult(
        emotion=top,
        confidence=scores[top],
        scores=scores,
        valence=valence,
        arousal=arousal,
        dominant_emotions=dominant,
        path=path,
    )


def _run_backbone(audio: np.ndarray, sr: int, config: EmotionConfig) -> EmotionResult:
    model = ModelRegistry.instance().get("ser_backbone")
    valence, arousal = model(audio, sr)
    logits = _va_to_logits(valence, arousal, config.emotion_labels)
    va = np.array([valence, arousal], dtype=np.float32)
    return _build_emotion_result(logits, va, config, "backbone")
