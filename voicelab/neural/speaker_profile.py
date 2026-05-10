from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from voicelab.core.model_registry import ModelRegistry
from voicelab.schema import SpeakerProfile


# Custom model head as defined by audeering model card
class _ModelHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class _AgeGenderModel:
    """Wrapper following audeering/wav2vec2-large-robust-24-ft-age-gender usage."""

    def __init__(self):
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
                self.age = _ModelHead(config, 1)
                self.gender = _ModelHead(config, 3)
                self.init_weights()

            def forward(self, input_values):
                hidden = self.wav2vec2(input_values)[0].mean(dim=1)
                age = self.age(hidden)
                gender = torch.softmax(self.gender(hidden), dim=1)
                return torch.hstack([age, gender])

        model_id = "audeering/wav2vec2-large-robust-24-ft-age-gender"
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = _Net.from_pretrained(model_id)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def predict(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Returns [age(0-1), p_female, p_male, p_child]."""
        y = self.processor(audio, sampling_rate=sr)["input_values"][0]
        y = torch.from_numpy(y.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            out = self.model(y)
        return out.cpu().numpy()[0]


def _load_gender_age():
    # License: CC-BY-NC-SA-4.0 — temporary until own model is trained
    return _AgeGenderModel()


def _load_lang_id():
    from speechbrain.inference.classifiers import EncoderClassifier
    return EncoderClassifier.from_hparams(
        source="speechbrain/lang-id-voxlingua107-ecapa",
        savedir="voicelab/models/lang-id",
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )


ModelRegistry.instance().register("gender_age", _load_gender_age)
ModelRegistry.instance().register("lang_id", _load_lang_id)


def get_speaker_profile(
    audio: np.ndarray, embedding: np.ndarray, sr: int
) -> SpeakerProfile:
    gender, age_range = _predict_gender_age(audio, sr)
    language = _predict_language(audio, sr)
    return SpeakerProfile(
        embedding=embedding,
        gender=gender,
        age_range=age_range,
        language=language,
        accent="unknown",
    )


def _predict_gender_age(audio: np.ndarray, sr: int) -> tuple[str, str]:
    model_data = ModelRegistry.instance().get("gender_age")
    if callable(model_data) and not isinstance(model_data, tuple):
        return model_data(audio, sr)
    # result: [age(0-1), p_female, p_male, p_child]
    result = model_data.predict(audio, sr)
    age_years = float(result[0]) * 100
    p_female, p_male = float(result[1]), float(result[2])
    gender = "M" if p_male >= p_female else "F"
    return gender, _age_to_range(age_years)


def _age_to_range(age_years: float) -> str:
    if age_years < 18:
        return "0-18"
    if age_years < 25:
        return "18-25"
    if age_years < 35:
        return "25-35"
    if age_years < 50:
        return "35-50"
    return "50+"


def _predict_language(audio: np.ndarray, sr: int) -> str:
    model = ModelRegistry.instance().get("lang_id")
    tensor = torch.from_numpy(audio).unsqueeze(0).float()
    with torch.no_grad():
        _, _, _, labels = model.classify_batch(tensor)
    return labels[0] if labels else "unknown"
