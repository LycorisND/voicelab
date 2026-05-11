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
    # openai/whisper-small — Apache 2.0, robust on emotional/expressive speech
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    # Pre-build language token map: token_id → ISO code (e.g. 50266 → "ja")
    lang_map = {}
    for tid, tok in processor.tokenizer.added_tokens_decoder.items():
        s = str(tok)
        if s.startswith("<|") and s.endswith("|>") and tid < model.config.vocab_size:
            code = s[2:-2]
            if len(code) == 2 or len(code) == 3:
                lang_map[tid] = code
    return processor, model, lang_map


ModelRegistry.instance().register("gender_age", _load_gender_age)
ModelRegistry.instance().register("lang_id", _load_lang_id)


def get_speaker_profile(
    audio: np.ndarray,
    embedding: np.ndarray,
    sr: int,
    pitch_features=None,
) -> SpeakerProfile:
    gender, age_range = _predict_gender_age(audio, sr, pitch_features)
    language = _predict_language(audio, sr)
    return SpeakerProfile(
        embedding=embedding,
        gender=gender,
        age_range=age_range,
        language=language,
        accent="unknown",
    )


def _predict_gender_age(audio: np.ndarray, sr: int, pitch_features=None) -> tuple[str, str]:
    model_data = ModelRegistry.instance().get("gender_age")
    if callable(model_data) and not isinstance(model_data, tuple):
        return model_data(audio, sr)
    # result: [age(0-1), p_female, p_male, p_child]
    result = model_data.predict(audio, sr)
    age_neural = float(result[0]) * 100
    p_female, p_male = float(result[1]), float(result[2])
    gender = "M" if p_male >= p_female else "F"
    age_years = _apply_f0_age_correction(age_neural, gender, pitch_features)
    return gender, _age_to_range(age_years)


def _apply_f0_age_correction(age_neural: float, gender: str, pitch_features) -> float:
    """Blend neural age with F0-based acoustic heuristic.

    The audeering model is trained on natural speech corpora and struggles with
    expressive/animated voice acting, falsetto, or high-pitched characters.
    F0 (fundamental frequency) is a reliable physical correlate of vocal age:
    pre-pubertal voices sit above 230 Hz for both sexes; post-pubertal males
    drop sharply to 85-130 Hz. When the two estimates disagree by >15 years,
    we shift the blend 60 % toward the acoustic signal.
    Sources: Titze 1994, Robb & Saxman 1985.
    """
    if pitch_features is None:
        return age_neural
    f0 = getattr(pitch_features, "f0_mean", 0.0)
    if f0 < 60 or f0 > 700:
        return age_neural

    if gender == "M":
        if f0 > 240:
            age_f0 = 12.0   # pre-pubertal male
        elif f0 > 165:
            age_f0 = 17.0   # adolescent male
        elif f0 > 115:
            age_f0 = 23.0   # young adult male
        elif f0 > 90:
            age_f0 = 38.0   # mid-age adult male
        else:
            age_f0 = 58.0   # older male
    else:
        if f0 > 260:
            age_f0 = 14.0   # pre-pubertal female
        elif f0 > 220:
            age_f0 = 21.0   # young adult female
        elif f0 > 185:
            age_f0 = 33.0   # mid-age adult female
        elif f0 > 165:
            age_f0 = 48.0   # older adult female
        else:
            age_f0 = 58.0   # elderly female

    if abs(age_neural - age_f0) > 15:
        return 0.6 * age_neural + 0.4 * age_f0
    return age_neural


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
    model_data = ModelRegistry.instance().get("lang_id")
    if callable(model_data) and not isinstance(model_data, tuple):
        return model_data(audio, sr)
    processor, model, lang_map = model_data
    device = next(model.parameters()).device
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    with torch.no_grad():
        enc = model.model.encoder(input_features)
        dec_in = torch.tensor(
            [[model.config.decoder_start_token_id]], device=device
        )
        logits = model.proj_out(
            model.model.decoder(
                input_ids=dec_in,
                encoder_hidden_states=enc.last_hidden_state,
            ).last_hidden_state[:, -1, :]
        )
    lang_scores = {tid: logits[0, tid].item() for tid in lang_map}
    best_tid = max(lang_scores, key=lang_scores.get)
    return lang_map[best_tid]
