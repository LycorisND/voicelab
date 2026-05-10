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
