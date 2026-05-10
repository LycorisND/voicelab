from __future__ import annotations
import numpy as np
import torch
from voicelab.core.model_registry import ModelRegistry
from voicelab.schema import SpeakerProfile


def _load_gender_age():
    from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
    model_id = "facebook/wav2vec2-base"
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id, num_labels=8)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return processor, model


def _load_lang_id():
    from speechbrain.inference.classifiers import EncoderClassifier
    return EncoderClassifier.from_hparams(
        source="speechbrain/lang-id-voxlingua107-ecapa",
        savedir="voicelab/models/lang-id",
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )


ModelRegistry.instance().register("gender_age", _load_gender_age)
ModelRegistry.instance().register("lang_id", _load_lang_id)

_GENDER_AGE_LABELS = {
    0: ("M", "18-25"), 1: ("M", "25-35"), 2: ("M", "35-50"), 3: ("M", "50+"),
    4: ("F", "18-25"), 5: ("F", "25-35"), 6: ("F", "35-50"), 7: ("F", "50+"),
}


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
    # Mock returns a simple callable
    if callable(model_data) and not isinstance(model_data, tuple):
        return model_data(audio, sr)
    processor, model = model_data
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    idx = int(logits.argmax(-1).item())
    return _GENDER_AGE_LABELS.get(idx, ("unknown", "unknown"))


def _predict_language(audio: np.ndarray, sr: int) -> str:
    model = ModelRegistry.instance().get("lang_id")
    tensor = torch.from_numpy(audio).unsqueeze(0).float()
    with torch.no_grad():
        _, _, _, labels = model.classify_batch(tensor)
    return labels[0] if labels else "unknown"
