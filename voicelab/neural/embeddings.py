from __future__ import annotations
import numpy as np
import torch
from voicelab.core.model_registry import ModelRegistry


def _load_ecapa():
    from speechbrain.inference.speaker import EncoderClassifier
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="voicelab/models/ecapa-tdnn",
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )
    model.eval()
    return model


ModelRegistry.instance().register("ecapa", _load_ecapa)


def get_speaker_embedding(audio: np.ndarray, sr: int) -> np.ndarray:
    """Return 192-dim L2-normalised speaker embedding via ECAPA-TDNN."""
    model = ModelRegistry.instance().get("ecapa")
    tensor = torch.from_numpy(audio).unsqueeze(0).float()
    with torch.no_grad():
        emb = model.encode_batch(tensor)
    if isinstance(emb, tuple):
        emb = emb[0]
    # Handle both numpy arrays (from mock) and torch tensors (from real model)
    if isinstance(emb, np.ndarray):
        vec = emb.squeeze().astype(np.float32)
    else:
        vec = emb.squeeze().cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-8)
