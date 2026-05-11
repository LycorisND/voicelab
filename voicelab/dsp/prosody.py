import numpy as np
import librosa
import webrtcvad
from voicelab.schema import ProsodyFeatures


def extract_prosody(audio: np.ndarray, sr: int) -> ProsodyFeatures:
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    tempo_bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])

    energy_profile = librosa.feature.rms(y=audio)[0].astype(np.float32)

    pause_ratio, pause_count = _detect_pauses(audio, sr)

    return ProsodyFeatures(
        tempo_bpm=tempo_bpm,
        pause_ratio=pause_ratio,
        pause_count=pause_count,
        energy_profile=energy_profile,
    )


def _detect_pauses(
    audio: np.ndarray, sr: int, frame_ms: int = 30, aggressiveness: int = 2
) -> tuple[float, int]:
    """Use WebRTC VAD to count pauses. Audio must be 16 kHz (VAD requirement)."""
    if sr != 16000:
        raise ValueError(f"WebRTC VAD requires 16 kHz audio, got {sr} Hz. Resample before calling.")
    vad = webrtcvad.Vad(aggressiveness)
    frame_len = int(sr * frame_ms / 1000)
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return 1.0, 1

    speech = 0
    non_speech = 0
    in_pause = False
    pause_count = 0

    for i in range(n_frames):
        frame = audio[i * frame_len : (i + 1) * frame_len]
        pcm = (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        try:
            is_speech = vad.is_speech(pcm, sr)
        except Exception:
            is_speech = True

        if is_speech:
            speech += 1
            in_pause = False
        else:
            non_speech += 1
            if not in_pause:
                in_pause = True
                pause_count += 1

    total = speech + non_speech
    return (non_speech / total if total > 0 else 0.0), pause_count
