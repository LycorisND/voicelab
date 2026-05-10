from __future__ import annotations
import numpy as np
from voicelab.schema import (
    AnalysisResult, AudioMetadata, Config,
    HealthIndicators, SpeakerProfile,
)
from voicelab.core.audio_io import load_audio, detect_clipping, estimate_snr
from voicelab.dsp.spectral import extract_spectral_features
from voicelab.dsp.pitch import extract_pitch
from voicelab.dsp.prosody import extract_prosody


class VoiceAnalyzer:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()

    def analyze(self, path: str) -> AnalysisResult:
        audio, sr = load_audio(path)
        duration = len(audio) / sr

        metadata = AudioMetadata(
            sample_rate=sr,
            duration=duration,
            snr_db=estimate_snr(audio, sr),
            clipping_detected=detect_clipping(audio),
            mono=True,
        )

        pitch = extract_pitch(audio, sr)
        spectral = extract_spectral_features(audio, sr, n_mfcc=self.config.n_mfcc)
        prosody = extract_prosody(audio, sr)

        if self.config.neural:
            speaker, health = self._run_neural(audio, sr, pitch)
        else:
            speaker = SpeakerProfile(
                embedding=np.zeros(192, dtype=np.float32),
                gender="unknown", age_range="unknown",
                language="unknown", accent="unknown",
            )
            health = HealthIndicators(0.0, 0.0, 0.0, [])

        return AnalysisResult(
            duration=duration,
            pitch=pitch,
            spectral=spectral,
            prosody=prosody,
            speaker=speaker,
            health=health,
            metadata=metadata,
        )

    def _run_neural(self, audio: np.ndarray, sr: int, pitch):
        from voicelab.neural.embeddings import get_speaker_embedding
        from voicelab.neural.speaker_profile import get_speaker_profile
        from voicelab.neural.health import get_health_indicators
        embedding = get_speaker_embedding(audio, sr)
        speaker = get_speaker_profile(audio, embedding, sr)
        health = get_health_indicators(audio, pitch, sr)
        return speaker, health
