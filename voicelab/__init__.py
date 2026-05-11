from voicelab.schema import (
    AnalysisResult, FrameResult, Config,
    PitchFeatures, SpectralFeatures, ProsodyFeatures,
    SpeakerProfile, HealthIndicators, AudioMetadata,
    EmotionResult, EmotionFrame, EmotionConfig,
)
from voicelab.analysis.voice_analyzer import VoiceAnalyzer
from voicelab.analysis.voice_stream import VoiceStream
from voicelab.emotion.analyzer import EmotionAnalyzer
from voicelab.emotion.stream import EmotionStream

__version__ = "0.1.0"
__all__ = [
    "analyze", "stream", "analyze_emotion", "emotion_stream",
    "Config", "EmotionConfig",
    "AnalysisResult", "FrameResult", "EmotionResult", "EmotionFrame",
]

_analyzer_cache: dict[str, VoiceAnalyzer] = {}
_emotion_analyzer_cache: dict[str, EmotionAnalyzer] = {}


def analyze(path: str, config: Config | None = None) -> AnalysisResult:
    """Analyse an audio file and return a fully populated AnalysisResult."""
    cfg = config or Config()
    key = f"{cfg.device}-{cfg.neural}-{cfg.n_mfcc}"
    if key not in _analyzer_cache:
        _analyzer_cache[key] = VoiceAnalyzer(cfg)
    return _analyzer_cache[key].analyze(path)


def stream(config: Config | None = None) -> VoiceStream:
    """Return a VoiceStream context manager for real-time microphone processing."""
    return VoiceStream(config or Config())


def analyze_emotion(path: str, config: EmotionConfig | None = None) -> EmotionResult:
    """Analyse emotion in an audio file. Backbone path by default; fusion if fast=True."""
    cfg = config or EmotionConfig()
    key = f"{cfg.fast}-{cfg.dominant_threshold}-{','.join(cfg.emotion_labels)}"
    if key not in _emotion_analyzer_cache:
        _emotion_analyzer_cache[key] = EmotionAnalyzer(cfg)
    return _emotion_analyzer_cache[key].analyze(path)


def emotion_stream(config: EmotionConfig | None = None) -> EmotionStream:
    """Return an EmotionStream context manager for real-time emotion recognition."""
    return EmotionStream(config or EmotionConfig())
