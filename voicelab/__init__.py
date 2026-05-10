from voicelab.schema import (
    AnalysisResult,
    FrameResult,
    Config,
    PitchFeatures,
    SpectralFeatures,
    ProsodyFeatures,
    SpeakerProfile,
    HealthIndicators,
    AudioMetadata,
)
from voicelab.analysis.voice_analyzer import VoiceAnalyzer
from voicelab.analysis.voice_stream import VoiceStream

__version__ = "0.1.0"
__all__ = [
    "analyze", "stream", "Config",
    "AnalysisResult", "FrameResult",
]

_analyzer_cache: dict[str, VoiceAnalyzer] = {}


def analyze(path: str, config: Config | None = None) -> AnalysisResult:
    """Analyse an audio file and return a fully populated AnalysisResult."""
    cfg = config or Config()
    cache_key = f"{cfg.device}-{cfg.neural}-{cfg.n_mfcc}"
    if cache_key not in _analyzer_cache:
        _analyzer_cache[cache_key] = VoiceAnalyzer(cfg)
    return _analyzer_cache[cache_key].analyze(path)


def stream(config: Config | None = None) -> VoiceStream:
    """Return a VoiceStream context manager for real-time microphone processing."""
    return VoiceStream(config or Config())
