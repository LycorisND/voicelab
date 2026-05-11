# voicelab/emotion/analyzer.py
from __future__ import annotations
from voicelab.schema import AnalysisResult, EmotionConfig, EmotionResult


class EmotionAnalyzer:
    def __init__(self, config: EmotionConfig | None = None) -> None:
        self.config = config or EmotionConfig()

    def analyze(self, path: str) -> EmotionResult:
        """File → EmotionResult. Backbone if fast=False, fusion if fast=True."""
        if self.config.fast:
            from voicelab.analysis.voice_analyzer import VoiceAnalyzer
            from voicelab.schema import Config
            analysis = VoiceAnalyzer(Config(neural=True)).analyze(path)
            return self.analyze_result(analysis)
        from voicelab.core.audio_io import load_audio
        from voicelab.emotion.backbone import _run_backbone
        audio, sr = load_audio(path)
        return _run_backbone(audio, sr, self.config)

    def analyze_result(self, result: AnalysisResult) -> EmotionResult:
        """Existing AnalysisResult → EmotionResult via fusion (no audio reload)."""
        from voicelab.emotion.fusion import _run_fusion
        return _run_fusion(result, self.config)
