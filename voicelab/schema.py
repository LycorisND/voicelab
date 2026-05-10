from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import numpy as np


@dataclass
class PitchFeatures:
    f0_mean: float
    f0_std: float
    f0_min: float
    f0_max: float
    f0_contour: np.ndarray
    jitter_local: float
    shimmer_local: float
    hnr: float
    voiced_fraction: float


@dataclass
class SpectralFeatures:
    mfcc: np.ndarray          # shape [n_frames, n_mfcc]
    centroid_mean: float
    rolloff_mean: float
    zcr_mean: float
    rms_mean: float


@dataclass
class ProsodyFeatures:
    tempo_bpm: float
    pause_ratio: float
    pause_count: int
    energy_profile: np.ndarray


@dataclass
class SpeakerProfile:
    embedding: np.ndarray     # 192-dim ECAPA-TDNN vector
    gender: Literal["M", "F", "unknown"]
    age_range: str            # e.g. "25-35"
    language: str             # ISO 639-1 e.g. "en", "ru"
    accent: str


@dataclass
class HealthIndicators:
    dysphonia_score: float    # 0.0–1.0
    fatigue_index: float      # 0.0–1.0
    hoarseness: float         # 0.0–1.0
    pathology_flags: list[str] = field(default_factory=list)


@dataclass
class AudioMetadata:
    sample_rate: int
    duration: float
    snr_db: float
    clipping_detected: bool
    mono: bool


@dataclass
class AnalysisResult:
    duration: float           # same value as metadata.duration; kept at top level for convenience
    pitch: PitchFeatures
    spectral: SpectralFeatures
    prosody: ProsodyFeatures
    speaker: SpeakerProfile
    health: HealthIndicators
    metadata: AudioMetadata


@dataclass
class FrameResult:
    """Emitted per chunk in real-time mode."""
    timestamp: float
    pitch: float              # F0 in Hz; float('nan') if unvoiced
    energy: float             # RMS amplitude
    is_voiced: bool
    mfcc: np.ndarray          # shape [n_mfcc]


@dataclass
class Config:
    device: str = field(default_factory=lambda: "cuda" if __import__("torch").cuda.is_available() else "cpu")
    neural: bool = True       # Whether to run neural models
    n_mfcc: int = 13
    chunk_size: int = 512     # samples per chunk (32ms @ 16kHz)
    hop_size: int = 256       # 50% overlap
