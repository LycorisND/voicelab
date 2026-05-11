# VoiceLab

Deep voice analysis SDK for Python — acoustics, prosody, speaker profiling, emotion, and health.

## Installation

```bash
pip install -e .
```

Requires Python 3.11+, PyTorch, and Transformers.

## Quick Start

### Full voice analysis

```python
import voicelab as vl

result = vl.analyze("speech.wav")

print(result.pitch.f0_mean)           # 185.3 Hz
print(result.pitch.jitter_local)      # 0.004
print(result.spectral.centroid_mean)  # 2400.0 Hz
print(result.prosody.tempo_bpm)       # 142.0
print(result.speaker.gender)          # "M" or "F"
print(result.speaker.age_range)       # "25-35"
print(result.speaker.language)        # "en"
print(result.health.dysphonia_score)  # 0.0–1.0
print(result.metadata.snr_db)         # 34.2
```

### Emotion recognition

```python
import voicelab as vl

# Backbone path — high accuracy, uses wav2vec2 (default)
result = vl.analyze_emotion("speech.wav")
print(result.emotion)            # "joy"
print(result.confidence)         # 0.61
print(result.valence)            # 0.72   (−1.0 … +1.0)
print(result.arousal)            # 0.55   (−1.0 … +1.0)
print(result.dominant_emotions)  # ["joy"]
print(result.scores)             # {"anger": 0.05, "joy": 0.61, ...}

# Fusion path — low latency, uses ECAPA + MLP
fast_result = vl.analyze_emotion("speech.wav", vl.EmotionConfig(fast=True))
```

### Real-time streaming

```python
import voicelab as vl

# Acoustic features per chunk
with vl.stream() as s:
    for frame in s:
        print(frame.timestamp, frame.pitch, frame.energy, frame.is_voiced)

# Emotion per chunk
with vl.emotion_stream() as s:
    for frame in s:
        print(frame.timestamp, frame.emotion, frame.valence, frame.arousal)
```

## API Reference

### Functions

| Function | Returns | Description |
|---|---|---|
| `analyze(path, config?)` | `AnalysisResult` | Full offline analysis of an audio file |
| `stream(config?)` | `VoiceStream` | Context manager for real-time microphone streaming |
| `analyze_emotion(path, config?)` | `EmotionResult` | Emotion recognition from an audio file |
| `emotion_stream(config?)` | `EmotionStream` | Context manager for real-time emotion streaming |

### Config

```python
@dataclass
class Config:
    device: str = "cpu" | "cuda"   # auto-detected
    neural: bool = True             # enable neural models (embeddings, age/gender, health)
    n_mfcc: int = 13
    chunk_size: int = 512           # samples per streaming chunk (32ms @ 16kHz)
    hop_size: int = 256             # 50% overlap

@dataclass
class EmotionConfig:
    fast: bool = False              # True → fusion MLP path; False → wav2vec2 backbone
    emotion_labels: list[str] = ["anger", "disgust", "fear", "joy",
                                  "sadness", "surprise", "neutral"]
    dominant_threshold: float = 0.2 # labels above this score go into dominant_emotions
    device: str = "cpu" | "cuda"   # auto-detected
```

### Result dataclasses

#### AnalysisResult

```python
@dataclass
class AnalysisResult:
    duration: float
    pitch: PitchFeatures
    spectral: SpectralFeatures
    prosody: ProsodyFeatures
    speaker: SpeakerProfile
    health: HealthIndicators
    metadata: AudioMetadata
```

#### PitchFeatures

```python
@dataclass
class PitchFeatures:
    f0_mean: float          # mean fundamental frequency, Hz
    f0_std: float           # standard deviation of F0
    f0_min: float
    f0_max: float
    f0_contour: np.ndarray  # per-frame F0 values
    jitter_local: float     # cycle-to-cycle F0 variation (0.0–1.0)
    shimmer_local: float    # cycle-to-cycle amplitude variation (0.0–1.0)
    hnr: float              # harmonics-to-noise ratio, dB
    voiced_fraction: float  # fraction of voiced frames (0.0–1.0)
```

#### SpectralFeatures

```python
@dataclass
class SpectralFeatures:
    mfcc: np.ndarray        # shape [n_frames, n_mfcc]
    centroid_mean: float    # spectral centroid, Hz
    rolloff_mean: float     # spectral rolloff, Hz
    zcr_mean: float         # zero-crossing rate
    rms_mean: float         # RMS amplitude
```

#### ProsodyFeatures

```python
@dataclass
class ProsodyFeatures:
    tempo_bpm: float
    pause_ratio: float      # fraction of signal that is silence
    pause_count: int
    energy_profile: np.ndarray
```

#### SpeakerProfile

```python
@dataclass
class SpeakerProfile:
    embedding: np.ndarray   # 192-dim ECAPA-TDNN speaker vector
    gender: "M" | "F" | "unknown"
    age_range: str          # "0-18" | "18-25" | "25-35" | "35-50" | "50+"
    language: str           # ISO 639-1 code, e.g. "en", "ru", "ja"
    accent: str
```

Age estimation blends a neural model with an F0-based acoustic heuristic. When the
two estimates disagree by more than 15 years (common in expressive/animated speech),
the blend shifts 60% toward the neural estimate and 40% toward the acoustic signal.

#### HealthIndicators

```python
@dataclass
class HealthIndicators:
    dysphonia_score: float  # 0.0–1.0; voice disorder likelihood
    fatigue_index: float    # 0.0–1.0
    hoarseness: float       # 0.0–1.0
    pathology_flags: list[str]
```

#### EmotionResult

```python
@dataclass
class EmotionResult:
    emotion: str                 # top-1 label
    confidence: float            # probability of top-1
    scores: dict[str, float]     # all labels → probability (sum ≈ 1.0)
    valence: float               # −1.0 … +1.0  (Russell's circumplex)
    arousal: float               # −1.0 … +1.0
    dominant_emotions: list[str] # labels with score > dominant_threshold
    path: str                    # "backbone" | "fusion"
```

#### EmotionFrame

```python
@dataclass
class EmotionFrame:
    timestamp: float
    emotion: str
    confidence: float
    scores: dict[str, float]
    valence: float
    arousal: float
    dominant_emotions: list[str]
    path: str                    # "backbone" | "fusion" | "fusion-lite"
```

#### FrameResult (streaming)

```python
@dataclass
class FrameResult:
    timestamp: float
    pitch: float        # F0 in Hz; float('nan') if unvoiced
    energy: float       # RMS amplitude
    is_voiced: bool
    mfcc: np.ndarray    # shape [n_mfcc]
```

## Architecture

```
voicelab/
├── core/
│   ├── audio_io.py         # load_audio, detect_clipping, estimate_snr
│   └── model_registry.py   # lazy-loading singleton registry
├── dsp/
│   ├── pitch.py            # RAPT-based F0 + jitter/shimmer/HNR
│   ├── spectral.py         # MFCC, centroid, rolloff, ZCR
│   └── prosody.py          # tempo, pauses, energy profile
├── neural/
│   ├── embeddings.py       # ECAPA-TDNN 192-dim speaker embedding
│   ├── speaker_profile.py  # gender/age (audeering) + language (Whisper)
│   └── health.py           # dysphonia, fatigue, hoarseness
├── emotion/
│   ├── backbone.py         # wav2vec2-large SER → VA → categorical
│   ├── fusion.py           # MLP(198) offline + MLP(15) real-time
│   ├── analyzer.py         # EmotionAnalyzer — offline orchestrator
│   └── stream.py           # EmotionStream — real-time orchestrator
├── analysis/
│   ├── voice_analyzer.py   # VoiceAnalyzer — offline orchestrator
│   └── voice_stream.py     # VoiceStream — real-time orchestrator
└── schema.py               # all public dataclasses
```

### Inference paths

**Backbone (default, `fast=False`):**
`wav2vec2-large-robust-12-ft-emotion-msp-dim` → [arousal, dominance, valence] →
inverse-distance to Russell's circumplex centroids → softmax categorical scores.

**Fusion (`fast=True`):**
ECAPA-TDNN embedding (192) + prosody features (6) = 198-dim →
randomly initialised MLP(198→128→64→[n_labels, 2]).
Intended for fine-tuning on own labelled data.

**Fusion-Lite (streaming):**
MFCC mean (13) + energy (1) + pitch (1) = 15-dim →
MLP(15→32→[n_labels, 2]). Runs on every 32ms chunk.

## Models used

| Model | License | Purpose |
|---|---|---|
| `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` | CC-BY-NC-4.0 | Backbone SER — replace before commercial use |
| `audeering/wav2vec2-large-robust-24-ft-age-gender` | CC-BY-NC-SA-4.0 | Age/gender — replace before commercial use |
| `openai/whisper-small` | Apache 2.0 | Language identification |
| `speechbrain/spkrec-ecapa-voxceleb` | Apache 2.0 | Speaker embeddings |

## Running tests

```bash
pytest tests/ -q
# 95 passed
```
