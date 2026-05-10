# VoiceLab SDK — Design Specification

**Date:** 2026-05-11  
**Status:** Approved  
**Scope:** Module 1 — Voice Analysis (foundation for SER, VC, TTS modules)

---

## 1. Overview

`voicelab` is a Python SDK for deep voice, speech, and emotion analysis, plus voice cloning and conversion. This spec covers **Module 1: Voice Analysis** — the foundational layer that all subsequent modules will depend on.

**Target users:** Python developers integrating voice analysis into their own systems.  
**Hardware:** Local NVIDIA GPU (primary), CPU fallback supported.  
**Processing modes:** Both offline (file-based) and real-time (microphone stream).

---

## 2. Public API

The public surface is minimal — complexity is hidden inside.

```python
import voicelab as vl

# Offline: file → full analysis result
result = vl.analyze("speech.wav")
print(result.pitch.f0_mean)        # e.g. 185.3 Hz
print(result.speaker.gender)       # "M" or "F"
print(result.health.dysphonia_score)  # 0.0–1.0

# Real-time: microphone → streaming frames
with vl.stream() as s:
    for frame in s:
        print(frame.pitch.f0, frame.energy)

# Configuration override
result = vl.analyze("speech.wav", config=vl.Config(device="cpu", neural=False))
```

---

## 3. Architecture

### 3.1 Layer Overview

```
┌─────────────────────────────────────────────────────┐
│  Public API  (VoiceAnalyzer, VoiceStream)           │
├──────────────────────┬──────────────────────────────┤
│  DSP Layer           │  Neural Layer                │
│  (librosa,           │  (HuBERT, Wav2Vec2,          │
│   parselmouth,       │   ECAPA-TDNN, SpeechBrain)   │
│   torchcrepe)        │                              │
├──────────────────────┴──────────────────────────────┤
│  Stream Engine  (unified chunk-based processor)     │
│  Offline: file → chunks                             │
│  Real-time: microphone → chunks                     │
├─────────────────────────────────────────────────────┤
│  Model Registry  (lazy load, in-memory cache, ONNX) │
└─────────────────────────────────────────────────────┘
```

### 3.2 Package Structure

```
voicelab/
├── __init__.py             # vl.analyze(), vl.stream(), vl.Config
├── core/
│   ├── stream_engine.py    # Unified chunk-based processing engine
│   ├── model_registry.py   # Lazy model loading, caching, ONNX export
│   └── audio_io.py         # File loading and microphone capture
├── dsp/
│   ├── pitch.py            # F0 (CREPE + pYIN), jitter, shimmer, HNR
│   ├── formants.py         # LPC-based formant analysis (F1–F4)
│   ├── spectral.py         # MFCC, centroid, rolloff, ZCR, RMS
│   └── prosody.py          # Tempo, pauses, rhythm, energy profile
├── neural/
│   ├── embeddings.py       # HuBERT speaker embeddings
│   ├── speaker_profile.py  # ECAPA-TDNN: gender, age, language, accent
│   └── health.py           # Voice pathology and fatigue detection
├── analysis/
│   ├── voice_analyzer.py   # Offline analysis orchestrator
│   └── voice_stream.py     # Real-time stream orchestrator
├── models/                 # Downloaded weights (gitignored)
└── utils/
    ├── visualization.py    # Pitch contour, spectrogram plots
    └── export.py           # JSON, CSV, HDF5 output
```

---

## 4. Stream Engine

The Stream Engine unifies offline and real-time processing. The same analysis code runs in both modes.

### 4.1 Data Flow

```
AudioSource (File / Microphone)
    │
    ▼
ChunkBuffer (sliding window, 50% overlap)
    │
    ├──▶ DSP Pipeline (synchronous, per-chunk, numpy/librosa, <5ms)
    │        └──▶ F0, MFCC, energy, formants, prosody markers
    │
    ├──▶ Neural Pipeline (async, batched, GPU)
    │        └──▶ HuBERT embeddings → speaker profile, health
    │
    ▼
FrameResult (dataclass)
    │
    ▼  (offline only)
Aggregation → AnalysisResult
```

### 4.2 Chunk Parameters (defaults)

| Parameter | Value | Rationale |
|---|---|---|
| Sample rate | 16 kHz | Standard for all speech SSL models |
| Chunk size | 512 samples (32ms) | Low enough for real-time, enough for pitch |
| Overlap | 50% | Smooth pitch/formant continuity across chunks |
| Neural batch | max 30 sec | Prevents GPU OOM on long files |

### 4.3 Threading Model

- DSP pipeline: synchronous in main thread (< 5ms per chunk)
- Neural pipeline: runs in a dedicated thread with a queue
- Real-time: producer (sounddevice callback) → queue → consumer (engine)
- Offline: file read in chunks, same consumer code path

---

## 5. Models

### 5.1 DSP Layer (no GPU required)

| Feature | Method | Library |
|---|---|---|
| F0 / Pitch | CREPE (neural) + pYIN (DSP) | `torchcrepe`, `librosa` |
| Jitter, Shimmer | Period-by-period analysis | `parselmouth` (Praat) |
| Formants F1–F4 | LPC Levinson-Durbin | `parselmouth` |
| MFCC (13–40 coeff.) | Mel-filterbank + DCT | `librosa` |
| Spectral centroid/rolloff | FFT-based | `librosa` |
| ZCR, Energy, RMS | Frame-level calculation | `numpy` |
| Pauses, Tempo | VAD + beat tracking | `webrtcvad`, `librosa` |

### 5.2 Neural Layer (GPU, lazy-loaded)

| Task | Model | Size | Notes |
|---|---|---|---|
| Speaker embeddings | `speechbrain/spkrec-ecapa-voxceleb` | 80 MB | EER 0.87% on VoxCeleb |
| Gender + Age | Fine-tuned Wav2Vec2 | 300 MB | ~90% gender accuracy |
| Language / Accent | `speechbrain/lang-id-voxlingua107-ecapa` | 80 MB | 107 languages |
| Voice pathologies | HuBERT-base + classifier | 360 MB | Saarbrücken Voice DB |
| General embeddings | `facebook/hubert-large-ls960-ft` | 1.2 GB | SOTA on SUPERB |

All models load on first use and stay cached in memory. Models are never downloaded at `import voicelab`.

---

## 6. Result Schema

```python
@dataclass
class AnalysisResult:
    duration: float

    pitch: PitchFeatures
    # f0_mean, f0_std, f0_min, f0_max, f0_contour (np.ndarray)
    # jitter_local, shimmer_local, HNR, voiced_fraction

    spectral: SpectralFeatures
    # mfcc (np.ndarray, shape [n_frames, n_mfcc])
    # centroid_mean, rolloff_mean, zcr_mean, rms_mean

    prosody: ProsodyFeatures
    # tempo_bpm, pause_ratio, pause_count, energy_profile (np.ndarray)

    speaker: SpeakerProfile
    # embedding (np.ndarray, 192-dim)
    # gender: Literal["M", "F", "unknown"]
    # age_range: str  e.g. "25-35"
    # language: str   e.g. "ru", "en"
    # accent: str     e.g. "standard", "southern"

    health: HealthIndicators
    # dysphonia_score: float  [0.0–1.0]
    # fatigue_index: float    [0.0–1.0]
    # hoarseness: float       [0.0–1.0]
    # pathology_flags: list[str]

    metadata: AudioMetadata
    # sample_rate, duration, snr_db, clipping_detected, mono

@dataclass
class FrameResult:
    """Emitted per chunk in real-time mode."""
    timestamp: float
    pitch: float          # F0 in Hz, NaN if unvoiced
    energy: float         # RMS
    is_voiced: bool
    mfcc: np.ndarray      # shape [n_mfcc]
```

---

## 7. Dependencies

```toml
[tool.poetry.dependencies]
python = ">=3.10"
torch = ">=2.1"
torchaudio = ">=2.1"
transformers = ">=4.40"
speechbrain = ">=1.0"
librosa = ">=0.10"
parselmouth = ">=0.4"
webrtcvad = ">=2.0"
torchcrepe = ">=0.0.20"
sounddevice = ">=0.4"
numpy = ">=1.24"
```

---

## 8. Testing Strategy

### 8.1 Unit Tests (pytest, CPU-only, runs in CI)

- DSP functions tested on synthetic signals with known ground truth
  - Sine wave at 220 Hz → verify F0 detection within ±2 Hz
  - White noise → verify voiced fraction ≈ 0
- Model Registry tested with mock models — no real weights downloaded in CI

### 8.2 Integration Tests

Five fixture WAV files in `tests/fixtures/`:
- `male_clean.wav` — clean male speech
- `female_clean.wav` — clean female speech
- `pathology_dysphonia.wav` — voice with dysphonia
- `silence.wav` — silence only
- `clipping.wav` — overloaded audio

Each fixture has expected output ranges checked after `vl.analyze()`.

### 8.3 Real-Time Test

Synthetic stream: `sounddevice.play(fixture)` → `vl.stream()` → compare frame aggregates with offline result (within 5% tolerance).

### 8.4 CI/CD

- GitHub Actions: `ruff` lint + unit tests on CPU (no GPU required)
- GPU integration tests: separate workflow, triggered manually

---

## 9. Roadmap (future modules)

This spec covers Module 1 only. Subsequent modules build on top of it:

| Module | Depends on |
|---|---|
| Module 2: SER (Speech Emotion Recognition) | Voice Analysis embeddings + prosody |
| Module 3: Voice Cloning (TTS) | Speaker embeddings + audio I/O |
| Module 4: Voice Conversion (VC) | Speaker embeddings + Stream Engine |

---

## 10. Out of Scope (this spec)

- Training or fine-tuning of any model
- Web API / REST server
- GUI or desktop application
- Speaker diarization (identifying multiple speakers)
- Transcription / ASR
