# VoiceLab SDK — Design Specification
## Module 2: Speech Emotion Recognition (SER)

**Depends on:** Module 1 (Voice Analysis) — ECAPA-TDNN embeddings, prosody, pitch, audio I/O, ModelRegistry, StreamEngine

---

## 1. Overview

Module 2 adds emotion recognition to the `voicelab` SDK. It produces per-file and per-frame emotion results with:
- Categorical labels (configurable, default 7: anger, disgust, fear, joy, sadness, surprise, neutral)
- Continuous VA-axes: valence (−1…+1) and arousal (−1…+1)
- Full probability distribution + dominant emotion list

Two inference paths share one output format:
- **Backbone path** — `wav2vec2-large` HuggingFace model, high accuracy, for offline use
- **Fusion path** — lightweight MLP on ECAPA embeddings + prosody features from Module 1, low latency, for real-time and fast offline

---

## 2. Public API

```python
import voicelab as vl

# Offline — accurate (backbone)
result = vl.analyze_emotion("speech.wav")
print(result.emotion, result.valence, result.scores)

# Offline — fast (fusion over Module 1 AnalysisResult)
result = vl.analyze_emotion("speech.wav", vl.EmotionConfig(fast=True))

# Real-time microphone
with vl.emotion_stream() as es:
    for frame in es:
        print(frame.emotion, frame.confidence)
```

New public symbols: `analyze_emotion`, `emotion_stream`, `EmotionConfig`, `EmotionResult`, `EmotionFrame`

---

## 3. Schema

Added to `voicelab/schema.py`:

```python
@dataclass
class EmotionResult:
    emotion: str                    # top-1 label
    confidence: float               # probability of top-1 (0.0–1.0)
    scores: dict[str, float]        # all labels → probability (sum ≈ 1.0)
    valence: float                  # −1.0 (negative) … +1.0 (positive)
    arousal: float                  # −1.0 (calm) … +1.0 (excited)
    dominant_emotions: list[str]    # all labels with score > dominant_threshold
    path: str                       # "backbone" | "fusion" | "fusion-lite"

@dataclass
class EmotionFrame:
    """Emitted per chunk in real-time emotion streaming."""
    timestamp: float
    emotion: str
    confidence: float
    scores: dict[str, float]
    valence: float
    arousal: float
    dominant_emotions: list[str]

@dataclass
class EmotionConfig:
    fast: bool = False              # True → fusion path; False → backbone path
    emotion_labels: list[str] = field(default_factory=lambda: [
        "anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"
    ])
    dominant_threshold: float = 0.2
    device: str = field(default_factory=lambda: "cuda" if __import__("torch").cuda.is_available() else "cpu")
```

**Invariants enforced at construction:**
- `sum(scores.values()) ≈ 1.0`
- `emotion in scores`
- `confidence == scores[emotion]`
- `−1.0 ≤ valence ≤ 1.0`, `−1.0 ≤ arousal ≤ 1.0`
- `all(e in scores for e in dominant_emotions)`

---

## 4. Architecture

### 4.1 Package Structure

```
voicelab/
  emotion/
    __init__.py
    backbone.py     # HF wav2vec2-large → classification + VA regression
    fusion.py       # MLP on ECAPA embedding + prosody features
    analyzer.py     # EmotionAnalyzer — offline orchestrator
    stream.py       # EmotionStream — real-time over VoiceStream
```

### 4.2 Backbone Path (`backbone.py`)

- **Model:** `audeering/wav2vec2-large-robust-12-ft-emotion-age-gender` (HuggingFace)
- **Input:** raw audio `np.ndarray` at 16 kHz
- **Output:** logits over `n_labels` classes + 2 VA regression values
- **Registry key:** `"ser_backbone"`
- **Function:** `_run_backbone(audio: np.ndarray, sr: int, config: EmotionConfig) -> EmotionResult`

### 4.3 Fusion Path (`fusion.py`)

- **Input vector (198-dim):** ECAPA embedding (192) + prosody scalars (tempo_bpm, pause_ratio, f0_mean, f0_std, hnr, rms_mean = 6)
- **MLP architecture:** `Linear(198→128) → ReLU → Linear(128→64) → ReLU` → two heads:
  - Classification: `Linear(64→n_labels)` → softmax
  - VA regression: `Linear(64→2)` → tanh
- **Weights:** randomly initialised (placeholder until fine-tuning); outputs are valid distributions but not semantically accurate
- **Registry key:** `"ser_fusion"`
- **Function:** `_run_fusion(result: AnalysisResult, config: EmotionConfig) -> EmotionResult`

### 4.4 Fusion-Lite Path (real-time)

Used inside `EmotionStream` where a full `AnalysisResult` is not available per-chunk. Input vector (15-dim): MFCC mean (13) + energy (1) + pitch (1) from `FrameResult`. MLP: `Linear(15→32) → ReLU → [Linear(32→n_labels) | Linear(32→2)]`. Same output schema. `path="fusion-lite"`.

### 4.5 Path Selection

| Call | Path |
|---|---|
| `analyze(path)` with `fast=False` | backbone |
| `analyze(path)` with `fast=True` | fusion (runs Module 1 internally) |
| `analyze_result(result: AnalysisResult)` | fusion |
| `EmotionStream` per-chunk | fusion-lite |

### 4.6 Shared Helper

`_build_emotion_result(logits, va, config, path) -> EmotionResult` — applies softmax, clips VA to [−1, 1], builds `scores` dict, selects top-1, filters `dominant_emotions` by threshold. Used by both backbone and fusion to guarantee identical output structure.

---

## 5. EmotionAnalyzer (Offline Orchestrator)

```python
class EmotionAnalyzer:
    def __init__(self, config: EmotionConfig | None = None) -> None

    def analyze(self, path: str) -> EmotionResult:
        """File → EmotionResult. Backbone if fast=False, fusion if fast=True."""

    def analyze_result(self, result: AnalysisResult) -> EmotionResult:
        """Existing AnalysisResult → EmotionResult via fusion (no re-loading audio)."""
```

`analyze(fast=True)` internally creates a `VoiceAnalyzer(Config(neural=True))` and calls `analyze_result()` on the output — no duplication of audio loading logic.

---

## 6. EmotionStream (Real-Time)

```python
class EmotionStream:
    def __init__(self, config: EmotionConfig | None = None, sr: int = 16000) -> None

    def _process_source(self, source: Iterator[np.ndarray]) -> Generator[EmotionFrame, None, None]

    # Context manager and __iter__ delegate to internal VoiceStream
    def __enter__(self) / __exit__(self, *_)
    def __iter__(self) -> Generator[EmotionFrame, None, None]
    def stop(self)
```

`EmotionStream` wraps `VoiceStream(Config(neural=False))` internally. Each `FrameResult` from `VoiceStream` is converted to an `EmotionFrame` via the fusion-lite MLP. No sounddevice dependency beyond what `VoiceStream` already handles.

---

## 7. ModelRegistry Keys

| Key | Loaded by | Content |
|---|---|---|
| `"ser_backbone"` | `backbone.py` module-level | `(processor, model)` tuple |
| `"ser_fusion"` | `fusion.py` module-level | `nn.Module` (MLP, eval mode) |
| `"ser_fusion_lite"` | `fusion.py` module-level | `nn.Module` (smaller MLP) |

All three lazy-loaded on first `get()` call, consistent with Module 1 pattern.

---

## 8. Testing Strategy

All tests CPU-only, no real model weights, consistent with Module 1 CI pattern.

| File | What it tests |
|---|---|
| `test_emotion_schema.py` | Dataclasses instantiate; invariants hold |
| `test_emotion_backbone.py` | `_run_backbone` with mock model → valid `EmotionResult`, `path="backbone"` |
| `test_emotion_fusion.py` | `_run_fusion` with mock MLP → valid `EmotionResult`, `path="fusion"` |
| `test_emotion_analyzer.py` | `EmotionAnalyzer.analyze()` both paths; `analyze_result()` |
| `test_emotion_stream.py` | `EmotionStream._process_source()` → `EmotionFrame` list, timestamps increase |
| `test_public_emotion_api.py` | `vl.analyze_emotion()`, `vl.emotion_stream()` surface tests |

**Key invariants checked in every test:**
- `abs(sum(result.scores.values()) - 1.0) < 1e-5`
- `result.emotion in result.scores`
- `result.confidence == result.scores[result.emotion]`
- `−1.0 ≤ result.valence ≤ 1.0`
- `−1.0 ≤ result.arousal ≤ 1.0`
- `all(s > config.dominant_threshold for s in [result.scores[e] for e in result.dominant_emotions])`

---

## 9. Dependencies

No new packages required — all already in `pyproject.toml`:
- `transformers` — backbone model loading
- `torch` — MLP inference
- `numpy` — vector operations

---

## 10. Out of Scope

- Fine-tuning or training on labelled emotion datasets
- Multi-speaker emotion tracking (diarization)
- Emotion intensity over time (temporal smoothing)
- Language-specific emotion models
