# Module 3: Voice Cloning (TTS) — Design Spec

**Date:** 2026-05-11  
**Status:** Approved

---

## 1. Overview

Module 3 adds voice cloning to the voicelab SDK. A user provides a reference audio file; the SDK extracts a language-agnostic **VoicePreset** capturing the speaker's timbral identity, emotional state, and prosody style. The preset can then be used to synthesize arbitrary text in any supported language using that speaker's voice and emotion.

**Target users:** Python developers who need to replicate a specific voice (e.g., a voice actor, a customer support agent) across languages or scripts.

**Module 4 (real-time pipeline)** builds on top of this: stream in speech → transcribe → translate → synthesize with Module 3. That is out of scope here.

---

## 2. Public API

```python
import voicelab as vl

# Extract voice preset from reference audio
preset = vl.create_preset("reference.wav", name="xinli")

# Inspect captured state
print(preset.emotion.emotion)   # "surprise"
print(preset.emotion.arousal)   # 1.0
print(preset.language)          # "ja"

# Persist and restore
preset.save("xinli.vpreset")
preset = vl.VoicePreset.load("xinli.vpreset")

# Synthesize text in any supported language
audio = vl.synthesize("Привет, как дела?", preset, lang="ru")
# → np.ndarray, float32, 16 kHz mono

# Save to file
vl.save_audio(audio, sr=16000, path="output.wav")
```

**Supported `lang` values:** `"en"`, `"ru"`, `"zh"`, `"ja"`, `"es"`, `"fr"`, `"ko"` (MeloTTS languages).

---

## 3. New Dataclasses (schema.py)

### VoicePreset

```python
@dataclass
class VoicePreset:
    name: str                       # user-assigned label
    voice_embedding: np.ndarray     # float32[192] — ECAPA-TDNN speaker vector
    tone_color: np.ndarray          # float32[256] — OpenVoice SE vector
    emotion: EmotionResult          # captured via Module 2 backbone
    prosody: ProsodyFeatures        # tempo, pauses, energy_profile, f0_contour
    language: str                   # ISO 639-1 source language of reference audio
    created_at: str                 # ISO 8601 timestamp

    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> "VoicePreset": ...
```

### SynthesisConfig

```python
@dataclass
class SynthesisConfig:
    lang: str = "en"                # target synthesis language
    speed: float = 1.0              # speech rate multiplier (0.5–2.0)
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
```

---

## 4. File Structure

```
voicelab/cloning/
├── __init__.py          # empty package marker
├── extractor.py         # create_preset: audio → VoicePreset
└── synthesizer.py       # synthesize: (text, preset, config) → np.ndarray
```

Modifications:
- `voicelab/schema.py` — add VoicePreset, SynthesisConfig
- `voicelab/__init__.py` — expose create_preset, synthesize, save_audio, VoicePreset, SynthesisConfig

---

## 5. Inference Pipelines

### create_preset (extractor.py)

```
VoiceAnalyzer(neural=True).analyze(path)
    → AnalysisResult.speaker.embedding   # ECAPA-TDNN, 192-dim
    → AnalysisResult.speaker.language    # ISO 639-1 via Whisper
    → AnalysisResult.prosody             # ProsodyFeatures

load_audio(path)
    → _extract_tone_color(audio, sr)     # OpenVoice SE extractor → 256-dim

analyze_emotion(path)                    # Module 2 backbone → EmotionResult

→ VoicePreset(name, voice_embedding, tone_color, emotion, prosody, language, created_at)
```

`analyze()` and `analyze_emotion()` both call `load_audio` internally; audio is loaded twice (acceptable — files are short reference clips, typically < 30 s).

### synthesize (synthesizer.py)

```
MeloTTS(text, lang, speed)       # base TTS → raw audio (neutral voice)
    ↓
OpenVoiceConverter(raw, tone_color)   # apply speaker timbral identity
    ↓
_adjust_prosody(audio, preset.prosody)  # scale tempo and pitch to match preset stats
→ np.ndarray (float32, 16 kHz)
```

`_adjust_prosody` applies a simple time-stretch + pitch-shift using `librosa` to bring the output closer to the preset's `tempo_bpm` and `f0_mean`.

---

## 6. Preset Serialization (.vpreset)

A `.vpreset` file is a NumPy `.npz` archive:

```
xinli.vpreset  (npz)
├── voice_embedding.npy    # float32[192]
├── tone_color.npy         # float32[256]
├── energy_profile.npy     # float32[n]   (prosody)
├── f0_contour.npy         # float32[n]   (prosody)
└── meta.json              # all scalar fields: name, language, created_at,
                           #   prosody scalars, emotion dict
```

`meta.json` is stored as a bytes entry inside the `.npz`. The file can be read with `numpy.load` alone — no OpenVoice import required for loading.

---

## 7. ModelRegistry Entries

```python
ModelRegistry.instance().register("se_extractor", _load_se_extractor)  # OpenVoice
ModelRegistry.instance().register("vc_converter", _load_vc_converter)  # OpenVoice
ModelRegistry.instance().register("melo_tts",     _load_melo_tts)      # MeloTTS
```

All three use lazy loading, consistent with existing pattern in Module 1 & 2.

---

## 8. Dependencies

| Package | License | Purpose |
|---|---|---|
| `openvoice` | MIT | Tone color (SE) extraction + voice conversion |
| `melo` | MIT | Multi-lingual base TTS |
| `librosa` | ISC | Tempo/pitch adjustment in `_adjust_prosody` |

`librosa` is already a common audio dependency; verify it is in `pyproject.toml`.

---

## 9. Testing Strategy

All unit tests use mock ModelRegistry entries — no real model downloads required.

| Test file | What it covers |
|---|---|
| `tests/test_cloning_schema.py` | VoicePreset construction, save/load round-trip, SynthesisConfig defaults |
| `tests/test_cloning_extractor.py` | create_preset with mocked SE extractor, emotion, prosody |
| `tests/test_cloning_synthesizer.py` | synthesize with mocked MeloTTS + converter; output shape, dtype, sr |
| `tests/test_public_cloning_api.py` | vl.create_preset, vl.synthesize, vl.save_audio surface tests |

Integration test with real audio (`xinli_0020.wav`) is marked `@pytest.mark.slow` and excluded from default `pytest` run.

---

## 10. Out of Scope (this module)

- Real-time voice cloning / streaming synthesis (Module 4)
- Fine-tuning or training any model
- Multi-speaker presets
- Preset versioning or migration
- REST API or GUI
