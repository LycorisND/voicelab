# Module 3: Voice Cloning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `vl.create_preset(path, name)` and `vl.synthesize(text, preset, lang)` to the voicelab SDK — extract a language-agnostic VoicePreset from reference audio and synthesize new speech with that voice and emotion.

**Architecture:** OpenVoice v2 (MIT) extracts a tone color SE vector from the reference speaker; MeloTTS (MIT) synthesizes base audio in the target language; OpenVoice's ToneColorConverter applies the stored tone color. Preset is serialized as a ZIP archive (`.vpreset`) containing numpy arrays + JSON metadata. All models loaded lazily via ModelRegistry.

**Tech Stack:** `openvoice`, `melo-tts`, `librosa` (already in pyproject.toml), `torch`, `numpy`, `voicelab.core.model_registry.ModelRegistry`, `voicelab.analysis.voice_analyzer.VoiceAnalyzer`, `voicelab.emotion` (Module 2)

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Modify | `voicelab/schema.py` | Add `VoicePreset` (with save/load), `SynthesisConfig` |
| Create | `voicelab/cloning/__init__.py` | Empty package marker |
| Create | `voicelab/cloning/extractor.py` | `create_preset`, `_extract_tone_color`, registry entry `"vc_converter"` |
| Create | `voicelab/cloning/synthesizer.py` | `synthesize`, `_adjust_pitch`, registry entry `"melo_tts"` |
| Modify | `voicelab/core/audio_io.py` | Add `save_audio(audio, sr, path)` |
| Modify | `voicelab/__init__.py` | Expose `create_preset`, `synthesize`, `save_audio`, `VoicePreset`, `SynthesisConfig` |
| Modify | `pyproject.toml` | Add `openvoice`, `melo-tts`, `huggingface-hub` dependencies |
| Create | `tests/test_cloning_schema.py` | VoicePreset construction + save/load round-trip, SynthesisConfig defaults |
| Create | `tests/test_cloning_extractor.py` | `create_preset` and `_extract_tone_color` with mock registry |
| Create | `tests/test_cloning_synthesizer.py` | `synthesize` and `_adjust_pitch` with mock registry |
| Create | `tests/test_public_cloning_api.py` | Surface tests for public symbols |

---

## Task 1: Schema — VoicePreset + SynthesisConfig + serialization

**Files:**
- Modify: `voicelab/schema.py`
- Create: `tests/test_cloning_schema.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cloning_schema.py
import io, json, zipfile
import numpy as np
import pytest
from voicelab.schema import (
    VoicePreset, SynthesisConfig, EmotionResult, ProsodyFeatures,
)


def _make_emotion() -> EmotionResult:
    return EmotionResult(
        emotion="joy", confidence=0.6,
        scores={"joy": 0.6, "neutral": 0.4},
        valence=0.5, arousal=0.4,
        dominant_emotions=["joy"], path="backbone",
    )


def _make_prosody() -> ProsodyFeatures:
    return ProsodyFeatures(
        tempo_bpm=130.0, pause_ratio=0.1, pause_count=3,
        energy_profile=np.ones(100, dtype=np.float32),
    )


def _make_preset(**overrides) -> VoicePreset:
    defaults = dict(
        name="test",
        voice_embedding=np.zeros(192, dtype=np.float32),
        tone_color=np.zeros(256, dtype=np.float32),
        emotion=_make_emotion(),
        prosody=_make_prosody(),
        f0_mean=180.0,
        language="en",
        created_at="2026-05-11T00:00:00+00:00",
    )
    return VoicePreset(**{**defaults, **overrides})


def test_voice_preset_construction():
    p = _make_preset()
    assert p.name == "test"
    assert p.voice_embedding.shape == (192,)
    assert p.tone_color.shape == (256,)
    assert p.language == "en"
    assert p.f0_mean == 180.0


def test_synthesis_config_defaults():
    cfg = SynthesisConfig()
    assert cfg.lang == "en"
    assert cfg.speed == 1.0
    assert cfg.device in ("cpu", "cuda")


def test_synthesis_config_custom():
    cfg = SynthesisConfig(lang="ja", speed=1.2)
    assert cfg.lang == "ja"
    assert cfg.speed == 1.2


def test_voice_preset_save_load_roundtrip(tmp_path):
    p = _make_preset()
    path = str(tmp_path / "test.vpreset")
    p.save(path)
    p2 = VoicePreset.load(path)
    assert p2.name == p.name
    assert p2.language == p.language
    assert p2.f0_mean == p.f0_mean
    assert p2.created_at == p.created_at
    np.testing.assert_array_equal(p2.voice_embedding, p.voice_embedding)
    np.testing.assert_array_equal(p2.tone_color, p.tone_color)
    np.testing.assert_array_equal(p2.prosody.energy_profile, p.prosody.energy_profile)


def test_voice_preset_save_load_emotion(tmp_path):
    p = _make_preset()
    path = str(tmp_path / "test.vpreset")
    p.save(path)
    p2 = VoicePreset.load(path)
    assert p2.emotion.emotion == "joy"
    assert p2.emotion.confidence == 0.6
    assert p2.emotion.valence == 0.5
    assert p2.emotion.dominant_emotions == ["joy"]


def test_voice_preset_vpreset_is_zip(tmp_path):
    p = _make_preset()
    path = str(tmp_path / "test.vpreset")
    p.save(path)
    assert zipfile.is_zipfile(path)
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    assert "voice_embedding.npy" in names
    assert "tone_color.npy" in names
    assert "energy_profile.npy" in names
    assert "meta.json" in names


def test_voice_preset_prosody_scalars_preserved(tmp_path):
    p = _make_preset()
    path = str(tmp_path / "test.vpreset")
    p.save(path)
    p2 = VoicePreset.load(path)
    assert p2.prosody.tempo_bpm == 130.0
    assert p2.prosody.pause_ratio == 0.1
    assert p2.prosody.pause_count == 3
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_cloning_schema.py -v
```

Expected: `ImportError: cannot import name 'VoicePreset' from 'voicelab.schema'`

- [ ] **Step 3: Add VoicePreset and SynthesisConfig to schema.py**

Append to `voicelab/schema.py` (after `EmotionConfig`):

```python
@dataclass
class SynthesisConfig:
    lang: str = "en"
    speed: float = 1.0
    device: str = field(
        default_factory=lambda: "cuda" if __import__("torch").cuda.is_available() else "cpu"
    )


@dataclass
class VoicePreset:
    name: str
    voice_embedding: np.ndarray    # float32[192] — ECAPA-TDNN
    tone_color: np.ndarray         # float32[256] — OpenVoice SE vector
    emotion: "EmotionResult"
    prosody: "ProsodyFeatures"
    f0_mean: float                 # from PitchFeatures.f0_mean
    language: str                  # ISO 639-1
    created_at: str                # ISO 8601

    def save(self, path: str) -> None:
        import io, json, zipfile
        meta = {
            "name": self.name,
            "language": self.language,
            "created_at": self.created_at,
            "f0_mean": self.f0_mean,
            "prosody": {
                "tempo_bpm": self.prosody.tempo_bpm,
                "pause_ratio": self.prosody.pause_ratio,
                "pause_count": self.prosody.pause_count,
            },
            "emotion": {
                "emotion": self.emotion.emotion,
                "confidence": self.emotion.confidence,
                "scores": self.emotion.scores,
                "valence": self.emotion.valence,
                "arousal": self.emotion.arousal,
                "dominant_emotions": self.emotion.dominant_emotions,
                "path": self.emotion.path,
            },
        }
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            for arr_name, arr in [
                ("voice_embedding", self.voice_embedding),
                ("tone_color", self.tone_color),
                ("energy_profile", self.prosody.energy_profile),
            ]:
                buf = io.BytesIO()
                np.save(buf, arr)
                zf.writestr(f"{arr_name}.npy", buf.getvalue())
            zf.writestr("meta.json", json.dumps(meta).encode("utf-8"))

    @staticmethod
    def load(path: str) -> "VoicePreset":
        import io, json, zipfile
        with zipfile.ZipFile(path, "r") as zf:
            voice_embedding = np.load(io.BytesIO(zf.read("voice_embedding.npy")))
            tone_color = np.load(io.BytesIO(zf.read("tone_color.npy")))
            energy_profile = np.load(io.BytesIO(zf.read("energy_profile.npy")))
            meta = json.loads(zf.read("meta.json").decode("utf-8"))
        prosody = ProsodyFeatures(
            tempo_bpm=meta["prosody"]["tempo_bpm"],
            pause_ratio=meta["prosody"]["pause_ratio"],
            pause_count=int(meta["prosody"]["pause_count"]),
            energy_profile=energy_profile.astype(np.float32),
        )
        emotion = EmotionResult(
            emotion=meta["emotion"]["emotion"],
            confidence=meta["emotion"]["confidence"],
            scores=meta["emotion"]["scores"],
            valence=meta["emotion"]["valence"],
            arousal=meta["emotion"]["arousal"],
            dominant_emotions=meta["emotion"]["dominant_emotions"],
            path=meta["emotion"]["path"],
        )
        return VoicePreset(
            name=meta["name"],
            voice_embedding=voice_embedding.astype(np.float32),
            tone_color=tone_color.astype(np.float32),
            emotion=emotion,
            prosody=prosody,
            f0_mean=float(meta["f0_mean"]),
            language=meta["language"],
            created_at=meta["created_at"],
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_cloning_schema.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add voicelab/schema.py tests/test_cloning_schema.py
git commit -m "feat: add VoicePreset and SynthesisConfig dataclasses with serialization"
```

---

## Task 2: Extractor — create_preset + vc_converter registry

**Files:**
- Create: `voicelab/cloning/__init__.py`
- Create: `voicelab/cloning/extractor.py`
- Create: `tests/test_cloning_extractor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cloning_extractor.py
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from voicelab.core.model_registry import ModelRegistry
from voicelab.schema import (
    VoicePreset, AnalysisResult, AudioMetadata,
    PitchFeatures, SpectralFeatures, ProsodyFeatures,
    SpeakerProfile, HealthIndicators, EmotionResult, EmotionConfig,
)
import voicelab.cloning.extractor  # noqa: F401 — triggers register()


@pytest.fixture(autouse=True)
def clean_registry():
    ModelRegistry.instance().clear()
    yield
    ModelRegistry.instance().clear()


def _mock_vc_converter():
    import torch

    class _MockConverter:
        def convert(self, audio_src_path, src_se, tgt_se, output_path, tau=0.3, message=""):
            import shutil
            shutil.copy(audio_src_path, output_path)

    class _MockSeExt:
        def get_se(self, audio_path, vc_model, target_dir="/tmp", vad=True):
            return torch.zeros(1, 256, 1), "mock"

    base_ses = {lang: torch.zeros(1, 256, 1) for lang in ["en", "zh", "ja", "es", "fr", "ko"]}
    return _MockConverter(), _MockSeExt(), base_ses


def _fake_analysis_result() -> AnalysisResult:
    return AnalysisResult(
        duration=2.0,
        pitch=PitchFeatures(180.0, 10.0, 100.0, 300.0,
                            np.ones(100, dtype=np.float32), 0.01, 0.02, 15.0, 0.8),
        spectral=SpectralFeatures(np.zeros((10, 13), dtype=np.float32), 2000.0, 4000.0, 0.05, 0.1),
        prosody=ProsodyFeatures(130.0, 0.1, 3, np.ones(100, dtype=np.float32)),
        speaker=SpeakerProfile(np.ones(192, dtype=np.float32) * 0.1, "M", "25-35", "en", "unknown"),
        health=HealthIndicators(0.1, 0.1, 0.1, []),
        metadata=AudioMetadata(16000, 2.0, 30.0, False, True),
    )


def _fake_emotion() -> EmotionResult:
    return EmotionResult(
        emotion="joy", confidence=0.6,
        scores={"joy": 0.6, "neutral": 0.4},
        valence=0.5, arousal=0.4,
        dominant_emotions=["joy"], path="backbone",
    )


def test_extract_tone_color_shape(sine_440_wav):
    ModelRegistry.instance().register("vc_converter", _mock_vc_converter)
    from voicelab.cloning.extractor import _extract_tone_color
    path, _, _ = sine_440_wav
    tc = _extract_tone_color(path)
    assert tc.shape == (256,)
    assert tc.dtype == np.float32


def test_create_preset_returns_voice_preset(sine_440_wav):
    ModelRegistry.instance().register("vc_converter", _mock_vc_converter)
    path, _, _ = sine_440_wav
    with patch("voicelab.cloning.extractor.VoiceAnalyzer") as mock_va, \
         patch("voicelab.cloning.extractor.vl") as mock_vl:
        mock_va.return_value.analyze.return_value = _fake_analysis_result()
        mock_vl.analyze_emotion.return_value = _fake_emotion()
        from voicelab.cloning.extractor import create_preset
        preset = create_preset(path, name="test_voice")
    assert isinstance(preset, VoicePreset)
    assert preset.name == "test_voice"


def test_create_preset_embedding_shape(sine_440_wav):
    ModelRegistry.instance().register("vc_converter", _mock_vc_converter)
    path, _, _ = sine_440_wav
    with patch("voicelab.cloning.extractor.VoiceAnalyzer") as mock_va, \
         patch("voicelab.cloning.extractor.vl") as mock_vl:
        mock_va.return_value.analyze.return_value = _fake_analysis_result()
        mock_vl.analyze_emotion.return_value = _fake_emotion()
        from voicelab.cloning.extractor import create_preset
        preset = create_preset(path, name="x")
    assert preset.voice_embedding.shape == (192,)
    assert preset.tone_color.shape == (256,)


def test_create_preset_language_from_analysis(sine_440_wav):
    ModelRegistry.instance().register("vc_converter", _mock_vc_converter)
    path, _, _ = sine_440_wav
    with patch("voicelab.cloning.extractor.VoiceAnalyzer") as mock_va, \
         patch("voicelab.cloning.extractor.vl") as mock_vl:
        mock_va.return_value.analyze.return_value = _fake_analysis_result()
        mock_vl.analyze_emotion.return_value = _fake_emotion()
        from voicelab.cloning.extractor import create_preset
        preset = create_preset(path, name="x")
    assert preset.language == "en"


def test_create_preset_emotion_captured(sine_440_wav):
    ModelRegistry.instance().register("vc_converter", _mock_vc_converter)
    path, _, _ = sine_440_wav
    with patch("voicelab.cloning.extractor.VoiceAnalyzer") as mock_va, \
         patch("voicelab.cloning.extractor.vl") as mock_vl:
        mock_va.return_value.analyze.return_value = _fake_analysis_result()
        mock_vl.analyze_emotion.return_value = _fake_emotion()
        from voicelab.cloning.extractor import create_preset
        preset = create_preset(path, name="x")
    assert preset.emotion.emotion == "joy"
    assert preset.f0_mean == 180.0


def test_create_preset_created_at_is_iso(sine_440_wav):
    ModelRegistry.instance().register("vc_converter", _mock_vc_converter)
    path, _, _ = sine_440_wav
    with patch("voicelab.cloning.extractor.VoiceAnalyzer") as mock_va, \
         patch("voicelab.cloning.extractor.vl") as mock_vl:
        mock_va.return_value.analyze.return_value = _fake_analysis_result()
        mock_vl.analyze_emotion.return_value = _fake_emotion()
        from voicelab.cloning.extractor import create_preset
        preset = create_preset(path, name="x")
    from datetime import datetime
    datetime.fromisoformat(preset.created_at)  # raises if invalid
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_cloning_extractor.py -v
```

Expected: `ModuleNotFoundError: No module named 'voicelab.cloning'`

- [ ] **Step 3: Create package and extractor**

```bash
mkdir -p voicelab/cloning
touch voicelab/cloning/__init__.py
```

Create `voicelab/cloning/extractor.py`:

```python
# voicelab/cloning/extractor.py
from __future__ import annotations
import tempfile
from datetime import datetime, timezone
import numpy as np
import torch
from voicelab.core.model_registry import ModelRegistry
from voicelab.schema import VoicePreset


def _load_vc_converter():
    """Load OpenVoice v2 ToneColorConverter + se_extractor + base speaker SEs."""
    from huggingface_hub import snapshot_download
    from openvoice.api import ToneColorConverter
    from openvoice import se_extractor as se_ext
    import os

    ckpt_dir = snapshot_download(
        "myshell-ai/OpenVoiceV2",
        local_dir="voicelab/models/openvoice_v2",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    converter = ToneColorConverter(
        f"{ckpt_dir}/converter/config.json", device=device
    )
    converter.load_ckpt(f"{ckpt_dir}/converter/checkpoint.pth")

    _SE_FILES = {
        "en": "en-default.pth",
        "zh": "zh.pth",
        "ja": "jp.pth",
        "es": "es.pth",
        "fr": "fr.pth",
        "ko": "kr.pth",
    }
    base_ses: dict[str, object] = {}
    for lang, fname in _SE_FILES.items():
        pth = f"{ckpt_dir}/base_speakers/ses/{fname}"
        if os.path.exists(pth):
            base_ses[lang] = torch.load(pth, map_location=device)

    return converter, se_ext, base_ses


ModelRegistry.instance().register("vc_converter", _load_vc_converter)


def _extract_tone_color(audio_path: str) -> np.ndarray:
    """Run OpenVoice SE extractor on audio_path → float32[256]."""
    converter, se_ext, _ = ModelRegistry.instance().get("vc_converter")
    se_tensor, _ = se_ext.get_se(
        audio_path,
        converter,
        target_dir=tempfile.gettempdir(),
        vad=True,
    )
    return se_tensor.squeeze().cpu().detach().numpy().astype(np.float32).reshape(-1)


def create_preset(path: str, name: str) -> VoicePreset:
    """Analyse reference audio and build a VoicePreset."""
    from voicelab.analysis.voice_analyzer import VoiceAnalyzer
    from voicelab.schema import Config
    import voicelab as vl

    result = VoiceAnalyzer(Config(neural=True)).analyze(path)
    emotion = vl.analyze_emotion(path)
    tone_color = _extract_tone_color(path)

    return VoicePreset(
        name=name,
        voice_embedding=result.speaker.embedding,
        tone_color=tone_color,
        emotion=emotion,
        prosody=result.prosody,
        f0_mean=result.pitch.f0_mean,
        language=result.speaker.language,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_cloning_extractor.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add voicelab/cloning/__init__.py voicelab/cloning/extractor.py tests/test_cloning_extractor.py
git commit -m "feat: add cloning extractor — create_preset and vc_converter registry"
```

---

## Task 3: Synthesizer — synthesize + pitch adjustment

**Files:**
- Create: `voicelab/cloning/synthesizer.py`
- Create: `tests/test_cloning_synthesizer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cloning_synthesizer.py
import os
import numpy as np
import pytest
import soundfile as sf
from voicelab.core.model_registry import ModelRegistry
from voicelab.schema import (
    VoicePreset, SynthesisConfig, EmotionResult, ProsodyFeatures,
)
import voicelab.cloning.extractor   # noqa: F401
import voicelab.cloning.synthesizer  # noqa: F401


@pytest.fixture(autouse=True)
def clean_registry():
    ModelRegistry.instance().clear()
    yield
    ModelRegistry.instance().clear()


def _mock_vc_converter():
    import torch

    class _MockConverter:
        def convert(self, audio_src_path, src_se, tgt_se, output_path, tau=0.3, message=""):
            import shutil
            shutil.copy(audio_src_path, output_path)

    class _MockSeExt:
        def get_se(self, audio_path, vc_model, target_dir="/tmp", vad=True):
            return torch.zeros(1, 256, 1), "mock"

    base_ses = {lang: torch.zeros(1, 256, 1) for lang in ["en", "zh", "ja", "es", "fr", "ko"]}
    return _MockConverter(), _MockSeExt(), base_ses


def _mock_melo_factory():
    class _MockTTS:
        class hps:
            class data:
                spk2id = {
                    "EN-Default": 0, "ZH": 0, "JP": 0, "ES": 0, "FR": 0, "KR": 0,
                }

        def tts_to_file(self, text, speaker_id, output_path, speed=1.0):
            audio = (np.sin(2 * np.pi * 200 * np.arange(16000) / 16000) * 0.1).astype(np.float32)
            sf.write(output_path, audio, 16000)

    def factory(lang, device):
        return _MockTTS()

    return factory


def _make_preset() -> VoicePreset:
    return VoicePreset(
        name="test",
        voice_embedding=np.zeros(192, dtype=np.float32),
        tone_color=np.zeros(256, dtype=np.float32),
        emotion=EmotionResult("joy", 0.6, {"joy": 0.6, "neutral": 0.4},
                              0.5, 0.4, ["joy"], "backbone"),
        prosody=ProsodyFeatures(130.0, 0.1, 3, np.ones(100, dtype=np.float32)),
        f0_mean=180.0,
        language="en",
        created_at="2026-05-11T00:00:00+00:00",
    )


def test_synthesize_returns_ndarray():
    ModelRegistry.instance().register("vc_converter", _mock_vc_converter)
    ModelRegistry.instance().register("melo_tts", _mock_melo_factory)
    from voicelab.cloning.synthesizer import synthesize
    audio = synthesize("Hello world", _make_preset(), SynthesisConfig(lang="en"))
    assert isinstance(audio, np.ndarray)


def test_synthesize_output_dtype():
    ModelRegistry.instance().register("vc_converter", _mock_vc_converter)
    ModelRegistry.instance().register("melo_tts", _mock_melo_factory)
    from voicelab.cloning.synthesizer import synthesize
    audio = synthesize("Hello world", _make_preset(), SynthesisConfig(lang="en"))
    assert audio.dtype == np.float32


def test_synthesize_output_nonempty():
    ModelRegistry.instance().register("vc_converter", _mock_vc_converter)
    ModelRegistry.instance().register("melo_tts", _mock_melo_factory)
    from voicelab.cloning.synthesizer import synthesize
    audio = synthesize("Hello world", _make_preset(), SynthesisConfig(lang="en"))
    assert len(audio) > 0


def test_synthesize_invalid_lang_raises():
    ModelRegistry.instance().register("vc_converter", _mock_vc_converter)
    ModelRegistry.instance().register("melo_tts", _mock_melo_factory)
    from voicelab.cloning.synthesizer import synthesize
    with pytest.raises(ValueError, match="Unsupported lang"):
        synthesize("Hi", _make_preset(), SynthesisConfig(lang="xx"))


def test_adjust_pitch_no_change_when_zero_target():
    from voicelab.cloning.synthesizer import _adjust_pitch
    audio = (np.sin(2 * np.pi * 200 * np.arange(16000) / 16000)).astype(np.float32)
    result = _adjust_pitch(audio, 16000, target_f0=0.0)
    np.testing.assert_array_equal(result, audio)


def test_adjust_pitch_returns_float32():
    from voicelab.cloning.synthesizer import _adjust_pitch
    audio = (np.sin(2 * np.pi * 200 * np.arange(16000) / 16000)).astype(np.float32)
    result = _adjust_pitch(audio, 16000, target_f0=300.0)
    assert result.dtype == np.float32
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_cloning_synthesizer.py -v
```

Expected: `ModuleNotFoundError: No module named 'voicelab.cloning.synthesizer'`

- [ ] **Step 3: Create synthesizer.py**

Create `voicelab/cloning/synthesizer.py`:

```python
# voicelab/cloning/synthesizer.py
from __future__ import annotations
import os
import tempfile
import numpy as np
import torch
from voicelab.core.audio_io import load_audio
from voicelab.core.model_registry import ModelRegistry
from voicelab.schema import VoicePreset, SynthesisConfig

_LANG_CONFIG: dict[str, tuple[str, str]] = {
    "en": ("EN-Default", "en-default"),
    "zh": ("ZH",         "zh"),
    "ja": ("JP",         "jp"),
    "es": ("ES",         "es"),
    "fr": ("FR",         "fr"),
    "ko": ("KR",         "kr"),
}


def _load_melo_factory():
    """Returns a factory: (lang, device) → MeloTTS model (lazy per language)."""
    _cache: dict[str, object] = {}

    def get_model(lang: str, device: str) -> object:
        key = f"{lang}:{device}"
        if key not in _cache:
            from melo.api import TTS
            _cache[key] = TTS(language=lang.upper(), device=device)
        return _cache[key]

    return get_model


ModelRegistry.instance().register("melo_tts", _load_melo_factory)


def _get_melo(lang: str, device: str) -> object:
    return ModelRegistry.instance().get("melo_tts")(lang, device)


def _adjust_pitch(audio: np.ndarray, sr: int, target_f0: float) -> np.ndarray:
    """Shift pitch toward target F0 mean (capped at ±6 semitones)."""
    if target_f0 <= 0:
        return audio
    import librosa
    f0 = librosa.yin(audio, fmin=60, fmax=500, sr=sr)
    voiced = f0[f0 > 0]
    if len(voiced) == 0:
        return audio
    current_f0 = float(np.median(voiced))
    if current_f0 <= 0:
        return audio
    n_steps = float(np.clip(12.0 * np.log2(target_f0 / current_f0), -6.0, 6.0))
    if abs(n_steps) < 0.5:
        return audio
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps).astype(np.float32)


def synthesize(
    text: str,
    preset: VoicePreset,
    config: SynthesisConfig | None = None,
) -> np.ndarray:
    """Synthesize text in preset speaker's voice. Returns float32 at 16 kHz."""
    cfg = config or SynthesisConfig()
    lang = cfg.lang
    if lang not in _LANG_CONFIG:
        raise ValueError(f"Unsupported lang '{lang}'. Choose from: {sorted(_LANG_CONFIG)}")

    melo_speaker_key, _ = _LANG_CONFIG[lang]
    tts = _get_melo(lang, cfg.device)
    speaker_id = tts.hps.data.spk2id[melo_speaker_key]

    src_fd, src_path = tempfile.mkstemp(suffix=".wav")
    os.close(src_fd)
    try:
        tts.tts_to_file(text, speaker_id, src_path, speed=cfg.speed)

        converter, _, base_ses = ModelRegistry.instance().get("vc_converter")
        src_se = base_ses.get(lang)
        tgt_se = torch.from_numpy(preset.tone_color).reshape(1, -1, 1).to(cfg.device)

        out_fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(out_fd)
        try:
            converter.convert(
                audio_src_path=src_path,
                src_se=src_se,
                tgt_se=tgt_se,
                output_path=out_path,
                tau=0.3,
            )
            audio, sr = load_audio(out_path)
        finally:
            os.unlink(out_path)
    finally:
        os.unlink(src_path)

    return _adjust_pitch(audio, sr, preset.f0_mean)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_cloning_synthesizer.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add voicelab/cloning/synthesizer.py tests/test_cloning_synthesizer.py
git commit -m "feat: add synthesizer — MeloTTS + OpenVoice tone color conversion + pitch adjustment"
```

---

## Task 4: save_audio utility

**Files:**
- Modify: `voicelab/core/audio_io.py`
- Modify: `tests/test_audio_io.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_audio_io.py`:

```python
def test_save_audio_creates_file(tmp_path):
    from voicelab.core.audio_io import save_audio
    audio = np.zeros(16000, dtype=np.float32)
    path = str(tmp_path / "out.wav")
    save_audio(audio, sr=16000, path=path)
    assert os.path.exists(path)


def test_save_audio_roundtrip(tmp_path):
    from voicelab.core.audio_io import save_audio, load_audio
    audio = (np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype(np.float32)
    path = str(tmp_path / "roundtrip.wav")
    save_audio(audio, sr=16000, path=path)
    loaded, sr = load_audio(path)
    assert sr == 16000
    assert loaded.shape == audio.shape
    assert loaded.dtype == np.float32
```

Also add `import os` at the top of `tests/test_audio_io.py` if not already present.

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_audio_io.py::test_save_audio_creates_file -v
```

Expected: `ImportError: cannot import name 'save_audio'`

- [ ] **Step 3: Add save_audio to audio_io.py**

Append to `voicelab/core/audio_io.py`:

```python
def save_audio(audio: np.ndarray, sr: int, path: str) -> None:
    """Save float32 mono audio array to WAV file."""
    sf.write(path, audio, sr)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_audio_io.py -v
```

Expected: all tests pass (including the two new ones)

- [ ] **Step 5: Commit**

```bash
git add voicelab/core/audio_io.py tests/test_audio_io.py
git commit -m "feat: add save_audio utility to audio_io"
```

---

## Task 5: Public API + dependencies + final tests

**Files:**
- Modify: `voicelab/__init__.py`
- Modify: `pyproject.toml`
- Create: `tests/test_public_cloning_api.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_public_cloning_api.py
import numpy as np
import pytest
import soundfile as sf
from unittest.mock import patch, MagicMock
from voicelab.core.model_registry import ModelRegistry
from voicelab.schema import (
    VoicePreset, SynthesisConfig, EmotionResult, ProsodyFeatures,
    AnalysisResult, AudioMetadata, PitchFeatures, SpectralFeatures,
    SpeakerProfile, HealthIndicators,
)
import voicelab.cloning.extractor    # noqa: F401
import voicelab.cloning.synthesizer  # noqa: F401
import voicelab.neural.embeddings    # noqa: F401
import voicelab.neural.speaker_profile  # noqa: F401
import voicelab.neural.health        # noqa: F401
import voicelab.emotion.backbone     # noqa: F401
import voicelab.emotion.fusion       # noqa: F401


@pytest.fixture(autouse=True)
def clean_registry():
    ModelRegistry.instance().clear()
    yield
    ModelRegistry.instance().clear()


def _register_all_mocks():
    import torch
    from unittest.mock import MagicMock

    # Module 1 mocks
    m = MagicMock()
    m.encode_batch.return_value = (np.ones((1, 192), dtype=np.float32) * 0.1,)
    ModelRegistry.instance().register("ecapa", lambda: m)
    ModelRegistry.instance().register("gender_age", lambda: (lambda *a: ("M", "25-35")))
    ModelRegistry.instance().register("lang_id", lambda: (lambda audio, sr: "en"))
    ModelRegistry.instance().register("health", lambda: (lambda *a: {"dysphonia": 0.1, "fatigue": 0.1, "hoarseness": 0.1}))

    # Module 2 mocks
    n = 7
    logits = np.zeros(n); logits[3] = 2.0
    va = np.array([0.4, 0.5])
    ModelRegistry.instance().register("ser_backbone", lambda: (lambda audio, sr: (0.4, 0.6)))
    ModelRegistry.instance().register("ser_fusion", lambda: (lambda feat: (logits.copy(), va.copy())))
    ModelRegistry.instance().register("ser_fusion_lite", lambda: (lambda feat: (logits.copy(), va.copy())))

    # Module 3 mocks
    class _MockConverter:
        def convert(self, audio_src_path, src_se, tgt_se, output_path, tau=0.3, message=""):
            import shutil; shutil.copy(audio_src_path, output_path)

    class _MockSeExt:
        def get_se(self, audio_path, vc_model, target_dir="/tmp", vad=True):
            return torch.zeros(1, 256, 1), "mock"

    base_ses = {lang: torch.zeros(1, 256, 1) for lang in ["en", "zh", "ja", "es", "fr", "ko"]}
    ModelRegistry.instance().register("vc_converter", lambda: (_MockConverter(), _MockSeExt(), base_ses))

    class _MockTTS:
        class hps:
            class data:
                spk2id = {"EN-Default": 0, "ZH": 0, "JP": 0, "ES": 0, "FR": 0, "KR": 0}
        def tts_to_file(self, text, speaker_id, output_path, speed=1.0):
            audio = (np.sin(2 * np.pi * 200 * np.arange(16000) / 16000) * 0.1).astype(np.float32)
            sf.write(output_path, audio, 16000)

    ModelRegistry.instance().register("melo_tts", lambda: (lambda lang, device: _MockTTS()))


def test_create_preset_in_public_api():
    import voicelab as vl
    assert hasattr(vl, "create_preset")


def test_synthesize_in_public_api():
    import voicelab as vl
    assert hasattr(vl, "synthesize")


def test_save_audio_in_public_api():
    import voicelab as vl
    assert hasattr(vl, "save_audio")


def test_voice_preset_importable():
    import voicelab as vl
    p = vl.VoicePreset(
        name="x",
        voice_embedding=np.zeros(192, dtype=np.float32),
        tone_color=np.zeros(256, dtype=np.float32),
        emotion=EmotionResult("joy", 0.6, {"joy": 0.6, "neutral": 0.4}, 0.5, 0.4, ["joy"], "backbone"),
        prosody=ProsodyFeatures(130.0, 0.1, 3, np.ones(100, dtype=np.float32)),
        f0_mean=180.0, language="en", created_at="2026-05-11T00:00:00+00:00",
    )
    assert p.name == "x"


def test_synthesis_config_importable():
    import voicelab as vl
    cfg = vl.SynthesisConfig(lang="ja")
    assert cfg.lang == "ja"


def test_create_preset_end_to_end(sine_440_wav):
    _register_all_mocks()
    import voicelab as vl
    path, _, _ = sine_440_wav
    preset = vl.create_preset(path, name="test_e2e")
    assert isinstance(preset, vl.VoicePreset)
    assert preset.name == "test_e2e"
    assert preset.voice_embedding.shape == (192,)
    assert preset.tone_color.shape == (256,)


def test_synthesize_end_to_end(sine_440_wav):
    _register_all_mocks()
    import voicelab as vl
    path, _, _ = sine_440_wav
    preset = vl.create_preset(path, name="e2e")
    audio = vl.synthesize("Hello", preset, vl.SynthesisConfig(lang="en"))
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert len(audio) > 0


def test_save_audio_writes_file(tmp_path):
    import voicelab as vl
    import os
    audio = np.zeros(16000, dtype=np.float32)
    out = str(tmp_path / "out.wav")
    vl.save_audio(audio, sr=16000, path=out)
    assert os.path.exists(out)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_public_cloning_api.py::test_create_preset_in_public_api -v
```

Expected: `AttributeError: module 'voicelab' has no attribute 'create_preset'`

- [ ] **Step 3: Update voicelab/__init__.py**

Replace `voicelab/__init__.py` with:

```python
from voicelab.schema import (
    AnalysisResult, FrameResult, Config,
    PitchFeatures, SpectralFeatures, ProsodyFeatures,
    SpeakerProfile, HealthIndicators, AudioMetadata,
    EmotionResult, EmotionFrame, EmotionConfig,
    VoicePreset, SynthesisConfig,
)
from voicelab.analysis.voice_analyzer import VoiceAnalyzer
from voicelab.analysis.voice_stream import VoiceStream
from voicelab.emotion.analyzer import EmotionAnalyzer
from voicelab.emotion.stream import EmotionStream
from voicelab.core.audio_io import save_audio

__version__ = "0.1.0"
__all__ = [
    "analyze", "stream", "analyze_emotion", "emotion_stream",
    "create_preset", "synthesize", "save_audio",
    "Config", "EmotionConfig", "SynthesisConfig",
    "AnalysisResult", "FrameResult", "EmotionResult", "EmotionFrame",
    "VoicePreset",
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


def create_preset(path: str, name: str) -> VoicePreset:
    """Extract a VoicePreset from reference audio (voice + emotion + prosody)."""
    from voicelab.cloning.extractor import create_preset as _create
    return _create(path, name)


def synthesize(
    text: str,
    preset: VoicePreset,
    config: SynthesisConfig | None = None,
) -> "import numpy; numpy.ndarray":
    """Synthesize text in preset speaker's voice. Returns float32 ndarray at 16 kHz."""
    from voicelab.cloning.synthesizer import synthesize as _synthesize
    return _synthesize(text, preset, config)
```

> **Note:** The return type annotation on `synthesize` above uses an invalid inline import. Replace the return type with just `object` or use a forward reference. The correct signature is:

```python
def synthesize(
    text: str,
    preset: VoicePreset,
    config: SynthesisConfig | None = None,
):
    """Synthesize text in preset speaker's voice. Returns float32 ndarray at 16 kHz."""
    from voicelab.cloning.synthesizer import synthesize as _synthesize
    return _synthesize(text, preset, config)
```

- [ ] **Step 4: Update pyproject.toml dependencies**

In `pyproject.toml`, add to `[tool.poetry.dependencies]`:

```toml
openvoice = ">=0.1"
melo-tts = ">=0.0.1"
huggingface-hub = ">=0.20"
```

Install the new deps:

```bash
poetry add openvoice melo-tts huggingface-hub
```

- [ ] **Step 5: Run all tests**

```bash
pytest tests/ -q
```

Expected: all tests pass (prior 95 + new cloning tests). Run count will be ≥ 115.

- [ ] **Step 6: Commit**

```bash
git add voicelab/__init__.py pyproject.toml poetry.lock tests/test_public_cloning_api.py
git commit -m "feat: expose Module 3 public API — create_preset, synthesize, save_audio"
```

---

## Self-Review Checklist (already run — fixes applied inline)

1. **Spec coverage:**
   - VoicePreset ✓ (Task 1)
   - SynthesisConfig ✓ (Task 1)
   - `.vpreset` serialization ✓ (Task 1 — zipfile-based)
   - `_extract_tone_color` ✓ (Task 2)
   - `create_preset` ✓ (Task 2)
   - `synthesize` ✓ (Task 3)
   - `_adjust_pitch` ✓ (Task 3)
   - `save_audio` ✓ (Task 4)
   - Public API ✓ (Task 5)
   - ModelRegistry entries for `vc_converter` and `melo_tts` ✓
   - pyproject.toml deps ✓

2. **Placeholder scan:** No TBD/TODO found.

3. **Type consistency:**
   - `VoicePreset.tone_color` is `np.ndarray` shape `(256,)` throughout
   - `_mock_vc_converter` returns `(converter, se_ext, base_ses)` — matches `_load_vc_converter` return signature in all 3 test files ✓
   - `_mock_melo_factory` returns a `(lang, device) → TTS` callable — matches `_load_melo_factory` pattern ✓
   - `synthesize` in `__init__.py` delegates to `voicelab.cloning.synthesizer.synthesize` ✓
