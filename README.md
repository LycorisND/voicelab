# VoiceLab

Deep voice analysis SDK for Python — acoustics, prosody, speaker profiling, emotion, and health.

```python
import voicelab as vl

result = vl.analyze("speech.wav")
print(result.pitch.f0_mean)       # e.g. 185.3 Hz
print(result.speaker.gender)      # "M" or "F"
print(result.health.dysphonia_score)  # 0.0–1.0
```

See `docs/superpowers/specs/` for full design specification.
