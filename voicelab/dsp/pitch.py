import numpy as np
import parselmouth
from parselmouth.praat import call
from voicelab.schema import PitchFeatures


def extract_pitch(audio: np.ndarray, sr: int) -> PitchFeatures:
    snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=float(sr))

    pitch_obj = snd.to_pitch(time_step=0.01, pitch_floor=50.0, pitch_ceiling=600.0)
    f0_values = pitch_obj.selected_array["frequency"]  # 0 where unvoiced
    voiced_mask = f0_values > 0
    voiced_f0 = f0_values[voiced_mask]

    if len(voiced_f0) == 0:
        return PitchFeatures(
            f0_mean=0.0, f0_std=0.0, f0_min=0.0, f0_max=0.0,
            f0_contour=f0_values, jitter_local=0.0,
            shimmer_local=0.0, hnr=0.0, voiced_fraction=0.0,
        )

    jitter_local = 0.0
    shimmer_local = 0.0
    try:
        pp = call(snd, "To PointProcess (periodic, cc)", 50.0, 600.0)
        n_points = call(pp, "Get number of points")
        if n_points >= 3:
            jitter_local = float(
                call(pp, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
            )
            shimmer_local = float(
                call([snd, pp], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
            )
    except Exception:
        pass

    hnr = 0.0
    try:
        harm = call(snd, "To Harmonicity (cc)", 0.01, 50.0, 0.1, 1.0)
        hnr_val = call(harm, "Get mean", 0.0, 0.0)
        if hnr_val is not None and not np.isnan(hnr_val):
            hnr = float(np.clip(hnr_val, -20.0, 60.0))
    except Exception:
        pass

    return PitchFeatures(
        f0_mean=float(np.mean(voiced_f0)),
        f0_std=float(np.std(voiced_f0)),
        f0_min=float(np.min(voiced_f0)),
        f0_max=float(np.max(voiced_f0)),
        f0_contour=f0_values,
        jitter_local=jitter_local,
        shimmer_local=shimmer_local,
        hnr=hnr,
        voiced_fraction=float(np.sum(voiced_mask) / len(f0_values)),
    )
