import numpy as np
import parselmouth


def extract_formants(
    audio: np.ndarray, sr: int, n_formants: int = 4
) -> dict[str, float]:
    """Return mean F1–F4 in Hz. Returns 0.0 for untracked formants."""
    snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=float(sr))
    formant = snd.to_formant_burg(
        time_step=0.01,
        max_number_of_formants=float(n_formants),
        maximum_formant=5500.0,
        window_length=0.025,
        pre_emphasis_from=50.0,
    )
    times = formant.ts()
    values: dict[str, list[float]] = {f"F{i + 1}": [] for i in range(n_formants)}
    for t in times:
        for i in range(n_formants):
            val = formant.get_value_at_time(
                formant_number=i + 1, time=float(t), unit=parselmouth.FormantUnit.HERTZ
            )
            if val is not None and not np.isnan(val) and val > 0:
                values[f"F{i + 1}"].append(val)
    return {k: float(np.mean(v)) if v else 0.0 for k, v in values.items()}
