import numpy as np
import librosa
from voicelab.schema import SpectralFeatures


def extract_spectral_features(
    audio: np.ndarray, sr: int, n_mfcc: int = 13
) -> SpectralFeatures:
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc).T  # [n_frames, n_mfcc]
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
    rms = float(np.mean(librosa.feature.rms(y=audio)))
    return SpectralFeatures(
        mfcc=mfcc,
        centroid_mean=centroid,
        rolloff_mean=rolloff,
        zcr_mean=zcr,
        rms_mean=rms,
    )
