import librosa
from joblib import Parallel
import numpy as np
import random

import DeepInstruments as di

def perceptual_cqt(
        file_path,
        decision_duration,
        fmin,
        hop_duration,
        n_bins_per_octave,
        n_octaves,
        sr):
    y, y_sr = librosa.load(file_path)
    if y_sr != sr:
        y = librosa.resample(y, y_sr, sr)
    hop_length = hop_duration * sr
    n_bins = n_octaves * n_bins_per_octave
    freqs = librosa.cqt_frequencies(
            bins_per_octave=n_bins_per_octave,
            fmin=fmin,
            n_bins=n_bins)
    CQT = librosa.hybrid_cqt(
            y,
            bins_per_octave=n_bins_per_octave,
            fmin=fmin,
            hop_length=hop_length,
            n_bins=n_bins,
            sr=sr)
    audio_features = librosa.perceptual_weighting(
            CQT ** 2,
            freqs,
            ref_power=1.0)
    n_hops = audio_features.shape[1]
    decision_length = decision_duration * sr / hop_length
    if n_hops < decision_length:
        padding = np.zeros(n_bins, decision_length - n_hops)
        audio_features = np.hstack((audio_features, padding))
        n_hops = decision_length
    return np.transpose(audio_features)
