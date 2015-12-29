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
    decision_length = decision_duration * sr / hop_length
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
    if n_hops < decision_length:
        padding = np.zeros(n_bins, decision_length - n_hops)
        audio_features = np.hstack((audio_features, padding))
        n_hops = decision_length
    n_windows = int(n_hops / decision_length)
    n_hops_truncated = n_windows * decision_length
    audio_features = np.transpose(audio_features)
    audio_features = audio_features[:n_hops_truncated, :]
    new_shape = (n_windows, decision_length, n_bins)
    audio_features = np.reshape(audio_features, new_shape)
    if len(y)<(n_windows * decision_duration * sr):
        padding = np.zeros(n_windows * decision_duration * sr - len(y))
        y = np.hstack((y, padding))
    y_truncated = y[:n_windows * decision_duration * sr]
    y_abs2 = y_truncated * y_truncated
    y_abs2 = np.reshape(y_abs2, (n_windows, decision_duration * sr))
    y_levels = np.sqrt(np.sum(y_abs2, axis=1))
    y_levels /= np.max(y_levels)
    threshold_lin = 10.0**(silence_threshold / 10.0)
    window_bools = y_levels > threshold_lin
    audio_features = audio_features[window_bools, :, :]
    return np.transpose(audio_features, (0, 2, 1))


class ChunkGenerator(object):
    def __init__(self,
                 hop_length
                 decision_duration,
                 sr):
        chunk_length = decision_duration * sr / hop_length
        self.chunk_length = chunk_length
