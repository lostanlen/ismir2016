from joblib import Parallel, delayed
import glob
import librosa
import numpy as np
import os
import sys

dataset_dir = '/Users/vlostan/datasets/solosDb8'
wav_paths = get_paths(dataset_dir, 'wav')


def get_paths(dir, extension):
    walk = os.walk(dir)
    regex = '*.' + extension
    return [p for d in walk for p in glob.glob(os.path.join(d[0], regex))]


fmin = 55  # in Hertz
n_octaves = 7
n_bins_per_octave = 24
sr = 32000.0  # in Hertz
hop_time = 0.016  # in seconds
decision_time = 2.048  # in seconds


def perceptual_cqt(
        file_path,
        decision_duration,
        fmin,
        hop_duration,
        n_bins,
        n_bins_per_octave,
        sr):
    y, y_sr = librosa.load(file_path)
    if y_sr != sr:
        y = librosa.resample(y, y_sr, sr)
    hop_length = hop_time * sr
    decision_length = decision_time * sr / hop_length
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
    n_windows = int(n_hops / decision_length)
    n_hops_truncated = n_windows * decision_length
    audio_features = np.transpose(audio_features)
    audio_features = audio_features[:n_hops_truncated, :]
    new_shape = (n_windows, decision_length, n_bins)
    audio_features = np.reshape(audio_features, new_shape)
    return np.transpose(audio_features, (0, 2, 1))
