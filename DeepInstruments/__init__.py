import librosa
import matplotlib.pyplot as plt
import numpy as np
import os

dataset_folder = '/Users/vlostan/datasets/solosDb'
instrument_folder = 'As'
file_name = '1095.wav'

file_path = os.path.join(dataset_folder, instrument_folder, file_name)

fmin = librosa.note_to_hz('A1')
n_octaves = 7
n_bins_per_octave = 24
n_bins = n_octaves * n_bins_per_octave
sr = 32000.0  # in Hertz
hop_time = 0.020  # in seconds
hop_length = hop_time * sr
decision_time = 2.000  # in seconds
decision_length = decision_time * sr / hop_length

freqs = librosa.cqt_frequencies(
        bins_per_octave=n_bins_per_octave,
        fmin=fmin,
        n_bins=n_bins)

y, y_sr = librosa.load(file_path)
if y_sr != sr:
    y = librosa.resample(y, y_sr, sr)


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

audio_features = audio_features[:, :n_hops_truncated]
new_shape = (n_bins, decision_length, n_windows)
audio_features = np.reshape(audio_features, new_shape)

