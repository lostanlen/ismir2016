import librosa
import matplotlib.pyplot as plt
import numpy as np
import os

dataset_dir = "/Users/vlostan/datasets/solosDb_kapokotrain"

kapoko_dirs = ["Cl", "Co", "Fh", "Gt", "Ob", "Pn", "Tr", "Vl"]


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
instrument_dirs = kapoko_dirs
n_instruments = len(instrument_dirs)
id_instrument = 0
instrument_dir = instrument_dirs[id_instrument]
instrument_path = os.path.join(dataset_dir, instrument_dir)
file_names = os.listdir(instrument_path)
id_file = 0
file_name = file_names[id_file]
file_path = os.path.join(instrument_path, file_name)


def perceptual_cqt(
        file_path,
        freqs,
        hop_length,
        n_bins,
        n_bins_per_octave):
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
    audio_features = np.transpose(audio_features)
    audio_features = audio_features[:n_hops_truncated, :]
    new_shape = (n_windows, decision_length, n_bins)
    audio_features = np.reshape(audio_features, new_shape)
    return np.transpose(audio_features, (0, 2, 1))

