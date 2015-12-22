import librosa
import numpy as np

from DeepInstruments import audio
def check_silence_threshold(
        file_paths,
        decision_duration,
        hop_duration,
        silence_threshold,
        sr):
    silences = [ audio.extract_silence(
        file_path,
        decision_duration,
        silence_threshold,
        sr) for file_path in file_paths ]


def extract_silence(
        file_path,
        decision_duration,
        silence_threshold,
        sr):
    y, y_sr = librosa.load(file_path)
    if y_sr != sr:
        y = librosa.resample(y, y_sr, sr)
    n_windows = int(len(y) / (decision_duration*sr))
    y_abs = np.abs(y)[:n_windows * decision_duration * sr]
    y_abs2 = y_abs**2
    y_abs2 = np.reshape(y_abs2, (n_windows, decision_duration * sr))
    y_levels = np.mean(y_abs2, axis=1)
    y_levels = y_levels / np.max(y_levels)
    window_bools = np.reshape(y_levels, (len(y_levels),1)) > silence_threshold
    broadcaster = np.ones((1, decision_duration * sr), dtype = bool)
    sample_bools = np.ndarray.flatten(window_bools * broadcaster)
    return y[np.logical_not(sample_bools)]

def perceptual_cqt(
        file_path,
        decision_duration,
        fmin,
        hop_duration,
        n_bins_per_octave,
        n_octaves,
        sr,
        silence_threshold):
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
        audio_features = np.hstack(audio_features, padding)
        n_hops = decision_length
    n_windows = int(n_hops / decision_length)
    n_hops_truncated = n_windows * decision_length
    audio_features = np.transpose(audio_features)
    audio_features = audio_features[:n_hops_truncated, :]
    new_shape = (n_windows, decision_length, n_bins)
    audio_features = np.reshape(audio_features, new_shape)
    y_abs = np.abs(y)[:n_windows * decision_duration * sr]
    y_abs2 = y_abs**2
    y_abs2 = np.reshape(y_abs2, (n_windows, decision_duration * sr))
    y_levels = np.mean(y_abs2, axis=1)
    y_levels = y_levels / np.max(y_levels)
    window_bools = np.reshape(y_levels, (len(y_levels),1)) > silence_threshold
    audio_features = audio_features[window_bools, :, :]
    return np.transpose(audio_features, (0, 2, 1))