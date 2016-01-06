import librosa
import numpy as np

sr, X_stereo = track.audio_data
X_stereo = X_stereo.astype(np.float32)
X_mono = np.sum(X_stereo, axis=1) / (32768.0 * 2)

def get_X(
        file_path,
        decision_duration,
        fmin,
        hop_duration,
        n_bins_per_octave,
        n_octaves,
        track):
    x_stereo = track.audio_data
    x_stereo = x_stereo.astype(np.float32)
    x_mono = np.sum(x_stereo, axis=1) / (32768.0 * 2)
    if x_mono.shape[0] < (decision_duration * sr):
        padding = np.zeros(x_mono.shape[0] - decision_duration * sr)
        x_mono =
    n_bins = n_octaves * n_bins_per_octave
    hop_length = hop_duration * sr
    freqs = librosa.cqt_frequencies(
            bins_per_octave=n_bins_per_octave,
            fmin=fmin,
            n_bins=n_bins)
    CQT = librosa.hybrid_cqt(
            x_mono,
            bins_per_octave=n_bins_per_octave,
            fmin=fmin,
            hop_length=hop_length,
            n_bins=n_bins,
            sr=sr)
    X = librosa.perceptual_weighting(
            CQT ** 2,
            freqs,
            ref_power=1.0)
    n_hops = X.shape[1]
    decision_length = int(decision_duration * sr / hop_length)
    if n_hops < decision_length:
        padding = np.zeros(n_bins, decision_length - n_hops)
        X = np.hstack((X, padding))
    X = X.astype(np.float32)
    return X
