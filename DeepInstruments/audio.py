import joblib
import librosa
import numpy as np
import os
import warnings


def get_X(decision_length,
          fmin,
          hop_length,
          n_bins_per_octave,
          n_octaves,
          track_or_path):
    if isinstance(track_or_path, basestring):
        x_mono, sr = librosa.core.load(track_or_path, sr=None, mono=True)
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            (sr, x_stereo) = track_or_path.audio_data
            warnings.resetwarnings()
        x_stereo = x_stereo.astype(np.float32)
        x_mono = np.sum(x_stereo, axis=1) / (32768.0 * 2)
    if x_mono.shape[0] < decision_length:
        padding_length = x_mono.shape[0] - decision_length
        padding = np.zeros(padding_length, dtype=np.float32)
        x_mono = np.hstack((x_mono, padding))
    n_bins = n_octaves * n_bins_per_octave
    freqs = librosa.cqt_frequencies(
            bins_per_octave=n_bins_per_octave,
            fmin=fmin,
            n_bins=n_bins)
    CQT = librosa.cqt(
            x_mono,
            bins_per_octave=n_bins_per_octave,
            fmin=fmin,
            hop_length=hop_length,
            n_bins=n_bins,
            sr=sr)
    #X = librosa.perceptual_weighting(
    #        CQT ** 2,
    #        freqs,
    #        ref_power=1.0)
    X = CQT
    X = X.astype(np.float32)
    return X


cachedir = os.path.expanduser('~/joblib')
memory = joblib.Memory(cachedir=cachedir, verbose=0)
cached_get_X = memory.cache(get_X)