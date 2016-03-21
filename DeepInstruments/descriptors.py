import os

import joblib
import librosa
import numpy as np

import DeepInstruments as di


def get_X(paths):
    delayed_get_descriptors = \
        joblib.delayed(di.descriptors.cached_get_descriptors)
    X = joblib.Parallel(n_jobs=-1, verbose=10)(
            delayed_get_descriptors(path) for path in paths)
    return np.vstack(X)


def get_descriptors(path):
    waveform, sr = librosa.core.load(path, sr=22050, mono=True)
    mfcc = librosa.feature.mfcc(waveform, sr)
    delta_mfcc = librosa.feature.delta(mfcc)
    deltadelta_mfcc = librosa.feature.delta(mfcc, order=2)
    bandwidth = librosa.feature.spectral_bandwidth(waveform, sr)
    centroid = librosa.feature.spectral_centroid(waveform, sr)
    contrast = librosa.feature.spectral_contrast(waveform, sr)
    rolloff = librosa.feature.spectral_rolloff(waveform, sr)
    zcr = librosa.feature.zero_crossing_rate(waveform, sr)
    return np.hstack(
            (np.mean(mfcc, axis=1),
             np.mean(delta_mfcc, axis=1),
             np.mean(deltadelta_mfcc, axis=1),
             np.mean(bandwidth),
             np.std(bandwidth),
             np.mean(centroid),
             np.std(centroid),
             np.mean(contrast),
             np.std(contrast),
             np.mean(rolloff),
             np.std(rolloff),
             np.mean(zcr),
             np.std(zcr))
    )

cachedir = os.path.expanduser('~/joblib')
memory = joblib.Memory(cachedir=cachedir, verbose=0)
cached_get_descriptors = memory.cache(get_descriptors)


def get_y(path):
    set_path = path.split("medleydb-single-instruments", 1)[1]
    class_path = set_path[1:].split("/", 1)[1]
    return int(class_path[1])