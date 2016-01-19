import DeepInstruments as di
import joblib
import librosa
import numpy as np
import os


def get_paths(training_or_test):
    set_path = os.path.join(os.path.expanduser("~"),
                             "datasets",
                             "medleydb-single-instruments",
                             training_or_test)
    paths = [
        [os.path.join(path, name)
         for (path, subdir, names)
         in os.walk(os.path.join(set_path, class_name))
         for name in names]
        for class_name in os.listdir(set_path)]
    paths = [path for class_path in paths for path in class_path]
    return paths


def get_X(paths):
    X = [joblib.delayed(di.descriptors.cached_get_descriptors)(path)
         for path in paths]
    return np.vstack(X)


def get_Y(paths):
    Y = map(di.descriptors.get_y, paths)
    return np.vstack(Y)


def get_descriptors(path):
    waveform, sr = librosa.core.load(path)
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
             np.mean(centroid),
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
    class_path = path.split("medleydb-single-instruments", 1)[1]
    return int(class_path[2])