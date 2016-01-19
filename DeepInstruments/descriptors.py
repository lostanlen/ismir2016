import DeepInstruments as di
import joblib
import librosa
import numpy as np
import os


def get_training_descriptors():
    training_path = os.path.join(os.path.expanduser("~"),
                                 "datasets",
                                 "medleydb-single-instruments",
                                 "training")
    chunk_paths = [
        [os.path.join(path, name)
         for (path, subdir, names)
         in os.walk(os.path.join(training_path, class_name))
         for name in names]
        for class_name in os.listdir(training_path)]
    chunk_paths = [path for class_path in chunk_paths for path in class_path]

    X = joblib.Parallel(n_jobs=-1, verbose=10)(
        joblib.delayed(di.descriptors.cached_get_descriptors)(chunk_path)
        for chunk_path in chunk_paths)
    return X


def get_descriptors(chunk_path):
    waveform, sr = librosa.core.load(chunk_path)
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
