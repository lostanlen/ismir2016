import DeepInstruments as di
import librosa
import numpy as np
import os

training_path = os.path.join(os.path.expanduser("~"),
                             "datasets",
                             "medleydb-single-instruments",
                             "training")
set_path = training_path

chunk_paths = [
    [os.path.join(path, name)
     for (path, subdir, names) in os.walk(os.path.join(set_path, class_name))
     for name in names]
    for class_name in os.listdir(set_path)]

chunk_path = chunk_paths[0][0]
waveform, sr = librosa.core.load(chunk_path)
mfcc = librosa.feature.mfcc(waveform, sr)
delta_mfcc = librosa.feature.delta(mfcc)
deltadelta_mfcc = librosa.feature.delta(mfcc, order=2)
mean_mfcc = np.mean(mfcc, axis=-1)
mean_delta_mfcc = np.mean(delta_mfcc, axis=1)
mean_deltadelta_mfcc = np.mean(deltadelta_mfcc, axis=1)
bandwidth = librosa.feature.spectral_bandwidth(waveform, sr)
mean_bandwidth = np.mean(bandwidth, axis=1)
std_bandwidth = np.std(bandwidth, axis=1)
centroid = librosa.feature.spectral_centroid(waveform, sr)
mean_centroid = np.mean(centroid, axis=1)
std_centroid = np.std(centroid, axis=1)
contrast = librosa.feature.spectral_contrast(waveform, sr)
mean_contrast = np.mean(contrast, axis=1)
std_contrast = np.std(contrast, axis=1)
rolloff = librosa.feature.spectral_rolloff(waveform, sr)
mean_rolloff = np.mean(rolloff, axis=1)
std_rolloff = np.std(rolloff, axis=1)
zcr = librosa.feature.zero_crossing_rate(y, sr)
mean_zcr = np.mean(zcr)
std_zcr = np.std(zcr)
descriptors = np.hstack(
        (mean_mfcc,
         mean_delta_mfcc,
         mean_deltadelta_mfcc,
         mean_bandwidth,
         std_bandwidth,
         mean_centroid,
         std_centroid,
         mean_contrast,
         std_contrast,
         mean_rolloff,
         std_rolloff,
         mean_zcr,
         std_zcr)
)