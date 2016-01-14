import DeepInstruments as di
import joblib
import librosa
import numpy as np
import os


def chunk_stems(decision_length, stems):
    X = []
    Y = []
    delayed_get_X = joblib.delayed(di.audio.cached_get_X)
    for class_stems in training_stems:
        X.append(joblib.Parallel(n_jobs=-1)(
            delayed_get_X(decision_length, fmin, hop_length,
                          n_bins_per_octave, n_octaves, stem)
            for stem in class_stems
        ))
        Y.append([di.singlelabel.get_Y(stem) for stem in class_stems])
    X_mat = np.hstack([X_file for X_class in X for X_file in X_class])
    self.X_mean = np.mean(X_mat)
    self.X_std = np.std(X_mat)
    for instrument_id in range(len(X)):
        X[instrument_id] = [(X_file-self.X_mean) / self.X_std
                            for X_file in X[instrument_id]]
    self.X = X
    self.Y = Y
    self.indices = di.singlelabel.get_indices(Y, decision_length)
    n_instruments = len(X)
    durations = []
    for instrument_id in range(n_instruments):
        file_lengths = map(float, map(len, self.indices[instrument_id]))
        durations.append(file_lengths / np.sum(file_lengths))
    self.durations = durations