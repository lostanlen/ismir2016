import DeepInstruments as di
import librosa
import numpy as np

decision_length = 131072  # in samples
fmin = 55  # in Hz
hop_length = 1024  # in samples
n_bins_per_octave = 12
n_octaves = 8

X = di.audio.get_X(decision_length,
                   fmin,
                   hop_length,
                   n_bins_per_octave,
                   n_octaves,
                   "research/1395.wav")

X = (X - np.min(X)) / (np.max(X) - np.min(X))
librosa.display.specshow(1 - np.sqrt(X), cmap="gray")