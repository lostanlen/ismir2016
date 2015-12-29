import librosa
import keras
import numpy as np
import os

import DeepInstruments as di

# Audio settings
fmin = librosa.note_to_hz('A1')  # in Hertz
n_octaves = 7
n_bins_per_octave = 24
sr = 32000.0  # in Hertz
hop_duration = 0.016  # in seconds
decision_duration = 2.048  # in seconds
silence_threshold = -30 # in dB
instrument_list = ['Cl', 'Co', 'Fh', 'Gt', 'Ob', 'Pn', 'Tr', 'Vl']

# Compute audio features on training set
solosDb8train_dir = '~/datasets/solosDb8/train'
train_file_paths = di.symbolic.get_paths(solosDb8train_dir, instrument_list, 'wav')
(X_sdbtrain_list, Ys_sdbtrain_list) = di.solosdb.get_XY(
        train_file_paths,
        instrument_list, decision_duration, fmin, hop_duration,
        n_bins_per_octave, n_octaves, sr)