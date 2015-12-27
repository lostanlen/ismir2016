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

# Deep learning settings
graph = di.learning.build_graph(
    X_height = 168,
    X_width = 128,
    conv1_channels = 50,
    conv1_height = 96,
    conv1_width = 32,
    pool1_height = 8,
    pool1_width = 8,
    conv2_channels = 30,
    conv2_height = 7,
    conv2_width = 7,
    pool2_height = 3,
    pool2_width = 3,
    drop1_proportion = 0.5,
    dense1_channels = 256,
    drop2_proportion = 0.5,
    dense2_channels = 8
)

# Compute audio features on training set
solosDb8train_dir = '~/datasets/solosDb8/train'
train_file_paths = di.symbolic.get_paths(solosDb8train_dir, instrument_list, 'wav')
(X_train, Y_train) = di.solosdb.get_XY(
        train_file_paths,
        instrument_list, decision_duration, fmin, hop_duration,
        n_bins_per_octave, n_octaves, silence_threshold, sr)

# Train deep network
adagrad = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-06)
graph.compile(loss={'Y': 'mean_squared_error'}, optimizer=adagrad)
history = graph.fit({'X': X_train, 'Y': Y_train}, nb_epoch=1, batch_size=32)

# Compute audio features on test set
solosDb8test_dir = '~/datasets/solosDb8/test'
test_file_paths = di.symbolic.get_paths(solosDb8test_dir, instrument_list, 'wav')
(X_test, Y_test) = di.solosdb.get_XY(
        test_file_paths,
        instrument_list, decision_duration, fmin, hop_duration,
        n_bins_per_octave, n_octaves, silence_threshold, sr)

# Evaluate deep network
score = graph.evaluate({'X': X_test, 'Y': Y_test}, batch_size=32)