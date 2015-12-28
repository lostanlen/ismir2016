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
(X_train, Y_train) = di.solosdb.get_XY(
        train_file_paths,
        instrument_list, decision_duration, fmin, hop_duration,
        n_bins_per_octave, n_octaves, silence_threshold, sr)

# Compute audio features on test set
solosDb8test_dir = '~/datasets/solosDb8/test'
test_file_paths = di.symbolic.get_paths(solosDb8test_dir, instrument_list, 'wav')
(X_test, Y_test) = di.solosdb.get_XY(
        test_file_paths,
        instrument_list, decision_duration, fmin, hop_duration,
        n_bins_per_octave, n_octaves, silence_threshold, sr)

graph = di.learning.build_graph(
    X_height=168,
    X_width=128,
    conv1_channels=100,
    conv1_height=96,
    conv1_width=32,
    pool1_height=8,
    pool1_width=8,
    conv2_channels=100,
    conv2_height=8,
    conv2_width=8,
    pool2_height=3,
    pool2_width=3,
    dense1_channels=512,
    drop1_proportion=0.25,
    dense2_channels=64,
    drop2_proportion=0.25,
    dense3_channels=8)

graph.compile(loss={'Y': 'categorical_crossentropy'}, optimizer="adagrad")
history = graph.fit(
        {'X': X_train, 'Y': Y_train},
        nb_epoch=100,
        batch_size=128,
        verbose=1)

test_prediction = graph.predict({'X': X_test})
Y_test_predicted = test_prediction["Y"]

import sklearn
y_test_true = np.argmax(Y_test, axis=1)
y_test_predicted = np.argmax(Y_test_predicted, axis=1)
n_classes = np.size(Y_train, 1)
labels = range(n_classes)
cm = sklearn.metrics.confusion_matrix(y_test_true, y_test_predicted, labels)