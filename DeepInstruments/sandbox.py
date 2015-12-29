import librosa
import keras
import numpy as np
import os
import sklearn

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
(X_sdbtrain_list, Y_sdbtrain_list, Xtrain_mean, Xtrain_std) = di.solosdb.get_XY(
        train_file_paths,
        instrument_list, decision_duration, fmin, hop_duration,
        n_bins_per_octave, n_octaves, sr)

# Compute audio features on test set
solosDb8test_dir = '~/datasets/solosDb8/test'
test_file_paths = di.symbolic.get_paths(solosDb8test_dir, instrument_list, 'wav')
(X_sdbtest_list, Y_sdbtest_list) = di.solosdb.get_XY(
        test_file_paths,
        instrument_list, decision_duration, fmin, hop_duration,
        n_bins_per_octave, n_octaves, sr)

n_instruments = len(instrument_list)
for instrument_id in range(n_instruments):
    X_instrument = X_sdbtest_list[instrument_id]
    X_instrument = np.transpose(X_instrument)
    (n_hops, n_bins) = X_instrument.shape
    hop_length = hop_duration * sr
    decision_length = decision_duration * sr / hop_length
    n_windows = int(n_hops / decision_length)
    n_hops_truncated = n_windows * decision_length
    X_instrument = X_instrument[:n_hops_truncated, :]
    new_shape = (n_windows, decision_length, n_bins)
    X_instrument = np.reshape(X_instrument, new_shape)
    X_instrument = np.transpose(X_instrument, (0, 2, 1))
    X_sdbtest_list[instrument_id] = X_instrument
    # TODO: discard silent windows
    Y_instrument = Y_sdbtest_list[instrument_id]
    Y_instrument = np.ones((n_windows, 1)) * Y_instrument
    Y_sdbtest_list[instrument_id] = Y_instrument

X_test = np.vstack(X_sdbtest_list)
X_test = (X_test - X_train_mean) / X_train_std
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1], X_test.shape[2]))
Y_test = np.vstack(Y_sdbtest_list)



# Build generator
datagen = di.learning.ChunkGenerator(decision_duration, hop_duration, silence_threshold)

# Build graph model
graph = di.learning.build_graph(
    X_height=168,
    X_width=128,
    conv1_channels=100,
    conv1_height=96,
    conv1_width=32,
    pool1_height=7,
    pool1_width=7,
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

# Compile model
graph.compile(loss={'Y': 'categorical_crossentropy'}, optimizer="adagrad")

# Train model
from keras.utils.generic_utils import Progbar

n_epochs = 100
batch_size = 128
epoch_size = 4096
every_n_epoch = 5
labels = range(len(instrument_list))


for epoch_id in range(n_epochs):
    dataflow = datagen.flow(
        X_sdbtrain_list,
        Y_sdbtrain_list,
        batch_size=batch_size,
        seed=None,
        epoch_size=epoch_size)
    print 'Epoch ', 1 + epoch_id
    progbar = keras.utils.generic_utils.Progbar(epoch_size)
    batch_id = 0
    for (X_batch, Y_batch) in dataflow:
        batch_id += 1
        loss = graph.train_on_batch({"X": X_batch, "Y": Y_batch})
        progbar.update(batch_id * batch_size)

    if np.mod(epoch_id, every_n_epoch) == 0:
        # Get training accuracy
        dataflow = datagen.flow(
            X_sdbtrain_list,
            Y_sdbtrain_list,
            batch_size=batch_size,
            seed=None,
            epoch_size=epoch_size)
        y_train_true = np.zeros(epoch_size, dtype=int)
        y_train_predicted = np.zeros(epoch_size, dtype=int)
        batch_id = 0
        for (X_batch, Y_batch) in dataflow:
            Y_batch_predicted = graph.predict_on_batch({"X": X_batch})
            Y_batch_predicted = np.hstack(Y_batch_predicted)
            y_batch_predicted = np.argmax(Y_batch_predicted, axis=1)
            batch_range = xrange(batch_id*batch_size, (batch_id+1)*batch_size)
            y_train_predicted[batch_range] = y_batch_predicted
            y_batch_true = np.argmax(Y_batch, axis=1)
            y_train_true[batch_range] = y_batch_true
            batch_id += 1

        cm = sklearn.metrics.confusion_matrix(y_train_true, y_train_predicted, labels)
        cm = cm.astype(np.float64)
        cm = cm / cm.sum(axis=1)[:, np.newaxis]
        train_accuracy = np.mean(np.diag(cm))
        print "train accuracy = ", train_accuracy

        # Get test accracy
        test_prediction = graph.predict({"X": X_test})
        Y_test_predicted = test_prediction["Y"]
        y_test_predicted = np.argmax(Y_test_predicted, axis=1)
        y_test_true = np.argmax(Y_test, axis=1)
        cm = sklearn.metrics.confusion_matrix(y_test_true, y_test_predicted, labels)
        cm = cm.astype(np.float64)
        cm = cm / cm.sum(axis=1)[:, np.newaxis]
        test_accuracy = np.mean(np.diag(cm))
        print "test accuracy = ", test_accuracy
