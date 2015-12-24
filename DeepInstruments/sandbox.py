import librosa
import keras
import numpy as np
import os

import DeepInstruments as di

fmin = librosa.note_to_hz('A1')  # in Hertz
n_octaves = 7
n_bins_per_octave = 24
sr = 32000.0  # in Hertz
hop_duration = 0.016  # in seconds
decision_duration = 2.048  # in seconds
silence_threshold = -30 # in dB

instrument_list = ['Cl', 'Co', 'Fh', 'Gt', 'Ob', 'Pn', 'Tr', 'Vl']
solosDb8train_dir = '~/datasets/solosDb8/train'
file_paths = di.symbolic.get_paths(solosDb8train_dir, instrument_list, 'wav')
file_path = file_paths[0]

(X_train, Y_train) = di.solosdb.get_XY(
        file_paths,
        instrument_list, decision_duration, fmin, hop_duration,
        n_bins_per_octave, n_octaves, silence_threshold, sr)

X_train = np.reshape(X_train, (X_train.shape[0], np.prod(X_train.shape[1:])))

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adagrad

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(256, input_dim=21504, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(128, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(8, init='uniform'))
model.add(Activation('softmax'))
adagrad = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-06)
model.compile(loss='mean_squared_error', optimizer=adagrad)
model.fit(X_train, Y_train, nb_epoch=10, batch_size=16)


graph.compile(loss={'Y':'mean_squared_error'}, optimizer=adagrad)
history = graph.fit({'X':X_train, 'Y':Y_train}, nb_epoch=1)