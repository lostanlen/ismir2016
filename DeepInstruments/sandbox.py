import librosa

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

import DeepInstruments as di

fmin = librosa.note_to_hz('A1')  # in Hertz
n_octaves = 7
n_bins_per_octave = 24
sr = 32000.0  # in Hertz
hop_duration = 0.016  # in seconds
decision_duration = 2.048  # in seconds
silence_threshold = 1e-3

instrument_list = ['Cl', 'Co', 'Fh', 'Gt', 'Ob', 'Pn', 'Tr', 'Vl']

solosDb8train_dir = '~/datasets/solosDb8/train'
(X_train, Y_train) = di.solosdb.get_XY(
        di.symbolic.get_paths(solosDb8train_dir, instrument_list, 'wav'),
        instrument_list, decision_duration, fmin, hop_duration,
        n_bins_per_octave, n_octaves, sr)



conv1_width = 32
conv1_height = 96
conv1_channels = 50

pool1_width = 8
pool1_height = 8

conv2_width = 8
conv2_height = 8
conv2_channels = 30

pool2_width = 3
pool2_height = 3

drop_proportion = 0.5

dense1_input = 256

dense2_input = n_instruments
