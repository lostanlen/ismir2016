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

from keras.models import Graph

from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D


X_height = 168
X_width = 128
conv1_channels = 50
conv1_height = 96
conv1_width = 32
pool1_height = 8
pool1_width = 8
conv2_channels = 30
conv2_height = 7
conv2_width = 7
pool2_height = 3
pool2_width = 3
drop1_proportion = 0.5
dense1_channels = 256
drop2_proportion = 0.5
dense2_channels = 8

graph = Graph()

graph.add_input(name="X", input_shape=(1, X_height, X_width))

conv1 = Convolution2D(conv1_channels, conv1_height, conv1_width)
graph.add_node(conv1, name="conv1", input="X")

relu1 = LeakyReLU()
graph.add_node(relu1, name="relu1", input="conv1")

pool1 = MaxPooling2D(pool_size=(pool1_height, pool1_width))
graph.add_node(pool1, name="pool1", input="relu1")

conv2 = Convolution2D(conv2_channels, conv2_height, conv2_width)
graph.add_node(conv2, name="conv2", input="pool1")

relu2 = LeakyReLU()
graph.add_node(relu2, name="relu2", input="conv2")

pool2 = MaxPooling2D(pool_size=(pool2_height, pool2_width))
graph.add_node(pool2, name="pool2", input="relu2")

flatten = Flatten()
graph.add_node(flatten, name="flatten", input="pool2")

drop1 = Dropout(drop1_proportion)
graph.add_node(drop1, name="drop1", input="flatten")

dense1 = Dense(dense1_channels, activation="relu")
graph.add_node(dense1, name="dense1", input="drop1")

drop2 = Dropout(drop2_proportion)
graph.add_node(drop2, name="drop2", input="dense1")

dense2 = Dense(dense2_channels, activation="relu")
graph.add_node(dense2, name="dense2", input="drop2")

graph.add_output(name="Y", input="dense2")


adagrad = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-06)
graph.compile(loss={'Y':'mean_squared_error'}, optimizer=adagrad)
history = graph.fit({'X':X_train, 'Y':Y_train}, nb_epoch=1, batch_size=32)