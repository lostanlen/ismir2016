import DeepInstruments as di
import librosa
import matplotlib.pyplot as plt
import numpy as np
import sklearn.ensemble
import keras
from keras.models import Graph
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.core import Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import math
import random

(test_stems, training_stems) = di.singlelabel.get_stems()

datagen = di.singlelabel.ScalogramGenerator(decision_length, fmin,
                                            hop_length, n_bins_per_octave,
                                            n_octaves, training_stems)


conv1_channels = 10
conv1_height = 3
conv1_width = 3
X_height = 96
X_width = 128

graph = Graph()

# Input
graph.add_input(name="X", input_shape=(1, X_height, X_width))
# graph.add_input(name="Z", input_shape=(1, X_height, X_width))
# graph.add_input(name="G", input_shape=(1, X_height, X_width))

# Shared layers
conv1 = Convolution2D(conv1_channels, conv1_height, conv1_width,
                      border_mode='same')
graph.add_node(conv1, name="conv1", input="X")

relu1 = LeakyReLU()
graph.add_node(relu1, name="relu1", input="conv1")

pool1_X = MaxPooling2D(pool_size=(pool1_height, pool1_width))
graph.add_node(pool1_X, name="pool1_X", input="relu1")

# Layers towards pitch
# pool1_Z = MaxPooling2D(pool_size=(pool1_height, pool1_width))
# graph.add_node(pool1_Z, name="pool1_Z", input="Z")
#
# pool1_G = MaxPooling2D(pool_size=(pool1_height, pool1_width))
# graph.add_node(pool1_G, name="pool1_G", input="G")

graph.add_output(name='masked_Z', input='pool1_X')

graph.compile(loss={'masked_Z': 'mse'}, optimizer="sgd")
