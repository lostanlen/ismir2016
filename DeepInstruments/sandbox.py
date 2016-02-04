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
from keras.layers.convolutional import Convolution2D, AveragePooling1D,\
                                       MaxPooling2D, MaxPooling1D, \
                                        AveragePooling2D
from keras.layers.core import LambdaMerge
from keras.layers.core import Permute, Reshape
import math
import random


X_height = 96
X_width = 128
conv1_channels = 2
conv1_height = 3
conv1_width = 3
pool1_height = 3
pool1_width = 3


graph = Graph()

# Input
graph.add_input(name="X", input_shape=(1, X_height, X_width))
graph.add_input(name="Z", input_shape=(1, X_height, X_width))
graph.add_input(name="G", input_shape=(1, X_height, X_width))


# Shared layers
conv1 = Convolution2D(conv1_channels, conv1_height, conv1_width,
                      border_mode='same')
graph.add_node(conv1, name="conv1", input="X")

relu1 = LeakyReLU()
graph.add_node(relu1, name="relu1", input="conv1")

pool1_X = MaxPooling2D(pool_size=(pool1_height, pool1_width))
graph.add_node(pool1_X, name="pool1_X", input="relu1")

reshaped_X = Reshape((conv1_channels, 32*42))
graph.add_node(reshaped_X, name="reshaped_X", input="pool1_X")

collapsed_X = AveragePooling1D(pool_length=2, stride=2)
graph.add_node(collapsed_X, name="collapsed_X", input="reshaped_X")

pool1_Z = MaxPooling2D(pool_size=(pool1_height, pool1_width))
graph.add_node(pool1_Z, name="pool1_Z", input="Z")

reshaped_Z = Reshape((1, 32*42))
graph.add_node(reshaped_Z, name="reshaped_Z", input="pool1_Z")

pool1_G = MaxPooling2D(pool_size=(pool1_height, pool1_width))
graph.add_node(pool1_G, name="pool1_G", input="G")

reshaped_G = Reshape((1, 32*42))
graph.add_node(reshaped_G, name="reshaped_G", input="pool1_G")

melodic_error = LambdaMerge([collapsed_X, reshaped_Z, reshaped_G],
                            di.learning.substract_and_mask)
graph.add_node(melodic_error, name="melodic_error")

graph.add_output(name="melodic_error", input="melodic_error")

graph.compile(loss={"melodic_error": "mse"}, optimizer="sgd")
