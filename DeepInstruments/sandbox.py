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
conv2_channels = 10
conv2_height = 8
conv2_width = 8
pool2_height = 8
pool2_width = 8
dense1_channels = 32
drop1_proportion = 0.5
dense2_channels = 16
drop2_proportion = 0.5
dense3_channels = 8


graph = di.learning.build_graph(
    X_height,
    X_width,
    conv1_channels,
    conv1_height,
    conv1_width,
    pool1_height,
    pool1_width,
    conv2_channels,
    conv2_height,
    conv2_width,
    pool2_height,
    pool2_width,
    dense1_channels,
    drop1_proportion,
    dense2_channels,
    drop2_proportion,
    dense3_channels)