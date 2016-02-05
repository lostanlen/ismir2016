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