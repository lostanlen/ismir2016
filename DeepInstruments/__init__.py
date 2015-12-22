from joblib import Memory, Parallel, delayed
import glob
import keras
import librosa
import numpy as np
import numpy.matlib
import os

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD