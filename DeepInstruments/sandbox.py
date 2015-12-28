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



from keras.models import Graph, Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import numpy as np
import sklearn


X_height = 168
X_width = 128
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
drop1_proportion = 0.5
dense1_channels = 256
drop2_proportion = 0.5
dense2_channels = 8

X = np.reshape(X_train, (X_train.shape[0], X_train.shape[2]*X_train.shape[3]))
graph = Graph()
graph.add_input(name="X", input_shape=(1, X_height, X_width))
flatten = Flatten()
graph.add_node(flatten, name="flatten", input="X")
dense2 = Dense(8, activation="softmax")
graph.add_node(dense2, name="dense2", input="flatten")
graph.add_output(name="Y", input="dense2")

# Train deep network
graph.compile(loss={'Y': 'categorical_crossentropy'}, optimizer="rmsprop")

history = graph.fit(
        {'X': X_train, 'Y': Y_train},
        nb_epoch=2,
        batch_size=128,
        verbose=1)

model = Sequential()
model.add(Dense(output_dim=8, input_dim=21504, init="glorot_uniform"))
model.add(Activation("tanh"))
model.compile(loss="categorical_crossentropy", optimizer="sgd")

model.fit(X_train, Y_train, nb_epoch=5, verbose=1)
Y_train_predicted = model.predict(X_train, verbose=1)
y_true = np.argmax(Y_train, axis=1)
y_predicted = np.argmax(Y_train_predicted, axis=1)

# Predict on training set
train_prediction = graph.predict({'X': X_train}, batch_size=1, verbose=1)
Y_train_predicted = train_prediction["Y"]

y_true = np.argmax(Y_train, axis=1)
y_predicted = np.argmax(Y_train_predicted, axis=1)
n_classes = np.size(Y_train, 1)
labels = range(n_classes)
cm = sklearn.metrics.confusion_matrix(y_true, y_predicted, labels)
