from joblib import Memory, Parallel, delayed
import glob
import keras
import librosa
import numpy as np
import numpy.matlib
import os

import audio
import rwc
import solosdb
import symbolic

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD


solosDb8test_dir = '~/datasets/solosDb8/test'


fmin = librosa.note_to_hz('A1')  # in Hertz
n_octaves = 7
n_bins_per_octave = 24
sr = 32000.0  # in Hertz
hop_duration = 0.016  # in seconds
decision_duration = 2.048  # in seconds
instrument_list = ['Cl', 'Co', 'Fh', 'Gt', 'Ob', 'Pn', 'Tr', 'Vl']
n_instruments = len(instrument_list)

solosDb8train_dir = '~/datasets/solosDb8/train'
(X_train, Y_train) = solosdb.get_XY(
        symbolic.get_paths(solosDb8train_dir, instrument_list, 'wav'),
        instrument_list,
        decision_duration, fmin, hop_duration, n_bins_per_octave, n_octaves, sr)
input_shape = X_train.shape[1:]

rwc8_dir = '~/datasets/rwc8/'
rwc_paths = symbolic.get_paths(rwc8_dir, instrument_list, 'wav')
pooling_strides = np.array([2, 2])
rwc_offsets = dict(Cl=librosa.note_to_midi('D3'),
                          Co=librosa.note_to_midi('E1'),
                          Fh=librosa.note_to_midi('D2'),
                          Gt=librosa.note_to_midi('E2'),
                          Ob=librosa.note_to_midi('Bb3'),
                          Pn=librosa.note_to_midi('A0'),
                          Tr=librosa.note_to_midi('F#3'),
                          Vl=librosa.note_to_midi('G3'))

file_paths = get_paths('~/datasets/rwc8', instrument_list, 'wav')
midis = [get_RWC_midi(p, rwc_offsets) for p in file_paths]
