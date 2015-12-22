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
instrument_list = ['Cl', 'Co', 'Fh', 'Gt', 'Ob', 'Pn', 'Tr', 'Vl']
n_instruments = len(instrument_list)

solosDb8train_dir = '~/datasets/solosDb8/train'
(X_train, Y_train) = di.solosdb.get_XY(
        di.symbolic.get_paths(solosDb8train_dir, instrument_list, 'wav'),
        instrument_list, decision_duration, fmin, hop_duration,
        n_bins_per_octave, n_octaves, sr)
input_shape = X_train.shape[1:]

rwc8_dir = '~/datasets/rwc8/'
rwc_paths = di.symbolic.get_paths(rwc8_dir, instrument_list, 'wav')
midis = [ di.rwc.get_midi(p, di.rwc.midi_offsets) for p in rwc_paths ]
