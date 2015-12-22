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

solosDb8train_dir = '~/datasets/solosDb8/train'
solosDb8test_dir = '~/datasets/solosDb8/test'
rwc8_dir = '~/datasets/rwc8/'
memory = Memory(cachedir='/tmp/joblib')
cached_cqt = memory.cache(perceptual_cqt, verbose=0)
fmin = librosa.note_to_hz('A1')  # in Hertz
n_octaves = 7
n_bins_per_octave = 24
sr = 32000.0  # in Hertz
hop_duration = 0.016  # in seconds
decision_duration = 2.048  # in seconds
instrument_list = ['Cl', 'Co', 'Fh', 'Gt', 'Ob', 'Pn', 'Tr', 'Vl']
n_instruments = len(instrument_list)
solosDb_train_paths = get_paths(solosDb8train_dir, instrument_list, 'wav')

(X_train, Y_train) = get_solosDb_XY(
        train_paths,
        instrument_list,
        decision_duration, fmin, hop_duration, n_bins_per_octave, n_octaves, sr)

input_shape = X_train.shape[1:]

model = Sequential()
model.add(Dense(8, input_shape=(21504,), init="uniform", activation="softmax"))
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
model.fit(X_train, Y_train, batch_size=32, nb_epoch=1)

rwc_paths = get_paths(rwc8_dir, instrument_list, 'wav')
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

def get_solosDb_XY(
        file_paths,
        instrument_list,
        decision_duration,
        fmin,
        hop_duration,
        n_bins_per_octave,
        n_octaves,
        sr):
    # Run perceptual CQT in parallel with joblib
    # n_jobs = -1 means that all CPUs are used
    file_cqts = Parallel(n_jobs=-1, verbose=20)(delayed(cached_cqt)(
        file_path,
        decision_duration,
        fmin,
        hop_duration,
        n_bins_per_octave,
        n_octaves,
        sr) for file_path in file_paths)
    # Reduce all CQTs into one
    X = np.vstack(file_cqts)
    # Reshape to Theano-friendly format
    new_shape = X.shape
    new_shape = (new_shape[0], 1, new_shape[1], new_shape[2])
    X = np.reshape(X, new_shape)
    file_instruments = [get_instrument(p, instrument_list) for p in file_paths]
    n_items_per_file = [cqt.shape[0] for cqt in file_cqts]
    Y = expand_instruments(file_instruments, n_items_per_file)
    return (X, Y)

def get_rwc_Z(
        file_paths,
        fmin,
        n_bins_per_octave,
        n_octaves,
        pooling_strides,
        rwc_offsets):
    cqt_midimin = librosa.hz_to_midi(fmin)
    n_bins = n_bins_per_octave * n_octaves
    n_rows = n_bins / np.prod(pooling_strides)
    midis = [ get_RWC_midi(p, rwc_offsets) for p in file_paths ]
    n_files = len(file_paths)
    onehots = np.zeros((n_files, n_rows))
    for file_index in range(n_files):
        midi = midis[file_index]
        row = int(((midi - cqt_midimin) / n_bins) * n_rows)
        onehots[file_index, row] = 1.0
    return onehots

def get_paths(dir, instrument_list, extension):
    dir = os.path.expanduser(dir)
    walk = os.walk(dir)
    regex = '*.' + extension
    file_paths = [p for d in walk for p in glob.glob(os.path.join(d[0], regex))]
    return [p for p in file_paths if os.path.split(os.path.split(p)[0])[1] in instrument_list]

