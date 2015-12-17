from joblib import Memory, Parallel, delayed
import glob
import keras
import librosa
import numpy as np
import numpy.matlib
import os

solosDb8train_dir = '~/datasets/solosDb8/train'
solosDb8test_dir = '~/datasets/solosDb8/test'
memory = Memory(cachedir='solosDb8_train')
cached_cqt = memory.cache(perceptual_cqt, verbose=0)

fmin = librosa.note_to_hz('A1')  # in Hertz
n_octaves = 7
n_bins_per_octave = 24
sr = 32000.0  # in Hertz
hop_duration = 0.016  # in seconds
decision_duration = 2.048  # in seconds
instrument_list = ['Cl', 'Co', 'Fh', 'Gt', 'Ob', 'Pn', 'Tr', 'Vl']
(X_train, Y_train) = get_XY(
        solosDb8train_dir,
        instrument_list,
        decision_duration, fmin, hop_duration, n_bins_per_octave, n_octaves, sr)

input_shape = X_train.shape[1:]

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D

model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape = input_shape))



def get_XY(
        dataset_dir,
        instrument_list,
        decision_duration,
        fmin,
        hop_duration,
        n_bins_per_octave,
        n_octaves,
        sr):
    file_paths = get_paths(dataset_dir, instrument_list, 'wav')
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

def get_paths(dir, instrument_list, extension):
    dir = os.path.expanduser(dir)
    walk = os.walk(dir)
    regex = '*.' + extension
    file_paths = [p for d in walk for p in glob.glob(os.path.join(d[0], regex))]
    return [p for p in file_paths if os.path.split(os.path.split(p)[0])[1] in instrument_list]


def get_instrument(file_path, instrument_list):
    instrument_str = os.path.split(os.path.split(file_path)[0])[1]
    n_instruments = len(instrument_list)
    instrument_onehot = np.zeros(n_instruments)
    instrument_onehot[instrument_list.index(instrument_str)] = 1.0
    return instrument_onehot

def expand_instruments(fs, ns):
    items = [ numpy.matlib.repmat(fs[i], ns[i], 1) for i in range(len(fs))]
    return np.vstack(items)


def perceptual_cqt(
        file_path,
        decision_duration,
        fmin,
        hop_duration,
        n_bins_per_octave,
        n_octaves,
        sr):
    y, y_sr = librosa.load(file_path)
    if y_sr != sr:
        y = librosa.resample(y, y_sr, sr)
    hop_length = hop_duration * sr
    decision_length = decision_duration * sr / hop_length
    n_bins = n_octaves * n_bins_per_octave
    freqs = librosa.cqt_frequencies(
            bins_per_octave=n_bins_per_octave,
            fmin=fmin,
            n_bins=n_bins)
    CQT = librosa.hybrid_cqt(
            y,
            bins_per_octave=n_bins_per_octave,
            fmin=fmin,
            hop_length=hop_length,
            n_bins=n_bins,
            sr=sr)
    audio_features = librosa.perceptual_weighting(
            CQT ** 2,
            freqs,
            ref_power=1.0)
    n_hops = audio_features.shape[1]
    n_windows = int(n_hops / decision_length)
    n_hops_truncated = n_windows * decision_length
    audio_features = np.transpose(audio_features)
    audio_features = audio_features[:n_hops_truncated, :]
    new_shape = (n_windows, decision_length, n_bins)
    audio_features = np.reshape(audio_features, new_shape)
    return np.transpose(audio_features, (0, 2, 1))
