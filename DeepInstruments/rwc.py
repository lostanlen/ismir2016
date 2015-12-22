from joblib import Memory, Parallel, delayed
import numpy as np
import os

import audio
import symbolic

def get_midi(file_path, offset_dictionary):
    instrument_str = os.path.split(os.path.split(file_path)[0])[1]
    file_index = os.path.split(os.path.split(file_path)[1])[1].split('_')[1].split('.')[0]
    # we substract 1 because RWC has one-based indexing
    return offset_dictionary[instrument_str] + int(file_index) - 1

def get_XY(
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
    memory = Memory(cachedir='/tmp/joblib')
    cached_cqt = memory.cache(audio.perceptual_cqt, verbose=0)
    file_cqts = Parallel(n_jobs=-1, verbose=20)(delayed(cached_cqt)(
        file_path,
        decision_duration,
        fmin,
        hop_duration,
        n_bins_per_octave,
        n_octaves,
        sr) for file_path in file_paths)
    # Get first window of each file and stack according to first dimension
    X = np.vstack([file_cqt[0, :, :] for file_cqt in file_cqts])
    # Reshape to Theano-friendly format
    new_shape = X.shape
    new_shape = (new_shape[0], 1, new_shape[1], new_shape[2])
    X = np.reshape(X, new_shape)
    file_instruments = [symbolic.get_instrument(p, instrument_list) for p in file_paths]
    Y = np.vstack(file_instruments)
    return (X, Y)