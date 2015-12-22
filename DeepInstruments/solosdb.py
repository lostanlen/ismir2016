from joblib import Memory, Parallel, delayed
import numpy as np

import audio
import os
import symbolic

def get_XY(
        file_paths,
        instrument_list,
        decision_duration,
        fmin,
        hop_duration,
        n_bins_per_octave,
        n_octaves,
        silence_threshold,
        sr):
    # Run perceptual CQT in parallel with joblib
    # n_jobs = -1 means that all CPUs are used
    memory = Memory(cachedir=os.path.expanduser('~/joblib'))
    cached_cqt = memory.cache(audio.perceptual_cqt, verbose=0)
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
    file_instruments = [symbolic.get_instrument(p, instrument_list) for p in
                        file_paths]
    n_items_per_file = [cqt.shape[0] for cqt in file_cqts]
    Y = symbolic.expand_instruments(file_instruments, n_items_per_file)
    return (X, Y)
