from joblib import Memory, Parallel, delayed
import numpy as np
import os

from DeepInstruments import audio, symbolic

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
    memory = Memory(cachedir=os.path.expanduser('~/joblib'))
    cached_cqt = memory.cache(audio.perceptual_cqt, verbose=0)
    X_files = Parallel(n_jobs=-1, verbose=20)(delayed(cached_cqt)(
            file_path,
            decision_duration,
            fmin,
            hop_duration,
            n_bins_per_octave,
            n_octaves,
            sr) for file_path in file_paths)
    # Gather file CQTs according to each class
    n_files = len(file_paths)
    file_range = range(n_files)
    n_instruments = len(instrument_list)
    instr_range = range(n_instruments)
    Y_files = [symbolic.get_instrument(p, instrument_list) for p in file_paths]
    X_list = [np.hstack([X_files[f] for f in file_range if Y_files[f]==i]) for i in instr_range]
    # Generate one-hot vectors for each class
    Y_list = [np.zeros((1, n_instruments), dtype=np.float32) for i in instr_range]
    for i in instr_range:
        Y_list[i][0, i] = 1.0
    # Standardize globally
    X_global = np.hstack(X_files)
    X_mean = np.mean(X_global)
    X_var = np.std(X_global)
    X_list = [(X-X_mean)/X_var for X in X_list]
    return (X_list, Y_list, X_mean, X_var)
