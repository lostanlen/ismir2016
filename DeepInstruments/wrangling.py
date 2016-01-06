import DeepInstruments as di
import joblib
import numpy as np
import os


def chunk_test_set(X_test_list, Y_test_list, hop_duration, sr):
    pass


def get_XY(file_paths,
           instrument_list, decision_duration, fmin, hop_duration,
           n_bins_per_octave, n_octaves, sr):
    # Run perceptual CQT in parallel with joblib
    # n_jobs = -1 means that all CPUs are used
    memory = joblib.Memory(cachedir=os.path.expanduser('~/joblib'))
    cached_cqt = memory.cache(di.audio.perceptual_cqt, verbose=0)
    X_files = joblib.Parallel(n_jobs=-1, verbose=0)(
            joblib.delayed(cached_cqt)(file_path,
                                       decision_duration,
                                       fmin,
                                       hop_duration,
                                       n_bins_per_octave,
                                       n_octaves,
                                       sr) for file_path in file_paths)
    # Gather file CQTs X according to each class
    # Generate one-hot vectors Y for each class
    n_files = len(file_paths)
    file_range = range(n_files)
    n_instruments = len(instrument_list)
    instr_range = range(n_instruments)
    Y_files = \
        [di.symbolic.get_instrument(p, instrument_list) for p in file_paths]
    X_list = []
    Y_list = []
    for i in instr_range:
        X_instrument = [X_files[f] for f in file_range if Y_files[f] == i]
        Y_instrument = np.zeros((1, n_instruments), dtype=np.float32)
        Y_instrument[0, i] = 1.0
        X_list.append(np.hstack(X_instrument))
        Y_list.append(Y_instrument)
    return (X_list, Y_list)


def instrument_stems(instrument_names, track):
    stem_names = [s.instrument.name for s in track.stems]
    n_instruments = len(instrument_names)
    n_stems = len(stem_names)
    instrument_matches = []
    for instrument_index in range(n_instruments):
        instrument_name = instrument_names[instrument_index]
        instrument_match = []
        for stem_index in range(n_stems):
            stem_name = stem_names[stem_index]
            if stem_name == instrument_name:
                instrument_match.append(stem_index)
        instrument_matches.append(instrument_match)
    return instrument_matches
