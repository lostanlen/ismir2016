import numpy as np
import numpy.matlib
import os


def expand_instruments(fs, ns):
    items = [ numpy.matlib.repmat(fs[i], ns[i], 1) for i in range(len(fs))]
    return np.vstack(items)


def get_instrument(file_path, instrument_list):
    instrument_str = os.path.split(os.path.split(file_path)[0])[1]
    n_instruments = len(instrument_list)
    instrument_onehot = np.zeros(n_instruments)
    instrument_onehot[instrument_list.index(instrument_str)] = 1.0
    return instrument_onehot


def get_paths(dir, instrument_list, extension):
    dir = os.path.expanduser(dir)
    walk = os.walk(dir)
    regex = '*.' + extension
    file_paths = [p for d in walk for p in glob.glob(os.path.join(d[0], regex))]
    return [p for p in file_paths if os.path.split(os.path.split(p)[0])[1] in instrument_list]