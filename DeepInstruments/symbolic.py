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