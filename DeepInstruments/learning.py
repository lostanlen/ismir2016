import DeepInstruments as di
import numpy as np
import scipy.signal


def substract_and_mask(args):
    return (args[0] - args[1]) * args[2]

def window(X_batch, start, full_width, top_width=None):
    if not top_width:
        top_width = full_width / 3
    alpha = 1.0 - float(top_width) / full_width
    phi = np.zeros((1, 1, X_batch.shape[2], 1))
    support = xrange(start, start + full_width)
    phi[0, 0, support, 0] = scipy.signal.tukey(full_width, alpha=alpha)
    return phi
