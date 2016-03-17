import DeepInstruments as di
import numpy as np
import scipy.signal


def substract_and_mask(args):
    return (args[0] - args[1]) * args[2]


def train_on_batch(graph, is_spiral, X_batch, Y_batch, offsets):
    if is_spiral:
        Q = 12
        X0 = X_batch[:, :, (0*Q):(7*Q), :] - offsets[0]
        X1 = X_batch[:, :, (1*Q):(8*Q), :] - offsets[1]
        loss = graph.train_on_batch({
            "X0": X0,
            "X1": X1,
            "Y": Y_batch})
        return loss
    else:
        X = X_batch - offsets[0]
        loss = graph.train_on_batch({"X": X, "Y": Y_batch})
        return loss


def window(X_batch, start, full_width, top_width=None):
    if not top_width:
        top_width = full_width / 3
    alpha = 1.0 - float(top_width) / full_width
    phi = np.zeros((1, 1, X_batch.shape[2], 1))
    support = xrange(start, start + full_width)
    phi[0, 0, support, 0] = scipy.signal.tukey(full_width, alpha=alpha)
    return phi
