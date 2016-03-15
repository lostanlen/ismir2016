import DeepInstruments as di
import numpy as np
import scipy.signal


def build_graph(
        is_spiral,
        n_bins_per_octave,
        n_octaves,
        X_width,
        conv1_channels,
        conv1_height,
        conv1_width,
        pool1_height,
        pool1_width,
        conv2_channels,
        conv2_height,
        conv2_width,
        pool2_height,
        pool2_width,
        dense1_channels,
        dense2_channels):
    if is_spiral:
        module = di.spiral
    else:
        module = di.scalogram
    return module.build_graph(
                n_bins_per_octave,
                n_octaves,
                X_width,
                conv1_channels,
                conv1_height,
                conv1_width,
                pool1_height,
                pool1_width,
                conv2_channels,
                conv2_height,
                conv2_width,
                pool2_height,
                pool2_width,
                dense1_channels,
                dense2_channels)


def predict(graph, is_spiral, X_test, offset):
    if is_spiral:
        Q = 12
        X0 = X_test * window(X_test, Q, 0) - offset/6
        X1 = X_test * window(X_test, Q, 1*Q) - offset/6
        X2 = X_test * window(X_test, Q, 2*Q) - offset/6
        X3 = X_test * window(X_test, Q, 3*Q) - offset/6
        X4 = X_test * window(X_test, Q, 4*Q) - offset/6
        X5 = X_test * window(X_test, Q, 5*Q) - offset/6
        class_probs = graph.predict({"X0": X0, "X1": X1, "X2": X2,
                                     "X3": X3, "X4": X4, "X5": X5})["Y"]
    else:
        X = X_test - offset
        class_probs = graph.predict({"X": X})["Y"]
    return class_probs


def substract_and_mask(args):
    return (args[0] - args[1]) * args[2]


def train_on_batch(graph, is_spiral, X_batch, Y_batch, offset):
    if is_spiral:
        Q = 12
        X0 = X_batch * window(X_batch, Q, 0) - offset/6
        X1 = X_batch * window(X_batch, Q, 1*Q) - offset/6
        X2 = X_batch * window(X_batch, Q, 2*Q) - offset/6
        X3 = X_batch * window(X_batch, Q, 3*Q) - offset/6
        X4 = X_batch * window(X_batch, Q, 4*Q) - offset/6
        X5 = X_batch * window(X_batch, Q, 5*Q) - offset/6
        loss = graph.train_on_batch({"X0": X0, "X1": X1, "X2": X2,
                                     "X3": X3, "X4": X4, "X5": X5,
                                     "Y": Y_batch})
        return loss
    else:
        X = X_batch - offset
        loss = graph.train_on_batch({"X": X, "Y": Y_batch})
        return loss


def window(X_batch, top_width, start):
    phi = np.zeros((1, 1, X_batch.shape[2], 1))
    support = xrange(start, start + 3*top_width)
    phi[0, 0, support, 0] = scipy.signal.tukey(3*top_width, alpha=1.0/3)
    return phi