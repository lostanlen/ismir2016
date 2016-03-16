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


def predict(graph, is_spiral, X_test, offsets):
    if is_spiral:
        Q = 12
        X1 = X_test[:, :, (0*Q):(4*Q), :] - offsets[0]
        X2 = X_test[:, :, (1*Q):(5*Q), :] - offsets[1]
        X3 = X_test[:, :, (2*Q):(6*Q), :] - offsets[2]
        X4 = X_test[:, :, (3*Q):(7*Q), :] - offsets[3]
        X5 = X_test[:, :, (4*Q):(8*Q), :] - offsets[4]
        class_probs = graph.predict({
            "X1": X1,
            "X2": X2,
            "X3": X3,
            "X4": X4,
            "X5": X5})["Y"]
    else:
        X = X_test - offsets[0]
        class_probs = graph.predict({"X": X})["Y"]
    return class_probs


def substract_and_mask(args):
    return (args[0] - args[1]) * args[2]


def train_on_batch(graph, is_spiral, X_batch, Y_batch, offsets):
    if is_spiral:
        Q = 12
        X1 = X_batch[:, :, (0*Q):(4*Q), :] - offsets[0]
        X2 = X_batch[:, :, (1*Q):(5*Q), :] - offsets[1]
        X3 = X_batch[:, :, (2*Q):(6*Q), :] - offsets[2]
        X4 = X_batch[:, :, (3*Q):(7*Q), :] - offsets[3]
        X5 = X_batch[:, :, (4*Q):(8*Q), :] - offsets[4]
        loss = graph.train_on_batch({
            "X1": X1,
            "X2": X2,
            "X3": X3,
            "X4": X4,
            "X5": X5,
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
