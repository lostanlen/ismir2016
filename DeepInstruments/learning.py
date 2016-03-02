import DeepInstruments as di
import numpy as np


def build_graph(
        is_spiral,
        is_Z_supervision,
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
        drop1_proportion,
        dense1_channels,
        drop2_proportion,
        dense2_channels):
    if is_spiral:
        module = di.spiral
    else:
        module = di.scalogram
    return module.build_graph(
                is_Z_supervision,
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
                drop1_proportion,
                dense1_channels,
                drop2_proportion,
                dense2_channels)


def predict(graph, is_spiral, is_Z_supervision, X_test):
    if is_spiral:
        if is_Z_supervision:
            pass
        else:
            # X0 = X_test[:, :, xrange(0*12, 2*12), :]
            X1 = X_test[:, :, xrange(1*12, 3*12), :]
            X2 = X_test[:, :, xrange(2*12, 4*12), :]
            X3 = X_test[:, :, xrange(3*12, 5*12), :]
            X4 = X_test[:, :, xrange(4*12, 6*12), :]
            X5 = X_test[:, :, xrange(5*12, 7*12), :]
            # X6 = X_test[:, :, xrange(6*12, 8*12), :]
            y_predicted = np.argmax(
                graph.predict({"X1": X1, "X2": X2, "X3": X3,
                               "X4": X4, "X5": X5})["Y"],
                axis=1)
            return y_predicted
    else:
        if is_Z_supervision:
            y_predicted = di.singlelabel.predict(graph, X_test)
            return y_predicted
        else:
            y_predicted = np.argmax(graph.predict({"X": X_test})["Y"], axis=1)
            return y_predicted


def substract_and_mask(args):
    return (args[0] - args[1]) * args[2]


def train_on_batch(graph, is_spiral, is_Z_supervision,
                   X_batch, Y_batch, Z_batch, G_batch, masked_output):
    if is_spiral:
        # X0 = X_batch[:, :, xrange(0*12, 2*12), :]
        X1 = X_batch[:, :, xrange(1*12, 3*12), :]
        X2 = X_batch[:, :, xrange(2*12, 4*12), :]
        X3 = X_batch[:, :, xrange(3*12, 5*12), :]
        X4 = X_batch[:, :, xrange(4*12, 6*12), :]
        X5 = X_batch[:, :, xrange(5*12, 7*12), :]
        # X6 = X_batch[:, :, xrange(6*12, 8*12), :]
        if is_Z_supervision:
            pass
        else:
            loss = graph.train_on_batch({"X1": X1, "X2": X2,
                                         "X3": X3, "X4": X4, "X5": X5,
                                         "Y": Y_batch})
            return loss
    else:
        if is_Z_supervision:
            loss = graph.train_on_batch({"X": X_batch,
                                         "Y": Y_batch,
                                         "Z": Z_batch,
                                         "G": G_batch,
                                         "zero": masked_output})
            return loss
        else:
            loss = graph.train_on_batch({"X": X_batch, "Y": Y_batch})
            return loss
