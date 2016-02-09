import DeepInstruments as di
import keras
from keras.models import Graph
from keras.layers.advanced_activations import LeakyReLU, ParametricSoftplus
from keras.layers.core import Dense, Dropout, Flatten, LambdaMerge, Reshape
from keras.layers.core import Permute
from keras.layers.convolutional import AveragePooling1D, Convolution2D, \
                                       MaxPooling2D

import math
import numpy as np
import random


def build_graph(
        X_height,
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
        drop1_proportion,
        dense2_channels,
        drop2_proportion,
        dense3_channels):
    graph = Graph()

    # Input
    graph.add_input(name="X", input_shape=(1, X_height, X_width))
    graph.add_input(name="Z", input_shape=(1, X_height, X_width))
    graph.add_input(name="G", input_shape=(1, X_height, X_width))

    # Shared layers
    conv1 = Convolution2D(conv1_channels, conv1_height, conv1_width,
                          border_mode="same")
    graph.add_node(conv1, name="conv1", input="X")

    relu1 = LeakyReLU()
    graph.add_node(relu1, name="relu1", input="conv1")

    pool1_X = MaxPooling2D(pool_size=(pool1_height, pool1_width))
    graph.add_node(pool1_X, name="pool1_X", input="relu1")

    # Layers towards instrument target
    conv2 = Convolution2D(conv2_channels, conv2_height, conv2_width)
    graph.add_node(conv2, name="conv2", input="pool1_X")

    relu2 = LeakyReLU()
    graph.add_node(relu2, name="relu2", input="conv2")

    pool2 = MaxPooling2D(pool_size=(pool2_height, pool2_width))
    graph.add_node(pool2, name="pool2", input="relu2")

    flatten = Flatten()
    graph.add_node(flatten, name="flatten", input="pool2")

    dense1 = Dense(dense1_channels, activation="relu")
    graph.add_node(dense1, name="dense1", input="flatten")

    drop1 = Dropout(drop1_proportion)
    graph.add_node(drop1, name="drop1", input="dense1")

    dense2 = Dense(dense2_channels, activation="relu")
    graph.add_node(dense2, name="dense2", input="drop1")

    drop2 = Dropout(drop2_proportion)
    graph.add_node(drop2, name="drop2", input="dense2")

    dense3 = Dense(dense3_channels, activation="softmax")
    graph.add_node(dense3, name="dense3", input="drop2")

    # Pooling of symbolic activations Z (piano-roll) and G (melody gate)
    pool1_Z = MaxPooling2D(pool_size=(pool1_height, pool1_width))
    graph.add_node(pool1_Z, name="pool1_Z", input="Z")

    pool1_G = MaxPooling2D(pool_size=(pool1_height, pool1_width))
    graph.add_node(pool1_G, name="pool1_G", input="G")

    # Layers towards melodic target
    flat_shape = (pool1_X.output_shape[1],
                pool1_X.output_shape[2] * pool1_X.output_shape[3])
    reshaped_X = Reshape(dims=flat_shape)
    graph.add_node(reshaped_X, name="reshaped_X", input="pool1_X")

    collapsed_X = AveragePooling1D(pool_length=conv1_channels)
    graph.add_node(collapsed_X, name="collapsed_X", input="reshaped_X")

    softplus_X = ParametricSoftplus()
    graph.add_node(softplus_X, name="softplus_X", input="collapsed_X")

    rectangular_shape = (1, pool1_X.output_shape[2], pool1_X.output_shape[3])
    toplevel_X = Reshape(dims=rectangular_shape)
    graph.add_node(toplevel_X, name="toplevel_X", input="softplus_X")

    melodic_error = LambdaMerge([toplevel_X, pool1_Z, pool1_G],
                                di.learning.substract_and_mask)
    graph.add_node(melodic_error, name="melodic_error",
                   inputs=["toplevel_X", "pool1_Z", "pool1_G"])

    graph.add_output(name="Y", input="dense3")
    graph.add_output(name="zero", input="melodic_error")

    return graph


def substract_and_mask(args):
    return (args[0] - args[1]) * args[2]


def run_graph(X_train_list, Y_train_list, X_test, Y_test,
              batch_size, datagen, epoch_size, every_n_epoch,
              graph, n_epochs):
    loss_history = []
    train_accuracies_history = []
    test_accuracies_history = []
    for epoch_id in xrange(n_epochs):
        dataflow = datagen.flow(
            X_train_list,
            Y_train_list,
            batch_size=batch_size,
            epoch_size=epoch_size)
        print 'Epoch ', 1 + epoch_id
        progbar = keras.utils.generic_utils.Progbar(epoch_size)
        batch_id = 0
        for (X_batch, Y_batch) in dataflow:
            batch_id += 1
            loss = graph.train_on_batch({"X": X_batch, "Y": Y_batch})
            progbar.update(batch_id * batch_size)
        print "Training loss = ", loss
        loss_history.append(loss)
        if np.mod(epoch_id+1, every_n_epoch) == 0:
            train_accuracies = di.singlelabel.training_accuracies(
                    X_train_list, Y_train_list,
                    batch_size, datagen, epoch_size, graph)
            train_accuracies_history.append(train_accuracies)
            test_accuracies = di.singlelabel.test_accuracies(
                    X_test, Y_test, batch_size, epoch_size, graph)
            test_accuracies_history.append(test_accuracies)
    return loss_history, train_accuracies_history, test_accuracies_history
