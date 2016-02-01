import DeepInstruments as di
import keras
from keras.models import Graph
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
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

    # Shared layers
    conv1 = Convolution2D(conv1_channels, conv1_height, conv1_width)
    graph.add_node(conv1, name="conv1", input="X")

    relu1 = LeakyReLU()
    graph.add_node(relu1, name="relu1", input="conv1")

    pool1 = MaxPooling2D(pool_size=(pool1_height, pool1_width))
    graph.add_node(pool1, name="pool1", input="relu1")

    # Layers towards instrument target
    conv2 = Convolution2D(conv2_channels, conv2_height, conv2_width)
    graph.add_node(conv2, name="conv2", input="pool1")

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

    graph.add_output(name="Y", input="dense3")

    return graph


def elementwise_substract_and_mask(x, z, g):
    return (x - z) * g


subtract_and_mask = np.vectorize(elementwise_substract_and_mask)


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
