from keras.models import Graph
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import numpy as np
import sklearn

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

    dense3 = Dense(dense2_channels, activation="softmax")
    graph.add_node(dense3, name="dense3", input="drop2")

    graph.add_output(name="Y", input="dense3")

    return graph

def confusion_matrix(Y_true, Y_predicted):
    y_true = np.argmax(Y_true, axis=1)
    y_predicted = np.argmax(Y_predicted, axis=1)
    n_classes = np.size(Y_true, 1)
    labels = range(n_classes)
    cm = sklearn.metrics.confusion_matrix(y_true, y_predicted, labels)
    return cm / cm.sum(axis=1)[:, np.newaxis]