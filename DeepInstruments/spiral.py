import DeepInstruments as di
import keras
from keras.models import Graph
from keras.layers.advanced_activations import LeakyReLU, ParametricSoftplus
from keras.layers.core import Dense, Dropout, Flatten, LambdaMerge, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D


def build_graph(
        X_channels,
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
        drop1_proportion,
        dense1_channels,
        drop2_proportion,
        dense2_channels):
    graph = Graph()

    # Input
    graph.add_input(name="X", input_shape=(X_channels, X_height, X_width))
    graph.add_input(name="Z", input_shape=(X_channels, X_height, X_width))
    graph.add_input(name="G", input_shape=(X_channels, X_height, X_width))

    # Shared layers
    conv1 = Convolution2D(conv1_channels, conv1_height, conv1_width,
                          border_mode="same", activation="relu")
    graph.add_node(conv1, name="conv1", input="X")

    pool1_X = MaxPooling2D(pool_size=(pool1_height, pool1_width))
    graph.add_node(pool1_X, name="pool1_X", input="conv1")

    # Layers towards instrument target
    conv2 = Convolution2D(conv2_channels, conv2_height, conv2_width,
                          border_mode="same", activation="relu")
    graph.add_node(conv2, name="conv2", input="pool1_X")

    pool2 = MaxPooling2D(pool_size=(pool2_height, pool2_width))
    graph.add_node(pool2, name="pool2", input="conv2")

    flatten = Flatten()
    graph.add_node(flatten, name="flatten", input="pool2")

    drop1 = Dropout(drop1_proportion)
    graph.add_node(drop1, name="drop1", input="flatten")

    dense1 = Dense(dense1_channels, activation="relu")
    graph.add_node(dense1, name="dense1", input="drop1")

    drop2 = Dropout(drop2_proportion)
    graph.add_node(drop2, name="drop2", input="dense1")

    dense2 = Dense(dense2_channels, activation="softmax")
    graph.add_node(dense2, name="dense2", input="drop2")

    graph.add_output(name="Y", input="dense2")

    return graph


def substract_and_mask(args):
    return (args[0] - args[1]) * args[2]
