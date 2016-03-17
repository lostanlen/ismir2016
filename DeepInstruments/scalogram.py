import DeepInstruments as di
import keras
from keras.models import Graph
from keras.layers.advanced_activations import LeakyReLU, ParametricSoftplus
from keras.layers.core import Dense, Dropout, Flatten, LambdaMerge, Reshape
from keras.layers.convolutional import AveragePooling1D,\
    Convolution2D, MaxPooling2D, AveragePooling2D
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization


def build_graph(
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
    graph = Graph()

    # Input
    X_height = n_octaves * n_bins_per_octave
    X_shape = (1, X_height, X_width)
    graph.add_input(name="X", input_shape=X_shape)

    # Shared layers
    init = "he_normal"
    conv1 = Convolution2D(conv1_channels, conv1_height, conv1_width,
                          border_mode="valid", init=init)
    graph.add_node(conv1, name="conv1", input="X")

    relu1 = LeakyReLU()
    graph.add_node(relu1, name="relu1", input="conv1")

    pool1 = MaxPooling2D(pool_size=(pool1_height, pool1_width))
    graph.add_node(pool1, name="pool1", input="relu1")

    conv2 = Convolution2D(conv2_channels, conv2_height, conv2_width,
                          border_mode="same", init=init)
    graph.add_node(conv2, name="conv2", input="pool1")

    relu2 = LeakyReLU()
    graph.add_node(relu2, name="relu2", input="conv2")

    pool2 = MaxPooling2D(pool_size=(pool2_height, pool2_width))
    graph.add_node(pool2, name="pool2", input="relu2")

    flatten = Flatten()
    graph.add_node(flatten, name="flatten", input="pool2")

    drop1 = Dropout(0.5)
    graph.add_node(drop1, name="drop1", input="flatten")

    dense1 = Dense(dense1_channels, init="lecun_uniform")
    graph.add_node(dense1, name="dense1", input="drop1")

    relu3 = LeakyReLU()
    graph.add_node(relu3, name="relu3", input="dense1")

    drop2 = Dropout(0.5)
    graph.add_node(drop2, name="drop2", input="relu3")

    dense2 = Dense(dense2_channels,
                   init="lecun_uniform", activation="softmax")
    graph.add_node(dense2, name="dense2", input="drop2")

    # Outputs
    graph.add_output(name="Y", input="dense2")
    return graph


def predict(graph, X_test, offsets):
    X = X_test - offsets[0]
    class_probs = graph.predict({"X": X})["Y"]
    return class_probs


def train_on_batch(graph, X_batch, Y_batch, offsets):
    X = X_batch - offsets[0]
    loss = graph.train_on_batch({"X": X, "Y": Y_batch})
    return loss