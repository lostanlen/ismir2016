import DeepInstruments as di
import keras
from keras.models import Graph
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.core import Dense, Dropout, Flatten, LambdaMerge, Reshape
from keras.layers.core import Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.convolutional import MaxPooling1D, AveragePooling1D
from keras.regularizers import ActivityRegularizer, WeightRegularizer
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.constraints import maxnorm


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
    graph.add_input(name="X0", input_shape=(1, 7*n_bins_per_octave, X_width))
    graph.add_input(name="X1", input_shape=(1, 7*n_bins_per_octave, X_width))

    # Octave-wise convolutional layers
    init = "he_normal"
    conv1_X0 = Convolution2D(conv1_channels[0], conv1_height, conv1_width,
                             border_mode="valid", init=init)
    graph.add_node(conv1_X0, name="conv1_X0", input="X0")
    conv1_X1 = Convolution2D(conv1_channels[1], conv1_height, conv1_width,
                             border_mode="valid", init=init)
    graph.add_node(conv1_X1, name="conv1_X1", input="X1")

    relu1 = LeakyReLU()
    graph.add_node(relu1, name="relu1",
                   inputs=["conv1_X0",
                           "conv1_X1"],
                   merge_mode="concat", concat_axis=1)

    pool1 = MaxPooling2D(pool_size=(pool1_height, pool1_width))
    graph.add_node(pool1, name="pool1", input="relu1")

    conv2 = Convolution2D(conv2_channels, conv2_height, conv2_width,
                          border_mode="same", init=init)
    graph.add_node(conv2, name="conv2", input="pool1")

    relu2 = LeakyReLU()
    graph.add_node(relu2, name="relu2", input="conv2")

    pool2 = MaxPooling2D(pool_size=(pool2_height, pool2_width))
    graph.add_node(pool2, name="pool2", input="relu2")

    # Multi-layer perceptron with dropout
    flatten = Flatten()
    graph.add_node(flatten, name="flatten", input="pool2")

    drop1 = Dropout(0.5)
    graph.add_node(drop1, name="drop1", input="flatten")

    dense1 = Dense(dense1_channels, init="lecun_uniform")
    graph.add_node(dense1, name="dense1", input="flatten")

    relu3 = LeakyReLU()
    graph.add_node(relu3, name="relu3", input="dense1")

    drop2 = Dropout(0.5)
    graph.add_node(drop2, name="drop2", input="relu3")

    dense2 = Dense(dense2_channels,
                   activation="softmax", init="lecun_uniform")
    graph.add_node(dense2, name="dense2", input="drop2")

    # Output
    graph.add_output(name="Y", input="dense2")

    return graph


def predict(graph, X_test, offsets):
    Q = 12
    X0 = X_test[:, :, (0*Q):(7*Q), :] - offsets[0]
    X1 = X_test[:, :, (1*Q):(8*Q), :] - offsets[1]
    class_probs = graph.predict({
        "X0": X0,
        "X1": X1})["Y"]
    return class_probs


def train_on_batch(graph, X_batch, Y_batch, offsets):
    Q = 12
    X0 = X_batch[:, :, (0*Q):(7*Q), :] - offsets[0]
    X1 = X_batch[:, :, (1*Q):(8*Q), :] - offsets[1]
    loss = graph.train_on_batch({
        "X0": X0,
        "X1": X1,
        "Y": Y_batch})
    return loss
