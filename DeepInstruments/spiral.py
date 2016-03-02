import DeepInstruments as di
import keras
from keras.models import Graph
from keras.layers.advanced_activations import LeakyReLU, ParametricSoftplus
from keras.layers.core import Dense, Dropout, Flatten, LambdaMerge, Reshape
from keras.layers.core import Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.convolutional import MaxPooling1D, AveragePooling1D


def build_graph(
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
    graph = Graph()
    assert n_octaves == 8
    assert conv1_height == (n_bins_per_octave+1)
    assert not is_Z_supervision

    # Input
    X_height = n_bins_per_octave * 2
    for octave_index in range(1, n_octaves - 2):
        name = "X" + str(octave_index)
        graph.add_input(name=name, input_shape=(1, X_height, X_width))

    # Octave-wise convolutional layers
    #conv1_X0 = Convolution2D(conv1_channels, conv1_height, conv1_width,
    #                         border_mode="valid")
    #graph.add_node(conv1_X0, name="conv1_X0", input="X0")
    conv1_X1 = Convolution2D(conv1_channels, conv1_height, conv1_width,
                             border_mode="valid")
    graph.add_node(conv1_X1, name="conv1_X1", input="X1")
    conv1_X2 = Convolution2D(conv1_channels, conv1_height, conv1_width,
                             border_mode="valid")
    graph.add_node(conv1_X2, name="conv1_X2", input="X2")
    conv1_X3 = Convolution2D(conv1_channels, conv1_height, conv1_width,
                             border_mode="valid")
    graph.add_node(conv1_X3, name="conv1_X3", input="X3")
    conv1_X4 = Convolution2D(conv1_channels, conv1_height, conv1_width,
                             border_mode="valid", activation="relu")
    graph.add_node(conv1_X4, name="conv1_X4", input="X4")
    conv1_X5 = Convolution2D(conv1_channels, conv1_height, conv1_width,
                             border_mode="valid")
    graph.add_node(conv1_X5, name="conv1_X5", input="X5")
    #conv1_X6 = Convolution2D(conv1_channels, conv1_height, conv1_width,
    #                         border_mode="valid")
    #graph.add_node(conv1_X6, name="conv1_X6", input="X6")

    # Spiral concatenation and pooling
    relu1 = LeakyReLU()
    graph.add_node(relu1, name="relu1",
                   inputs=["conv1_X1", "conv1_X2", "conv1_X3",
                           "conv1_X4", "conv1_X5"], concat_axis=2)

    pool1_X = MaxPooling2D(pool_size=(pool1_height, pool1_width))
    graph.add_node(pool1_X, name="pool1_X", input="relu1")

    # Time-frequency convolutions
    conv2 = Convolution2D(conv2_channels, conv2_height, conv2_width,
                          border_mode="valid")
    graph.add_node(conv2, name="conv2", input="pool1_X")

    relu2 = LeakyReLU()
    graph.add_node(relu2, name="relu2", input="conv2")

    pool2 = MaxPooling2D(pool_size=(pool2_height, pool2_width))
    graph.add_node(pool2, name="pool2", input="relu2")

    # Multi-layer perceptron with dropout
    flatten = Flatten()
    graph.add_node(flatten, name="flatten", input="pool2")

    drop1 = Dropout(drop1_proportion)
    graph.add_node(drop1, name="drop1", input="flatten")

    dense1 = Dense(dense1_channels)
    graph.add_node(dense1, name="dense1", input="drop1")

    relu3 = LeakyReLU()
    graph.add_node(relu3, name="relu3", input="dense1")

    drop2 = Dropout(drop2_proportion)
    graph.add_node(drop2, name="drop2", input="relu3")

    dense2 = Dense(dense2_channels, activation="softmax")
    graph.add_node(dense2, name="dense2", input="drop2")

    # Output
    graph.add_output(name="Y", input="dense2")

    return graph