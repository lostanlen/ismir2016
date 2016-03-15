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
    for octave_index in range(0, n_octaves - 2):
        name = "X" + str(octave_index)
        graph.add_input(name=name, input_shape=(1, X_height, X_width))

    # Octave-wise convolutional layers
    init = "he_normal"
    conv1_X0 = Convolution2D(conv1_channels, conv1_height, conv1_width,
                             border_mode="valid", init=init)
    graph.add_node(conv1_X0, name="conv1_X0", input="X0")
    relu1_X0 = LeakyReLU()
    graph.add_node(relu1_X0, name="relu1_X0", input="conv1_X0")
    pool1_X0 = MaxPooling2D(pool_size=(pool1_height, pool1_width))
    graph.add_node(pool1_X0, name="pool1_X0", input="relu1_X0")
    conv2_X0 = Convolution2D(conv2_channels, conv2_height, conv2_width,
                             border_mode="valid", init=init)
    graph.add_node(conv2_X0, name="conv2_X0", input="pool1_X0")

    conv1_X1 = Convolution2D(conv1_channels, conv1_height, conv1_width,
                             border_mode="valid", init=init)
    graph.add_node(conv1_X1, name="conv1_X1", input="X1")
    relu1_X1 = LeakyReLU()
    graph.add_node(relu1_X1, name="relu1_X1", input="conv1_X1")
    pool1_X1 = MaxPooling2D(pool_size=(pool1_height, pool1_width))
    graph.add_node(pool1_X1, name="pool1_X1", input="relu1_X1")
    conv2_X1 = Convolution2D(conv2_channels, conv2_height, conv2_width,
                             border_mode="valid", init=init)
    graph.add_node(conv2_X1, name="conv2_X1", input="pool1_X1")

    conv1_X2 = Convolution2D(conv1_channels, conv1_height, conv1_width,
                             border_mode="valid", init=init)
    graph.add_node(conv1_X2, name="conv1_X2", input="X2")
    relu1_X2 = LeakyReLU()
    graph.add_node(relu1_X2, name="relu1_X2", input="conv1_X2")
    pool1_X2 = MaxPooling2D(pool_size=(pool1_height, pool1_width))
    graph.add_node(pool1_X2, name="pool1_X2", input="relu1_X2")
    conv2_X2 = Convolution2D(conv2_channels, conv2_height, conv2_width,
                             border_mode="valid", init=init)
    graph.add_node(conv2_X2, name="conv2_X2", input="pool1_X2")

    conv1_X3 = Convolution2D(conv1_channels, conv1_height, conv1_width,
                             border_mode="valid", init=init)
    graph.add_node(conv1_X3, name="conv1_X3", input="X3")
    relu1_X3 = LeakyReLU()
    graph.add_node(relu1_X3, name="relu1_X3", input="conv1_X3")
    pool1_X3 = MaxPooling2D(pool_size=(pool1_height, pool1_width))
    graph.add_node(pool1_X3, name="pool1_X3", input="relu1_X3")
    conv2_X3 = Convolution2D(conv2_channels, conv2_height, conv2_width,
                             border_mode="valid", init=init)
    graph.add_node(conv2_X3, name="conv2_X3", input="pool1_X3")

    conv1_X4 = Convolution2D(conv1_channels, conv1_height, conv1_width,
                             border_mode="valid", init=init)
    graph.add_node(conv1_X4, name="conv1_X4", input="X4")
    relu1_X4 = LeakyReLU()
    graph.add_node(relu1_X4, name="relu1_X4", input="conv1_X4")
    pool1_X4 = MaxPooling2D(pool_size=(pool1_height, pool1_width))
    graph.add_node(pool1_X4, name="pool1_X4", input="relu1_X4")
    conv2_X4 = Convolution2D(conv2_channels, conv2_height, conv2_width,
                             border_mode="valid", init=init)
    graph.add_node(conv2_X4, name="conv2_X4", input="pool1_X4")

    conv1_X5 = Convolution2D(conv1_channels, conv1_height, conv1_width,
                             border_mode="valid", init=init)
    graph.add_node(conv1_X5, name="conv1_X5", input="X5")
    relu1_X5 = LeakyReLU()
    graph.add_node(relu1_X5, name="relu1_X5", input="conv1_X5")
    pool1_X5 = MaxPooling2D(pool_size=(pool1_height, pool1_width))
    graph.add_node(pool1_X5, name="pool1_X5", input="relu1_X5")
    conv2_X5 = Convolution2D(conv2_channels, conv2_height, conv2_width,
                             border_mode="valid", init=init)
    graph.add_node(conv2_X5, name="conv2_X5", input="pool1_X5")

    # Spiral concatenation and pooling
    relu2 = LeakyReLU()
    graph.add_node(relu2, name="relu2",
                   inputs=["conv2_X0", "conv2_X1", "conv2_X2",
                           "conv2_X3", "conv2_X4", "conv2_X5"],
                   merge_mode="sum", concat_axis=1)

    # Multi-layer perceptron with dropout
    flatten = Flatten()
    graph.add_node(flatten, name="flatten",
                   input="pool2", concat_axis=1)

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
