import DeepInstruments as di
import keras
from keras.models import Graph
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Dropout, Flatten, LambdaMerge, Reshape
from keras.layers.core import Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.convolutional import MaxPooling1D, AveragePooling1D
from keras.regularizers import ActivityRegularizer, WeightRegularizer
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


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
    X0_bn = BatchNormalization(mode=1)
    graph.add_node(X0_bn, name="X0_bn", input="X0")
    conv1_X0 = Convolution2D(conv1_channels, conv1_height, conv1_width,
                             border_mode="valid", init=init)
    graph.add_node(conv1_X0, name="conv1_X0", input="X0_bn")

    X1_bn = BatchNormalization(mode=1)
    graph.add_node(X1_bn, name="X1_bn", input="X1")
    conv1_X1 = Convolution2D(conv1_channels, conv1_height, conv1_width,
                             border_mode="valid", init=init)
    graph.add_node(conv1_X1, name="conv1_X1", input="X1_bn")

    X2_bn = BatchNormalization(mode=1)
    graph.add_node(X2_bn, name="X2_bn", input="X2")
    conv1_X2 = Convolution2D(conv1_channels, conv1_height, conv1_width,
                             border_mode="valid", init=init)
    graph.add_node(conv1_X2, name="conv1_X2", input="X2_bn")

    X3_bn = BatchNormalization(mode=1)
    graph.add_node(X3_bn, name="X3_bn", input="X3")
    conv1_X3 = Convolution2D(conv1_channels, conv1_height, conv1_width,
                             border_mode="valid", init=init)
    graph.add_node(conv1_X3, name="conv1_X3", input="X3_bn")

    X4_bn = BatchNormalization(mode=1)
    graph.add_node(X4_bn, name="X4_bn", input="X4")
    conv1_X4 = Convolution2D(conv1_channels, conv1_height, conv1_width,
                             border_mode="valid", init=init)
    graph.add_node(conv1_X4, name="conv1_X4", input="X4_bn")

    X5_bn = BatchNormalization(mode=1)
    graph.add_node(X5_bn, name="X5_bn", input="X5")
    conv1_X5 = Convolution2D(conv1_channels, conv1_height, conv1_width,
                             border_mode="valid", init=init)
    graph.add_node(conv1_X5, name="conv1_X5", input="X5_bn")

    # Spiral concatenation and pooling
    relu1 = LeakyReLU()
    graph.add_node(relu1, name="relu1",
                   inputs=["conv1_X0", "conv1_X1", "conv1_X2",
                           "conv1_X3", "conv1_X4", "conv1_X5"],
                   merge_mode="sum", concat_axis=1)

    pool1_X = MaxPooling2D(pool_size=(pool1_height, pool1_width))
    graph.add_node(pool1_X, name="pool1_X", input="relu1")

    # Time-frequency convolutions
    conv2 = Convolution2D(conv2_channels, conv2_height, conv2_width,
                          border_mode="valid", init=init)
    graph.add_node(conv2, name="conv2", input="pool1_X")

    relu2 = LeakyReLU()
    graph.add_node(relu2, name="relu2", input="conv2")

    pool2 = MaxPooling2D(pool_size=(pool2_height, pool2_width))
    graph.add_node(pool2, name="pool2", input="relu2")

    # Multi-layer perceptron with dropout
    flatten = Flatten()
    graph.add_node(flatten, name="flatten", input="pool2")

    dense1 = Dense(dense1_channels, init="lecun_uniform")
    graph.add_node(dense1, name="dense1", input="flatten")

    relu3 = LeakyReLU()
    graph.add_node(relu3, name="relu3", input="dense1")

    dense2 = Dense(dense2_channels,
                   activation="softmax", init="lecun_uniform",
                   W_regularizer=l2(0.05))
    graph.add_node(dense2, name="dense2", input="relu3")

    # Output
    graph.add_output(name="Y", input="dense2")

    return graph
