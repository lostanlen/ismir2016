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
        Q,
        js,
        X_width,
        conv1_channels,
        conv1_height,
        conv1_width,
        pool1_height,
        pool1_width,
        conv2_channels,
        conv2_width,
        pool2_height,
        pool2_width,
        dense1_channels,
        dense2_channels):
    graph = Graph()

    # Inputs
    Xs_shape = (1, (js[0,1]-js[0,0])*Q, X_width)
    graph.add_input(name="Xs_1", input_shape=Xs_shape)
    graph.add_input(name="Xs_2", input_shape=Xs_shape)
    Xf_shape = (1, (js[2,0]-js[2,1])*Q, X_width)
    graph.add_input(name="Xf", input_shape=Xf_shape)

    # Octave-wise convolutional layers for the source
    init = "he_normal"
    conv1_X1 = Convolution2D(conv1_channels[0], conv1_height, conv1_width,
                             border_mode="valid", init=init)
    graph.add_node(conv1_X1, name="conv1_X1", input="Xs_1")
    conv1_X2 = Convolution2D(conv1_channels[0], conv1_height, conv1_width,
                             border_mode="valid", init=init)
    graph.add_node(conv1_X2, name="conv1_X2", input="Xs_2")

    relu1_s = LeakyReLU()
    graph.add_node(relu1_s, name="relu1_s",
                   inputs=["conv1_X1",
                           "conv1_X2"],
                   merge_mode="concat", concat_axis=1)

    pool1_s = MaxPooling2D(pool_size=(pool1_height, pool1_width))
    graph.add_node(pool1_s, name="pool1_s", input="relu1_s")

    conv2_height = pool1_s.output_shape[2] - 2*Q / pool1_height
    conv2_s = Convolution2D(conv2_channels[0], conv2_height, conv2_width,
                            border_mode="same", init=init)
    graph.add_node(conv2_s, name="conv2_s", input="pool1_s")

    relu2_s = LeakyReLU()
    graph.add_node(relu2_s, name="relu2_s", input="conv2_s")

    pool2_s = MaxPooling2D(pool_size=(pool2_height, pool2_width))
    graph.add_node(pool2_s, name="pool2_s", input="relu2_s")

    flatten_s = Flatten()
    graph.add_node(flatten_s, name="flatten_s", input="pool2_s")

    # Filter layers
    conv1_f = Convolution2D(conv1_channels[1], Xf_shape[1],
                            conv1_width, border_mode="valid", init=init)
    graph.add_node(conv1_f, name="conv1_f", input="Xf")

    relu1_f = LeakyReLU()
    graph.add_node(relu1_f, name="relu1_f", input="conv1_f")

    pool1_f = MaxPooling2D(pool_size=(1, pool1_width))
    graph.add_node(pool1_f, name="pool1_f", input="relu1_f")

    conv2_f = Convolution2D(conv2_channels[1], 1,
                            conv2_width, border_mode="same", init=init)
    graph.add_node(conv2_f, name="conv2_f", input="pool1_f")

    relu2_f = LeakyReLU()
    graph.add_node(relu2_f, name="relu2_f", input="conv2_f")

    pool2_f = MaxPooling2D(pool_size=(1, pool2_width))
    graph.add_node(pool2_f, name="pool2_f", input="relu2_f")

    flatten_f = Flatten()
    graph.add_node(flatten_f, name="flatten_f", input="pool2_f")

    # Multi-layer perceptron with dropout
    drop1 = Dropout(0.5)
    graph.add_node(drop1, name="drop1",
                   inputs=["flatten_s", "flatten_f"])

    dense1 = Dense(dense1_channels, init="lecun_uniform")
    graph.add_node(dense1, name="dense1", input="drop1")

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


def predict(graph, X_test, Q, js, offsets):
    Xs_1 = X_test[:, :, (js[0,0]*Q):(js[0,1]*Q), :] - offsets[0]
    Xs_2 = X_test[:, :, (js[1,0]*Q):(js[1,1]*Q), :] - offsets[1]
    Xf = X_test[:, :, (js[2,0]*Q):(js[2,1]*Q), :] - offsets[2]
    class_probs = graph.predict({
        "Xs_1": Xs_1,
        "Xs_2": Xs_2,
        "Xf": Xf})["Y"]
    return class_probs


def train_on_batch(graph, X_batch, Y_batch, Q, js, offsets):
    Xs_1 = X_batch[:, :, (js[0,0]*Q):(js[0,1]*Q), :] - offsets[0]
    Xs_2 = X_batch[:, :, (js[1,0]*Q):(js[1,1]*Q), :] - offsets[1]
    Xf = X_batch[:, :, (js[2,0]*Q):(js[2,1]*Q), :] - offsets[2]
    loss = graph.train_on_batch({
        "Xs_1": Xs_1,
        "Xs_2": Xs_2,
        "Xf": Xf,
        "Y": Y_batch})
    return loss
