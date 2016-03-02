import DeepInstruments as di
import keras
from keras.models import Graph
from keras.layers.advanced_activations import LeakyReLU, ParametricSoftplus
from keras.layers.core import Dense, Dropout, Flatten, LambdaMerge, Reshape
from keras.layers.convolutional import AveragePooling1D,\
    Convolution2D, MaxPooling2D


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

    # Input
    X_height = n_octaves * n_bins_per_octave
    graph.add_input(name="X", input_shape=(1, X_height, X_width))
    if is_Z_supervision:
        graph.add_input(name="Z", input_shape=(1, X_height, X_width))
        graph.add_input(name="G", input_shape=(1, X_height, X_width))

    # Shared layers
    conv1 = Convolution2D(conv1_channels, conv1_height, conv1_width,
                          border_mode="valid")
    graph.add_node(conv1, name="conv1", input="X")

    relu1 = LeakyReLU()
    graph.add_node(relu1, name="relu1", input="conv1")

    pool1_X = MaxPooling2D(pool_size=(pool1_height, pool1_width))
    graph.add_node(pool1_X, name="pool1_X", input="relu1")

    # Layers towards instrument target
    conv2 = Convolution2D(conv2_channels, conv2_height, conv2_width,
                          border_mode="valid")
    graph.add_node(conv2, name="conv2", input="pool1_X")

    relu2 = LeakyReLU()
    graph.add_node(relu2, name="relu2", input="conv2")

    pool2 = MaxPooling2D(pool_size=(pool2_height, pool2_width))
    graph.add_node(pool2, name="pool2", input="relu2")

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

    if is_Z_supervision:
        # Pooling of symbolic activations Z (piano-roll) and G (melody gate)
        pool1_Z = MaxPooling2D(pool_size=(pool1_height, pool1_width))
        graph.add_node(pool1_Z, name="pool1_Z", input="Z")

        pool1_G = MaxPooling2D(pool_size=(pool1_height, pool1_width))
        graph.add_node(pool1_G, name="pool1_G", input="G")

        # Layers towards melodic target
        flat_shape = (relu2.output_shape[1],
                      relu2.output_shape[2] * relu2.output_shape[3])
        reshaped_X = Reshape(dims=flat_shape)
        graph.add_node(reshaped_X, name="reshaped_X", input="relu2")

        collapsed_X = AveragePooling1D(pool_length=conv1_channels)
        graph.add_node(collapsed_X, name="collapsed_X", input="reshaped_X")

        softplus_X = ParametricSoftplus()
        graph.add_node(softplus_X, name="softplus_X", input="collapsed_X")

        rectangular_shape = (1, pool1_X.output_shape[2],
                             pool1_X.output_shape[3])
        toplevel_X = Reshape(dims=rectangular_shape)
        graph.add_node(toplevel_X, name="toplevel_X", input="softplus_X")

        melodic_error = LambdaMerge([toplevel_X, pool1_Z, pool1_G],
                                    di.learning.substract_and_mask)
        graph.add_node(melodic_error, name="melodic_error",
                       inputs=["pool1_Z", "pool1_Z", "pool1_G"])

    # Outputs
    graph.add_output(name="Y", input="dense2")
    if is_Z_supervision:
        graph.add_output(name="zero", input="melodic_error")

    return graph