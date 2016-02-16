import DeepInstruments as di
import keras
from keras.models import Graph
from keras.layers.advanced_activations import LeakyReLU, ParametricSoftplus
from keras.layers.core import Dense, Dropout, Flatten, LambdaMerge, Reshape
from keras.layers.core import Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.convolutional import MaxPooling1D, AveragePooling1D


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
        drop1_proportion,
        dense1_channels,
        drop2_proportion,
        dense2_channels):
    graph = Graph()

    # Input
    X_height = n_bins_per_octave * n_octaves
    graph.add_input(name="X", input_shape=(1, X_height, X_width))
    graph.add_input(name="Z", input_shape=(1, X_height, X_width))
    graph.add_input(name="G", input_shape=(1, X_height, X_width))

    # Spiral transformation
    spiral_X = Reshape((n_octaves, n_bins_per_octave, X_width))
    graph.add_node(spiral_X, name="spiral_X", input="X")

    # Shared layers
    conv1 = Convolution2D(conv1_channels, conv1_height, conv1_width,
                          border_mode="same", activation="relu")
    graph.add_node(conv1, name="conv1", input="spiral_X")

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

    # Pooling of symbolic activations Z (piano-roll) and G (melody gate)
    pool1_Z = MaxPooling2D(pool_size=(pool1_height, pool1_width))
    graph.add_node(pool1_Z, name="pool1_Z", input="Z")

    Z_height = n_bins_per_octave / pool1_height
    Z_width = X_width / pool1_width
    reshaped_Z = Reshape((n_octaves, Z_height * Z_width))
    graph.add_node(reshaped_Z, name="reshaped_Z", input="pool1_Z")

    chroma_Z = MaxPooling1D(pool_length=n_octaves)
    graph.add_node(chroma_Z, name="chroma_Z", input="reshaped_Z")

    toplevel_Z = Reshape((1, Z_height, Z_width))
    graph.add_node(toplevel_Z, name="toplevel_Z", input="chroma_Z")

    pool1_G = MaxPooling2D(pool_size=(pool1_height, pool1_width))
    graph.add_node(pool1_G, name="pool1_G", input="G")

    reshaped_G = Reshape((n_octaves, Z_height * Z_width))
    graph.add_node(reshaped_G, name="reshaped_G", input="pool1_G")

    chroma_G = MaxPooling1D(pool_length=n_octaves)
    graph.add_node(chroma_G, name="chroma_G", input="reshaped_G")

    toplevel_G = Reshape((1, Z_height, Z_width))
    graph.add_node(toplevel_G, name="toplevel_G", input="chroma_G")

    # Layers towards melodic target
    flat_shape = (pool1_X.output_shape[1],
                  pool1_X.output_shape[2] * pool1_X.output_shape[3])
    reshaped_X = Reshape(dims=flat_shape)
    graph.add_node(reshaped_X, name="reshaped_X", input="conv2")

    collapsed_X = AveragePooling1D(pool_length=conv2_channels)
    graph.add_node(collapsed_X, name="collapsed_X", input="reshaped_X")

    softplus_X = ParametricSoftplus()
    graph.add_node(softplus_X, name="softplus_X", input="collapsed_X")

    toplevel_X = Reshape(dims=(1, Z_height, Z_width))
    graph.add_node(toplevel_X, name="toplevel_X", input="softplus_X")

    melodic_error = LambdaMerge([toplevel_X, toplevel_Z, toplevel_G],
                                di.learning.substract_and_mask)
    graph.add_node(melodic_error, name="melodic_error",
                   inputs=["toplevel_X", "toplevel_Z", "toplevel_G"])

    # Outputs
    graph.add_output(name="Y", input="dense2")
    graph.add_output(name="zero", input="Z")

    return graph
