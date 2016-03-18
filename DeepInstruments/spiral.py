from keras.models import Graph
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D


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
    X2d_shape = (1, (js[0,1]-js[0,0])*Q, X_width)
    graph.add_input(name="X2d", input_shape=X2d_shape)

    X1d_shape = (1, (js[1,1]-js[1,0])*Q, X_width)
    graph.add_input(name="X1d", input_shape=X1d_shape)

    Xsp_shape = (1, (js[2,1]-js[2,0])*Q, X_width)
    graph.add_input(name="Xsp_1", input_shape=Xsp_shape)
    graph.add_input(name="Xsp_2", input_shape=Xsp_shape)
    graph.add_input(name="Xsp_3", input_shape=Xsp_shape)

    # 2D convolutions (typically for the whole spectrum)
    init = "he_normal"
    conv1_2d = Convolution2D(conv1_channels[0], conv1_height, conv1_width,
                             border_mode="valid", init=init)
    graph.add_node(conv1_2d, name="conv1_2d", input="X2d")

    relu1_2d = LeakyReLU()
    graph.add_node(relu1_2d, name="relu1_2d", input="conv1_2d")

    pool1_2d = MaxPooling2D(pool_size=(pool1_height, pool1_width))
    graph.add_node(pool1_2d, name="pool1_2d", input="relu1_2d")

    conv2_height = pool1_2d.output_shape[2] - 2*Q / pool1_height
    conv2_2d = Convolution2D(conv2_channels[0], conv2_height, conv2_width,
                             border_mode="same", init=init)
    graph.add_node(conv2_2d, name="conv2_2d", input="pool1_2d")

    relu2_2d = LeakyReLU()
    graph.add_node(relu2_2d, name="relu2_2d", input="conv2_2d")

    pool2_2d = MaxPooling2D(pool_size=(pool2_height, pool2_width))
    graph.add_node(pool2_2d, name="pool2_2d", input="relu2_2d")

    flatten_2d = Flatten()
    graph.add_node(flatten_2d, name="flatten_2d", input="pool2_2d")

    # 1D convolutions (typically for the filter only)
    conv1_1d = Convolution2D(conv1_channels[1], X1d_shape[1],
                             conv1_width, border_mode="valid", init=init)
    graph.add_node(conv1_1d, name="conv1_1d", input="X1d")

    relu1_1d = LeakyReLU()
    graph.add_node(relu1_1d, name="relu1_1d", input="conv1_1d")

    pool1_1d = MaxPooling2D(pool_size=(1, pool1_width))
    graph.add_node(pool1_1d, name="pool1_1d", input="relu1_1d")

    conv2_1d = Convolution2D(conv2_channels[1], 1,
                             conv2_width, border_mode="same", init=init)
    graph.add_node(conv2_1d, name="conv2_1d", input="pool1_1d")

    relu2_1d = LeakyReLU()
    graph.add_node(relu2_1d, name="relu2_1d", input="conv2_1d")

    pool2_1d = MaxPooling2D(pool_size=(1, pool2_width))
    graph.add_node(pool2_1d, name="pool2_1d", input="relu2_1d")

    flatten_1d = Flatten()
    graph.add_node(flatten_1d, name="flatten_1d", input="pool2_1d")

    # Spiral convolutions (typically for the source only)
    conv1_sp1 = Convolution2D(conv1_channels[2], conv1_height,
                              conv1_width, border_mode="valid", init=init)
    graph.add_node(conv1_sp1, name="conv1_sp1", input="Xsp_1")

    conv1_sp2 = Convolution2D(conv1_channels[2], conv1_height,
                              conv1_width, border_mode="valid", init=init)
    graph.add_node(conv1_sp2, name="conv1_sp2", input="Xsp_2")

    conv1_sp3 = Convolution2D(conv1_channels[2], conv1_height,
                              conv1_width, border_mode="valid", init=init)
    graph.add_node(conv1_sp3, name="conv1_sp3", input="Xsp_3")

    relu1_sp = LeakyReLU()
    graph.add_node(relu1_sp, name="relu1_sp",
                   inputs=["conv1_sp1", "conv1_sp2", "conv1_sp3"],
                   merge_mode="sum")

    pool1_sp = MaxPooling2D(pool_size=(pool1_height, pool1_width))
    graph.add_node(pool1_sp, name="pool1_sp", input="relu1_sp")

    conv2_sp = Convolution2D(conv2_channels[2], pool1_sp.output_shape[2],
                             conv2_width, border_mode="same", init=init)
    graph.add_node(conv2_sp, name="conv2_sp", input="pool1_sp")

    relu2_sp = LeakyReLU()
    graph.add_node(relu2_sp, name="relu2_sp", input="conv2_sp")

    pool2_sp = MaxPooling2D(pool_size=(1, pool2_width))
    graph.add_node(pool2_sp, name="pool2_sp", input="relu2_sp")

    flatten_sp = Flatten()
    graph.add_node(flatten_sp, name="flatten_sp", input="pool2_sp")

    # Multi-layer perceptron with dropout
    drop1 = Dropout(0.5)
    graph.add_node(drop1, name="drop1",
                   inputs=["flatten_2d", "flatten_1d", "flatten_sp"])

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
    X2d = X_test[:, :, (js[0,0]*Q):(js[0,1]*Q), :] - offsets[0]
    X1d = X_test[:, :, (js[1,0]*Q):(js[1,1]*Q), :] - offsets[2]
    Xsp_1 = X_test[:, :, (js[2,0]*Q):(js[2,1]*Q), :] - offsets[0]
    Xsp_2 = X_test[:, :, (js[3,0]*Q):(js[3,1]*Q), :] - offsets[1]
    Xsp_3 = X_test[:, :, (js[4,0]*Q):(js[4,1]*Q), :] - offsets[1]
    class_probs = graph.predict({
        "X2d": X2d,
        "X1d": X1d,
        "Xsp_1": Xsp_1,
        "Xsp_2": Xsp_2,
        "Xsp_3": Xsp_3})["Y"]
    return class_probs


def train_on_batch(graph, X_batch, Y_batch, Q, js, offsets):
    X2d = X_batch[:, :, (js[0,0]*Q):(js[0,1]*Q), :] - offsets[0]
    X1d = X_batch[:, :, (js[1,0]*Q):(js[1,1]*Q), :] - offsets[2]
    Xsp_1 = X_batch[:, :, (js[2,0]*Q):(js[2,1]*Q), :] - offsets[0]
    Xsp_2 = X_batch[:, :, (js[3,0]*Q):(js[3,1]*Q), :] - offsets[1]
    Xsp_3 = X_batch[:, :, (js[4,0]*Q):(js[4,1]*Q), :] - offsets[1]
    loss = graph.train_on_batch({
        "X2d": X2d,
        "X1d": X1d,
        "Xsp_1": Xsp_1,
        "Xsp_2": Xsp_2,
        "Xsp_3": Xsp_3,
        "Y": Y_batch})
    return loss
