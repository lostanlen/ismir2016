from keras.models import Graph

from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

def build_graph(
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
        drop_proportion,
        dense1_channels,
        dense2_channels):
    graph = Graph()

    graph.add_input(name='X', input_shape=(1, X_height, X_width))

    conv1 = Convolution2D(conv1_channels, conv1_height, conv1_width)
    graph.add_node(conv1, name="conv1", input="X")