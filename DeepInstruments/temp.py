
if arch == 1:  # spiral
    conv1_channels = [0, 0, 160]
elif arch == 2:  # 1d
    conv1_channels = [0, 224, 0]
elif arch == 3:  # spiral & 1d
    conv1_channels = [0, 128, 128]
elif arch == 4:  # 2d
    conv1_channels = [96, 0, 0]
elif arch == 5:  # 2d & spiral
    conv1_channels = [96, 0, 96]
elif arch == 6:  # 2d & 1d
    conv1_channels = [96, 96, 0]
elif arch == 7:  # 2d & 1d & spiral
    conv1_channels = [96, 96, 96]
elif arch == 8: # 2d (more parameters)
    conv1_channels = [144, 0, 0]

conv2_channels = conv1_channels
dense1_channels = 128

is_sp = arch in [1,    3,    5,    7]
is_1d = arch in [   2, 3,       6, 7]
is_2d = arch in [         4, 5, 6, 7, 8]
js = np.matrix([[0, 8], [5, 8], [1, 3], [2, 4], [3, 5]])
if not is_2d:
    js[0, :] = 0
if not is_1d:
    js[1, :] = 0
if not is_sp:
    js[2:, :] = 0
print js

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    offsets = [
         np.nanmean(X_test[:, :, (js[0,0]*Q):(js[0,1]*Q), :]),
         np.nanmean(X_test[:, :, (js[1,0]*Q):(js[1,1]*Q), :]),
         np.nanmean(X_test[:, :, (js[2,0]*Q):(js[2,1]*Q), :]),
         np.nanmean(X_test[:, :, (js[3,0]*Q):(js[3,1]*Q), :]),
         np.nanmean(X_test[:, :, (js[4,0]*Q):(js[4,1]*Q), :])]

# Build ConvNet as a Keras graph, compile it with Theano
graph = di.learning.build_graph(Q, js, X_width,
    conv1_channels, conv1_height, conv1_width,
    pool1_height, pool1_width,
    conv2_channels, conv2_height, conv2_width,
    pool2_height, pool2_width,
    dense1_channels, dense2_channels, alpha)
print graph.summary()