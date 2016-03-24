import DeepInstruments as di
import numpy as np
import warnings

is_2d = True
is_1d = False
is_spiral = False

# Parameters for ConvNet
arch = is_2d * 4 + is_1d * 2 + is_spiral

if   arch == 1:  # spiral
    conv1_channels = [0, 0, 68] # 103k parameters
elif arch == 2:  # 1d
    conv1_channels = [0, 104, 0] # 97k parameters
elif arch == 3:  # sp & 1d
    conv1_channels = [0, 72, 48] # 104k parameters
elif arch == 4:  # 2d
    conv1_channels = [36, 0, 0] # 95k parameters
elif arch == 5:  # 2d & spiral
    conv1_channels = [32, 0, 40] # 104k parameters
elif arch == 6:  # 2d & 1d
    conv1_channels = [28, 56, 0] # 99k parameters
elif arch == 7:  # 2d & 1d & spiral
    conv1_channels = [24, 48, 32] # 104k parameters


conv1_height = [5, 3] # resp for 2d, spiral
conv1_width = 3
pool1_height = 2
pool1_width = 6
conv2_channels = conv1_channels
conv2_height = [5, 3] # resp for 2d, 1d, spiral
conv2_width = 7
pool2_height = 2
pool2_width = 6
dense1_channels = 32
alpha = 0.3 # for LeakyReLU
js = np.matrix([[0, 8], [5, 8], [1, 3], [2, 4], [3, 5]])
if not is_2d:
    js[0, :] = 0
if not is_1d:
    js[1, :] = 0
if not is_spiral:
    js[2:, :] = 0

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    offsets = [
         np.nanmean(X_test[:, :, (js[0,0]*Q):(js[0,1]*Q), :]),
         np.nanmean(X_test[:, :, (js[1,0]*Q):(js[1,1]*Q), :]),
         np.nanmean(X_test[:, :, (js[2,0]*Q):(js[2,1]*Q), :]),
         np.nanmean(X_test[:, :, (js[3,0]*Q):(js[3,1]*Q), :]),
         np.nanmean(X_test[:, :, (js[4,0]*Q):(js[4,1]*Q), :])]

# Parameters for learning
batch_size = 32
epoch_size = 8192
n_epochs = 20
optimizer = "adam"

# I/O sizes
X_width = decision_length / hop_length
dense2_channels = 8
names = [name.split(" ")[0] for name in di.singlelabel.names]

# Build ConvNet as a Keras graph, compile it with Theano
graph = di.learning.build_graph(Q, js, X_width,
    conv1_channels, conv1_height, conv1_width, pool1_height, pool1_width,
    conv2_channels, conv2_height, conv2_width, pool2_height, pool2_width,
    dense1_channels, dense2_channels, alpha)
graph.compile(loss={"Y": "categorical_crossentropy"}, optimizer=optimizer)

# Train ConvNet
from keras.utils.generic_utils import Progbar
batch_losses = np.zeros(epoch_size / batch_size)
chunk_accuracies_history = []
file_accuracies_history = []
mean_loss = float("inf")

for epoch_id in xrange(n_epochs):
    dataflow = datagen.flow(batch_size=batch_size, epoch_size=epoch_size)
    print "\nEpoch ", 1 + epoch_id
    progbar = Progbar(epoch_size)
    batch_id = 0
    for (X_batch, Y_batch) in dataflow:
        loss = di.learning.train_on_batch(graph, X_batch, Y_batch, Q, js, offsets)
        batch_losses[batch_id] = loss[0]
        progbar.update(batch_id * batch_size)
        batch_id += 1
    if np.mean(batch_losses) < mean_loss:
        mean_loss = np.mean(batch_losses)
        std_loss = np.std(batch_losses)
    else:
        break
    print "\nTraining loss = ", mean_loss, " +/- ", std_loss

    # Measure test accuracies
    class_probs = di.learning.predict(graph, X_test, Q, js, offsets)
    y_predicted = np.argmax(class_probs, axis=1)
    chunk_accuracies = di.singlelabel.chunk_accuracies(y_predicted, y_test)
    chunk_accuracies_history.append(chunk_accuracies)
    file_accuracies = di.singlelabel.file_accuracies(test_paths, class_probs,
                                                     y_test,
                                                     method="geometric_mean")
    file_accuracies_history.append(file_accuracies)
    mean_file_accuracy = np.mean(file_accuracies)
    std_file_accuracy = np.std(file_accuracies)
    mean_chunk_accuracy = np.mean(chunk_accuracies)
    std_chunk_accuracy = np.std(chunk_accuracies)
    print "----------------------------"
    print "            CHUNK     FILE  "
    for name_index in range(len(names)):
        print names[name_index],\
            " " * (9 - len(names[name_index])),\
            " " * (chunk_accuracies[name_index] < 100.0),\
            round(chunk_accuracies[name_index], 1), "  ",\
            " " * (file_accuracies[name_index] < 100.0),\
            round(file_accuracies[name_index], 1), "  "
    print "----------------------------"
    print "GLOBAL      ",\
        round(mean_chunk_accuracy, 1), "    ",\
        round(mean_file_accuracy, 1)
    print "std      (+/-" +\
        str(round(0.5 * std_chunk_accuracy, 1)) +\
        ") (+/-" +\
        str(round(0.5 * std_file_accuracy, 1)) + ")"

# Save final scores
final_chunk_score = chunk_accuracies_history[-1]
final_mean_chunk_score = np.mean(final_chunk_score)
final_file_score = file_accuracies_history[-1]
final_mean_file_score = np.mean(final_file_score)