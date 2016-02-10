import DeepInstruments as di
import numpy as np
from keras.utils.generic_utils import Progbar

X_height = 96
X_width = 128
conv1_channels = 80
conv1_height = 3
conv1_width = 3
pool1_height = 3
pool1_width = 3
conv2_channels = 40
conv2_height = 8
conv2_width = 8
pool2_height = 3
pool2_width = 35
drop1_proportion = 0.5
dense1_channels = 64
drop2_proportion = 0.5
dense2_channels = 8

batch_size = 512
decision_length = 131072  # in samples
epoch_size = 8192
every_n_epoch = 10
fmin = 55 # in Hz
hop_length = 1024 #  in samples
n_bins_per_octave = 12
n_epochs = 20
n_octaves = 8
# optimizer = "adagrad"

graph = di.learning.build_graph(
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
    drop1_proportion,
    dense1_channels,
    drop2_proportion,
    dense2_channels)

graph.compile(loss={"Y": "categorical_crossentropy",
                    "zero": "mse"}, optimizer="adam")

(test_stems, training_stems) = di.singlelabel.get_stems()


datagen = di.singlelabel.ScalogramGenerator(
        decision_length, fmin, hop_length, n_bins_per_octave, n_octaves,
        training_stems)

from keras.utils.generic_utils import Progbar
mean_training_loss_history = []

dataflow = datagen.flow(batch_size=batch_size, epoch_size=epoch_size)
batch_losses = np.zeros(batch_size)

for epoch_id in xrange(n_epochs):
    dataflow = datagen.flow(batch_size=batch_size, epoch_size=epoch_size)
    print 'Epoch ', 1 + epoch_id
    progbar = keras.utils.generic_utils.Progbar(epoch_size)
    batch_id = 0
    for (X_batch, Y_batch) in dataflow:
        batch_id += 1
        loss = graph.train_on_batch({"X": X_batch, "Y": Y_batch})
        batch_losses[batch_id] = loss[0]
        progbar.update(batch_id * batch_size)
    mean_loss = np.mean(batch_losses)
    std_loss = np.std(batch_losses)
    print "Training loss = ", mean_loss, " +/- ", std_loss
    mean_training_loss_history.append(mean_loss)
