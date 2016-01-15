import DeepInstruments as di
import keras
from librosa.display import specshow
import matplotlib.pyplot as plt
import numpy as np


batch_size = 512
decision_length = 131072  # in samples
epoch_size = 8192
every_n_epoch = 10
fmin = 55 # in Hz
hop_length = 1024 #  in samples
n_bins_per_octave = 12
n_epochs = 20
n_octaves = 8
optimizer = "adagrad"
sr = 32000
conv1_channels = 100
conv1_height = 96
conv1_width = 32
pool1_height = 3
pool1_width = 3
conv2_channels = 100
conv2_height = 8
conv2_width = 8
pool2_height = 8
dense1_channels = 512
drop1_proportion = 0.5
dense2_channels = 64
drop2_proportion = 0.5


(test_stems, training_stems) = di.singlelabel.split_stems(
    di.singlelabel.names, di.singlelabel.test_discarded,
    di.singlelabel.training_discarded, di.singlelabel.training_to_test)

datagen = di.singlelabel.ScalogramGenerator(decision_length, fmin,
                                            hop_length, n_bins_per_octave,
                                            n_octaves, training_stems)

X_chunks, Y_chunks = datagen.chunk(test_stems)

graph = di.learning.build_graph(
    X_height=96,
    X_width=128,
    conv1_channels=16,
    conv1_height=48,
    conv1_width=16,
    pool1_height=3,
    pool1_width=3,
    conv2_channels=16,
    conv2_height=8,
    conv2_width=8,
    pool2_height=3,
    pool2_width=3,
    dense1_channels=256,
    drop1_proportion=0.5,
    dense2_channels=64,
    drop2_proportion=0.5,
    dense3_channels=8)

graph.compile(loss={'Y': 'categorical_crossentropy'}, optimizer="sgd")


# Train model
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
        batch_losses[batch_id] = loss
        progbar.update(batch_id * batch_size)
    mean_loss = np.mean(batch_losses)
    std_loss = np.std(batch_losses)
    print "Training loss = ", mean_loss, " +/- ", std_loss
    mean_training_loss_history.append(mean_loss)