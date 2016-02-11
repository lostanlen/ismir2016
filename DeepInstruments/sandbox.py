import DeepInstruments as di
import numpy as np
from keras.utils.generic_utils import Progbar

# Parameters for audio
decision_length = 131072  # in samples
fmin = 55  # in Hz
hop_length = 1024  # in samples
n_bins_per_octave = 12
n_octaves = 8

# Parameters for ConvNet
conv1_channels = 40
conv1_height = 36
conv1_width = 4
pool1_height = 4
pool1_width = 4
conv2_channels = 40
conv2_height = 12
conv2_width = 8
pool2_height = 6
pool2_width = 16
drop1_proportion = 0.5
dense1_channels = 64
drop2_proportion = 0.5

# Parameters for learning
batch_size = 128
n_epochs = 10
optimizer = "adam"
samples_per_epoch = 8192
mask_weight = 1.0e3

# I/O sizes
X_height = n_bins_per_octave * n_octaves
X_width = decision_length / hop_length
mask_height = X_height / pool1_height
mask_width = X_width / pool1_width
dense2_channels = 8

# Build ConvNet as a Keras graph, compile it with Theano
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
                    "zero": "mse"}, optimizer=optimizer)

# Get single-label split (MedleyDB for training, solosDb for test
(test_stems, training_stems) = di.singlelabel.get_stems()

# Compute audio features and retrieve melodies on the training set
datagen = di.singlelabel.ScalogramGenerator(
        decision_length, fmin, hop_length, mask_weight,
        n_bins_per_octave, n_octaves,
        training_stems)

# Compute audio features on the test set
test_paths = di.singlelabel.get_paths("test")
X_test = datagen.get_X(test_paths)
y_test = np.hstack(map(di.descriptors.get_y, test_paths))


# Train ConvNet
graph.fit_generator(datagen, samples_per_epoch, n_epochs, verbose=1)

from keras.utils.generic_utils import Progbar
mean_training_loss_history = []
batch_losses = np.zeros(epoch_size / batch_size)

for epoch_id in xrange(n_epochs):
    dataflow = datagen.flow(batch_size=batch_size, epoch_size=epoch_size)
    print "Epoch ", 1 + epoch_id
    progbar = Progbar(epoch_size)
    batch_id = 0
    for (X_batch, Y_batch, Z_batch, G_batch) in dataflow:
        loss = graph.train_on_batch({"X": X_batch,
                                     "Y": Y_batch,
                                     "Z": Z_batch,
                                     "G": G_batch,
                                     "zero": np.zeros((batch_size, 1, 32, 42))})
        batch_losses[batch_id] = loss[0]
        progbar.update(batch_id * batch_size)
        batch_id += 1
    mean_loss = np.mean(batch_losses)
    std_loss = np.std(batch_losses)
    print "Training loss = ", mean_loss, " +/- ", std_loss
    mean_training_loss_history.append(mean_loss)
