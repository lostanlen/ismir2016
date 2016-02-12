import DeepInstruments as di
import numpy as np

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
epoch_size = 8192
n_epochs = 20
optimizer = "adam"
mask_weight = 100

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
masked_output = np.zeros((batch_size, 1, mask_height, mask_width))

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
from keras.utils.generic_utils import Progbar
mean_training_loss_history = []
batch_losses = np.zeros(epoch_size / batch_size)
test_accuracies_history = []
training_accuracies_history = []

for epoch_id in xrange(n_epochs):
    dataflow = datagen.flow(batch_size=batch_size, epoch_size=epoch_size)
    print "\nEpoch ", 1 + epoch_id
    progbar = Progbar(epoch_size)
    batch_id = 0
    for (X_batch, Y_batch, Z_batch, G_batch) in dataflow:
        loss = graph.train_on_batch({"X": X_batch,
                                     "Y": Y_batch,
                                     "Z": Z_batch,
                                     "G": G_batch,
                                     "zero": masked_output})
        batch_losses[batch_id] = loss[0]
        progbar.update(batch_id * batch_size)
        batch_id += 1
    mean_loss = np.mean(batch_losses)
    std_loss = np.std(batch_losses)
    print "\nTraining loss = ", mean_loss, " +/- ", std_loss
    mean_training_loss_history.append(mean_loss)
    # Measure training accuracy
    training_accuracies = di.singlelabel.training_accuracies(
            batch_size, datagen, epoch_size, graph)
    training_accuracies_history.append(training_accuracies)
    print "Training accuracies: \n", training_accuracies

    # Measure test accuracies
    test_accuracies = di.singlelabel.test_accuracies(graph, X_test, y_test)
    test_accuracies_history.append(test_accuracies)
    print "Test accuracies: \n", test_accuracies
    mean_test_accuracy = np.mean(test_accuracies)
    std_test_accuracy = np.std(test_accuracies)
    print "GLOBAL TEST ACCURACY: ",\
        mean_test_accuracy, " +/- ", std_test_accuracy

final_score = test_accuracies_history[-1]
final_mean_score = np.mean(final_score)

# Save results
np.savez(
    '40_40_weight100.npz',
    decision_length=decision_length,
    fmin=fmin,
    hop_length=hop_length,
    n_bins_per_octave=n_bins_per_octave,
    n_octaves=n_octaves,
    conv1_channels=conv1_channels,
    conv1_height=conv1_height,
    conv1_width=conv1_width,
    pool1_height=pool1_height,
    pool1_width=pool1_width,
    conv2_channels=conv2_channels,
    conv2_height=conv2_height,
    conv2_width=conv2_width,
    pool2_height=pool2_height,
    pool2_width=pool2_width,
    drop1_proportion=drop1_proportion,
    dense1_channels=dense1_channels,
    drop2_proportion=drop2_proportion,
    batch_size=batch_size,
    epoch_size=epoch_size,
    n_epochs=n_epochs,
    optimizer=optimizer,
    mask_weight=mask_weight,
    mean_training_loss_history=mean_training_loss_history,
    test_accuracies_history=test_accuracies_history,
    training_accuracies_history=training_accuracies_history,
    final_score=final_score,
    final_mean_score=final_mean_score)
