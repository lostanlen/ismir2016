import DeepInstruments as di
import keras
from librosa.display import specshow
import matplotlib.pyplot as plt
import numpy as np
import sklearn.ensemble

import DeepInstruments.singlelabel

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
        batch_losses[batch_id] = loss[0]
        progbar.update(batch_id * batch_size)
    mean_loss = np.mean(batch_losses)
    std_loss = np.std(batch_losses)
    print "Training loss = ", mean_loss, " +/- ", std_loss
    mean_training_loss_history.append(mean_loss)


# Evaluate random forrest
training_paths = DeepInstruments.singlelabel.get_paths("training")
X_training = di.descriptors.get_X(training_paths)
Y_training = di.descriptors.get_Y(training_paths)

X_means = np.mean(X_training, axis=0)
X_stds = np.std(X_training, axis=0)
X_training = (X_training - X_means) / X_stds

test_paths = DeepInstruments.singlelabel.get_paths("test")
X_test = di.descriptors.get_X(test_paths)
X_test = (X_test - X_means) / X_stds
Y_test = di.descriptors.get_Y(test_paths)

n_trials = 10
confusion_matrices = []

for trial_index in range(n_trials):
    clf = sklearn.ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=100)
    clf = clf.fit(X_training, Y_training)
    Y_predicted = clf.predict(X_test)

    cm = sklearn.metrics.confusion_matrix(Y_test, Y_predicted).astype("float")
    cmn = cm / np.sum(cm, axis=0)
    confusion_matrices.append(cmn)
    accuracies = np.diag(cmn)
    mean_accuracy = round(100 * np.mean(accuracies), 1)

diags = map(np.diag, confusion_matrices)
accuracy_report = np.vstack(diags)
accuracy_means = np.mean(accuracy_report, axis=0)
accuracy_stds = np.std(accuracy_report, axis=0)

global_mean_accuracy = 100 * np.mean(accuracy_report)
global_std_accuracy = 100 * np.std(np.mean(accuracy_report, axis=1))