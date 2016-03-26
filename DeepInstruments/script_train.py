import DeepInstruments as di
import numpy as np
import warnings
import pickle

# Parameters for learning
batch_size = 32
epoch_size = 8192
n_epochs = 20
optimizer = "adam"

# I/O sizes
X_width = decision_length / hop_length
dense2_channels = 8
names = [name.split(" ")[0] for name in di.singlelabel.names]

n_trials = 5
conv1_height = [5, 3]  # resp for 2d, spiral
conv1_width = 5
pool1_height = 3
pool1_width = 5
conv2_height = [5, 5]  # resp for 2d, spiral
conv2_width = 5
pool2_height = 3
pool2_width = 5
alpha = 0.3  # for LeakyReLU
js = np.matrix([[0, 8], [5, 8], [1, 3], [2, 4], [3, 5]])

loss_report = []
chunk_report = []
file_report = []
report = {}

for trial in range(0, n_trials):
    print "*********************************************************"
    print "                           TRIAL", 1+trial
    loss_trial = []
    chunk_trial = []
    file_trial = []
    for arch in range(1, 9):
        print "========================================================="
        print "                      TRIAL ", 1+trial, " ARCH ", arch
        if arch == 1:  # spiral
            conv1_channels = [0, 0, 32]
        elif arch == 2:  # 1d
            conv1_channels = [0, 32, 0]
        elif arch == 3:  # spiral & 1d
            conv1_channels = [0, 32, 32]
        elif arch == 4:  # 2d
            conv1_channels = [32, 0, 0]
        elif arch == 5:  # 2d & spiral
            conv1_channels = [32, 0, 32]
        elif arch == 6:  # 2d & 1d
            conv1_channels = [32, 32, 0]
        elif arch == 7:  # 2d & 1d & spiral
            conv1_channels = [32, 32, 32]
        elif arch == 8: # 2d (more parameters)
            conv1_channels = [48, 0, 0]

        conv2_channels = conv1_channels
        dense1_channels = 64

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
        graph.compile(loss={"Y": "categorical_crossentropy"},
                      optimizer=optimizer)


        # Train ConvNet
        from keras.utils.generic_utils import Progbar
        batch_losses = np.zeros(epoch_size / batch_size)
        loss_history = []
        mean_loss = float("inf")
        for epoch_id in xrange(n_epochs):
            dataflow = datagen.flow(batch_size=batch_size,
                                    epoch_size=epoch_size)
            print "\nEpoch ", 1 + epoch_id
            progbar = Progbar(epoch_size)
            batch_id = 0
            for (X_batch, Y_batch) in dataflow:
                loss = di.learning.train_on_batch(graph, X_batch,
                                                  Y_batch, Q, js, offsets)
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
        file_accuracies = di.singlelabel.file_accuracies(test_paths,
            class_probs, y_test, method="geometric_mean")
        mean_file_accuracy = np.mean(file_accuracies)
        mean_chunk_accuracy = np.mean(chunk_accuracies)
        loss_history.append(mean_loss)
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
        loss_trial.append(loss_history)
        chunk_trial.append(chunk_accuracies)
        file_trial.append(file_accuracies)
    loss_report.append(loss_trial)
    chunk_report.append(chunk_trial)
    file_report.append(file_trial)
    report["loss"] = loss_report
    report["chunk"] = chunk_report
    report["file"] = file_report
    pickle.dump(report, open("Table2_K32.p", "wb"))
