import DeepInstruments as di
import joblib
import numpy as np
import os
import sacred

ex = sacred.Experiment("DeepInstruments")

memory = joblib.Memory(cachedir=os.path.expanduser('~/joblib'))


@ex.automain
@memory.cache
def run(batch_size,
        decision_duration,
        epoch_size,
        every_n_epoch,
        fmin,
        hop_duration,
        instrument_list,
        n_bins_per_octave,
        n_epochs,
        n_octaves,
        optimizer,
        silence_threshold,
        sr,
        test_dirs,
        train_dirs,
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
        dense1_channels,
        drop1_proportion,
        dense2_channels,
        drop2_proportion):
    # Get training file paths
    train_file_paths = \
        di.symbolic.get_paths(train_dirs, instrument_list, 'wav')

    # Compute features and labels in training set, list them per class
    (X_train_list, Y_train_list) = \
        di.wrangling.get_XY(train_file_paths, instrument_list,
                            decision_duration, fmin, hop_duration,
                            n_bins_per_octave, n_octaves, sr)

    # Standardize globally the training set
    X_global = np.hstack(X_train_list)
    X_mean = np.mean(X_global)
    X_std = np.std(X_global)
    X_train_list = [(X-X_mean)/X_std for X in X_train_list]

    # Build chunk generator
    datagen = di.learning.ChunkGenerator(decision_duration, hop_duration,
                                         silence_threshold)

    # Compute features and labels in test set, list them per class
    test_file_paths = \
        di.symbolic.get_paths(test_dirs, instrument_list, 'wav')
    (X_test_list, Y_test_list) = \
        di.wrangling.get_XY(test_file_paths, instrument_list,
                            decision_duration, fmin, hop_duration,
                            n_bins_per_octave, n_octaves, sr)

    # Standardize globally the test set according mean and std of training set
    X_test_list = [(X-X_mean)/X_std for X in X_test_list]

    # Format the test set into chunks, discard silent chunks
    (X_test, Y_test) = di.wrangling.chunk_test_set(X_test_list, Y_test_list,
                                                   hop_duration, sr)

    # Build deep learning model as a Keras graph
    input_height = n_bins_per_octave * n_octaves
    input_width = decision_duration / hop_duration
    graph = \
        di.learning.build_graph(input_height, input_width,
                                conv1_channels, conv1_height, conv1_width,
                                pool1_height, pool1_width,
                                conv2_channels, conv2_height, conv2_width,
                                pool2_height, pool2_width,
                                dense1_channels, drop1_proportion,
                                dense2_channels, drop2_proportion,
                                dense3_channels=len(instrument_list))

    # Compile model
    graph.compile(loss={"Y": "categorical_crossentropy"}, optimizer=optimizer)

    # Train model, monitor accuracy
    report = di.learning.run_graph(X_train_list, Y_train_list, X_test, Y_test,
                                   batch_size, datagen, epoch_size,
                                   every_n_epoch, graph, n_epochs)

    return report
