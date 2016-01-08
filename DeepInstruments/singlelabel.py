import collections
import joblib
import medleydb
import numpy as np
import os
import sklearn


cachedir = os.path.expanduser('~/joblib')
memory = joblib.Memory(cachedir=cachedir, verbose=0)


def confusion_matrix(Y_true, Y_predicted):
    y_true = np.argmax(Y_true, axis=1)
    y_predicted = np.argmax(Y_predicted, axis=1)
    n_classes = np.size(Y_true, 1)
    labels = range(n_classes)
    cm = sklearn.metrics.confusion_matrix(y_true, y_predicted, labels)
    cm = cm.astype('float64')
    return cm / cm.sum(axis=1)[:, np.newaxis]


@memory.cache
def melody_annotation_durations():
    session = medleydb.sql.session()
    stems = session.query(medleydb.sql.model.Stem).all()
    melodic_stems = \
        [stem for stem in stems if stem.rank and not stem.track.has_bleed]
    melodic_stem_names = [stem.instrument.name for stem in melodic_stems]
    unique_melodic_names = np.unique(melodic_stem_names)
    n_melodic_names = len(unique_melodic_names)
    n_melodic_stems = len(melodic_stems)
    melodic_counts = np.zeros(n_melodic_names)
    for stem_index in range(n_melodic_stems):
        name = melodic_stem_names[stem_index]
        stem = melodic_stems[stem_index]
        melody_3rd_definition = stem.track.melodies[2]
        f0 = np.vstack(melody_3rd_definition.annotation_data)[:, stem.rank]
        melodic_counts[unique_melodic_names==name] += (f0 > 0).sum()
    sorting = melodic_counts.argsort()
    sorted_melodic_names = unique_melodic_names[sorting]
    sorted_melodic_counts = melodic_counts[sorting]
    sorted_melodic_durations = sorted_melodic_counts * 256.0 / 44100
    counter = collections.Counter(melodic_stem_names)
    n_files_per_name = [ counter[name] for name in sorted_melodic_names ]
    tuples = (sorted_melodic_names,
              n_files_per_name,
              sorted_melodic_durations)
    return np.transpose(np.vstack(tuples))


def train_accuracy(X_train_list, Y_train_list,
                   batch_size, datagen, epoch_size, graph):
    labels = len(Y_train_list)
    dataflow = datagen.flow(X_train_list, Y_train_list,
        batch_size=batch_size,
        epoch_size=epoch_size)
    y_train_true = np.zeros(epoch_size, dtype=int)
    y_train_predicted = np.zeros(epoch_size, dtype=int)
    batch_id = 0
    for (X_batch, Y_batch) in dataflow:
        Y_batch_predicted = graph.predict_on_batch({"X": X_batch})
        Y_batch_predicted = np.hstack(Y_batch_predicted)
        y_batch_predicted = np.argmax(Y_batch_predicted, axis=1)
        batch_range = xrange(batch_id*batch_size, (batch_id+1)*batch_size)
        y_train_predicted[batch_range] = y_batch_predicted
        y_batch_true = np.argmax(Y_batch, axis=1)
        y_train_true[batch_range] = y_batch_true
        batch_id += 1
    cm = sklearn.metrics.confusion_matrix(y_train_true, y_train_predicted,
                                          labels)
    cm = cm.astype(np.float64)
    cm = cm / cm.sum(axis=1)[:, np.newaxis]
    train_accuracies = np.diag(cm)
    mean_accuracy = np.mean(train_accuracies)
    std_accuracy = np.std(train_accuracies)
    print "train accuracy = ", mean_accuracy, " +/- ", std_accuracy
    return train_accuracies


def test_accuracy(X_test, Y_test, batch_size, epoch_size, graph):
    labels = range(Y_test.shape[1])
    test_prediction = graph.predict({"X": X_test})
    Y_test_predicted = test_prediction["Y"]
    y_test_predicted = np.argmax(Y_test_predicted, axis=1)
    y_test_true = np.argmax(Y_test, axis=1)
    cm = sklearn.metrics.confusion_matrix(y_test_true, y_test_predicted,
                                          labels)
    cm = cm.astype(np.float64)
    cm = cm / cm.sum(axis=1)[:, np.newaxis]
    test_accuracies = np.diag(cm)
    mean_accuracy = np.mean(test_accuracies)
    std_accuracy = np.std(test_accuracies)
    print "test mean accuracy = ", mean_accuracy, " +/- ", std_accuracy
    return test_accuracies
