import numpy as np
import sklearn


def confusion_matrix(Y_true, Y_predicted):
    y_true = np.argmax(Y_true, axis=1)
    y_predicted = np.argmax(Y_predicted, axis=1)
    n_classes = np.size(Y_true, 1)
    labels = range(n_classes)
    cm = sklearn.metrics.confusion_matrix(y_true, y_predicted, labels)
    cm = cm.astype('float64')
    return cm / cm.sum(axis=1)[:, np.newaxis]


def evaluate(graph,
             datagen,
             X_train_list,
             Y_train_list,
             X_test,
             Y_test,
             batch_size,
             epoch_size):
    labels = range(Y_test.shape[1])

    # Get training accuracy
    dataflow = datagen.flow(X_train_list,
        Y_train_list,
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

    cm = sklearn.metrics.confusion_matrix(y_train_true, y_train_predicted, labels)
    cm = cm.astype(np.float64)
    cm = cm / cm.sum(axis=1)[:, np.newaxis]
    train_accuracies = np.diag(cm)
    train_mean_accuracy = np.mean(train_accuracies)
    train_std_accuracy = np.std(train_accuracies)
    print "train mean accuracy = ", train_mean_accuracy

    # Get test accuracy
    test_prediction = graph.predict({"X": X_test})
    Y_test_predicted = test_prediction["Y"]
    y_test_predicted = np.argmax(Y_test_predicted, axis=1)
    y_test_true = np.argmax(Y_test, axis=1)
    cm = sklearn.metrics.confusion_matrix(y_test_true, y_test_predicted, labels)
    cm = cm.astype(np.float64)
    cm = cm / cm.sum(axis=1)[:, np.newaxis]
    test_accuracies = np.diag(cm)
    test_mean_accuracy = np.mean(test_accuracies)
    test_std_accuracy = np.std(test_accuracies)
    print "test mean accuracy = ", test_mean_accuracy

    return train_accuracies, test_accuracies
