import DeepInstruments as di
import sklearn.ensemble
import numpy as np

# Evaluate random forest
training_paths = di.singlelabel.get_paths("training")
X_training = di.descriptors.get_X(training_paths)
y_training = np.hstack(map(di.descriptors.get_y, training_paths))

X_means = np.mean(X_training, axis=0)
X_stds = np.std(X_training, axis=0)
X_training = (X_training - X_means) / X_stds

test_paths = di.singlelabel.get_paths("test")
X_test = di.descriptors.get_X(test_paths)
X_test = (X_test - X_means) / X_stds
y_test = np.hstack(map(di.descriptors.get_y, test_paths))

n_trials = 10
confusion_matrices = []

for trial_index in range(n_trials):
    clf = sklearn.ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=100)
    clf = clf.fit(X_training, y_training)
    y_predicted = clf.predict(X_test)

    cm = sklearn.metrics.confusion_matrix(y_test, y_predicted).astype("float")
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