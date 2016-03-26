import numpy as np
import pickle

report = pickle.load(open("Table2.p", "r"))
loss_report = report["loss"]
chunk_report = report["chunk"]
file_report = report["file"]

n_instruments = 8
n_archs = 8
n_trials = file_report.__len__()

loss_tensor = np.zeros((n_archs, 1, n_trials))
chunk_tensor = np.zeros((n_archs, n_instruments, n_trials))

for arch in range(n_archs):
    for trial in range(n_trials):
        loss_tensor[arch, :, trial] = loss_report[trial][arch][0]
        chunk_tensor[arch, :, trial] = chunk_report[trial][arch]

# Get the average performance on trials
average = np.mean(np.mean(chunk_tensor, axis=1), axis=1)
stddev = np.std(np.mean(chunk_tensor, axis=1), axis=1)

# Get the trials with best training loss
best_loss_ids = np.argmax(loss_tensor, axis=2)
best_trials = np.zeros((n_archs, n_instruments))
for arch in range(n_archs):
    best_trials[arch] = chunk_tensor[arch, :, best_loss_ids[arch]]


# Get the training loss corresponding to best trial
best_losses = np.zeros((n_archs,))
best_trial_ids = np.argmax(np.mean(chunk_tensor, axis=1), axis=1)
for arch in range(n_archs):
    best_losses[arch] = loss_tensor[arch, 0, best_trial_ids[arch]]