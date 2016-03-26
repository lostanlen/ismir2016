import numpy as np
import pickle

report = pickle.load(open("Table2_K32.p", "r"))
loss_report = report["loss"]
chunk_report = report["chunk"]
file_report = report["file"]

n_instruments = 8
n_archs = 8
n_trials = file_report.__len__()

loss_tensor = np.zeros((n_archs, 1, n_trials))
chunk_tensor = np.zeros((n_archs, n_instruments, n_trials))
file_tensor = np.zeros((n_archs, n_instruments, n_trials))

for arch in range(n_archs):
    for trial in range(n_trials):
        loss_tensor[arch, :, trial] = loss_report[trial][arch][0]
        chunk_tensor[arch, :, trial] = chunk_report[trial][arch]
        file_tensor[arch, :, trial] = file_report[trial][arch]

# Get the chunk-wise average performance on trials
chunk_avg = np.mean(np.mean(chunk_tensor, axis=1), axis=1)
chunk_std = np.std(np.mean(chunk_tensor, axis=1), axis=1)

# Get the file-wise average performance on trials
file_avg = np.mean(file_tensor, axis=2)
file_std = np.std(file_tensor, axis=2)