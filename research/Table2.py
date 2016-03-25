import numpy as np
import pickle

report = pickle.load(open("research/Table2.p", "r"))
loss_report = report["loss"]
chunk_report = report["chunk"]
file_report = report["file"]

n_instruments = 8
n_archs = 6
n_trials = 10

chunk_tensor = np.zeros((n_archs, n_instruments, n_trials))
for arch in range(n_archs):
    for trial in range(n_trials):
        chunk_tensor[arch, :, trial] = chunk_report[trial][arch][-1]


