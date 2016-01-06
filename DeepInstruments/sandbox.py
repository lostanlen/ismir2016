import numpy as np
import os
os.environ["MEDLEYDB_PATH"] = os.path.expanduser("~/datasets/MedleyDB")
import medleydb.sql

instrument_list = ['female_singer', 'acoustic_guitar', 'violin']

session = medleydb.sql.session()
track = session.query(medleydb.sql.model.Track).first()

# get X
instrument_list, decision_duration, fmin, hop_duration,
           n_bins_per_octave, n_octaves, sr
def get_X(file_path, decision_duration, fmin, hop_duration, n_bins_per_octave,
          n_octaves, sr):


sr, X_stereo = track.audio_data
X_stereo = X_stereo.astype(np.float32)
X_mono = np.sum(X_stereo, axis=1) / (32768.0 * 2)


track.audio_data
for track in mtrack_list:
    print([s.instrument for s in track.stems])
    X = track.audio_data
