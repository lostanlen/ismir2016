import os
os.environ["MEDLEYDB_PATH"] = os.path.expanduser("~/datasets/MedleyDB")
import medleydb


instrument_list = ['female_singer', 'acoustic_guitar', 'violin']
mtrack_list = medleydb.load_all_multitracks()

for track in mtrack_list:
    print([s.instrument for s in track.stems])