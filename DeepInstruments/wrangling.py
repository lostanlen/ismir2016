import DeepInstruments as di
import librosa
import numpy as np
import os
import shutil


def chunk_stems(dataset_path,
                decision_hop,
                decision_length,
                stems):
    activation_hop = 2048
    for instrument_id in range(len(stems)):
        instrument_name = di.singlelabel.names[instrument_id]
        instrument_id_str = repr('%(i)02d' % {"i": instrument_id})[1:-1]
        instrument_folder = instrument_id_str + "_" + instrument_name
        instrument_folder_path = os.path.join(dataset_path,
                                              training_or_test,
                                              instrument_folder)
        try:
            os.makedirs(instrument_folder_path)
        except OSError:
            shutil.rmtree(instrument_folder_path)
            os.makedirs(instrument_folder_path)
        for file_id in range(len(stems[instrument_id])):
            file_id_str = repr('%(i)02d' % {"i": file_id})[1:-1]
            stem = stems[instrument_id][file_id]
            track_name = stem.track.name
            stem_name = stem.name
            file_folder = file_id_str + "_" + track_name + "_" + stem_name
            file_folder_path = os.path.join(dataset_path,
                                            training_or_test,
                                            instrument_folder,
                                            file_folder)
            try:
                os.makedirs(file_folder_path)
            except OSError:
                shutil.rmtree(file_folder_path)
                os.makedirs(file_folder_path)
            Y = np.vstack(stem.track.activations_data)[:, stem.name[1:]]
            half_x_hop = int(0.5 * decision_length)
            Y_hop = int(0.5 * float(decision_hop) / activation_hop)
            Y_id = Y_hop
            chunk_id = 0
            sr, x = stem.audio_data
            while Y_id < (len(Y) - 2*Y_hop):
                if Y[Y_id] > 0.5:
                    x_id = int(Y_id * activation_hop)
                    x_range = xrange(x_id-half_x_hop, x_id+half_x_hop)
                    x_chunk = np.transpose(x[x_range, :])
                    chunk_id_str = repr('%(i)03d' % {"i": chunk_id})[1:-1]
                    chunk_str = instrument_name + \
                                "_" + \
                                track_name + \
                                "_" + \
                                stem_name + \
                                "_chunk" + \
                                chunk_id_str + \
                                ".wav"
                    chunk_path = os.path.join(dataset_path,
                                              training_or_test,
                                              instrument_folder,
                                              file_folder,
                                              chunk_str)
                    print(chunk_path)
                    librosa.output.write_wav(chunk_path,
                                             x_chunk,
                                             sr,
                                             norm=False)
                    Y_id += Y_hop
                    chunk_id += 1
                Y_id += Y_hop


def export_singlelabel_dataset():
    dataset_path = os.path.join(os.path.expanduser("~"),
                                "datasets",
                                "medleydb-single-instruments")
    decision_hop = 65536
    decision_length = 131072
    training_stems, test_stems = di.singlelabel.get_stems()
    chunk_stems(dataset_path, decision_hop, decision_length, training_stems)
    chunk_stems(dataset_path, decision_hop, decision_length, test_stems)
