import collections
import joblib
import librosa
import math
import medleydb
import medleydb.sql
import numpy as np
import os
import sklearn

import DeepInstruments as di

names = [u'clarinet',
         u'distorted electric guitar',
         u'female singer',
         u'flute',
         u'piano',
         u'tenor saxophone',
         u'trumpet',
         u'violin']

medleydb_discarded = [
    # Clarinet
    u'MusicDelta_InTheHalloftheMountainKing_STEM_09.wav',  # 2 players
    # Distorted electric guitar
    u'Meaxis_TakeAStep_STEM_03.wav',  # two players
    u'HopAlong_SisterCities_STEM_06.wav',  # is arguably not distorted
    u'TheSoSoGlos_Emergency_04.wav',  # two players
    u'TablaBreakbeatScience_Animoog_STEM_03.wav',  # prepared guitar I think
    u'Creepoid_OldTree_STEM_06.wav',  # is clean guitar at the start
    u'MusicDelta_Britpop_STEM_03.wav',  # is arguably not distorted at all
    u'MusicDelta_Rockabilly_STEM_03.wav',  # is overdrive, not distortion
    # Female singer
    u'DreamersOfTheGhetto_HeavyLove_STEM_08.wav',  # is a vocal FX track
    u'LizNelson_ImComingHome_STEM_02.wav',  # has bleed
    u'LizNelson_Rainfall_STEM_05.wav',  # has bleed
    # Flute
    u'MusicDelta_InTheHalloftheMountainKing_STEM_07.wav',  # 2 players
    # Piano
    u'MatthewEntwistle_TheArch_STEM_18.wav',  # has huge reverb
    # Trumpet
    u'MusicDelta_FusionJazz_STEM_06.wav',  # has sordina
    # Violin
    # (empty)

]

medleydb_movedtotest = [
    # Distorted electric guitar
    u'AClassicEducation_NightOwl_STEM_03.wav',
    u'AClassicEducation_NightOwl_STEM_04.wav',
    u'AClassicEducation_NightOwl_STEM_06.wav',
    u'AClassicEducation_NightOwl_STEM_07.wav',
    u'BigTroubles_Phantom_STEM_03.wav',
    u'BigTroubles_Phantom_STEM_05.wav',
    u'BigTroubles_Phantom_STEM_07.wav',
    u'Meaxic_TakeAStep_STEM_03.wav',
    u'Meaxic_YouListen_STEM_06.wav',
    u'PurlingHiss_Lolita_STEM_03.wav',
    u'PurlingHiss_Lolita_STEM_04.wav',
    u'TheScarletBrand_LesFleursDuMal_STEM_04.wav',
    u'TheScarletBrand_LesFleursDuMal_STEM_05.wav',
    # Female singer
    u'ClaraBerryAndWooldog_AirTraffic_STEM_08.wav',
    u'ClaraBerryAndWooldog_AirTraffic_STEM_07.wav',
    u'ClaraBerryAndWooldog_Boys_STEM_06.wav',
    u'ClaraBerryAndWooldog_Stella_STEM_07.wav',
    u'ClaraBerryAndWooldog_TheBadGuys_STEM_02.wav',
    u'ClaraBerryAndWooldog_WaltzForMyVictims_STEM_05.wav',
    u'LizNelson_Coldwar_STEM_02.wav',
    u'LizNelson_ImComingHome_STEM_01.wav',
    u'LizNelson_ImComingHome_STEM_03.wav',
    u'LizNelson_ImComingHome_STEM_04.wav',
    u'LizNelson_Rainfall_STEM_01.wav',
    u'LizNelson_Rainfall_STEM_02.wav',
    u'LizNelson_Rainfall_STEM_03.wav',
]


cachedir = os.path.expanduser('~/joblib')
memory = joblib.Memory(cachedir=cachedir, verbose=0)

class ScalogramGenerator(object):
    def __init__(self,
                 decision_length,
                 fmin,
                 hop_length,
                 n_bins_per_octave,
                 n_octaves,
                 training_stems):
        self.decision_length = float(decision_length)
        self.fmin = fmin
        self.hop_length = float(hop_length)
        self.n_bins_per_octave = n_bins_per_octave
        self.n_octaves = n_octaves
        X = []
        Y = []
        Z = []
        G = []
        delayed_get_X = joblib.delayed(di.audio.cached_get_X)
        for class_stems in training_stems:
            X.append(joblib.Parallel(n_jobs=-1, verbose=10)(
                delayed_get_X(decision_length, fmin, hop_length,
                              n_bins_per_octave, n_octaves, stem)
                for stem in class_stems
            ))
            Y.append([di.singlelabel.get_Y(stem) for stem in class_stems])
            Z.append([di.symbolic.get_Z(fmin, hop_length, n_bins_per_octave,
                                        n_octaves, stem)
                      for stem in class_stems])
            G.append([di.symbolic.get_G(hop_length, n_bins_per_octave,
                                        n_octaves, stem)
                      for stem in class_stems])
        X_mat = np.hstack([X_file for X_class in X for X_file in X_class])
        self.X_mean = np.mean(X_mat)
        self.X_std = np.std(X_mat)
        for instrument_id in range(len(X)):
            X[instrument_id] = [(X_file-self.X_mean) / self.X_std
                                for X_file in X[instrument_id]]
        self.X = X
        self.Y = Y
        self.Z = Z
        self.G = G
        self.indices = di.singlelabel.get_indices(Y, decision_length)
        n_instruments = len(X)
        durations = []
        for instrument_id in range(n_instruments):
            file_lengths = map(float, map(len, self.indices[instrument_id]))
            durations.append(file_lengths / np.sum(file_lengths))
        self.durations = durations

    def flow(self, batch_size=512, epoch_size=4096):
        half_X_hop = int(0.5 * self.decision_length / self.hop_length)
        n_batches = int(math.ceil(float(epoch_size) / batch_size))
        n_bins = self.X[0][0].shape[0]
        n_instruments = self.Y[0][0].shape[0]
        X_batch_size = (batch_size, 1, n_bins, 2 * half_X_hop)
        X_batch = np.zeros(X_batch_size, np.float32)
        Y_batch_size = (batch_size, n_instruments)
        Y_batch = np.zeros(Y_batch_size, np.float32)
        y_epoch_size = (n_batches, batch_size)
        y_epoch = np.random.randint(0, n_instruments, size=y_epoch_size)
        Z_batch = np.zeros(X_batch_size, np.float32)
        G_batch = np.zeros(X_batch_size, np.float32)
        for batch_id in range(n_batches):
            for sample_id in range(batch_size):
                instrument_id = y_epoch[batch_id, sample_id]
                n_files = len(self.indices[instrument_id])
                durations = self.durations[instrument_id]
                file_id = np.random.choice(n_files, p=durations)
                Y_id = np.random.choice(self.indices[instrument_id][file_id])
                X_id = int(Y_id * 2048.0 / self.hop_length)
                X_range = xrange(X_id-half_X_hop, X_id+half_X_hop)
                X_batch[sample_id, :, :] = \
                    self.X[instrument_id][file_id][:, X_range]
                Y_batch[sample_id, :] = \
                    self.Y[instrument_id][file_id][:, Y_id]
                Z_batch[sample_id, :] = \
                    self.Z[instrument_id][file_id][:, X_range]
                G_batch[sample_id, :] = \
                    self.G[instrument_id][file_id][:, X_range]
            yield X_batch, Y_batch, Z_batch, G_batch

    def get_X(self, paths):
        delayed_get_X = joblib.delayed(di.audio.cached_get_X)
        X_test = joblib.Parallel(n_jobs=-1, verbose=10)(
                delayed_get_X(self.decision_length,
                              self.fmin,
                              self.hop_length,
                              self.n_bins_per_octave,
                              self.n_octaves,
                              path)
                for path in paths)
        X_test = np.stack(X_test)[:, :, :-1]
        shape = X_test.shape
        new_shape = (shape[0], 1, shape[1], shape[2])
        X_test = np.reshape(X_test, new_shape)
        X_test = (X_test - self.X_mean) / self.X_std
        return X_test


def get_indices(Y, decision_length):
    indices_classes = []
    activation_hop = 2048
    half_trimming_length = int(0.5 * (decision_length / activation_hop))
    for Y_class in Y:
        indices_files = []
        for activations in Y_class:
            activation = np.max(activations, axis=0)
            left_bound = half_trimming_length
            right_bound = len(activation) - half_trimming_length
            indices = np.where(np.greater_equal(activation, 0.5))[0]
            indices = indices[np.where(
                    np.greater(indices, left_bound) &
                    np.less(indices, right_bound))[0]]
            indices_files.append(indices)
        indices_classes.append(indices_files)
    return indices_classes


def get_paths(training_or_test):
    set_path = os.path.join(os.path.expanduser("~"),
                             "datasets",
                             "medleydb-single-instruments",
                             training_or_test)
    paths = [
        [os.path.join(path, name)
         for (path, subdir, names)
         in os.walk(os.path.join(set_path, class_name))
         for name in names]
        for class_name in os.listdir(set_path)]
    paths = [path for class_path in paths for path in class_path]
    return paths


def get_Y(stem):
    track_activations = np.vstack(stem.track.activations_data)[:, 1:]
    stem_id = int(stem.name[1:])
    n_frames = track_activations.shape[0]
    n_instruments = len(di.singlelabel.names)
    activations = np.zeros((n_instruments, n_frames))
    instrument_id = di.singlelabel.names.index(stem.instrument.name)
    activations[instrument_id, :] = track_activations[:, stem_id - 1]
    return activations


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
    n_files_per_name = [counter[name] for name in sorted_melodic_names]
    tuples = (sorted_melodic_names,
              n_files_per_name,
              sorted_melodic_durations)
    return np.transpose(np.vstack(tuples))


def get_stems():
    session = medleydb.sql.session()
    stems = session.query(medleydb.sql.model.Stem).all()
    stems = [stem for stem in stems if not stem.track.has_bleed]
    training_stems = []
    test_stems = []
    for instrument_name in di.singlelabel.names:
        instrument_stems = [stem for stem in stems
                            if stem.instrument.name == instrument_name]
        training_instrument_stems = []
        test_instrument_stems = []
        for stem in instrument_stems:
            file_name = os.path.split(stem.audio_path)[1]
            if file_name in di.singlelabel.medleydb_discarded:
                pass
            else:
                if file_name in di.singlelabel.medleydb_movedtotest:
                    test_instrument_stems.append(stem)
                else:
                    training_instrument_stems.append(stem)
        training_stems.append(training_instrument_stems)
        test_stems.append(test_instrument_stems)
    return test_stems, training_stems


def test_accuracies(graph, X_test, y_true):
    Y_predicted = graph.predict({"X": X_test})["Y"]
    y_predicted = np.argmax(Y_predicted, axis=1)
    cm = sklearn.metrics.confusion_matrix(y_true, y_predicted)
    cm = cm.astype(np.float64)
    cm = cm / cm.sum(axis=1)[:, np.newaxis]
    test_accuracies = 100 * np.diag(cm)
    return test_accuracies


def training_accuracies(batch_size, datagen, epoch_size, graph):
    n_batches = int(math.ceil(float(epoch_size) / batch_size))
    n_instruments = datagen.Y[0][0].shape[0]
    dataflow = datagen.flow(batch_size=batch_size, epoch_size=epoch_size)
    y_train_predicted = np.zeros((n_batches, batch_size), dtype=int)
    y_train_true = np.zeros((n_batches, batch_size), dtype=int)
    batch_id = 0
    for (X_batch, Y_batch) in dataflow:
        Y_batch_predicted = graph.predict_on_batch({"X": X_batch})
        Y_batch_predicted = np.hstack(Y_batch_predicted)
        y_batch_predicted = np.argmax(Y_batch_predicted, axis=1)
        y_train_predicted[batch_id, :] = y_batch_predicted
        y_batch_true = np.argmax(Y_batch, axis=1)
        y_train_true[batch_id, :] = y_batch_true
        batch_id += 1
    y_train_predicted = np.ndarray.flatten(y_train_predicted)
    y_train_true = np.ndarray.flatten(y_train_true)
    cm = sklearn.metrics.confusion_matrix(y_train_true, y_train_predicted,
                                          range(n_instruments))
    cm = cm.astype(np.float64)
    cm = cm / cm.sum(axis=1)[:, np.newaxis]
    training_accuracies = 100 * np.diag(cm)
    return training_accuracies