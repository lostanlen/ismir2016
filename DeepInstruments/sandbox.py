import os
os.environ["MEDLEYDB_PATH"] = os.path.expanduser("~/datasets/MedleyDB")

import DeepInstruments as di
import librosa
import matplotlib.pyplot as plt
import numpy as np
import medleydb.sql

session = medleydb.sql.session()
stems = session.query(medleydb.sql.model.Stem).all()
training_paths = []
test_paths = []
for name in di.singlelabel.names:
    instrument_stems = [ stem for stem in stems
                         if stem.instrument.name == name ]
    training_paths.append([ os.path.split(stem.audio_path)[1]
                            for stem in instrument_stems
                            if stem.rank ])
    test_paths.append([ os.path.split(stem.audio_path)[1]
                        for stem in instrument_stems
                        if not stem.rank ])

training_discarded = [
    # Clean electric guitar
    u'CelestialShore_DieForUs_STEM_05.wav'  # has overdubs and shoegaze effects
    # Distorted electric guitar
    u'Meaxic_TakeAStep_STEM_03.wav'  # has overdubs (left/right channels)
    u'TheSoSoGLos_Emergency_STEM_04.wav' # has overdubs
    

]

discarded = [
    # Clarinet
    u'MusicDelta_Beethoven_STEM_09.wav',  # same song in training
    # Clean electric guitar
    u'AlexanderRoss_GoodbyeBolero_STEM_02.wav',  # same song in training
    u'CelestialShore_DieForUs_STEM_05.wav',  # has shoegaze effects
    u'AlexanderRoss_VelvetCurtain_STEM_10.wav',  # is actually a reverb track
    u'TheDistricts_Vermont_STEM_06.wav'  # same song in training
    # Distorted electric guitar
    u'AClassicEducation_NightOwl_STEM_03.wav',  # same song in training
    u'AClassicEducation_NightOwl_STEM_06.wav',  # same song in training
    u'AClassicEducation_NightOwl_STEM_07.wav'  # same song in training
    u'BigTroubles_Phantom_STEM_03.wav',  # same song in training
    u'BigTroubles_Phantom_STEM_03.wav',  # same song in training
    u'Creepoid_OldTree_STEM_05.wav',  # same song in training
    u'HopAlong_SisterCities_STEM_08',  # same song in training
    u'MusicDelta_Britpop_STEM_03.wav',  # is arguably not distorted at all
    u'MusicDelta_Rockabilly_STEM_03.wav',  # is overdrive, not distortion
    u'MusicDelta_SpeedMetal_STEM_03.wav',  # same song in training
    u'MusicDelta_Zeppelin_STEM_03.wav',  # same song in training
    u'PurlingHiss_Lolita_STEM_04.wav',  # same song in training
    u'TheScarletBrand_LesFleursDuMal_STEM_05.wav'  # same song in training

]

batch_size = 32
decision_length = 131072 # in samples
epoch_size = 4096
every_n_epoch = 1
fmin = 55 # in Hz
hop_length = 1024 # in samples
n_bins_per_octave = 12
n_epochs = 1
n_octaves = 8
optimizer = "adagrad"
silence_threshold = -0.7
sr = 32000
test_dirs = ["~/datasets/solosDb8/test"]
train_dirs = ["~/datasets/solosDb8/train"]
conv1_channels = 100
conv1_height = 96
conv1_width = 32
pool1_height = 7
pool1_width = 7
conv2_channels = 100
conv2_height = 8
conv2_width = 8
pool2_height = 8
dense1_channels = 512
drop1_proportion = 0.5
dense2_channels = 64
drop2_proportion = 0.5


session = medleydb.sql.session()
tracks = session.query(medleydb.sql.model.Track).all()
track = tracks[1]

# X (audio representation)
X = di.audio.get_X(decision_length, fmin, hop_length, n_bins_per_octave,
                   n_octaves, track)

# Y (all instrument activations)
activations = di.wrangling.get_activations(instrument_names, track)


# Melodic Z (piano-rolls, i.e. time-frequency activations)
pianorolls = di.wrangling.get_pianorolls(fmin, melodic_names,
                                         n_bins_per_octave, n_octaves, track)


# Non-melodic Z (non-melodic activations)
nonmelodic_activations = di.wrangling.get_activations(nonmelodic_names, track)