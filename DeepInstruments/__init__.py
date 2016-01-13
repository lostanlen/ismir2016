import os
os.environ["MEDLEYDB_PATH"] = os.path.join(os.path.expanduser("~"),
                                           "datasets", "MedleyDB")

wrong_names = [
    "CroqueMadame_Pilot(Lakelot)",
    "Phoenix_BrokenPledge-ChicagoReel"
]

fixed_names = [
    "CroqueMadame_Pilot",
    "Phoenix_BrokenPledgeChicagoReel"
]

suffix = "_ACTIVATION_CONF.lab"

activation_dir = os.path.join(os.environ["MEDLEYDB_PATH"],
                              "Annotations", "Instrument_Activations",
                              "ACTIVATION_CONF", "")

n_wrong_names = len(wrong_names)
name_index = 0
for name_index in range(n_wrong_names):
    wrong_path = activation_dir + wrong_names[name_index] + suffix
    fixed_path = activation_dir + fixed_names[name_index] + suffix
    try:
        os.rename(wrong_path, fixed_path)
    except OSError as e:
        if not os.path.exists(fixed_path):
            raise e



import DeepInstruments.audio
import DeepInstruments.learning
import DeepInstruments.main
import DeepInstruments.singlelabel
import DeepInstruments.symbolic
import DeepInstruments.wrangling