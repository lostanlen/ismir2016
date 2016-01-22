import os
import shutil
os.environ["MEDLEYDB_PATH"] = os.path.join(os.path.expanduser("~"),
                                           "datasets", "MedleyDB")

wrong_names = [
    "CroqueMadame_Pilot(Lakelot)",
    "JoelHelander_IntheAtticBedroom(SuitePartThree)",
    "Phoenix_BrokenPledge-ChicagoReel",
    "Phoenix_Elzic'sFarewell",
    "Phoenix_LarkOnTheStrand-DrummondCastle",
    "Phoenix_SeanCaughlin's-TheScartaglen"
]

fixed_names = [
    "CroqueMadame_Pilot",
    "JoelHelander_IntheAtticBedroom",
    "Phoenix_BrokenPledgeChicagoReel",
    "Phoenix_ElzicsFarewell",
    "Phoenix_LarkOnTheStrandDrummondCastle",
    "Phoenix_SeanCaughlinsTheScartaglen",
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


source_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           "patched_annotations",
                           "TheDistricts_Vermont_RANKING.txt")
destination_path = os.path.join(os.environ["MEDLEYDB_PATH"],
                                "Annotations",
                                "Stem_Rankings",
                                "TheDistricts_Vermont_RANKING.txt")
shutil.copyfile(source_path, destination_path)


import DeepInstruments.audio
import DeepInstruments.descriptors
import DeepInstruments.learning
import DeepInstruments.singlelabel
import DeepInstruments.symbolic
import DeepInstruments.wrangling
