import DeepInstruments as di
import fnmatch
import os

training_path = os.path.join(os.path.expanduser("~"),
                             "datasets",
                             "medleydb-single-instruments",
                             "training")
set_path = training_path
class_names = os.listdir(set_path)
chunk_paths = []
for class_name in class_names:
    class_path = os.path.join(set_path, class_name)
    paths = [os.path.join(path, name)
             for (path, subdir, names) in os.walk(class_path)
             for name in names]
    chunk_paths.append(paths)
