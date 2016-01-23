import numpy as np

def vote(y_test, y_predicted, test_paths):
    chunk_stem_names = [ path.split('chunk')[:-1] for path in test_paths]
    stem_dict = dict()
    