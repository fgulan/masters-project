import sys
import cro_mapper
import os
import unicodedata
import numpy as np
from scipy import misc

def _get_all_file_paths(path):
    file_paths = []
    for root, dirs, files in os.walk(path):
        for file_ in files:
            full_path = os.path.join(root, file_)
            if os.path.isfile(full_path) and full_path.endswith(".png"):
                file_paths.append(unicodedata.normalize('NFC', full_path))
    return file_paths
    
def load_dataset(path):
    print("Loading dataset at path:", path)
    files = _get_all_file_paths(path)
    X = []
    y = []
    for file in files:
        image = misc.imread(file, mode='F')
        X.append(image)
        folder_path = os.path.dirname(file)
        letter = os.path.basename(folder_path)
        letter_int = cro_mapper.map_letter_to_int(letter)
        y.append(letter_int)
    return np.asarray(X), np.asarray(y)
    