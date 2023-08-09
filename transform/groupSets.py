import os
import joblib
from skimage.io import imread
import json
import numpy as np

src = fr'{os.getcwd()}/transform/decks'
num_options = 3

data = dict()
data['data'] = []
data['label'] = []
pklname = f"train/data.pkl"

with open(os.path.join(src, 'isSetLookup.json')) as user_file:
    lookup = json.loads(user_file.read())

for subdir in os.listdir(src):
    current_path = os.path.join(src, subdir)
    if  os.path.isfile(current_path):
        continue

    for set in lookup.keys():
        filesInSet = []
        data['label'].append("set" if lookup[set] else "not a set")
        for id in set.split(','):
            im = imread(os.path.join(current_path, f"{id}.png"))
            filesInSet.append(im)
        data['data'].append(np.concatenate(filesInSet, axis=None))

joblib.dump(data, pklname)


