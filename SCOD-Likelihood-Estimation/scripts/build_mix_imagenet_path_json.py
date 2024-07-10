import glob
from tqdm import tqdm
import numpy as np
import os
import json
import sys


def build_path_json(root: str):
    """
    Constructs dictionary of (camera, label) pairs for the dataset.    

    Args:
        root (str, optional): Path to the dataset root.
    """
    os.chdir(root)

    outer_dict = {}

    types = ('./**/*.png',
             './**/*.PNG',
             './**/*.jpg',
             './**/*.JPG',
             './**/*.jpeg',
             './**/*.JPEG')

    files_grabbed = []
    for files in tqdm(types):
        files_grabbed.extend(glob.glob(files, recursive=True))

    ix = 0
    nr_in = 0
    nr_out = 0
    for file_path in tqdm(files_grabbed):
        label = int('imagenet' in file_path)
        outer_dict[ix] = {'camera': file_path, 'label': label}
        ix += 1
        
        if (label == 0):
            nr_out += 1
        elif (label == 1):
            nr_in += 1
        else:
            raise RuntimeError
        
    print(f"ID {nr_in}, OOD {nr_out}")
    perc = nr_out / nr_in    
        
    n_folders = 10
    c_folder = 0
    indexes = [i for i in range(ix)]
    np.random.shuffle(indexes)
    
    for i in indexes:
        if outer_dict[i]['label'] == 0:
            if np.random.rand() <= 0.5:    
                outer_dict[i]['folder'] = -1
            else:
                outer_dict[i]['folder'] = c_folder        
                c_folder += 1
        else:
            if np.random.rand() <= perc / 2.:
                outer_dict[i]['folder'] = c_folder        
                c_folder += 1
            else:
                outer_dict[i]['folder'] = -2

        if c_folder >= n_folders:
            c_folder = 0
         
    counts = {folder: {'id': 0, 'ood': 0} for folder in list(range(n_folders)) + [-2, -1]}

    for i in indexes:
        label = outer_dict[i]['label']
        folder = outer_dict[i]['folder']
        idood = 'id' if label == 1 else 'ood'
        counts[folder][idood] += 1
        
    for folder in counts:
        print(folder, counts[folder])    
    
    for folder in range(n_folders):
        valid_indexes = []
        for i in indexes:
            if outer_dict[i]['folder'] == -2:
                valid_indexes.append(i)

        np.random.shuffle(valid_indexes)
        for j in range(counts[folder]['ood']):
            outer_dict[valid_indexes[j]]['folder'] = folder
        for j in range(counts[folder]['ood'], 2*counts[folder]['ood']):
            outer_dict[valid_indexes[j]]['folder'] = folder
            outer_dict[valid_indexes[j]]['label'] = 0

    counts = {folder: {'id': 0, 'ood': 0} for folder in list(range(n_folders)) + [-2, -1]}

    for i in indexes:
        label = outer_dict[i]['label']
        folder = outer_dict[i]['folder']
        idood = 'id' if label == 1 else 'ood'
        counts[folder][idood] += 1
        
    for folder in counts:
        print(folder, counts[folder])
                
    print("Saving path dictionary to json ...")
    with open("paths.json", "w") as outfile:
        json.dump(outer_dict, outfile)


if __name__ == '__main__':
    assert len(
        sys.argv) == 2, f"Expected 1 command line argument, but received {len(sys.argv) - 1}"
    build_path_json(sys.argv[1])
