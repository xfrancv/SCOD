import numpy as np
import json
from helpers.metrics import Metrics

def load_result(path):
    data = np.load(path, allow_pickle=True)
    return data


def get_results(path, nearfar, ood_dataset, filter_paths=None):
    result = np.load(path, allow_pickle=True)
    id_y_pred, id_score, id_y_true = result['id']['test']    
    ood_y_pred, ood_score, ood_y_true = result['ood'][nearfar][ood_dataset]
    
    y_pred = np.concatenate([id_y_pred, ood_y_pred])
    score = np.concatenate([id_score, ood_score])
    y_true = np.concatenate([id_y_true, -np.ones_like(ood_y_true)])

    if filter_paths is None:
        return y_pred, score, y_true
    else:
        id_paths = result['id']['test_paths']
        ood_paths = result['ood'][nearfar][ood_dataset+'_paths']
        with open(filter_paths, 'r') as handle:
            data = json.load(handle)
        
        training_paths = []
        for sample in data.values():
            if sample['folder'] > 0:
                training_paths.append(sample['camera'][2:])
        
        training_paths = set(training_paths)
        
        paths = np.concatenate([id_paths, ood_paths])
        paths = [path.split()[0] for path in paths]
        
        is_test = [path not in training_paths for path in paths]
        return y_pred, score, y_true, is_test

def get_paths(path, nearfar, ood_dataset):
    result = np.load(path, allow_pickle=True)
    id_paths = result['id']['test_paths']
    ood_paths = result['ood'][nearfar][ood_dataset+'_paths']

    paths = np.concatenate([id_paths, ood_paths])
    paths = [path.split()[0] for path in paths]
    return paths

def get_metrics(result, nearfar, ood_dataset):
    id_y_pred, id_score, id_y_true = result['id']['test']
    ood_y_pred, ood_score, ood_y_true = result['ood'][nearfar][ood_dataset]

    y_pred = np.concatenate([id_y_pred, ood_y_pred])
    score = np.concatenate([id_score, ood_score])
    y_true = np.concatenate([id_y_true, -np.ones_like(ood_y_true)])

    metrics = Metrics(y_true=y_true, y_pred=y_pred, score=-score)
    return metrics