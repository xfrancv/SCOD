from torchvision import datasets
from base import BaseDataLoader
from data_loader.json_dataset import JSONDataset
import json


class JSONDataLoader(BaseDataLoader):
    def __init__(self,
                 data_dir,
                 batch_size,
                 shuffle,
                 stratified,
                 num_workers,
                 **kwargs):
        self.data_dir = data_dir
        path_json_path = self.data_dir + 'paths.json'
        with open(path_json_path, 'r') as handle:
            self.path_json = json.load(handle)

        self.dataset = JSONDataset(root=self.data_dir,
                                   path_json=self.path_json,
                                   **kwargs)
        super().__init__(self.dataset, batch_size, shuffle, num_workers,
                         self.dataset.get_sample_weights() if stratified else None)
