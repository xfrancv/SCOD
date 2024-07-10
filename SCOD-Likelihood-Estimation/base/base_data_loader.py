import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.dataloader import default_collate


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, num_workers, sample_weights=None, collate_fn=default_collate):
        self.batch_idx = 0
        self.n_samples = len(dataset)
        
        if sample_weights:
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            self.shuffle = None
        else:
            sampler = None
            self.shuffle = shuffle

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'sampler': sampler,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(**self.init_kwargs)