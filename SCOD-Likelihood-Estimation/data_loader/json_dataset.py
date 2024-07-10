import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
from PIL.Image import Image as PImage
import torch
from torch.utils.data import Dataset
import albumentations as A
import torchvision.transforms as T
from typing import Callable, Tuple, List, Dict
from collections import Counter
import warnings


class BasicAugment(object):
    """
    Albumentations wrapper for augmentation. Loads albumentations configuration and uses it to construct an augmentation process.
    """

    def __init__(self, config_path):
        """

        Args:
            config_path (str, optional): Path to albumentations configuration file.
        """
        self.transform = A.load(config_path)

    def __call__(self, a: np.array):
        result = self.transform(image=a)
        a = result['image']
        return a

    def __repr__(self):
        """
        Returns:
            str: Name of the object.
        """
        return self.__class__.__name__ + '()'


class PILToRGBArray(object):
    """
    Convert a ``PIL Image`` to `RGB` and then to a `np.array`.
    """

    def __call__(self, pic: Image) -> np.array:
        """
        Args:
            pic (PIL Image): Image to be converted.
        Returns:
            array: Converted image of shape [M,N,3].
        """
        return np.array(pic.convert('RGB'))

    def __repr__(self):
        """
        Returns:
            str: Name of the object.
        """
        return self.__class__.__name__ + '()'


class JSONDataset(Dataset):
    def __init__(self,
                 root: str,
                 path_json: Dict,
                 folders: List[str],
                 transform_path: str,
                 load_images_to_memory: bool,
                 to_array: Callable[[PImage], np.array] = PILToRGBArray()):
        """

        Args:
            root (str): Root path to the dataset folder.
            path_json (Dict): Path to json defining paths to data samples and corresponding labels.
            folders (List[str]): Folders from path_json to use. Used to separate data into train,test,val splits.
            transform_path (str): Path to albumentations configuration.
            load_images_to_memory (bool): If True, loads samples from disk to RAM.            
            to_array (Callable[[PImage], np.array], optional): Function to convert PIL Image to numpy array. Defaults to PILToRGBArray().
        """

        self.root = root
        self.path_json = path_json
        self.folders = folders
        self.load_images_to_memory = load_images_to_memory
        self.transform = BasicAugment(transform_path)
        self.to_array = to_array

        self.paths_and_labels = []
        for sample in self.path_json.values():
            sample_folder = sample['folder']
            camera_path = os.path.join(self.root, sample['camera'])
            label = sample['label']
            if sample_folder in folders:
                self.paths_and_labels.append(
                    (camera_path, label))
                
        self.data = []
        if self.load_images_to_memory:
            warnings.warn("Loading dataset to memory ...")
            for camera_path, label in tqdm(self.paths_and_labels):
                camera = self.load_image(camera_path=camera_path)
                self.data.append((camera, label))

    def load_image(self, camera_path: str) -> np.array:
        """
        Loads image from disk.

        Args:
            camera_path (str): Path to camera image.

        Returns:
            np.array: Camera image.
        """
        camera = self.to_array(Image.open(camera_path))
        return camera

    def __len__(self) -> int:
        """Returns number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        if self.load_images_to_memory:
            return len(self.data)
        else:
            return len(self.paths_and_labels)

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        """Loads a single sample from the dataset.

        Args:
            idx (int): Index of sample to load from internal memory.

        Returns:
            Tuple[torch.tensor, torch.tensor]: (image, label).

        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # if self.load_images_to_memory is False, every __getitem__ call must open a new PIL.Image
        # to increase the speed of the training, it is recommended to use self.load_images_to_memory=True
        # if the dataset is vast however, the entire dataset may not be able to fit into memory
        if self.load_images_to_memory:
            camera, label = self.data[idx]
        else:
            camera_path, label = self.paths_and_labels[idx]
            camera = self.load_image(camera_path=camera_path)

        camera = self.transform(camera)

        # create sample, convert class label to group label (e.g., we might not want to distinguish asphalt and tar roads)
        sample = (camera, label)

        assert camera.shape[0] == 3, f"Encountered image that does not have 3 color channels"

        return sample

    def get_sample_weights(self):
        label_list = []
        for camera_path, label in self.paths_and_labels:
            label_list.append(label)
        label_counter = Counter(label_list)
        weight_list = []
        for label in label_list:
            weight_list.append(10**10 / label_counter[label])
        return weight_list
