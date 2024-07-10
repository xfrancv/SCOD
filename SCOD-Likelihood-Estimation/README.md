# Theoretically Grounded SCOD - Likelihood estimation
This repository contains a Python library for training binary classifiers using PyTorch. The binary classifier is used to model the posterior probability $p(y={\rm ID}|x)$ or $p(y={\rm MIX}|x)$, which can then be used to extract the estimate $\hat{g}(x)$ of the likelihood ratio $g(x)=\frac{p_O(x)}{p_I(x)}$.

The trained models can then be exported into JIT using `serialize_model.py` and used within our fork of the OpenOOD benchmark.

## Installation
Use Anaconda to install all dependencies of the project.

```bash
$ conda env create --file environment.yaml
```

# Usage
## Data
The training code requires the user to provide path to a directory with the following structure:
```
data_root
│   paths.json
│
└───cifar100
│   │
│   └───train
│       │
│       └─── chair
│       │   └─── 0001.png
│       │   └─── 0002.png
│       │   └─── 0003.png
│       ...
│   
└───tin
    │ ...
```
The repository should contain the datasets (links suffice) and a **path.json** file. The JSON defines how to separate the data into parts for training, validation and evaluation. The structure of the **path.json** is as follows:
```JSON
{
    "0": {
        "camera": "./cifar100/train/chair/0408.png",
        "label": 1,
        "folder": 1
    },
    "1": {
        "camera": "./cifar100/train/chair/0230.png",
        "label": 1,
        "folder": 2
    },
    "2": {
        "camera": "./cifar100/train/chair/0278.png",
        "label": 1,
        "folder": 8
    },
    "149997": {
        "camera": "./tin/train/n01629819/images/n01629819_10.JPEG",
        "label": 0,
        "folder": 2
    },
    "149998": {
        "camera": "./tin/train/n01629819/images/n01629819_490.JPEG",
        "label": 0,
        "folder": 1
    },
    "149999": {
        "camera": "./tin/train/n01629819/images/n01629819_413.JPEG",
        "label": 0,
        "folder": 9
    }
}
```
Each sample has a unique number. The **label** entry defines whether the sample is ID (`"label": 1`) or OOD/MIX (`"label": 0`). The **folder** entry defines the separation into training, validation and evaluation parts. Folders -1 and -2 are reserved for evaluation. We use folders 0,1,...,7 for training and folders 8,9 for validation. The training and validation folders are defined by a configuration JSON; defining the entire training process (see below).

We generate the path JSONs using scripts in the `scripts` directory.

## Training
To train a model, run:
```bash
$ python train.py -c config.json
```
To resume training of a model, run:
```bash
$ python train.py -r path/to/checkpoint
```
After a model is trained, it can be exported to JIT by:
```bash
$ python serialize_model.py -c path/to/config.json -r path/to/checkpoint
```

## Config file format
Each training run is defined by a configuration file. The configuration file specifies what model to use, on what data to train, what optimizer to use, etc. An example of a full configuration file is shown below. All of the configuration files which we used are available in the `configs` directory.
```JSON
{
    "name": "Fixed_ImageNet_vs_Mix_Inaturalist_Locked",
    "n_gpu": 1,
    "arch": {
        "type": "TorchvisionResnet50",
        "args": {
            "method": 2
        }
    },
    "data_loader": {
        "type": "JSONDataLoader",
        "args": {
            "data_dir": "/mnt/personal/anonymized_username/bulk_data/imagenet_vs_inaturalist_mixture/",
            "batch_size": 20,
            "shuffle": true,
            "stratified": true,
            "num_workers": 16,
            "folders": [
                0, 1, 2, 3, 4, 5, 6, 7
            ],
            "transform_path": "data_loader/albumentations/training.json",
            "load_images_to_memory": false
        }
    },
    "valid_data_loader": {
        "type": "JSONDataLoader",
        "args": {
            "data_dir": "/mnt/personal/anonymized_username/bulk_data/imagenet_vs_inaturalist_mixture/",
            "batch_size": 20,
            "shuffle": false,
            "stratified": false,
            "num_workers": 16,
            "folders": [
                8, 9
            ],
            "transform_path": "data_loader/albumentations/validation.json",
            "load_images_to_memory": false
        }
    },
    "optimizer": {
        "type": "Adam",
        "args": {
            "lr": 0.003,
            "weight_decay": 0,
            "amsgrad": true
        }
    },
    "loss": "bce_loss",
    "metrics": [
        "accuracy"
    ],
    "lr_scheduler": {
        "type": "StepLR",
        "args": {
            "step_size": 50,
            "gamma": 0.1
        }
    },
    "trainer": {
        "epochs": 200,
        "len_epoch": 1000,
        "save_dir": "saved/",
        "save_period": 1000,
        "verbosity": 2,
        "monitor": "min val_loss",
        "early_stop": 50,
        "tensorboard": false
    }
}
```

# Notes
when training the likelihood ratio estimate $\hat{g}(x)$ for the ID CIFAR-10/100 datasets, we initialize the model to the appropriate ResNet-18 (different for each of the 3 folds). 