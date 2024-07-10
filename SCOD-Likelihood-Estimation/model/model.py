import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import torchvision
from base import BaseModel
from typing import Iterable, Tuple, List

import matplotlib.pyplot as plt
import numpy as np


class TorchvisionResnet50(BaseModel):
    def __init__(self, method,
                 freeze=True):
        super().__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        self.param_a = nn.Parameter(torch.tensor(
            torch.rand(1), requires_grad=True))
        self.register_buffer('method', torch.tensor(method))

        setattr(self.model, 'fc', nn.Sequential(nn.Dropout(
            p=0.2), nn.Linear(in_features=2048, out_features=1, bias=True)))

        if freeze:
            for child in self.model.children():
                for param in child.parameters():
                    param.requires_grad = False

            self.model.fc[1].weight.requires_grad = True
            self.param_a.requires_grad = True

    def forward(self, x):
        logit = self.model(x)
        sigmoid = torch.flatten(torch.sigmoid(logit))
        if self.method == 1:
            return sigmoid
        elif self.method == 2:
            return sigmoid/(1 + sigmoid*torch.abs(self.param_a))
        else:
            raise NotImplementedError


class Cifar10Resnet18(BaseModel):
    def __init__(self, method, fold, freeze=False):
        super().__init__()
        self.model = torch.jit.load(f'resnet18_cifar10_s{int(fold)}.pt')
        self.param_a = nn.Parameter(torch.tensor(
            torch.rand(1), requires_grad=True))
        self.register_buffer('method', torch.tensor(method))

        self.head = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(
            in_features=512, out_features=1, bias=True))

        if freeze:
            for child in self.model.children():
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        logit = self.head(self.model(x))
        sigmoid = torch.flatten(torch.sigmoid(logit))
        if self.method == 1:
            return sigmoid
        elif self.method == 2:
            return sigmoid/(1 + sigmoid*torch.abs(self.param_a))
        else:
            raise NotImplementedError


class Cifar100Resnet18(BaseModel):
    def __init__(self, method, fold, freeze=False):
        super().__init__()
        self.model = torch.jit.load(f'resnet18_cifar100_s{int(fold)}.pt')
        self.param_a = nn.Parameter(torch.tensor(
            torch.rand(1), requires_grad=True))
        self.register_buffer('method', torch.tensor(method))

        self.head = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(
            in_features=512, out_features=1, bias=True))

        if freeze:
            for child in self.model.children():
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        logit = self.head(self.model(x))
        sigmoid = torch.flatten(torch.sigmoid(logit))
        if self.method == 1:
            return sigmoid
        elif self.method == 2:
            return sigmoid/(1 + sigmoid*torch.abs(self.param_a))
        else:
            raise NotImplementedError
