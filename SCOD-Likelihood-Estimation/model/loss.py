import torch.nn.functional as F


def ce_loss(output, target):
    return F.cross_entropy(output, target)


def bce_loss(output, target):
    return F.binary_cross_entropy(output, target.float())
