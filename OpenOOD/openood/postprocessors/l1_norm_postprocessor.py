from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class L1NormPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)
        conf = torch.norm(output, p=1, dim=1)
        _, pred = torch.max(output, dim=1)
        return pred, conf
