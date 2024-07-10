from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class LikelihoodEstimatorPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
            output = net(data)                
            conf, pred = torch.max(output, dim=1)
            smax = torch.softmax(output, dim=1)
            smax = smax + 10**-10
            conf = -(smax[:,0]/smax[:,1])

            if 'model.param_a' in [i[0] for i in net.named_parameters()]:
                if net.model.method == 1:
                    pass
                elif net.model.method == 2:                    
                    pi_m = 1/2 # We trained the models with ratio ID / unlabeled mixture, approximately 50/50
                    a = torch.abs(net.model.param_a)
                    pi_o_hat = 1 + a - (a)/(pi_m)
                    compensation = (1-pi_m)/(pi_m*pi_o_hat)
                    conf *= compensation
                else:
                    raise NotImplementedError                    
            else:
                pass
            
            return pred, conf
