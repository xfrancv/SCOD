"""
Exports model into torch jit, which can be easily transferred and run without the need of the source code.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import numpy as np
import json
import sys


def main(config, output_path):
    # build model architecture
    model = config.init_obj('arch', module_arch)

    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    model.eval()

    with torch.no_grad():
        traced_cell = torch.jit.script(model)
        
    torch.jit.save(traced_cell, output_path)
    
    print(f"Finished exporting {output_path}")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-o', '--output', default='exported_model.pkl', type=str,
                      help='output path')
    
    parsed_args = args.parse_args()
    config = ConfigParser.from_args(args)
    main(config, parsed_args.output)
