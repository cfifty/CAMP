import argparse
import logging
import os
import pdb
import sys
import traceback
from typing import Optional, Text

import torch
from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from data import DataFold
from data.binding_data_multitask import MultitaskTaskSampleBatchIterable
from context_model_pretrain_loop import (
    train_loop,
    create_optimizer,
)
from models.molecule_transformer import mt_base_32, mt_large_32, mt_huge_32, mt_small_32, mt_medium_32, _molecule_transformer
from utils.cli_utils import add_train_cli_args, set_up_train_run


def make_model(model_size: Text, model_type: Text = 'MoleculeTransformer', device: Optional[torch.device] = None):
  if model_size == 'small':
    return mt_small_32(device=device, model_type=model_type)
  if model_size == 'medium':
    return mt_medium_32(device=device, model_type=model_type)
  if model_size == 'base':
    return mt_base_32(device=device, model_type=model_type)
  elif model_size == 'large':
    return mt_large_32(device=device, model_type=model_type)
  elif model_size == 'huge':
    return mt_huge_32(device=device, model_type=model_type)
  else:
    raise Exception(f'model size: {model_size} is not one of base, large, or huge. Not recognized.')


def load_model(path):
  model = make_model('huge', 'ContextTransformer_v2', device=torch.device('cuda:0'))
  model.load_state_dict(torch.load(path))
  model.eval()




if __name__ == '__main__':
  load_model('context_runs/ContextTransformer_v2_huge_5e-05_0.0_0.0_100_4096_ContextTransformer_v2_2023-04-13_01-21-47/best_model.pt')