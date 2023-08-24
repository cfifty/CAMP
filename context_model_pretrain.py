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
from models.molecule_transformer import mt_base_32, mt_large_32, mt_huge_32, mt_small_32, mt_medium_32
from utils.cli_utils import add_train_cli_args, set_up_train_run

SMALL_NUMBER = 1e-7
logger = logging.getLogger(__name__)

"""
**********Testing***********
**Context Modeling on Simulation Data**
1. python context_model_pretrain.py ../xsmall_simulation_dataset --batch_size 512 --model_size base --save-dir test --task-list-file datasets/simulation_data.json --model_type ContextTransformer_v1 --cuda 4

**Context Modeling on FS-Mol**
2. python context_model_pretrain.py ../small_fsmol_datasets --batch_sizes 256 256 --context_lengths 16 32 \
   --model_size base --save-dir test --num_epochs 100 --task-list-file datasets/fsmol-0.1.json \
   --model_type ContextTransformer_v2 --cuda 5

*************Training**************
1. Context Modeling on FS-Mol
    v2: python context_model_pretrain.py ../fsmol_datasets --batch_sizes 2048 1024 512 256 128 \
    --context_lengths 16 32 64 128 256 --model_size base --save-dir v2 --num_epochs 100 \
    --task-list-file datasets/fsmol-0.1.json --model_type ContextTransformer_v2 --cuda 3 --attention_dropout 0.1
    
    b: python context_model_pretrain.py ../fsmol_datasets --batch_sizes 256 256 256 256 128 \
    --context_lengths 16 32 64 128 256 --model_size base --save-dir v2 --num_epochs 100 \
    --task-list-file datasets/fsmol-0.1.json --model_type ContextTransformer_v2 --cuda 7 --attention_dropout 0.1

**** Orig Context Model ****
python context_model_pretrain.py ../fsmol_datasets --batch_sizes 256 256 256 256 256 \
    --context_lengths 256 128 64 32 16 --model_size base --save-dir layers_ablation --num_epochs 100 \
    --task-list-file datasets/fsmol-0.1.json --model_type ContextTransformer_orig --cuda 6 --attention_dropout 0.2 \
     --dropout 0.2 --warmup_steps 2000 --learning-rate 5e-5
    
    
**** with ECFP features ****
python context_model_pretrain.py ../fsmol_datasets --batch_sizes 256 256 256 256 256 \
    --context_lengths 256 128 64 32 16 --model_size base --save-dir gnn_ecfp_feat --num_epochs 100 \
    --task-list-file datasets/fsmol-0.1.json --model_type ContextTransformer_v2 --cuda 4 --attention_dropout 0.2 \
     --dropout 0.2 --warmup_steps 2000 --learning-rate 5e-5

**** WITH REBUTTAL AUGMENTATIONS ****
python context_model_pretrain.py ../fsmol_datasets --batch_sizes 256 256 256 256 256 \
    --context_lengths 256 128 64 32 16 --model_size base --save-dir gnn_ecfp_feat --num_epochs 100 \
    --task-list-file datasets/fsmol-0.1.json --model_type Rebuttal --cuda 0 --attention_dropout 0.0 \
     --dropout 0.2 --warmup_steps 2000 --learning-rate 1e-5
     


***To Test***
**Normal Transformer TEST**

python context_model_pretrain.py ../xsmall_simulation_dataset --batch_size 8 --model_size small --task-list-file datasets/simulation_data.json --cuda 6 --save-dir test --num_epochs 4 --model_type MoleculeTransformer

To extract from log file:
cat transformer_training_runs/FSMol_base_5e-05_256_Multitask_2023-02-21_19-07-00/train.log | grep "Mean train loss: " | sed 's/.*\(.......\)/\1/' 
"""


def make_model(args, model_size: Text, model_type: Text = 'MoleculeTransformer', device: Optional[torch.device] = None):
  if model_size == 'small':
    return mt_small_32(device=device, model_type=model_type, dropout=args.dropout, attention_dropout=args.attention_dropout)
  if model_size == 'medium':
    return mt_medium_32(device=device, model_type=model_type, dropout=args.dropout, attention_dropout=args.attention_dropout)
  if model_size == 'base':
    return mt_base_32(device=device, model_type=model_type, dropout=args.dropout, attention_dropout=args.attention_dropout)
    # return mt_base_32(device=device, model_type=model_type, dropout=args.dropout, attention_dropout=args.attention_dropout, metric=args.metric)
  elif model_size == 'large':
    return mt_large_32(device=device, model_type=model_type, dropout=args.dropout, attention_dropout=args.attention_dropout)
  elif model_size == 'huge':
    return mt_huge_32(device=device, model_type=model_type, dropout=args.dropout, attention_dropout=args.attention_dropout)
  else:
    raise Exception(f'model size: {model_size} is not one of base, large, or huge. Not recognized.')


def get_context_data_splits(dataset, context_lengths, batch_sizes, task_name_to_id, data_fold, device):
  """Return |context_length| datasets """
  rtn = {}
  # batch_sizes = [256, 256, 256, 256]
  for context_length, batch_size in zip(context_lengths, batch_sizes):
    rtn[context_length] = MultitaskTaskSampleBatchIterable(dataset, context_length=context_length, data_fold=data_fold,
                                                           task_name_to_id=task_name_to_id,
                                                           max_num_graphs=batch_size, device=device)
  return rtn


def add_train_loop_arguments(parser: argparse.ArgumentParser):
  parser.add_argument("--batch_size", type=int, default=256)
  parser.add_argument("--num_epochs", type=int, default=100)
  parser.add_argument("--patience", type=int, default=10)
  parser.add_argument("--cuda", type=int, default=5)
  parser.add_argument("--total_steps", type=int, default=120000)
  parser.add_argument("--context_lengths", type=int,nargs='+', default=[16, 32, 64, 128, 256])
  parser.add_argument("--batch_sizes", type=int, nargs='+', default=[2048, 1024, 512, 256, 128])
  parser.add_argument("--model_size", default='base')
  parser.add_argument("--model_type", default='MoleculeTransformer')
  parser.add_argument("--metric", default='None')
  parser.add_argument(
    "--learning-rate",
    type=float,
    default=0.00005,
    help="Learning rate for shared model components.",
  )
  parser.add_argument(
    "--weight_decay",
    type=float,
    default=0.0,
    help="Weight decay for AdamW.",
  )
  parser.add_argument("--warmup_steps", type=int, default=100)
  parser.add_argument(
    "--dropout",
    type=float,
    default=0.0,
    help="Dropout for molecular Transformer.",
  )
  parser.add_argument(
    "--attention_dropout",
    type=float,
    default=0.0,
    help="Attention Dropout for molecular Transformer.",
  )
  parser.add_argument(
    "--metric-to-use",
    type=str,
    choices=[
      "acc",
      "balanced_acc",
      "f1",
      "prec",
      "recall",
      "roc_auc",
      "avg_precision",
      "kappa",
    ],
    default="avg_precision",
    help="Metric to evaluate on validation data.",
  )


def main():
  parser = argparse.ArgumentParser(
    description="Train a Multitask GNN model.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )


  add_train_cli_args(parser)
  add_train_loop_arguments(parser)
  args = parser.parse_args()
  save_name = (
    f'{args.model_type}_{args.metric}_{args.model_size}_{args.learning_rate}_{args.dropout}_{args.total_steps}_{args.warmup_steps}_{args.context_lengths}_{args.batch_sizes}_{args.attention_dropout}')
  out_dir, fsmol_dataset, aml_run = set_up_train_run(f"{save_name}_{args.model_type}", args, torch=True)
  device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else "cpu")

  model = make_model(args, model_size=args.model_size, model_type=args.model_type, device=device)

  logger.info(f"\tNum parameters {sum(p.numel() for p in model.parameters())}")
  logger.info(f"\tDevice: {device}")
  logger.info(f"\tModel:\n{model}")

  train_task_name_to_id = {
    name: i for i, name in enumerate(fsmol_dataset.get_task_names(data_fold=DataFold.TRAIN))
  }
  valid_task_name_to_id = {
    name: i for i, name in enumerate(fsmol_dataset.get_task_names(data_fold=DataFold.VALIDATION))
  }
  # with [128, 64, 32, 16, 8], we have 6800 steps/epoch
  # 50 epochs will be 340,000 total steps
  # 100 epochs will be 680,000 total steps.
  optimizer, lr_scheduler = create_optimizer(
    model,
    lr=args.learning_rate,
    weight_decay=args.weight_decay,
    warmup_steps=args.warmup_steps,
    total_steps=args.total_steps,
  )
  if 'fsmol' in args.DATA_PATH:
    if 'ProtoICL' in args.model_type:
      loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    else:
      loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
  else:
    loss_fn = torch.nn.MSELoss(reduction='none')

  # Validate on the held-out molecules.
  _, best_model_state = train_loop(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    lr_scheduler=lr_scheduler,
    train_data=get_context_data_splits(fsmol_dataset, args.context_lengths, args.batch_sizes, train_task_name_to_id,
                                       DataFold.TRAIN, device),
    valid_data=get_context_data_splits(fsmol_dataset, args.context_lengths, args.batch_sizes, valid_task_name_to_id,
                                       DataFold.VALIDATION, device),
    max_num_epochs=args.num_epochs,
    patience=args.patience,
    aml_run=aml_run,
    out_dir=out_dir,
  )

  torch.save(best_model_state, os.path.join(out_dir, f"best_model.pt"))


if __name__ == "__main__":
  try:
    main()
  except Exception:
    _, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)
