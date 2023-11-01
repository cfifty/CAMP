import argparse
import logging
import os
import sys
import wandb

import torch
from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from data import DataFold
from CAMP_train_loop import train_loop
from utils.cli_utils import add_train_cli_args, set_up_train_run
from CAMP_train_utils import make_model, get_context_data_splits, add_train_loop_arguments, create_optimizer

SMALL_NUMBER = 1e-7
logger = logging.getLogger(__name__)

"""
To train on IGNITE:
python CAMP_train.py ../ignite_dataset --batch_sizes 256 256 256 256 256 \
    --context_lengths 256 128 64 32 16 --save-dir ignite --num_epochs 100 \
    --task-list-file datasets/ignite_data.json --model_type CAMP --cuda 3 --attention_dropout 0.2 \
     --dropout 0.2 --warmup_steps 2000 --learning-rate 5e-5
     
To test training on IGNITE:
python CAMP_train.py ../small_ignite_dataset --batch_sizes 256 \
    --context_lengths 16 --save-dir ignite --num_epochs 100 \
    --task-list-file datasets/ignite_data.json --model_type CAMP --cuda 0 --attention_dropout 0.2 \
     --dropout 0.2 --warmup_steps 2000 --learning-rate 5e-5
     
python CAMP_train.py ../small_fsmol_datasets --batch_sizes 256 \
  --context_lengths 16 --save-dir ignite --num_epochs 100 \
  --task-list-file datasets/fsmol-0.1.json --model_type CAMP --cuda 0 --attention_dropout 0.2 \
   --dropout 0.2 --warmup_steps 2000 --learning-rate 5e-5

To train:
python CAMP_train.py ../fsmol_datasets --batch_sizes 256 256 256 256 256 \
    --context_lengths 256 128 64 32 16 --save-dir CAMP --num_epochs 100 \
    --task-list-file datasets/fsmol-0.1.json --model_type CAMP --cuda 6 --attention_dropout 0.2 \
     --dropout 0.2 --warmup_steps 2000 --learning-rate 5e-5
     
To test:

python CAMP_train.py ../small_fsmol_datasets --batch_sizes 256 256 256 256 \
    --context_lengths 128 64 32 16 --save-dir CAMP --num_epochs 100 \
    --task-list-file datasets/fsmol-0.1.json --model_type CAMP --cuda 0 --attention_dropout 0.2 \
     --dropout 0.2 --warmup_steps 2000 --learning-rate 5e-5

"""

def init_wandb(args):
  wandb.init(
    project='camp',
    config={
      'dataset': args.DATA_PATH,
      'batch_sizes': args.batch_sizes,
      'context_lengths': args.context_lengths,
      'model': args.model_type,
      'lr': args.learning_rate,
      'dropout': args.dropout,
      'attn_dropout': args.attention_dropout,
      'warmup_steps': args.warmup_steps
    }
  )


def main():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  add_train_cli_args(parser)
  add_train_loop_arguments(parser)
  args = parser.parse_args()
  init_wandb(args)
  save_name = (
    f'{args.model_type}__{args.learning_rate}_{args.dropout}_{args.total_steps}_{args.warmup_steps}'
    f'_{args.context_lengths}_{args.batch_sizes}_{args.attention_dropout}')
  out_dir, fsmol_dataset, aml_run = set_up_train_run(f"{save_name}_{args.model_type}", args, torch=True)
  device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else "cpu")

  model = make_model(args, model_type=args.model_type, device=device)

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
  )
  if 'fsmol' in args.DATA_PATH:
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
  else:
    loss_fn = torch.nn.MSELoss(reduction='none')

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
  wandb.finish()
  torch.save(best_model_state, os.path.join(out_dir, f"best_model.pt"))


if __name__ == "__main__":
  main()
