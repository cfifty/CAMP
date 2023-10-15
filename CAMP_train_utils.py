import argparse
import sys
import torch

from functools import partial
from typing import Optional, Text, Tuple

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from data.binding_data_multitask import MultitaskTaskSampleBatchIterable
from models.context_model import mt_base_32


def compute_loss(loss_fn, preds: torch.float32, labels: torch.float32) -> torch.float32:
  """Compute mse loss between the predictions and the labels."""
  try:
    loss = loss_fn(preds.squeeze(dim=-1), labels)
  except:
    loss = loss_fn(preds, labels)
  mean_loss = torch.mean(loss)
  return mean_loss, loss


def linear_warmup(cur_step: int, warmup_steps: int = 0) -> float:
  if cur_step >= warmup_steps:
    return 1.0
  return cur_step / warmup_steps


def create_optimizer(model,
                     lr: float = 0.001,
                     weight_decay: float = 0.03,
                     warmup_steps: int = 10000,
                     ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
  param_list = []
  for param_name, param in model.named_parameters():
    param_list.append(param)
  opt = torch.optim.AdamW([{'params': param_list, "lr": lr, "weight_decay": weight_decay}])

  scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer=opt,
    lr_lambda=[partial(linear_warmup, warmup_steps=warmup_steps)]  # for shared paramters],
  )
  return opt, scheduler


def make_model(args, model_type: Text = 'MoleculeTransformer', device: Optional[torch.device] = None):
  return mt_base_32(device=device, model_type=model_type, dropout=args.dropout,
                    attention_dropout=args.attention_dropout)


def get_context_data_splits(dataset, context_lengths, batch_sizes, task_name_to_id, data_fold, device):
  """Return |context_length| datasets """
  rtn = {}
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
  parser.add_argument("--context_lengths", type=int, nargs='+', default=[16, 32, 64, 128, 256])
  parser.add_argument("--batch_sizes", type=int, nargs='+', default=[2048, 1024, 512, 256, 128])
  parser.add_argument("--model_type", default='CAMP')
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
