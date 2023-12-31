from __future__ import annotations

import logging
import os
import sys
import wandb

from typing import (
  Tuple,
  Optional,
  Iterable,
  Text,
)

import numpy as np
import torch
from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from utils.logging import PROGRESS_LOG_LEVEL
from models.abstract_torch_fsmol_model import TorchFSMolModelLoss
from utils.metric_logger import MetricLogger
from models.abstract_torch_fsmol_model import BatchFeaturesType, ModelStateType
from CAMP_train_utils import compute_loss

logger = logging.getLogger(__name__)


def train_one_epoch(
        model,
        context_length,
        loss_fn,
        data_iterable: Iterable[Tuple[BatchFeaturesType, torch.Tensor]],
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        max_num_steps: Optional[int] = None,
        quiet: bool = False,
        metric_name_prefix: str = "",
        aml_run=None,
) -> float:
  """Run the given model on the provided data loader.

  Args:
      model: Model to run things on.
      data_iterable: Iterable that provides the data we run on; data has been batched
          by an appropriate batcher.
      optimizer: Optional optimizer. If present, the given model will be trained.
      lr_scheduler: Optional learning rate scheduler around optimizer.
      max_num_steps: Optional number of steps. If not provided, will run until end of data loader.
  """
  if optimizer is None:
    model.eval()
  else:
    model.train()
  metric_logger = MetricLogger(
    log_fn=lambda msg: logger.log(PROGRESS_LOG_LEVEL, msg),
    aml_run=aml_run,
    quiet=quiet,
    metric_name_prefix=metric_name_prefix,
  )

  for batch_idx, (batch, labels) in enumerate(iter(data_iterable)):
    if labels.shape[0] < context_length:
      logger.log(logging.INFO, f'This dataset of size {labels.shape[0]} does not have sufficient training examples for '
                               'context length of {context_length}.')
      continue

    if max_num_steps is not None and batch_idx >= max_num_steps:
      break

    if optimizer is not None:
      optimizer.zero_grad()
    labels = labels.to(torch.float32)

    # print(batch.device)
    # logger.info(logging.INFO, labels.device)

    # Compute the loss.
    preds = model.forward_train(batch, labels, context_length=context_length)
    if not preds.shape[0]: continue  # Skip if no legit batches.
    mean_loss, loss = compute_loss(loss_fn, preds, labels)

    # Save statistics.
    metric_logger_loss = TorchFSMolModelLoss(label_loss=mean_loss)
    metric_logger.log_metrics(**metric_logger_loss.metrics_to_log)

    # Update parameters.
    if optimizer is not None:
      mean_loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
    if lr_scheduler is not None:
      lr_scheduler.step()

    # Print out loss every 50 steps.
    if batch_idx % 50 == 0:
      print(f'Loss at {batch_idx} is {mean_loss.item()}')

  try:
    total_loss = metric_logger.get_mean_metric_value("total_loss")
  except:
    logger.log(logging.INFO,
               f'NOTE: there does not exist {context_length} examples in this dataset; this should only occur during'
               f'testing.')
    total_loss = 0

  return total_loss


def compute_valid_loss(
        model,
        context_length,
        loss_fn,
        data_iterable: Iterable[Tuple[BatchFeaturesType, torch.Tensor]],
) -> float:
  model.eval()
  metric_logger = MetricLogger(
    log_fn=lambda msg: logger.log(PROGRESS_LOG_LEVEL, msg),
    aml_run=None,
    quiet=True,
    metric_name_prefix="",
  )
  for batch_idx, (batch, labels) in enumerate(iter(data_iterable)):
    if labels.shape[0] < context_length:
      logger.log(logging.INFO,
                 f'This dataset does not have sufficient validation examples for context length of {context_length}')
      continue
    with torch.no_grad():
      labels = labels.to(torch.float32)

      # Compute model predictions for this batch as well as the loss.
      preds = model.forward_train(batch, labels, context_length=context_length)
      if not preds.shape[0]: continue  # Skip if no legit batches.
      mean_loss, loss = compute_loss(loss_fn, preds, labels)

    metric_logger_loss = TorchFSMolModelLoss(label_loss=mean_loss)
    metric_logger.log_metrics(**metric_logger_loss.metrics_to_log)
  try:
    total_loss = metric_logger.get_mean_metric_value("total_loss")
  except:
    logger.log(logging.INFO,
               f'NOTE: there does not exist {context_length} examples in this dataset; this should only occur during'
               f'testing.')
    total_loss = 1e6
  return total_loss


def train_loop(
        model,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        train_data,
        valid_data,
        max_num_epochs: int = 101,
        patience: int = 5,
        aml_run=None,
        quiet: bool = False,
        out_dir: Text = '',
) -> Tuple[float, ModelStateType]:
  log_level = logging.INFO

  # Set up early stopping.
  initial_valid_loss = float("inf")
  best_valid_loss = initial_valid_loss
  logger.log(log_level, f"  Initial validation metric: {best_valid_loss:.5f}")
  best_model_state = model.state_dict()
  epochs_since_best = 0

  valid_losses = []

  for epoch in range(0, max_num_epochs):
    logger.log(log_level, f"== Epoch {epoch}")
    logger.log(log_level, f"  = Training")
    context_train_losses = []
    for context_length, train_split in train_data.items():
      train_loss = train_one_epoch(
        model,
        context_length,
        loss_fn,
        data_iterable=train_split,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        quiet=quiet,
        metric_name_prefix=f"train_{context_length}",
        aml_run=aml_run,
      )
      logger.log(log_level, f"  Mean train loss for context_length {context_length}: {train_loss:.5f}")
      context_train_losses.append(train_loss)
    logger.log(log_level, f"  Total train loss averaged across contexts: {np.mean(context_train_losses):.5f}")

    logger.log(log_level, f"  = Validation")
    context_valid_losses = []
    for context_length, valid_split in valid_data.items():
      valid_loss = compute_valid_loss(model, context_length, loss_fn, valid_split)
      logger.log(log_level, f"  Mean valid loss for context_length {context_length}: {valid_loss:.5f}")
      context_valid_losses.append(valid_loss)
    valid_loss = np.mean(context_valid_losses)
    logger.log(log_level, f"  Total valid loss averaged across contexts: {valid_loss:.5f}")
    valid_losses.append(valid_loss)

    wandb.log(
      {'val_loss': valid_loss, 'train_loss': np.mean(context_train_losses), 'lr': lr_scheduler.get_last_lr()[0]},
      step=epoch)

    # Consider early stopping.
    if valid_loss < best_valid_loss:
      logger.log(log_level, f"New best validation result {valid_loss:.5f} (increased from {best_valid_loss:.5f}).")
      best_valid_loss = valid_loss
      epochs_since_best = 0
      best_model_state = model.state_dict()
    else:
      epochs_since_best += 1
      logger.log(log_level, f"   Now had {epochs_since_best} epochs since best result.")
      if epochs_since_best >= patience:
        break

    # Save every 5 epochs.
    if epoch % 5 == 0:
      torch.save(best_model_state, os.path.join(out_dir, f"best_model.pt"))

  return best_valid_loss, best_model_state
