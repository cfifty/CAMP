from __future__ import annotations

import logging
import os
import sys
import time
from functools import partial
from typing import (
    Tuple,
    Dict,
    Optional,
    Iterable,
    Text,
    Type,
    Any,
)

import torch
from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from utils.logging import PROGRESS_LOG_LEVEL
from utils.metric_logger import MetricLogger
from models.abstract_torch_fsmol_model import AbstractTorchFSMolModel, BatchLossType, BatchOutputType, \
    BatchFeaturesType, ModelStateType

logger = logging.getLogger(__name__)


def linear_warmup(cur_step: int, warmup_steps: int = 0) -> float:
    if cur_step >= warmup_steps:
        return 1.0
    return cur_step / warmup_steps


def create_optimizer(
        model: AbstractTorchFSMolModel[BatchFeaturesType, BatchOutputType, BatchLossType],
        lr: float = 0.001,
        task_specific_lr: float = 0.005,
        warmup_steps: int = 1000,
        task_specific_warmup_steps: int = 100,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    # Split parameters into shared and task-specific ones:
    shared_parameters, task_spec_parameters = [], []
    for param_name, param in model.named_parameters():
        if model.is_param_task_specific(param_name):
            task_spec_parameters.append(param)
        else:
            shared_parameters.append(param)

    opt = torch.optim.Adam(
        [
            {"params": task_spec_parameters, "lr": task_specific_lr},
            {"params": shared_parameters, "lr": lr},
        ],
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt,
        lr_lambda=[
            partial(
                linear_warmup, warmup_steps=task_specific_warmup_steps
            ),  # for task specific paramters
            partial(linear_warmup, warmup_steps=warmup_steps),  # for shared paramters
        ],
    )

    return opt, scheduler


def save_model(
        path: str,
        model: AbstractTorchFSMolModel[BatchFeaturesType, BatchOutputType, BatchLossType],
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
) -> None:
    data = model.get_model_state()

    if optimizer is not None:
        data["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        data["epoch"] = epoch

    torch.save(data, path)


def load_model_weights(
        model: AbstractTorchFSMolModel[BatchFeaturesType, BatchOutputType, BatchLossType],
        path: str,
        load_task_specific_weights: bool,
        quiet: bool = False,
        device: Optional[torch.device] = None,
) -> None:
    checkpoint = torch.load(path, map_location=device)
    model.load_model_state(checkpoint, load_task_specific_weights, quiet)


def resolve_starting_model_file(
        model_file: str,
        model_cls: Type[AbstractTorchFSMolModel[BatchFeaturesType, BatchOutputType, BatchLossType]],
        out_dir: str,
        use_fresh_param_init: bool,
        config_overrides: Dict[str, Any] = {},
        device: Optional[torch.device] = None,
) -> str:
    # If we start from a fresh init, create a model, do a random init, and store that away somewhere:
    if use_fresh_param_init:
        logger.info("Using fresh model init.")
        model = model_cls.build_from_model_file(
            model_file=model_file, config_overrides=config_overrides, device=device
        )

        resolved_model_file = os.path.join(out_dir, f"fresh_init.pkl")
        save_model(resolved_model_file, model)

        # Hack to give AML some time to actually save.
        time.sleep(1)
    else:
        resolved_model_file = model_file
        logger.info(f"Using model weights loaded from {resolved_model_file}.")

    return resolved_model_file


def run_on_data_iterable(
        model: AbstractTorchFSMolModel[BatchFeaturesType, BatchOutputType, BatchLossType],
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
        if max_num_steps is not None and batch_idx >= max_num_steps:
            break

        if optimizer is not None:
            optimizer.zero_grad()

        predictions: BatchOutputType = model(batch)
        model_loss, label_loss = model.compute_loss(batch, predictions, labels)
        metric_logger.log_metrics(**model_loss.metrics_to_log)

        # Training step:
        if optimizer is not None:
            loss = model_loss.total_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
    return metric_logger.get_mean_metric_value("total_loss")


def compute_valid_loss(
        model: AbstractTorchFSMolModel[BatchFeaturesType, BatchOutputType, BatchLossType],
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
        with torch.no_grad():
            predictions: BatchOutputType = model(batch)
            model_loss, _ = model.compute_loss(batch, predictions, labels)
        metric_logger.log_metrics(**model_loss.metrics_to_log)
    return metric_logger.get_mean_metric_value("total_loss")


def train_loop(
        model: AbstractTorchFSMolModel[BatchFeaturesType, BatchOutputType, BatchLossType],
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        train_data: Iterable[Tuple[BatchFeaturesType, torch.Tensor]],
        valid_data: Iterable[Tuple[BatchFeaturesType, torch.Tensor]],
        max_num_epochs: int = 101,
        patience: int = 5,
        aml_run=None,
        quiet: bool = False,
        out_dir: Text = '',
) -> Tuple[float, ModelStateType]:
    if quiet:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    # Set up early stopping.
    initial_valid_loss = float("inf")
    best_valid_loss = initial_valid_loss
    logger.log(log_level, f"  Initial validation metric: {best_valid_loss:.5f}")
    best_model_state = model.state_dict()
    epochs_since_best = 0

    valid_losses = []
    train_losses = []

    for epoch in range(0, max_num_epochs):
        logger.log(log_level, f"== Epoch {epoch}")
        logger.log(log_level, f"  = Training")
        train_loss = run_on_data_iterable(
            model,
            data_iterable=train_data,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            quiet=quiet,
            metric_name_prefix="train_",
            aml_run=aml_run,
        )

        train_losses.append(train_loss)
        logger.log(log_level, f"  Mean train loss: {train_loss:.5f}")

        logger.log(log_level, f"  = Validation")
        valid_loss = compute_valid_loss(model, valid_data)
        logger.log(log_level, f"  Mean valid loss: {valid_loss:.5f}")
        valid_losses.append(valid_loss)
        # Consider early stopping.
        if valid_loss < best_valid_loss:
            logger.log(
                log_level,
                f"   New best validation result {valid_loss:.5f} (increased from {best_valid_loss:.5f}).",
            )
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
