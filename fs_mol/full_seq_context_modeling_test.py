import argparse
import logging
import pdb
import sys
import traceback
from typing import Text, Optional

import torch
from pyprojroot import here as project_root
import os

sys.path.insert(0, str(project_root()))

from fs_mol.data.fsmol_task import FSMolTaskSample
from fs_mol.data.multitask import get_multitask_inference_batcher
from fs_mol.models.abstract_torch_fsmol_model import resolve_starting_model_file
from fs_mol.models.gnn_multitask import GNNMultitaskModel
from fs_mol.multitask_train import eval_model_by_finetuning_on_task
from fs_mol.models.abstract_torch_fsmol_model import eval_full_seq_context_model
from fs_mol.utils.metrics import BinaryEvalMetrics
from fs_mol.utils.test_utils import add_eval_cli_args, eval_model, set_up_test_run
from fs_mol.models.abstract_torch_fsmol_model import (
  load_model_weights,
)
from fs_mol.utils.logging import prefix_log_msgs, set_up_logging
from models.molecule_transformer import mt_base_32, mt_large_32, mt_huge_32, mt_small_32, mt_medium_32, _molecule_transformer

logger = logging.getLogger(__name__)

"""
To Test:



# v2 mlcm m1 for FS-Mol datasets.
python fs_mol/full_seq_context_modeling_test.py . ../small_fsmol_datasets --save-dir ../test_full_context \
 --model_type ContextTransformer_v2 --model_path '../v2_mlcm/m1/best_model.pt' --train-sizes [128] --cuda 1

"""


def parse_command_line():
  parser = argparse.ArgumentParser(
    description="Test finetuning a GNN Multitask model on tasks.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )

  parser.add_argument(
    "TRAINED_MODEL",
    type=str,
    help="File to load model from (determines model architecture).",
  )

  add_eval_cli_args(parser)

  parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    help="Number of molecules per batch.",
  )
  parser.add_argument(
    "--use-fresh-param-init",
    action="store_true",
    help="Do not use trained weights, but start from a fresh, random initialisation.",
  )
  parser.add_argument(
    "--learning-rate",
    type=float,
    default=0.00005,
    help="Learning rate for shared model components.",
  )
  parser.add_argument(
    "--task-specific-lr",
    type=float,
    default=0.0001,
    help="Learning rate for shared model components.",
  )
  parser.add_argument("--model_type", default='MoleculeTransformer')
  parser.add_argument("--model_path", default='v2_mlcm/m1/best_model.pt')
  parser.add_argument("--use_embedding", type=bool, default=False)
  parser.add_argument("--cuda", type=int, default=5)

  return parser.parse_args()


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


def main():
  args = parse_command_line()
  out_dir, dataset = set_up_test_run("Multitask", args, torch=True)

  # Recreate the outdir.
  out_dir = os.path.join(args.save_dir, f'{args.model_path.split("/")[2]}_{args.train_sizes[0]}')
  os.makedirs(out_dir, exist_ok=True)
  set_up_logging(os.path.join(out_dir, f"eval_run.log"))

  device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
  model = make_model('base', args.model_type, device=device)
  model.load_state_dict(torch.load(args.model_path))

  def test_model_fn(
          task_sample: FSMolTaskSample, temp_out_folder: str, seed: int
  ) -> BinaryEvalMetrics:
    return eval_full_seq_context_model(
      model=model,
      task_sample=task_sample,
      # Set batch_size = train_sizes: slows down inference, but makes it simplier to work with a single variable.
      context_batcher=get_multitask_inference_batcher(max_num_graphs=args.train_sizes[0], device=device),
      test_batcher=get_multitask_inference_batcher(max_num_graphs=64, device=device),
      learning_rate=args.learning_rate,
      task_specific_learning_rate=args.task_specific_lr,
      metric_to_use="avg_precision",
      seed=seed,
      quiet=True,
      device=device,
    )

  eval_model(
    test_model_fn=test_model_fn,
    dataset=dataset,
    train_set_sample_sizes=args.train_sizes,
    out_dir=out_dir,
    num_samples=args.num_runs,
    valid_size_or_ratio=0.,
    seed=args.seed,
  )


if __name__ == "__main__":
  try:
    main()
  except Exception:
    _, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)
