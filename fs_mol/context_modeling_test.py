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

from context_model_pretrain import make_model

from fs_mol.data.fsmol_task import FSMolTaskSample
from fs_mol.data.multitask import get_multitask_inference_batcher
from fs_mol.models.abstract_torch_fsmol_model import resolve_starting_model_file
from fs_mol.models.gnn_multitask import GNNMultitaskModel
from fs_mol.multitask_train import eval_model_by_finetuning_on_task
from fs_mol.models.abstract_torch_fsmol_model import eval_context_model
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
**change context_length local variable in abstract_torch_fsmol_model as well as binding_data_multitask.


# v3
python fs_mol/context_modeling_test.py . ../fsmol_datasets --save-dir test --train-sizes [32] --model_type ContextTransformer_v2

# orig_DropPath r0 for FS-Mol datasets.
python fs_mol/context_modeling_test.py . ../fsmol_datasets --save-dir ../r0 \
--model_type ContextTransformer_orig --model_path '../orig_drop_path/r0/best_model.pt' --train-sizes [16]


# v2 mlcm **m0** for FS-Mol datasets.
python fs_mol/context_modeling_test.py . ../fsmol_datasets --save-dir ../v2_mlcm_eval \
--model_type ContextTransformer_v2 --model_path '../v2_mlcm/m0/best_model.pt' --train-sizes [128]

# v2 mlcm m1 for molNet few-shot datasets.
python fs_mol/context_modeling_test.py . ../molnet_fewshot --save-dir ../v2_molnet_eval \
--model_type ContextTransformer_v2 --model_path '../v2_mlcm/m1/best_model.pt' --train-sizes [64] \
--task-list-file datasets/molnet_fewshot.json

# v2 mlcm m2
python fs_mol/context_modeling_test.py . ../fsmol_datasets --save-dir ../v2_mlcm_eval \
--model_type ContextTransformer_v2 --model_path '../v2_mlcm/m2/best_model.pt' --train-sizes [16]

# v2 mlcm m3
python fs_mol/context_modeling_test.py . ../fsmol_datasets --save-dir ../v2_mlcm_eval \
--model_type ContextTransformer_v2 --model_path '../v2_mlcm/m3/best_model.pt' --train-sizes [16]


*******Rebuttal Models*********
python fs_mol/context_modeling_test.py . ../fsmol_datasets --save-dir MH128  --model_type ProtoICL \
--model_path 'mahalanobis/ProtoICL_mahalanobis_base_5e-05_0.0_400000_100_[128, 16]_[256, 256]_0.0_ProtoICL_2023-08-05_19-25-46/best_model.pt'  
"""


def parse_command_line():
  parser = argparse.ArgumentParser(
    description="Test finetuning a GNN Multitask model on tasks.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("--metric", default='None')
  parser.add_argument("--model_size", default='base')
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


# def make_model(model_size: Text, model_type: Text = 'MoleculeTransformer', device: Optional[torch.device] = None):
#   if model_size == 'small':
#     return mt_small_32(device=device, model_type=model_type)
#   if model_size == 'medium':
#     return mt_medium_32(device=device, model_type=model_type)
#   if model_size == 'base':
#     return mt_base_32(device=device, model_type=model_type)
#   elif model_size == 'large':
#     return mt_large_32(device=device, model_type=model_type)
#   elif model_size == 'huge':
#     return mt_huge_32(device=device, model_type=model_type)
#   else:
#     raise Exception(f'model size: {model_size} is not one of base, large, or huge. Not recognized.')


def main():
  args = parse_command_line()
  out_dir, dataset = set_up_test_run("Multitask", args, torch=True)

  # Recreate the outdir.
  out_dir = os.path.join(args.save_dir, f'{args.model_path.split("/")[2]}_{args.train_sizes[0]}')
  os.makedirs(out_dir, exist_ok=True)
  set_up_logging(os.path.join(out_dir, f"eval_run.log"))

  device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
  # model = make_model('base', args.model_type, device=device)
  model = make_model(args, model_size=args.model_size, model_type=args.model_type, device=device)
  # model.load_state_dict(torch.load(
  #   '/lfs/local/0/fifty/context_modeling/v3/ContextTransformer_v3_base_5e-05_0.0_0.0_100_256_ContextTransformer_v3_2023-04-20_13-51-50/best_model.pt'))
  # model.load_state_dict(torch.load(
  #   '/lfs/local/0/fifty/context_modeling/v2_full_dim/ContextTransformer_v2_base_5e-05_0.0_0.0_100_256_ContextTransformer_v2_2023-04-22_17-45-08/best_model.pt'))
  model.load_state_dict(torch.load(args.model_path))
  embedding_model = lambda x: x.node_features  # pass through or something.

  def test_model_fn(
          task_sample: FSMolTaskSample, temp_out_folder: str, seed: int
  ) -> BinaryEvalMetrics:
    return eval_context_model(
      model=model,
      embedding_model=embedding_model,
      task_sample=task_sample,
      batcher=get_multitask_inference_batcher(max_num_graphs=args.batch_size, device=device),
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
