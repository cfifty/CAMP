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

from data.fsmol_task import FSMolTaskSample
from data.multitask import get_multitask_inference_batcher
from models.abstract_torch_fsmol_model import eval_context_model
from utils.metrics import BinaryEvalMetrics
from utils.test_utils import add_eval_cli_args, eval_model, set_up_test_run
from utils.logging import prefix_log_msgs, set_up_logging

logger = logging.getLogger(__name__)

"""

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

def main():
  args = parse_command_line()
  out_dir, dataset = set_up_test_run("Multitask", args, torch=True)

  # Recreate the outdir.
  # out_dir = os.path.join(args.save_dir, f'{args.model_path.split("/")[2]}_{args.train_sizes[0]}')
  # os.makedirs(out_dir, exist_ok=True)

  # overwrite outdir to be the model dir: save-dir is now irrelevant.
  out_dir = '/'.join(args.model_path.split('/')[:-1])
  set_up_logging(os.path.join(out_dir, f"eval_run.log"))

  device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
  # model = make_model('base', args.model_type, device=device)
  model = make_model(args, model_size=args.model_size, model_type=args.model_type, device=device)
  # model.load_state_dict(torch.load(
  #   '/lfs/local/0/fifty/context_modeling/v3/ContextTransformer_v3_base_5e-05_0.0_0.0_100_256_ContextTransformer_v3_2023-04-20_13-51-50/best_model.pt'))
  # model.load_state_dict(torch.load(
  #   '/lfs/local/0/fifty/context_modeling/v2_full_dim/ContextTransformer_v2_base_5e-05_0.0_0.0_100_256_ContextTransformer_v2_2023-04-22_17-45-08/best_model.pt'))
  model.load_state_dict(torch.load(args.model_path, map_location=device))
  embedding_model = lambda x: x.node_features  # pass through or something.
  model.to(device)

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
