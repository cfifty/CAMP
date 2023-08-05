import argparse
import logging
import sys
from typing import List

import torch
from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from fs_mol.data import FSMolDataset
from fs_mol.models.abstract_torch_fsmol_model import resolve_starting_model_file
from fs_mol.models.protonet import PrototypicalNetwork
from fs_mol.utils.protonet_utils import (
    PrototypicalNetworkTrainer,
    evaluate_protonet_model,
)
from fs_mol.models.gnn_multitask import GNNMultitaskConfig, GNNMultitaskModel, create_model
from fs_mol.utils.test_utils import add_eval_cli_args, set_up_test_run
from fs_mol.models.abstract_torch_fsmol_model import (
    load_model_weights,
)

logger = logging.getLogger(__name__)

"""
To test:

***with embedding model***
-- Don't forget to change model path! --

python fs_mol/protonet_test.py large_dataset_embedding_training_runs/FSMol_ProtoNet_gnn+ecfp+fc_2022-12-11_00-09-09/best_validation.pt ../fsmol_datasets/ --use_embedding True --save-dir large_dataset_embedding_training_runs --cuda 7 
"""


def parse_command_line():
    parser = argparse.ArgumentParser(
        description="Test a Prototypical Network model on molecules.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "TRAINED_MODEL",
        type=str,
        help="File to load model from (determines model architecture).",
    )

    add_eval_cli_args(parser)

    parser.add_argument(
        "--batch-size",
        type=int,
        default=320,
        help="Maximum batch size to allow when running through inference on model.",
    )
    parser.add_argument(
        "--use-fresh-param-init",
        action="store_true",
        help="Do not use trained weights, but start from a fresh, random initialisation.",
    )
    # Add custom flags.
    parser.add_argument("--use_embedding", type=bool, default=False)
    parser.add_argument("--cuda", type=int, default=5)

    args = parser.parse_args()
    return args


def test(
    model: PrototypicalNetwork,
    embedding_model,
    dataset: FSMolDataset,
    save_dir: str,
    context_sizes: List[int],
    num_samples: int,
    seed: int,
    batch_size: int,
):
    """
    Same procedure as validation for prototypical networks. Each validation task is used to
    evaluate the model more than once, dependent on number of context sizes and samples.
    """

    return evaluate_protonet_model(
        model,
        embedding_model,
        dataset,
        support_sizes=context_sizes,
        num_samples=num_samples,
        seed=seed,
        batch_size=batch_size,
        save_dir=save_dir,
    )


def main():
    args = parse_command_line()
    out_dir, dataset = set_up_test_run("ProtoNet", args, torch=True)

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_weights_file = resolve_starting_model_file(
        model_file=args.TRAINED_MODEL,
        model_cls=PrototypicalNetworkTrainer,
        out_dir=out_dir,
        use_fresh_param_init=args.use_fresh_param_init,
        device=device,
    )

    model = PrototypicalNetworkTrainer.build_from_model_file(
        model_weights_file,
        device=device,
    )
    if args.use_embedding:
        # Create an embedding model.
        # MODEL_PATH = '../binding_models/mt_uw_binding_scores_epoch_50.pkl'
        MODEL_PATH = '../binding_models/large_dataset_embedding_model_50_epochs.pkl'
        embedding_model = GNNMultitaskModel.build_from_model_file(MODEL_PATH, device=device, quiet=False)
        load_model_weights(embedding_model, MODEL_PATH, load_task_specific_weights=False)
        gfe = embedding_model.graph_feature_extractor
        embedding_model.forward = lambda x: gfe.gnn(gfe.init_node_proj(x.node_features), x.adjacency_lists)[-1]
        embedding_model.eval()
    else:
        embedding_model = lambda x: x.node_features  # pass through or something.

    # For some reason, we need this...
    model.to(device)
    test(
        model,
        embedding_model,
        dataset,
        save_dir=out_dir,
        context_sizes=args.train_sizes,
        num_samples=args.num_runs,
        seed=args.seed,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        import pdb

        _, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
