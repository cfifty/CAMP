import argparse
import logging
import sys
import json

import torch
from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))


from fs_mol.models.abstract_torch_fsmol_model import (
    load_model_weights,
)
from fs_mol.models.gnn_multitask import GNNMultitaskConfig, GNNMultitaskModel, create_model


from fs_mol.modules.graph_feature_extractor import (
    add_graph_feature_extractor_arguments,
    make_graph_feature_extractor_config_from_args,
)
from fs_mol.utils.cli_utils import add_train_cli_args, set_up_train_run
from fs_mol.utils.protonet_utils import (
    PrototypicalNetworkTrainerConfig,
    PrototypicalNetworkTrainer,
)


logger = logging.getLogger(__name__)


"""
Training Scripts:
******with a pre-trained embedding******.

--Don't forget to change MODEL_PATH variable!--

1. To train on entire dataset:
python fs_mol/protonet_train.py ../fsmol_datasets/ --use_embedding True --save-dir embedding_training_runs

1.5 To train using the *large* embeddings:
python fs_mol/protonet_train.py ../fsmol_datasets/ --use_embedding True --save-dir large_dataset_embedding_training_runs

2. To train using the *large* embeddigns with only gnn features [removing ecfp and the unifying FC]:
python fs_mol/protonet_train.py ../fsmol_datasets/ --use_embedding True --save-dir large_dataset_embedding_training_runs --features gnn --cuda 2

2. To test...
python fs_mol/protonet_test.py embedding_training_runs/FSMol_ProtoNet_gnn+ecfp+fc_2022-11-14_23-47-08/best_validation.pt ../fsmol_datasets/ --use_embedding True --save-dir embedding_training_runs --cuda 5

****without pre-trained embedding model**** (i.e. learn its own embeddings).
1. To train on entire dataset:
python fs_mol/protonet_train.py ../fsmol_datasets/ --save-dir training_runs

1.5 To train on entire dataset with *ONLY* GNN features: 
python fs_mol/protonet_train.py ../fsmol_datasets/ --save-dir training_runs --features gnn --cuda 3 

2. To test...
python fs_mol/protonet_test.py training_runs/FSMol_ProtoNet_gnn+ecfp+fc_2022-11-14_23-46-49/best_validation.pt ../fsmol_datasets/ --save-dir training_runs --cuda 2

"""

def parse_command_line():
    parser = argparse.ArgumentParser(
        description="Train a Prototypical Network model on molecules.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_train_cli_args(parser)

    parser.add_argument(
        "--features",
        type=str,
        choices=[
            "gnn",
            "ecfp",
            "pc-descs",
            "ecfp+fc",
            "pc-descs+fc",
            "gnn+ecfp+fc",
            "gnn+ecfp+pc-descs+fc",
        ],
        default="gnn+ecfp+fc",
        help="Choice of features to use",
    )
    parser.add_argument(
        "--distance_metric",
        type=str,
        choices=["mahalanobis", "euclidean"],
        default="mahalanobis",
        help="Choice of distance to use.",
    )
    add_graph_feature_extractor_arguments(parser)

    parser.add_argument("--support_set_size", type=int, default=64, help="Size of support set")
    parser.add_argument(
        "--query_set_size",
        type=int,
        default=256,
        help="Size of target set. If -1, use everything but train examples.",
    )
    parser.add_argument(
        "--tasks_per_batch",
        type=int,
        default=16,
        help="Number of tasks to accumulate gradients for.",
    )

    parser.add_argument("--batch_size", type=int, default=256, help="Number of examples per batch.")
    parser.add_argument(
        "--num_train_steps", type=int, default=10000, help="Number of training steps."
    )
    parser.add_argument(
        "--validate_every",
        type=int,
        default=50,
        help="Number of training steps between model validations.",
    )
    parser.add_argument(
        "--validation-support-set-sizes",
        type=json.loads,
        default=[16, 128],
        help="JSON list selecting the number of datapoints sampled as support set data during evaluation through finetuning on the validation tasks.",
    )

    parser.add_argument(
        "--validation-query-set-size",
        type=int,
        default=512,
        help="Maximum number of datapoints sampled as query data during evaluation through finetuning on the validation tasks.",
    )

    parser.add_argument(
        "--validation-num-samples",
        type=int,
        default=5,
        help="Number of samples considered for each train set size for each validation task during evaluation through finetuning.",
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument(
        "--clip_value", type=float, default=1.0, help="Gradient norm clipping value"
    )
    parser.add_argument(
        "--pretrained_gnn",
        type=str,
        default=None,
        help="Path to a pretrained GNN model to use as a starting point.",
    )
    # Add custom flags.
    parser.add_argument("--use_embedding", type=bool, default=False)
    parser.add_argument("--cuda", type=int, default=5)

    args = parser.parse_args()
    return args


def make_trainer_config(args: argparse.Namespace) -> PrototypicalNetworkTrainerConfig:
    return PrototypicalNetworkTrainerConfig(
        graph_feature_extractor_config=make_graph_feature_extractor_config_from_args(args),
        used_features=args.features,
        distance_metric=args.distance_metric,
        batch_size=args.batch_size,
        tasks_per_batch=args.tasks_per_batch,
        support_set_size=args.support_set_size,
        query_set_size=args.query_set_size,
        validate_every_num_steps=args.validate_every,
        validation_support_set_sizes=tuple(args.validation_support_set_sizes),
        validation_query_set_size=args.validation_query_set_size,
        validation_num_samples=args.validation_num_samples,
        num_train_steps=args.num_train_steps,
        learning_rate=args.lr,
        clip_value=args.clip_value,
    )


def main():
    args = parse_command_line()
    config = make_trainer_config(args)

    out_dir, dataset, aml_run = set_up_train_run(
        f"ProtoNet_{config.used_features}", args, torch=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    model_trainer = PrototypicalNetworkTrainer(config=config).to(device)

    logger.info(f"\tDevice: {device}")
    logger.info(f"\tNum parameters {sum(p.numel() for p in model_trainer.parameters())}")
    logger.info(f"\tModel:\n{model_trainer}")

    if args.pretrained_gnn is not None:
        logger.info(f"Loading pretrained GNN weights from {args.pretrained_gnn}.")
        model_trainer.load_model_gnn_weights(path=args.pretrained_gnn, device=device)

    model_trainer.train_loop(embedding_model, out_dir, dataset, device, aml_run)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        import pdb

        _, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
