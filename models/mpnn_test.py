import sys

import pytest
import torch
import torch.fx

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))
from models.molecule_transformer import mt_base_32, mt_small_32


# from _utils_internal import get_relative_path
# from torchvision. import
# from torchvision.test.common_utils import *
# common_utils import cpu_and_gpu, freeze_rng_state, map_nested_tensor_object, needs_cuda, set_rng_seed
# from torchvision.models import get_model_builder


def get_data():
    input = torch.rand(8, 20, 32)
    batch_index_to_task = torch.tensor([0, 1, 2, 3, 0, 0, 1, 1], dtype=torch.int32)
    atom_topology = torch.round(torch.rand(8, 20, 13)).to(torch.int64)
    return input, batch_index_to_task, atom_topology



def test_load_from_file():
    """This is failing because the saved model is using 11 atom topologies and we've updated the model to use 13."""
    # model = mt_small_32(position_embedding='topology', num_tasks=8, device=torch.device('cpu'))
    # uninitialized_state_dict = model.state_dict()
    loaded_state_dict = torch.load('models/best_model.pt')
    # assert set(uninitialized_state_dict.keys()).intersection(set(loaded_state_dict.keys())) == set(
    #     loaded_state_dict.keys())
    print(loaded_state_dict)
    raise

    # for key in uninitialized_state_dict:
    #     assert torch.norm(uninitialized_state_dict[key]) != torch.norm(loaded_state_dict[key])

    # input, batch_index_to_task, atom_topology = get_data()
    # model.load_state_dict(loaded_state_dict)
    # output = model(input, atom_topology, batch_index_to_task)
    # assert output.shape == torch.Size([8])


if __name__ == "__main__":
    pytest.main([__file__])