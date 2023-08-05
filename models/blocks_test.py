# import sys
#
# import pytest
# import torch
# import torch.fx
#
# from pyprojroot import here as project_root
#
# sys.path.insert(0, str(project_root()))
# from models.blocks import get_MPNN_pos_emb
#
#
#
#
# def get_data():
#     input = torch.rand(8, 20, 32)
#     batch_index_to_task = torch.tensor([0, 1, 2, 3, 0, 0, 1, 1], dtype=torch.int32)
#     atom_topology = torch.round(torch.rand(8, 20, 13)).to(torch.int64)
#     return input, batch_index_to_task, atom_topology
#
#
# # @pytest.mark.parametrize("dev", cpu_and_gpu())
# # @pytest.mark
# def test_base_32_topology_embeddings():
#     # Batch of molecules of dimension [batch, seq, atom_dim]
#     input, batch_index_to_task, atom_topology = get_data()
#
#     model = mt_base_32(position_embedding='topology', num_tasks=4, device=torch.device('cpu'))
#     output = model(input, atom_topology, batch_index_to_task)
#     assert output.shape == torch.Size([8])
#
#
# def test_TM_base_32_topology_embeddings():
#     # Batch of molecules of dimension [batch, seq, atom_dim]
#     input, batch_index_to_task, atom_topology = get_data()
#
#     model = mt_base_32(position_embedding='topology', num_tasks=4, device=torch.device('cpu'), model_type='TMTransformer')
#     dist_matrix = torch.ones((8, 20, 20))
#     output = model(input, atom_topology, batch_index_to_task, dist_matrix)
#     assert output.shape == torch.Size([8])
#
#
# def test_base_32_lm_embeddings():
#     # Batch of molecules of dimension [batch, seq, atom_dim]
#     input = torch.rand(8, 20, 32)
#     batch_index_to_task = torch.tensor([0, 1, 2, 3, 0, 0, 1, 1], dtype=torch.int32)
#     atom_topology = torch.round(torch.rand(8, 20, 11)).to(torch.int64)
#
#     model = mt_base_32(position_embedding='lm', num_tasks=4, device=torch.device('cpu'))
#     print(model(input, atom_topology, batch_index_to_task).shape)
#
#
#
# if __name__ == "__main__":
#     pytest.main([__file__])
