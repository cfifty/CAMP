import sys

import pytest
import torch
import torch.fx

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))
from models.attention_mechanisms import MultiheadAttention, MATAttention, TDAttention
from models.blocks import TDEncoder, TDEncoderBlock


def get_data():
    input = torch.rand(8, 20, 32)
    return input

# @pytest.mark.parametrize("dev", cpu_and_gpu())
# @pytest.mark
def test_multihead_attention():
    input = get_data()
    multi_head_attn = MultiheadAttention(embed_dim=32, num_heads=4, dropout=0.1, batch_first=True)
    print(multi_head_attn(input, input, input))
    output, _ = multi_head_attn(input, input, input, need_weights=False)
    print(output.shape)
    assert output.shape == torch.Size([8, 20, 32])
    # raise


def test_mat_attention():
    input = get_data()
    dist = torch.ones((20, 20))
    adj = torch.ones((20, 20))
    # From TimeSformer: use these same parameters...
    mat_attn = MATAttention(dim=32, num_heads=8, qkv_bias=True, qk_scale=None, with_qkv=True, attn_drop=0.0,
                            proj_drop=0.0)
    output = mat_attn(input, dist, adj)
    assert output.shape == torch.Size([8, 20, 32])

def test_TDAttention():
    input = get_data()
    dist = torch.ones((8, 20, 20))
    top = torch.ones((8, 20, 14))
    # From TimeSformer: use these same parameters...
    mat_attn = TDAttention(dim=32, num_heads=8, qkv_bias=True, qk_scale=None, with_qkv=True, attn_drop=0.0,
                            proj_drop=0.0)
    output = mat_attn(input, dist, top)
    assert output.shape == torch.Size([8, 20, 32])


# def testMATEncoderBlock():
#     input = get_data()
#     dist = torch.ones((20, 20))
#     adj = torch.ones((20, 20))
#     mat_encoder_block = MATEncoderBlock(8, 32, 32, 0., 0.)
#     output = mat_encoder_block(input, dist, adj)
#     assert output.shape == torch.Size([8, 20, 32])

def testTDEncoder():
    input = torch.rand(8, 21, 32)  # Need to set 21 here b/c we add the cls token in the molecule_transformer class -- not the encoder.
    dist = torch.ones((8, 20, 20))
    atom_topologies = torch.ones((8, 20, 13))
    mat_encoder = TDEncoder(2, 8, 32, 32, 0.0, 0.0)
    output = mat_encoder(input, atom_topologies, dist)
    assert output.shape == torch.Size([8, 21, 32])

if __name__ == "__main__":
    pytest.main([__file__])
