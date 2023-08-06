import math
import sys
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.vision_transformer import MLPBlock

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from models.attention_mechanisms import MATAttention, TDAttention, ContextAttention
from modules.gnn import GNNConfig, GNN
from modules.graph_readout import GraphReadoutConfig
from modules.vit_utils import DropPath
from modules.graph_feature_extractor import GraphFeatureExtractorConfig, GraphFeatureExtractor

"""
args from MPNN:
Arguments: finetune_lr_scale=1.0, gnn_type='PNA', intermediate_dim=1024, learning_rate=5e-05, message_function_depth=1, metric_to_use='avg_precision', node_embed_dim=128, num_epochs=100, num_gnn_layers=10, num_heads=4, num_tail_layers=2, patience=10, per_head_dim=64, readout_head_dim=64, readout_num_heads=12, readout_output_dim=512, readout_type='combined', readout_use_all_states=True, save_dir='multitask_training_runs', seed=0, task_list_file='datasets/fsmol-0.1.json', task_specific_lr=0.0001)
"""


class PositionalEncoding(nn.Module):
  """Standard Transformer positional encoding (sin + cos)"""

  def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)

    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, 1, d_model)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)

  def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
    """
    Input: [batch_size, seq_length, hidden_dim]

    Pe: [seq_len, batch_size, hidden_dim].

    Output: [batch_size, seq_length, hidden_dim]
    """
    x = torch.transpose(x, 0, 1)
    x = x + self.pe[:x.size(0)]
    return torch.transpose(self.dropout(x), 0, 1)


class MPNNFeatureExtractor(nn.Module):
  """Encode the local topology around an atom with an MPNN."""

  def __init__(self, atom_dim, hidden_dim, num_heads):
    super().__init__()
    gnn_config = GNNConfig(type='PNA', num_edge_types=3, hidden_dim=128, num_heads=4,
                           per_head_dim=64, intermediate_dim=1024, message_function_depth=1,
                           num_layers=10)
    gnn_readout = GraphReadoutConfig(readout_type='combined', use_all_states=True, num_heads=12, head_dim=64,
                                     output_dim=hidden_dim)
    gfe_config = GraphFeatureExtractorConfig(initial_node_feature_dim=atom_dim, gnn_config=gnn_config,
                                             readout_config=gnn_readout, output_norm='off')
    self.gfe = GraphFeatureExtractor(gfe_config)

  def forward(self, x, *args, **kwargs) -> torch.Tensor:
    return self.gfe(x)


class ContextEncoder(nn.Module):
  """ContextEncoder."""

  def __init__(
          self,
          num_layers: int,
          num_heads: int,
          hidden_dim: int,
          mlp_dim: int,
          dropout: float,
          attention_dropout: float,
          norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
  ):
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    layers: OrderedDict[str, nn.Module] = OrderedDict()
    for i in range(num_layers):
      layers[f"encoder_layer_{i}"] = ContextEncoderBlock(
        num_heads,
        hidden_dim,
        mlp_dim,
        dropout,
        attention_dropout,
        norm_layer,
      )
    self.layers = nn.Sequential(layers)
    self.ln = norm_layer(hidden_dim)

  def forward(self, x, y, *args, **kwargs):
    torch._assert(x.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}")
    x = self.dropout(x)
    for layer in self.layers:
      x, y = layer.forward(x, y)
    return self.ln(x), y

class RebuttalEncoder(nn.Module):
  """I ain't fucking around anymore."""
  def __init__(
          self,
          num_layers: int,
          num_heads: int,
          hidden_dim: int,
          mlp_dim: int,
          dropout: float,
          attention_dropout: float,
          norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
  ):
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    layers: OrderedDict[str, nn.Module] = OrderedDict()
    for i in range(num_layers):
      layers[f"encoder_layer_{i}"] = EncoderBlock(
        num_heads,
        hidden_dim,
        mlp_dim,
        dropout,
        attention_dropout,
        norm_layer,
      )
    self.layers = nn.Sequential(layers)
    self.ln = norm_layer(hidden_dim)

  def forward(self, x: torch.Tensor, y, gather_idx, *args, **kwargs):
    cls_tokens = []
    x = self.dropout(x)
    for layer in self.layers:
      x = layer(x)
      # Extract the query from the sequence in each batch.
      query = torch.take_along_dim(x, gather_idx.reshape(-1, 1, 1), 1)
      cls_tokens.append(query)
    return self.ln(x), torch.cat(cls_tokens[-2:], dim=1)


class RebuttalEncoderBlock(nn.Module):
  """Really ain't fucking around."""

  def __init__(
          self,
          num_heads: int,
          hidden_dim: int,
          mlp_dim: int,
          dropout: float,
          attention_dropout: float,
          norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
  ):
    super().__init__()
    self.num_heads = num_heads

    # Attention block
    self.ln_1 = norm_layer(hidden_dim)
    self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
    self.dropout = nn.Dropout(dropout)
    self.drop_path1 = DropPath(0.1)

    # MLP block
    self.ln_2 = norm_layer(hidden_dim)
    self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)
    self.drop_path2 = DropPath(0.1)

  def forward(self, input: torch.Tensor):
    torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
    x = self.ln_1(input)
    x = input + self.drop_path1(self.dropout(self.self_attention(query=x, key=x, value=x, need_weights=False)[0]))
    x = x + self.drop_path2(self.mlp(self.ln_2(x)))
    return x


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation."""

  def __init__(
          self,
          num_layers: int,
          num_heads: int,
          hidden_dim: int,
          mlp_dim: int,
          dropout: float,
          attention_dropout: float,
          norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
  ):
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    layers: OrderedDict[str, nn.Module] = OrderedDict()
    for i in range(num_layers):
      layers[f"encoder_layer_{i}"] = EncoderBlock(
        num_heads,
        hidden_dim,
        mlp_dim,
        dropout,
        attention_dropout,
        norm_layer,
      )
    self.layers = nn.Sequential(layers)
    self.ln = norm_layer(hidden_dim)

  def forward(self, x: torch.Tensor, y: torch.Tensor = None, *args, **kwargs):
    torch._assert(x.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}")
    return self.ln(self.layers(self.dropout(x)))


class OrigEncoder(Encoder):
  def __init__(
          self,
          num_layers: int,
          num_heads: int,
          hidden_dim: int,
          mlp_dim: int,
          dropout: float,
          attention_dropout: float,
          norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
  ):
    super().__init__(num_layers, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout, norm_layer)
    layers: OrderedDict[str, nn.Module] = OrderedDict()
    for i in range(num_layers):
      layers[f"encoder_layer_{i}"] = OrigEncoderBlock(
        num_heads,
        hidden_dim,
        mlp_dim,
        dropout,
        attention_dropout,
        norm_layer,
      )
    self.layers = nn.Sequential(layers)

class ContextEncoderBlock(nn.Module):
  """Transformer encoder block."""

  def __init__(
          self,
          num_heads: int,
          hidden_dim: int,
          mlp_dim: int,
          dropout: float,
          attention_dropout: float,
          norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
  ):
    super().__init__()
    self.num_heads = num_heads

    # Attention block
    self.ln_1 = norm_layer(hidden_dim)
    self.self_attention = ContextAttention(hidden_dim, num_heads, qkv_bias=True, qk_scale=None,
                                           attn_drop=attention_dropout, proj_drop=0.)
    self.dropout = nn.Dropout(dropout)

    # MLP block
    self.ln_2 = norm_layer(hidden_dim)
    self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

  def forward(self, pre_attn_x: torch.Tensor, y: torch.Tensor):
    torch._assert(pre_attn_x.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {pre_attn_x.shape}")
    x = self.ln_1(pre_attn_x)
    x, y = self.self_attention(x, y)
    x = self.dropout(x)
    res_x = x + pre_attn_x

    x = self.ln_2(res_x)
    x = self.mlp(x)
    return res_x + x, y


class EncoderBlock(nn.Module):
  """Transformer encoder block."""

  def __init__(
          self,
          num_heads: int,
          hidden_dim: int,
          mlp_dim: int,
          dropout: float,
          attention_dropout: float,
          norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
  ):
    super().__init__()
    self.num_heads = num_heads

    # Attention block
    self.ln_1 = norm_layer(hidden_dim)
    self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
    self.dropout = nn.Dropout(dropout)

    # MLP block
    self.ln_2 = norm_layer(hidden_dim)
    self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

  def forward(self, input: torch.Tensor):
    torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
    x = self.ln_1(input)
    x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False)
    x = self.dropout(x)
    x = x + input

    y = self.ln_2(x)
    y = self.mlp(y)
    return x + y

class OrigEncoderBlock(nn.Module):
  """Transformer encoder block."""

  def __init__(
          self,
          num_heads: int,
          hidden_dim: int,
          mlp_dim: int,
          dropout: float,
          attention_dropout: float,
          norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
  ):
    super().__init__()
    self.num_heads = num_heads

    # Attention block
    self.ln_1 = norm_layer(hidden_dim)
    self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
    self.dropout = nn.Dropout(dropout)
    self.dp_1 = DropPath(0.1)

    # MLP block
    self.ln_2 = norm_layer(hidden_dim)
    self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)
    self.dp_2 = DropPath(0.1)

  def forward(self, input: torch.Tensor):
    torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
    x = self.ln_1(input)
    x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False)
    x = self.dropout(x)
    x = self.dp_1(x) + input

    y = self.ln_2(x)
    y = self.mlp(y)
    return x + self.dp_2(y)

#
# class TDEncoderBlock(EncoderBlock):
#   """Topological Distance Encoder."""
#
#   def __init__(
#           self,
#           num_heads: int,
#           hidden_dim: int,
#           mlp_dim: int,
#           dropout: float,
#           attention_dropout: float,
#           norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
#   ):
#     super().__init__(num_heads, hidden_dim, mlp_dim, dropout, attention_dropout, norm_layer)
#     # MHA uses 0. as proj_drop for the output projection after multiplication weights by values.
#     self.self_attention = TDAttention(hidden_dim, num_heads, qkv_bias=True, qk_scale=None,
#                                       attn_drop=attention_dropout, proj_drop=0.)
#
#   def forward(self, input: torch.Tensor, dist_matrix: torch.Tensor, topology_matrix: torch.Tensor):
#     torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
#     x = self.ln_1(input)
#     x = self.self_attention(x, dist_matrix, topology_matrix)
#     x = self.dropout(x)
#     x = x + input
#
#     y = self.ln_2(x)
#     y = self.mlp(y)
#     return x + y
