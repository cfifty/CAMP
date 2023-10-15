import sys
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn
from torchvision.models.vision_transformer import MLPBlock

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from modules.gnn import GNNConfig
from modules.graph_readout import GraphReadoutConfig
from modules.graph_feature_extractor import GraphFeatureExtractorConfig, GraphFeatureExtractor

class MPNNFeatureExtractor(nn.Module):
  """Encode the local topology around an atom with an MPNN."""

  def __init__(self, atom_dim, hidden_dim, num_heads):
    """Hard-code parameters to match the MT-Baseline in FS-Mol"""
    super().__init__()
    gnn_config = GNNConfig(type='PNA', num_edge_types=3, hidden_dim=128, num_heads=4,
                           per_head_dim=64, intermediate_dim=1024, message_function_depth=1,
                           num_layers=10)
    gnn_readout = GraphReadoutConfig(readout_type='combined', use_all_states=True, num_heads=12, head_dim=64,
                                     output_dim=hidden_dim)
    gfe_config = GraphFeatureExtractorConfig(initial_node_feature_dim=atom_dim, gnn_config=gnn_config,
                                             readout_config=gnn_readout, output_norm='off')
    self.gfe = GraphFeatureExtractor(gfe_config)

  def forward(self, x) -> torch.Tensor:
    return self.gfe(x)

class Encoder(nn.Module):
  """Transformer encoder."""

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

  def forward(self, x: torch.Tensor):
    torch._assert(x.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}")
    return self.ln(self.layers(self.dropout(x)))

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
