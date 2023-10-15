from enum import Enum
import math
import sys
from einops import rearrange
from functools import partial
from typing import Any, Callable, Optional, Text

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from models.blocks import Encoder, MPNNFeatureExtractor
from models.context_model_utils import tile_features_and_labels


class MoleculeTransformer(nn.Module):
  """Abstract class."""

  def __init__(
          self,
          atom_dim: int,
          num_layers: int,
          num_heads: int,
          hidden_dim: int,
          mlp_dim: int,
          device: torch.device,
          dropout: float = 0.0,
          attention_dropout: float = 0.0,
          norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
  ):
    super().__init__()
    self.atom_dim = atom_dim
    self.hidden_dim = hidden_dim
    self.mlp_dim = mlp_dim
    self.attention_dropout = attention_dropout
    self.dropout = dropout
    self.norm_layer = norm_layer

    self.gnn_extractor = MPNNFeatureExtractor(atom_dim, hidden_dim // 2, num_heads).to(device)
    self.encoder = Encoder(
      num_layers,
      num_heads,
      hidden_dim,
      mlp_dim,
      dropout,
      attention_dropout,
      norm_layer,
    )

  def forward(self, x, y, context_length, *args, **kwargs):
    raise NotImplementedError("this is an abstract class.")

class CAMP(MoleculeTransformer):
  """
  Implementation of CAMP as described in todo.xyz
  """
  def __init__(
          self,
          atom_dim: int,
          num_layers: int,
          num_heads: int,
          hidden_dim: int,
          mlp_dim: int,
          device: torch.device,
          dropout: float = 0.0,
          attention_dropout: float = 0.0,
          norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
  ):
    super().__init__(atom_dim, num_layers, num_heads, hidden_dim, mlp_dim, device, dropout, attention_dropout,
                     norm_layer)
    self.class_emb = torch.nn.Linear(in_features=1, out_features=hidden_dim//2, bias=False)
    self.label_emb = torch.nn.Parameter(torch.zeros(1, 1, hidden_dim//2))
    self.output_proj = torch.nn.Linear(in_features=hidden_dim, out_features=1, bias=False)

  def forward_train(self, x, y, context_length):
    """Forward method used during training."""
    # Step 1: Embed molecule graph of connected atom-level features into a single fixed-dimensional vector.
    x = self.gnn_extractor(x)

    # Step 2: Embed the labels.
    y = self.class_emb(torch.unsqueeze(y, 1))

    # Step 3: Reformualte a batch of (molecule, label) measurements into a sequence of length context_length.
    B, _ = x.shape
    x = x.reshape(B // context_length, context_length, -1)
    y = y.reshape(B // context_length, context_length, -1)

    # Step 4: Reformulate each sequence [(x_1, y_1),...,(x_n,y_n)] into n different training examples.
    x, y, gather_idx = tile_features_and_labels(x, y)

    # Step 5: Replace the label of the x_i we predict with a masked token.
    query_mask = torch.eye(context_length, device=x.device, dtype=torch.bool).repeat(B // context_length, 1).unsqueeze(
      -1)
    y += self.label_emb * query_mask - y * query_mask

    # Step 6: Concatenate molecule embeddings and label embeddings along the hidden dimension axis.
    x = torch.cat((x, y), dim=-1)

    # Step 7: Pass the concatenated sequence through the Transformer encoder context model.
    x = self.encoder(x)

    # Step 8: Extract the element in the output sequence that corresponds to the unknown x_i in the input sequence.
    y = torch.take_along_dim(x, gather_idx.reshape(-1, 1, 1), 1).squeeze()

    # Step 9: Learnable affine transformation from model output to label prediction.
    return self.output_proj(y)


  def forward_test(self, train_examples, test_examples, train_labels, test_labels, context_length):
    # Step 1: Extract molecular graph features with a GNN.
    train_examples = self.gnn_extractor(train_examples)
    test_examples = self.gnn_extractor(test_examples)

    B, C = train_examples.shape
    train_examples = train_examples.reshape(1, context_length, C)
    train_labels = self.class_emb(torch.unsqueeze(train_labels, -1)).reshape(1, context_length, -1)

    B, _ = test_examples.shape
    train_examples = train_examples.repeat((B, 1, 1))
    train_labels = train_labels.repeat((B, 1, 1))

    # 8.4.23 improvement: Use the full sequence rather than replacing the first element.
    features = torch.cat((test_examples.reshape((B, 1, -1)), train_examples), dim=1)
    labels = torch.cat((self.label_emb.repeat(B, 1, 1), train_labels), dim=1)
    x = torch.cat((features, labels), dim=-1)

    # Step 3: Pass inputs and labels through the context model.
    x = self.encoder(x, None)
    y = x[:, 0, :]

    # Step 7: Linear projection to determine the label.
    return self.output_proj(y)


def _molecule_transformer(
        atom_dim: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        weights,
        progress: bool,
        device: torch.device,
        model_type='MoleculeTransformer',
        **kwargs,
):
  if model_type == 'CAMP':
    model = CAMP(
      atom_dim=atom_dim,
      num_layers=num_layers,
      num_heads=num_heads,
      hidden_dim=hidden_dim,
      mlp_dim=mlp_dim,
      device=device,
      **kwargs,
    )
  else:
    raise Exception(f'model type: {model_type} is not recognized.')
  # if weights:
  # model.load_state_dict(weights.get_state_dict(progress=progress))
  model.to(device)
  return model


def mt_base_32(*, device=torch.device('cuda:0'), weights=None, progress: bool = True, model_type='MoleculeTransformer',
               **kwargs):
  """53705MiB in GPU"""
  return _molecule_transformer(
    atom_dim=32,
    num_layers=12,
    num_heads=12,
    hidden_dim=768,
    mlp_dim=3072,
    weights=weights,
    progress=progress,
    device=device,
    model_type=model_type,
    **kwargs,
  )