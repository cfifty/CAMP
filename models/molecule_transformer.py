from enum import Enum
import math
import sys
from einops import rearrange
from functools import partial
from typing import Any, Callable, Optional, Text

import torch
import torch.nn as nn
from torchvision.models._api import WeightsEnum

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from models.blocks import Encoder, ContextEncoder, MPNNFeatureExtractor, PositionalEncoding, RebuttalEncoder, \
  OrigEncoder


class MoleculeTransformer(nn.Module):
  """Molecular Transformer abstract class."""

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

    self.gnn_extractor = MPNNFeatureExtractor(atom_dim, hidden_dim, num_heads).to(device)

    self.encoder = ContextEncoder(
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


def compute_ETF(K, device):
  sub = (torch.eye(K) - 1 / K * torch.ones((K, K))).to(device) / pow(K - 1, 0.5)
  return sub


def replicate_batch(x, y):
  """Replicate the batch.

  Args:
    x: torch.Tensor of shape [B, N, C]
    y: torch.Tensor of shape [B, N, 1]
  Returns:
    (x,y,gather_idx) where x has shape [B*N, N, C].

  """
  B, N, C = x.shape

  # Repeat x so that batch_dim is b*n (n=seq length).
  x = x.repeat_interleave(N, dim=0)
  y = y.repeat_interleave(N, dim=0)
  # TODO(cfifty): fix this memory leak!
  gather_idx = torch.arange(N, device=x.device).repeat(B)

  return x, y, gather_idx


class ContextTransformer_v1(MoleculeTransformer):
  """*****Why this will not work well:*****

  The self-attention weights that perform a linear combination of the loss values are not functions of the loss
  values themselves. We do:
                          l_0 = w_0l_0 + w_1l_1 + ... + w_nl_n
  however, w_i is not a function of l_i, l_{i-1}, etc.

  Therefore, if the losses: l_0, l_1, ... change, but the molecular context remains constant, the weights will be the
  same.
  """
  """Context Modeling Molecular Transformer v1.

  This version only looks at the molecular embedding to determine attn_weights that then form a linear combination
  of each label.
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
          num_classes: int = 1000,
          norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
  ):
    super().__init__(atom_dim, num_layers, num_heads, hidden_dim, mlp_dim, device, dropout, attention_dropout,
                     norm_layer)
    self.encoder = ContextEncoder(
      num_layers,
      num_heads,
      hidden_dim,
      mlp_dim,
      dropout,
      attention_dropout,
      norm_layer,
    )

  def forward(self, x, y, context_length, *args, **kwargs):
    # Step 1: Extract molecular graph features with a GNN.
    x = self.gnn_extractor(x)
    B, C = x.shape

    # Step 2: Map the first extracted features to a "guess" label.
    x = x.reshape(B // context_length, context_length, C)
    y = y.reshape(B // context_length, context_length, 1)

    # Set the "masked" label to be the mean of all other labels in the context.
    y[:, 0, :] = torch.mean(y[:, 1:, :], dim=1)

    # Step 3: Pass inputs and labels through the context model.
    x, y = self.encoder(x, y, **kwargs)

    # Step 4: Extract refined "guess" label.
    y = y[:, 0]
    return y


class ContextTransformer_Orig(MoleculeTransformer):
  """Context Modeling Molecular Transformer v2.

  This version incorporates the label information into the molecular embedding itself as the first position to consider
  both the molecular embedding as well as the label information when generating attention weights.
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
    # Set gnn_extractor to have output_dim = hidden_dim -1 as we later concatenate this rep with the labels.
    self.gnn_extractor = MPNNFeatureExtractor(atom_dim, 384, num_heads).to(device)
    self.encoder = OrigEncoder(
      num_layers,
      num_heads,
      hidden_dim,
      mlp_dim,
      dropout,
      attention_dropout,
      norm_layer,
    )
    # self.pos_encoding_layer = PositionalEncoding(hidden_dim, dropout=dropout, max_len=256)
    self.class_emb = torch.nn.Linear(in_features=1, out_features=384, bias=False)
    self.label_emb = torch.nn.Parameter(torch.zeros(1, 1, 384))
    self.output_proj = torch.nn.Linear(in_features=hidden_dim, out_features=1, bias=False)

  def forward(self, x, y, context_length, *args, **kwargs):
    # Step 1: Extract molecular graph features with a GNN.
    x = self.gnn_extractor(x)

    # Combine features across modalities.
    y = self.class_emb(torch.unsqueeze(y, 1))

    # Step 2: Map the first extracted features to a "guess" label.
    B, _ = x.shape
    x = x.reshape(B // context_length, context_length, -1)
    y = y.reshape(B // context_length, context_length, -1)

    # Repeat each example |context| so that we predict on each molecule in the context.
    x, y, gather_idx = replicate_batch(x, y)

    # Add positional Embeddings.
    query_mask = torch.eye(context_length, device=x.device, dtype=torch.bool).repeat(B // context_length, 1).unsqueeze(
      -1)
    y += self.label_emb * query_mask - y * query_mask  # Replace the true label w/ the masked token.

    # Concatenate molecule embeddings and label embeddings along last axis.
    x = torch.cat((x, y), dim=-1)
    # x = self.pos_encoding_layer(x)  # TODO(cfifty): removing this is v2_no_pos_emb

    # Step 3: Pass inputs and labels through the context model.
    x = self.encoder(x, None, **kwargs)

    # Step 4: Extract refined "guess" label.
    # y = x[:, :, C:] # TODO(cfifty): this is simply v2.
    y = x  # TODO(cfifty): v2_full_dim.
    y = torch.take_along_dim(y, gather_idx.reshape(-1, 1, 1), 1).squeeze()

    # Step 7: Linear projection to determine the label.
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


class ProtoICL(ContextTransformer_Orig):
  def __init__(
          self,
          name: Text,
          atom_dim: int,
          num_layers: int,
          num_heads: int,
          hidden_dim: int,
          mlp_dim: int,
          device: torch.device,
          dropout: float = 0.0,
          attention_dropout: float = 0.0,
          metric: Text = '',
          norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
  ):
    super().__init__(atom_dim, num_layers, num_heads, hidden_dim, mlp_dim, device, dropout, attention_dropout,
                     norm_layer)
    self.name = name
    # bias & scale of cosine classifier
    self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0), requires_grad=True)
    self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10), requires_grad=True)
    # self.metric = 'mahalanobis'
    self.metric = metric

  def forward(self, x, y, context_length, *args, **kwargs):
    # Step 1: get features and labels.
    x = self.gnn_extractor(x)
    init_labels = y
    labels = torch.unsqueeze(y, 1)
    y = self.class_emb(torch.unsqueeze(y, 1))

    # Step 2: Map the first extracted features to a "guess" label.
    B, _ = x.shape
    x = x.reshape(B // context_length, context_length, -1)
    y = y.reshape(B // context_length, context_length, -1)
    labels = labels.reshape(B // context_length, context_length, -1).repeat_interleave(context_length, dim=0)

    # Repeat each example |context| so that we predict on each molecule in the context.
    x, y, gather_idx = replicate_batch(x, y)

    # Add positional Embeddings.
    query_mask = torch.eye(context_length, device=x.device, dtype=torch.bool).repeat(B // context_length, 1).unsqueeze(
      -1)
    y += self.label_emb * query_mask - y * query_mask  # Replace the true label w/ the masked token.

    # Concatenate molecule embeddings and label embeddings along last axis.
    x = torch.cat((x, y), dim=-1)

    # Step 3: Pass inputs and labels through the context model.
    x = self.encoder(x, None, **kwargs)

    # Step 4: Extract query and aggregate support into centroids.
    query = torch.take_along_dim(x, gather_idx.reshape(-1, 1, 1), 1)
    labels[query_mask] = -1  # Set queries to -1 to not get pulled into centroids.
    pos_centroids_mask = torch.where(labels == 1.0, 1.0, 0.0)
    neg_centroids_mask = torch.where(labels == 0.0, 1.0, 0.0)

    num_pos = (torch.sum(pos_centroids_mask, dim=1)).unsqueeze(-1)
    num_neg = (torch.sum(neg_centroids_mask, dim=1)).unsqueeze(-1)

    # Will get nans for batches with support belonging to a single class -- nan_to_num 0s out these b/c autodiff.
    # Handle illegitimate batches.
    legit_batches = torch.where(torch.logical_and(num_pos > 1, num_neg > 1), True, False).squeeze()
    num_pos = num_pos[legit_batches]
    num_neg = num_neg[legit_batches]
    pos_centroids_mask = pos_centroids_mask[legit_batches]
    neg_centroids_mask = neg_centroids_mask[legit_batches]
    query = query[legit_batches]
    x = x[legit_batches]
    labels = labels[legit_batches]

    # TODO(cfifty): check that all batches are legit? What if not a single one is...
    # Immediately return if no legit batches.
    if x.shape[0] == 0:
      return x, labels

    pos_centroids = torch.sum(x * pos_centroids_mask, dim=1) / torch.sum(pos_centroids_mask, dim=1)
    neg_centroids = torch.sum(x * neg_centroids_mask, dim=1) / torch.sum(neg_centroids_mask, dim=1)

    centroids = torch.stack((neg_centroids, pos_centroids), dim=1)

    if self.metric == 'cosine':
      query = torch.nn.functional.normalize(query, p=2, dim=query.dim() - 1, eps=1e-12)
      centroids = torch.nn.functional.normalize(centroids, p=2, dim=centroids.dim() - 1, eps=1e-12)
      scores = (query @ centroids.transpose(1, 2)).squeeze()
      scores = self.scale_cls * (scores + self.bias)
    elif self.metric == 'euclidean':
      scores = -1 * (query - centroids).pow(2).sum(dim=2)
    elif self.metric == 'mahalanobis':
      # Computing covariance is a bit more difficult; we'll have to 0-out entries in each centroid.
      e_neg = neg_centroids.unsqueeze(1)  # Expected value of the negative centroids.
      e_pos = pos_centroids.unsqueeze(1)  # Expected value o the positive centroids.

      # TODO(fifty): Begin weird black magic?
      B, S, D = x.shape
      support_mask = torch.where(labels != -1, 1.0, 0.0)
      e_support = torch.sum(x * support_mask, dim=1, keepdim=True) / (S - 1)
      full_cov = ((x * support_mask - e_support) * support_mask).transpose(1, 2) @ (
        ((x * support_mask - e_support) * support_mask)) / (S - 1)

      stable_term = 0.1 * torch.eye(D, device=x.device).unsqueeze(0)

      neg_lambda_k_tau = torch.minimum(num_neg / (num_neg + 1), torch.Tensor([0.1]).to(x.device))
      pos_lambda_k_tau = torch.minimum(num_pos / (num_pos + 1), torch.Tensor([0.1]).to(x.device))
      # TODO(cfifty): End black magic?

      # Do not divide by (num -1) b/c some examples have only a single example from the other class.
      cov_neg = ((x * neg_centroids_mask - e_neg) * neg_centroids_mask).transpose(1, 2) @ (
              (x * neg_centroids_mask - e_neg) * neg_centroids_mask) / (num_neg -1)
      cov_pos = ((x * pos_centroids_mask - e_pos) * pos_centroids_mask).transpose(1, 2) @ (
              (x * pos_centroids_mask - e_pos) * pos_centroids_mask) / (num_pos -1)

      S_inv_pos = torch.inverse(pos_lambda_k_tau * cov_pos + ((1 - pos_lambda_k_tau) * full_cov) + stable_term)
      S_inv_neg = torch.inverse(neg_lambda_k_tau * cov_neg + ((1 - neg_lambda_k_tau) * full_cov) + stable_term)

      d_pos = -1 * torch.einsum('bsd,bds->bs', ((query - e_pos) @ S_inv_pos), (query - e_pos).transpose(1, 2))
      d_neg = -1 * torch.einsum('bsd,bds->bs', ((query - e_neg) @ S_inv_neg), (query - e_neg).transpose(1, 2))
      scores = torch.cat((d_neg, d_pos), dim=1)

    return scores, init_labels[legit_batches]


class ContextTransformer_v2(MoleculeTransformer):
  """Context Modeling Molecular Transformer v2.

  This version incorporates the label information into the molecular embedding itself as the first position to consider
  both the molecular embedding as well as the label information when generating attention weights.
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
    # Set gnn_extractor to have output_dim = hidden_dim -1 as we later concatenate this rep with the labels.
    self.gnn_extractor = MPNNFeatureExtractor(atom_dim, 384, num_heads).to(device)
    self.encoder = OrigEncoder(
      num_layers,
      num_heads,
      hidden_dim,
      mlp_dim,
      dropout,
      attention_dropout,
      norm_layer,
    )
    # self.pos_encoding_layer = PositionalEncoding(hidden_dim, dropout=dropout, max_len=256)
    self.class_emb = torch.nn.Linear(in_features=1, out_features=384, bias=False)
    self.label_emb = torch.nn.Parameter(torch.zeros(1, 1, 384))
    self.output_proj = torch.nn.Linear(in_features=hidden_dim, out_features=1, bias=False)
    # self.ecfp_proj = nn.Sequential(nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.7), nn.Linear(1024, 512), nn.BatchNorm1d(512),
    #                                nn.ReLU(), nn.Dropout(0.7), nn.Linear(512, 384), nn.BatchNorm1d(384))
    self.ecfp_proj = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(0.7),
                                   nn.Linear(1024, 512),
                                   nn.ReLU(), nn.Dropout(0.7), nn.Linear(512, 384))
    # self.descriptors_proj = nn.Linear(200, 64)
    # self.join_proj = nn.Sequential(nn.Linear(2560, 1024), nn.ReLU(), nn.Dropout(dropout), nn.Linear(1024, 128))

  def forward(self, x, y, context_length, *args, **kwargs):
    # Post-GNN features: 512; ECFP Features: 2048; Descriptors Features: 200; Total: 2632.
    # Step 0: Get ecfp and molecule descriptors.
    # ecfp = self.ecfp_proj(x.fingerprints.to(torch.float32))
    ecfp = x.fingerprints
    ecfp = self.ecfp_proj(ecfp.to(torch.float32))

    # Step 1: Extract molecular graph features with a GNN.
    x = self.gnn_extractor(x)

    # Combine features across modalities: normalize to have unit norm.
    x = (ecfp + x)
    # x = torch.cat([ecfp, x], dim=1)
    # x = self.join_proj(x)

    y = self.class_emb(torch.unsqueeze(y, 1))

    # Step 2: Map the first extracted features to a "guess" label.
    B, _ = x.shape
    x = x.reshape(B // context_length, context_length, -1)
    y = y.reshape(B // context_length, context_length, -1)

    # Repeat each example |context| so that we predict on each molecule in the context.
    x, y, gather_idx = replicate_batch(x, y)

    # Add positional Embeddings.
    query_mask = torch.eye(context_length, device=x.device, dtype=torch.bool).repeat(B // context_length, 1).unsqueeze(
      -1)
    y += self.label_emb * query_mask - y * query_mask  # Replace the true label w/ the masked token.

    # Concatenate molecule embeddings and label embeddings along last axis.
    x = torch.cat((x, y), dim=-1)
    # x = self.pos_encoding_layer(x)  # TODO(cfifty): removing this is v2_no_pos_emb

    # Step 3: Pass inputs and labels through the context model.
    x = self.encoder(x, None, **kwargs)

    # Step 4: Extract refined "guess" label.
    # y = x[:, :, C:] # TODO(cfifty): this is simply v2.
    y = x  # TODO(cfifty): v2_full_dim.
    y = torch.take_along_dim(y, gather_idx.reshape(-1, 1, 1), 1).squeeze()

    # Step 7: Linear projection to determine the label.
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
    train_examples[:, 0, :] = test_examples

    x = train_examples
    y = train_labels
    y[:, 0, :] = self.label_emb
    x = torch.cat((x, y), dim=-1)

    # Step 3: Pass inputs and labels through the context model.
    x = self.encoder(x, None)
    y = x[:, 0, :]

    # Step 7: Linear projection to determine the label.
    return self.output_proj(y)


class ContextTransformer_v3(MoleculeTransformer):
  """Context Modeling Molecular Transformer v3.

  This version incorporates the label information into the molecular embedding itself as the first position to consider
  both the molecular embedding as well as the label information when generating attention weights.
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
    # Set gnn_extractor to have output_dim = hidden_dim -1 as we later concatenate this rep with the labels.
    self.gnn_extractor = MPNNFeatureExtractor(atom_dim, 384, num_heads).to(device)
    self.encoder = OrigEncoder(
      num_layers,
      num_heads,
      hidden_dim,
      mlp_dim,
      dropout,
      attention_dropout,
      norm_layer,
    )
    # self.pos_encoding_layer = PositionalEncoding(hidden_dim, dropout=dropout, max_len=256)
    self.class_emb = torch.nn.Linear(in_features=1, out_features=256, bias=False)
    self.label_emb = torch.nn.Parameter(torch.zeros(1, 1, 256))
    self.output_proj = torch.nn.Linear(in_features=hidden_dim, out_features=1, bias=False)
    self.ecfp_proj = nn.Sequential(nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.7),
                                   nn.Linear(1024, 512), nn.BatchNorm1d(512),
                                   nn.ReLU(), nn.Dropout(0.7), nn.Linear(512, 128), nn.BatchNorm1d(128))
    # self.descriptors_proj = nn.Linear(200, 64)
    # self.join_proj = nn.Sequential(nn.Linear(2560, 1024), nn.ReLU(), nn.Dropout(dropout), nn.Linear(1024, 128))

  def forward(self, x, y, context_length, *args, **kwargs):
    # Post-GNN features: 512; ECFP Features: 2048; Descriptors Features: 200; Total: 2632.
    # Step 0: Get ecfp and molecule descriptors.
    # ecfp = self.ecfp_proj(x.fingerprints.to(torch.float32))
    ecfp = x.fingerprints
    ecfp = self.ecfp_proj(ecfp.to(torch.float32))

    # Step 1: Extract molecular graph features with a GNN.
    x = self.gnn_extractor(x)

    # Combine features across modalities: normalize to have unit norm.
    x = torch.cat([ecfp, x], dim=1)
    y = self.class_emb(torch.unsqueeze(y, 1))

    # Step 2: Map the first extracted features to a "guess" label.
    B, _ = x.shape
    x = x.reshape(B // context_length, context_length, -1)
    y = y.reshape(B // context_length, context_length, -1)

    # Repeat each example |context| so that we predict on each molecule in the context.
    x, y, gather_idx = replicate_batch(x, y)

    # Add positional Embeddings.
    query_mask = torch.eye(context_length, device=x.device, dtype=torch.bool).repeat(B // context_length, 1).unsqueeze(
      -1)
    y += self.label_emb * query_mask - y * query_mask  # Replace the true label w/ the masked token.

    # Concatenate molecule embeddings and label embeddings along last axis.
    x = torch.cat((x, y), dim=-1)
    # x = self.pos_encoding_layer(x)  # TODO(cfifty): removing this is v2_no_pos_emb

    # Step 3: Pass inputs and labels through the context model.
    x = self.encoder(x, None, **kwargs)

    # Step 4: Extract refined "guess" label.
    # y = x[:, :, C:] # TODO(cfifty): this is simply v2.
    y = x  # TODO(cfifty): v2_full_dim.
    y = torch.take_along_dim(y, gather_idx.reshape(-1, 1, 1), 1).squeeze()

    # Step 7: Linear projection to determine the label.
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
    train_examples[:, 0, :] = test_examples

    x = train_examples
    y = train_labels
    y[:, 0, :] = self.label_emb
    x = torch.cat((x, y), dim=-1)

    # Step 3: Pass inputs and labels through the context model.
    x = self.encoder(x, None)
    y = x[:, 0, :]

    # Step 7: Linear projection to determine the label.
    return self.output_proj(y)


class RebuttalContextTransformer(ContextTransformer_v2):
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
    self.encoder = RebuttalEncoder(
      num_layers,
      num_heads,
      hidden_dim,
      mlp_dim,
      dropout,
      attention_dropout,
      norm_layer,
    )
    self.gnn_extractor = MPNNFeatureExtractor(atom_dim, 512, num_heads).to(device)

    readout_dim = 128
    self.readout = nn.Parameter(torch.empty((num_layers, hidden_dim, readout_dim)), requires_grad=True)
    torch.nn.init.kaiming_uniform_(self.readout, a=math.sqrt(5))
    self.output_proj = torch.nn.Linear(in_features=readout_dim * num_layers, out_features=1, bias=False)

    # For rebuttal model number 3 -- make hte label dim only 256.
    self.modality_MLP = nn.Sequential(nn.Linear(2560, 1024), nn.ReLU(), nn.Linear(1024, 512))
    self.class_emb = torch.nn.Linear(in_features=1, out_features=256, bias=False)
    self.label_emb = torch.nn.Parameter(torch.zeros(1, 1, 256))

  def forward(self, x, y, context_length, *args, **kwargs):
    # Post-GNN features: 512; ECFP Features: 2048; Descriptors Features: 200; Total: 2632.
    # Step 0: Get ecfp and molecule descriptors.
    ecfp = x.fingerprints

    # Step 1: Extract molecular graph features with a GNN.
    x = self.gnn_extractor(x)

    # Combine features across modalities.
    x = torch.cat([ecfp, x], dim=1)
    x = self.modality_MLP(x)
    y = self.class_emb(torch.unsqueeze(y, 1))

    # Step 2: Map the first extracted features to a "guess" label.
    B, _ = x.shape
    x = x.reshape(B // context_length, context_length, -1)
    y = y.reshape(B // context_length, context_length, -1)

    # Repeat each example |context| so that we predict on each molecule in the context.
    x, y, gather_idx = replicate_batch(x, y)

    # Add positional Embeddings.
    query_mask = torch.eye(context_length, device=x.device, dtype=torch.bool).repeat(B // context_length, 1).unsqueeze(
      -1)
    y += self.label_emb * query_mask - y * query_mask  # Replace the true label w/ the masked token.

    # Concatenate molecule embeddings and label embeddings along last axis.
    x = torch.cat((x, y), dim=-1)
    # x = self.pos_encoding_layer(x)  # TODO(cfifty): removing this is v2_no_pos_emb

    # Step 3: Pass inputs and labels through the context model.
    _, cls_tokens = self.encoder(x, None, gather_idx, **kwargs)
    query = torch.einsum('bsd,sda->bsa', cls_tokens, self.readout)
    query = rearrange(query, 'b s d -> b (s d)')

    # Step 7: Linear projection to determine the label.
    return self.output_proj(query)

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
    train_examples[:, 0, :] = test_examples

    x = train_examples
    y = train_labels
    y[:, 0, :] = self.label_emb
    x = torch.cat((x, y), dim=-1)

    # Step 3: Pass inputs and labels through the context model.
    x = self.encoder(x, None)
    y = x[:, 0, :]

    # Step 7: Linear projection to determine the label.
    return self.output_proj(y)


class RebuttalContextTransformer_v2(RebuttalContextTransformer):
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
    if hidden_dim == 1152:
      # Rebuttal_v2: make everything 384 & then concatenate together.
      self.gnn_extractor = MPNNFeatureExtractor(atom_dim, 384, num_heads).to(device)
      self.ecfp_proj = nn.Linear(2048, 384)
      self.class_emb = torch.nn.Linear(in_features=1, out_features=384, bias=False)
      self.label_emb = torch.nn.Parameter(torch.zeros(1, 1, 384))
    elif hidden_dim == 768:
      readout_dim = 128
      self.readout = nn.Parameter(torch.empty((2, hidden_dim, readout_dim)), requires_grad=True)
      torch.nn.init.kaiming_uniform_(self.readout, a=math.sqrt(5))
      self.output_proj = torch.nn.Linear(in_features=readout_dim * 2, out_features=1, bias=False)

      self.gnn_extractor = MPNNFeatureExtractor(atom_dim, 384, num_heads).to(device)
      self.ecfp_proj = nn.Linear(2048, 256)
      self.class_emb = torch.nn.Linear(in_features=1, out_features=128, bias=False)
      self.label_emb = torch.nn.Parameter(torch.zeros(1, 1, 128))
    else:
      raise Exception(f'hidden_dim {hidden_dim} is not recognized.')

  def forward(self, x, y, context_length, *args, **kwargs):
    # Post-GNN features: 512; ECFP Features: 2048; Descriptors Features: 200; Total: 2632.
    # Step 0: Get ecfp and molecule descriptors.
    ecfp = self.ecfp_proj(x.fingerprints.to(torch.float32))

    # Step 1: Extract molecular graph features with a GNN.
    x = self.gnn_extractor(x)

    # Combine features across modalities.
    x = torch.cat([ecfp, x], dim=1)
    y = self.class_emb(torch.unsqueeze(y, 1))

    # Step 2: Map the first extracted features to a "guess" label.
    B, _ = x.shape
    x = x.reshape(B // context_length, context_length, -1)
    y = y.reshape(B // context_length, context_length, -1)

    # Repeat each example |context| so that we predict on each molecule in the context.
    x, y, gather_idx = replicate_batch(x, y)

    # Add positional Embeddings.
    query_mask = torch.eye(context_length, device=x.device, dtype=torch.bool).repeat(B // context_length, 1).unsqueeze(
      -1)
    y += self.label_emb * query_mask - y * query_mask  # Replace the true label w/ the masked token.

    # Concatenate molecule embeddings and label embeddings along last axis.
    x = torch.cat((x, y), dim=-1)
    # x = self.pos_encoding_layer(x)  # TODO(cfifty): removing this is v2_no_pos_emb

    # Step 3: Pass inputs and labels through the context model.
    _, cls_tokens = self.encoder(x, None, gather_idx, **kwargs)
    query = torch.einsum('bsd,sda->bsa', cls_tokens, self.readout)
    query = rearrange(query, 'b s d -> b (s d)')

    # Step 7: Linear projection to determine the label.
    return self.output_proj(query)


def _molecule_transformer(
        atom_dim: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        weights: Optional[WeightsEnum],
        progress: bool,
        device: torch.device,
        model_type: Text = 'MoleculeTransformer',
        **kwargs: Any,
) -> MoleculeTransformer:
  # if weights is not None:
  #     _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
  #     assert weights.meta["min_size"][0] == weights.meta["min_size"][1]
  #     _ovewrite_named_param(kwargs, "image_size", weights.meta["min_size"][0])
  if model_type == 'ContextTransformer_v1':
    model = ContextTransformer_v1(
      atom_dim=atom_dim,
      num_layers=num_layers,
      num_heads=num_heads,
      hidden_dim=hidden_dim,
      mlp_dim=mlp_dim,
      device=device,
      **kwargs,
    )
  elif model_type == 'ContextTransformer_v2':
    model = ContextTransformer_v2(
      atom_dim=atom_dim,
      num_layers=num_layers,
      num_heads=num_heads,
      hidden_dim=hidden_dim,
      mlp_dim=mlp_dim,
      device=device,
      **kwargs,
    )
  elif model_type == 'Rebuttal':
    model = RebuttalContextTransformer(
      atom_dim=atom_dim,
      num_layers=num_layers,
      num_heads=num_heads,
      hidden_dim=hidden_dim,
      mlp_dim=mlp_dim,
      device=device,
      **kwargs,
    )
  elif model_type == 'ContextTransformer_orig':
    model = ContextTransformer_Orig(
      atom_dim=atom_dim,
      num_layers=num_layers,
      num_heads=num_heads,
      hidden_dim=hidden_dim,
      mlp_dim=mlp_dim,
      device=device,
      **kwargs,
    )
  elif model_type == 'ProtoICL':
    model = ProtoICL(
      name='ProtoICL',
      atom_dim=atom_dim,
      num_layers=num_layers,
      num_heads=num_heads,
      hidden_dim=hidden_dim,
      mlp_dim=mlp_dim,
      device=device,
      **kwargs,
    )
  elif model_type == 'Rebuttal_v2':
    model = RebuttalContextTransformer_v2(
      atom_dim=atom_dim,
      num_layers=num_layers,
      num_heads=num_heads,
      hidden_dim=hidden_dim,
      mlp_dim=mlp_dim,
      device=device,
      **kwargs,
    )
  elif model_type == 'ContextTransformer_v3':
    model = ContextTransformer_v3(
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


def mt_small_32(*, device=torch.device('cuda:0'), weights=None,
                progress: bool = True, model_type: Text = 'MoleculeTransformer',
                **kwargs: Any) -> MoleculeTransformer:
  return _molecule_transformer(
    atom_dim=32,
    num_layers=1,
    num_heads=1,
    hidden_dim=128,
    mlp_dim=128,
    weights=weights,
    progress=progress,
    device=device,
    model_type=model_type,
    **kwargs,
  )


def mt_medium_32(*, device=torch.device('cuda:0'), weights=None,
                 progress: bool = True, model_type: Text = 'MoleculeTransformer',
                 **kwargs: Any) -> MoleculeTransformer:
  """53705MiB in GPU"""
  return _molecule_transformer(
    atom_dim=32,
    num_layers=8,
    num_heads=8,
    hidden_dim=512,
    mlp_dim=2048,
    weights=weights,
    progress=progress,
    device=device,
    model_type=model_type,
    **kwargs,
  )


def mt_base_32(*, device=torch.device('cuda:0'), weights=None,
               progress: bool = True, model_type: Text = 'MoleculeTransformer',
               **kwargs: Any) -> MoleculeTransformer:
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


def mt_large_32(*, device=torch.device('cuda:0'), weights=None,
                progress: bool = True, model_type: Text = 'MoleculeTransformer',
                **kwargs: Any) -> MoleculeTransformer:
  """79049MiB in GPU"""
  return _molecule_transformer(
    atom_dim=32,
    num_layers=2,
    num_heads=16,
    hidden_dim=1152,
    mlp_dim=4096,
    weights=weights,
    progress=progress,
    device=device,
    model_type=model_type,
    **kwargs,
  )


def mt_huge_32(*, device=torch.device('cuda:0'), weights=None, progress: bool = True,
               model_type: Text = 'MoleculeTransformer',
               **kwargs: Any) -> MoleculeTransformer:
  """=( in GPU"""
  return _molecule_transformer(
    atom_dim=32,
    num_layers=32,
    num_heads=16,
    hidden_dim=1280,
    mlp_dim=5120,
    weights=weights,
    progress=progress,
    device=device,
    model_type=model_type,
    **kwargs,
  )