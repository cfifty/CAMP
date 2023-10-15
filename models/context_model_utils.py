import torch


def tile_features_and_labels(x, y):
  """
  Tile the features and lables along the sequence dimension.

  Example: the sequence [(x_1,y_1), (x_2, y_2), (x_3, y_3)] encodes 3 different testing paradigms.

  We can use [(x_1, y_1), (x_2, y_2), x_3] to predict y_3, [(x_2,y_2), (x_3,y_3), x_1] to predict y_1, etc.

  Args:
    x: torch.Tensor of shape [B, N, C]
    y: torch.Tensor of shape [B, N, 1]
  Returns:
    (x,y,gather_idx) where x has shape [B*N, N, C], y has shape [B*N, N, 1] and gather_idx is the NxN identity matrix
    stacked B times along the batch dimension.
  """
  B, N, C = x.shape

  # Repeat x so that batch_dim is b*n (n=seq length).
  x = x.repeat_interleave(N, dim=0)
  y = y.repeat_interleave(N, dim=0)
  gather_idx = torch.arange(N, device=x.device).repeat(B)
  return x, y, gather_idx



