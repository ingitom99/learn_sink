"""
nets.py contains the pytorch neural network classes for the generative and predictive networks.
"""

# Imports
import torch
import torch.nn as nn

# Generative network class
class gen_net(nn.Module):
  """
  Generative network class to create a data-creating neural network for the training process.
  """
  def __init__(self, dim_prior : int, dim : int, dust_const : float, skip_const : float):
    """
    dim_prior: dimension of the prior sample (i.e. the latent space)
    dim: dimension of the data
    dust_const: constant to add to the output of the network to prevent the data from containing zeros
    skip_const: constant to control influence of the skip connection
    """
    super(gen_net, self).__init__()
    self.dim_prior = dim_prior
    self.dim = dim
    self.dust_const = dust_const
    self.skip_const = skip_const
    self.length_prior = int(self.dim_prior**.5)
    self.length = int(self.dim**.5)
    self.l1 = nn.Sequential(nn.Linear(2*dim_prior, 2*dim), nn.Sigmoid())
    self.layers = [self.l1]

  def forward(self, x):

    # Creating a reshaped copy of the input to use as a skip connection
    x_0 = x.detach().clone().reshape(2, x.size(0), self.length_prior, self.length_prior)
    transform = torchvision.transforms.Resize((self.length, self.length), antialias=True)
    x_0 = torch.cat((transform(x_0[0]).reshape(x.size(0), self.dim), transform(x_0[1]).reshape(x.size(0), self.dim)), 1)

    # Forward pass
    for layer in self.layers:
      x = layer(x)
    
    # Normalization
    x_a = x[:, :self.dim]
    x_b = x[:, self.dim:]
    x_a = x_a / torch.unsqueeze(x_a.sum(dim=1), 1)
    x_b = x_b / torch.unsqueeze(x_b.sum(dim=1), 1)

    # Adding skip connection
    x_a = x_a + self.skip_const * nn.functional.relu(x_0[:, :self.dim])
    x_b = x_b + self.skip_const * nn.functional.relu(x_0[:, self.dim:])

    # Normalization and dusting
    x_a = x_a / torch.unsqueeze(x_a.sum(dim=1), 1)
    x_b = x_b / torch.unsqueeze(x_b.sum(dim=1), 1)
    x_a = x_a + self.dust_const
    x_b = x_b + self.dust_const
    x_a = x_a / torch.unsqueeze(x_a.sum(dim=1), 1)
    x_b = x_b / torch.unsqueeze(x_b.sum(dim=1), 1)
    x = torch.cat((x_a, x_b), dim=1)

    return x

# Predctive network class
class pred_net(nn.Module):
  """
  Network class for the predicting the centered log of a Sinkhorn scaling factor for a pair of distributions.
  """
  def __init__(self, dim, width):
    """
    dim: dimension of each probability distribution
    width: multiplier for width of the hidden layers
    """
    super(pred_net, self).__init__()
    self.dim = dim
    self.width = width
    self.l1 = nn.Sequential(nn.Linear(2*dim, width*dim), nn.BatchNorm1d(width*dim), nn.ELU())
    self.l2 = nn.Sequential(nn.Linear(width*dim, width*dim), nn.BatchNorm1d(width*dim), nn.ELU())
    self.l3 = nn.Sequential(nn.Linear(width*dim, width*dim), nn.BatchNorm1d(width*dim), nn.ELU())
    self.l4 = nn.Sequential(nn.Linear(width*dim, dim))
    self.layers = [self.l1, self.l2, self.l3, self.l4]

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return x