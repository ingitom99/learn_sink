"""
Utility functions for this project.
"""

import torch
import numpy as np
from skimage.draw import random_shapes
import matplotlib.pyplot as plt
  
def hilb_proj_loss(U, V):

    """
    Compute the mean Hilbert projective loss between pairs of vectors.

    Parameters
    ----------
    U : (n_samples, dim) torch.Tensor
        First set of vectors.
    V : (n_samples, dim) torch.Tensor
        Second set of vectors.
    
    Returns
    -------
    loss : float
        Loss value.
    """

    diff = U - V
    spectrum = torch.max(diff, dim=1)[0] - torch.min(diff, dim=1)[0]
    loss = spectrum.mean()

    return loss

def plot_XPT(X : torch.Tensor, P : torch.Tensor, T : torch.Tensor, dim : int
             ) -> None:

    """
    Plot and show a pair of probability distributions formatted as images
    followed by the corresponding target and prediction.

    Parameters
    ----------
    X : (2 * dim) torch.Tensor
        Pair of probability distributions.
    P : (dim) torch.Tensor  
        Prediction.
    T : (dim) torch.Tensor
        Target.
    dim : int
        Dimension of the probability distributions.
    """

    plt.figure()
    plt.title('Mu')
    plt.imshow(X[:784].cpu().detach().numpy().reshape(28, 28), cmap='magma')
    plt.colorbar()
    plt.show()
    plt.figure()
    plt.title('Nu')
    plt.imshow(X[784:].cpu().detach().numpy().reshape(28, 28), cmap='magma')
    plt.colorbar()
    plt.show()
    plt.figure()
    plt.title('T')
    plt.imshow(T.cpu().detach().numpy().reshape(28, 28), cmap='magma')
    plt.colorbar()
    plt.show()
    plt.figure()
    plt.title('P')
    plt.imshow(P.cpu().detach().numpy().reshape(28, 28), cmap='magma')
    plt.colorbar()
    plt.show()

    return None
    
def prior_sampler(n_samples : int, dim : int) -> torch.Tensor:

    """
    Sample from the prior distribution.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    dim : int
        Dimension of the samples.

    Returns
    -------
    samples : (n_samples, 2 * dim) torch.Tensor
        Samples from the prior distribution.
    """

    samples = torch.randn((n_samples, 2 * dim))
    
    return samples

def rando(n_samples, dim, dust_const):
  """
  Generate pairs of samples of randomly masked uniform random noise.

  Inputs:
    n_samples: int
      Number of samples
    dim: int
      Dimension of the samples
    dust_const: float
      Constant added to the samples to avoid zero values
  
  Returns:
    sample: torch tensor of shape (n_samples, 2 * dim)
      Pairs of samples
  """
  # Random probabilities for the Bernoulli masks
  bernoulli_p = torch.rand((n_samples, 1))
  bernoulli_p[bernoulli_p < 0.03] = 0.03
  multiplier = torch.randint(1, 6, (n_samples, 1))
  sample_a = torch.rand((n_samples, dim))
  mask_a = torch.bernoulli(bernoulli_p * torch.ones_like(sample_a))
  sample_a = (sample_a * mask_a)**multiplier
  sample_a /= torch.unsqueeze(sample_a.sum(dim=1), 1)
  sample_a = sample_a + dust_const
  sample_a /= torch.unsqueeze(sample_a.sum(dim=1), 1)
  sample_b = torch.rand((n_samples, dim))
  mask_b = torch.bernoulli(bernoulli_p * torch.ones_like(sample_b))
  sample_b = (sample_b * mask_b)**multiplier
  sample_b /= torch.unsqueeze(sample_b.sum(dim=1), 1)
  sample_b = sample_b + dust_const
  sample_b /= torch.unsqueeze(sample_b.sum(dim=1), 1)
  sample = torch.cat((sample_a, sample_b), dim=1)
  return sample


def random_shapes_loader(n_samples, dim, dust_const):
  """
  Generate a data set of pairs of samples of random shapes.

  Inputs:
    n_samples: int
      Number of samples
    dim: int
      Dimension of the samples
    dust_const: float
      Constant added to the samples to avoid zero values
  
  Returns:
    sample: torch tensor of shape (n_samples, 2 * dim)
      Pairs of samples
  """
  length = int(dim**.5)
  pairs = []
  for i in range(n_samples):
    image1 = random_shapes((length, length), max_shapes=8, min_shapes=2, min_size=4, max_size=12, channel_axis=None, allow_overlap=True)[0]
    image1 = image1.max() - image1
    image1 = image1 / image1.sum()
    image1 = image1 + dust_const
    image1 = image1 / image1.sum()
    image2= random_shapes((length, length), max_shapes=8, min_shapes=2, min_size=4, max_size=12, channel_axis=None, allow_overlap=True)[0]
    image2 = image2.max() - image2
    image2 = image2 + dust_const
    image2 = image2 /image2.sum()
    pair = np.concatenate((image1.flatten(), image2.flatten()))
    pairs.append(pair)
  pairs = np.array(pairs)
  sample = torch.tensor(pairs)
  return sample

def rn_rs(n_samples, dim, dust_const):
  """
  Generate a data set of pairs of samples of random shapes.

  Inputs:
    n_samples: int
      Number of samples
    dim: int
      Dimension of the samples
    dust_const: float
      Constant added to the samples to avoid zero values
  
  Returns:
    sample: torch tensor of shape (n_samples, 2 * dim)
      Pairs of samples
  """
  length = int(dim**.5)
  pairs = []
  for i in range(n_samples):
    image1 = random_shapes((length, length), max_shapes=10, min_shapes=2, min_size=2, max_size=10, channel_axis=None, allow_overlap=True, intensity_range=(200, 250))[0]
    image1 = image1.max() - image1
    image1 = image1 / image1.sum()
    image2= random_shapes((length, length), max_shapes=10, min_shapes=2, min_size=2, max_size=10, channel_axis=None, allow_overlap=True, intensity_range=(200, 250))[0]
    image2 = image2.max() - image2
    image2 = image2 / image2.sum()
    pair = np.concatenate((image1.flatten(), image2.flatten()))
    pairs.append(pair)
  pairs = np.array(pairs)
  sample_rs = torch.tensor(pairs)
  bernoulli_p = torch.rand((n_samples, 1))
  bernoulli_p[bernoulli_p < 0.03] = 0.03
  multiplier = torch.randint(1, 6, (n_samples, 1))
  sample_a = torch.rand((n_samples, dim))
  mask_a = torch.bernoulli(bernoulli_p * torch.ones_like(sample_a))
  sample_a = sample_a * mask_a
  sample_a /= torch.unsqueeze(sample_a.sum(dim=1), 1)
  sample_b = torch.rand((n_samples, dim))
  mask_b = torch.bernoulli(bernoulli_p * torch.ones_like(sample_b))
  sample_b = sample_b * mask_b
  sample_b /= torch.unsqueeze(sample_b.sum(dim=1), 1)
  sample_rn = torch.cat((sample_a, sample_b), dim=1)
  rs_rand_fact = torch.rand((n_samples, 1))
  rn_rand_fact = torch.rand((n_samples, 1))
  sample = rs_rand_fact * sample_rs  + rn_rand_fact * sample_rn
  sample_a = sample[:, :784]
  sample_b = sample[:, 784:]
  sample_a = sample_a / torch.unsqueeze(sample_a.sum(dim=1), 1)
  sample_b = sample_b / torch.unsqueeze(sample_b.sum(dim=1), 1)
  sample_a = sample_a + dust_const
  sample_b = sample_b + dust_const
  sample_a = sample_a / torch.unsqueeze(sample_a.sum(dim=1), 1)
  sample_b = sample_b / torch.unsqueeze(sample_b.sum(dim=1), 1)
  sample = torch.cat((sample_a, sample_b), dim=1)
  return sample

def test_sampler(test_set, n_samples):
  """
  Sample n_samples pairs of samples from a given dataset.

  Inputs:
    test_set: torch tensor of shape (n, dim)
      Dataset to sample from
    n_samples: int
      Number of samples
  
  Returns:
    test_sample: torch tensor of shape (n_samples, 2 * dim)
      Sample of pairs
  """
  rand_mask = torch.randint(low=0, high=len(test_set), size=(n_samples,2))
  test_sample = test_set[rand_mask]
  test_sample = torch.flatten(test_sample, start_dim=1)
  return test_sample