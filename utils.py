import torch
import numpy as np
from skimage.draw import random_shapes
import matplotlib.pyplot as plt

def hilb_proj_loss(u, v):
  diff = u - v
  spectrum = torch.max(diff, dim=1)[0] - torch.min(diff, dim=1)[0]
  loss = spectrum.mean()
  return loss

def plot_XPT(X, P, T):
  fig, ax = plt.subplots(2, 2)
  ax[0][0].set_title('Mu')
  ax[0][1].set_title('Nu')
  ax[1][0].set_title('P')
  ax[1][1].set_title('T')
  ax[0][0].imshow(X[0, :784].cpu().detach().numpy().reshape(28, 28), cmap='magma')
  ax[0][1].imshow(X[0, 784:].cpu().detach().numpy().reshape(28,28), cmap='magma')
  ax[1][0].imshow(P[0].cpu().detach().numpy().reshape(28, 28), cmap='magma')
  ax[1][1].imshow(T[0].cpu().detach().numpy().reshape(28,28), cmap='magma')
  plt.show()
  return None

  
def prior_sampler(n_samples, dim):
  sample = torch.randn((n_samples, 2 * dim))
  return sample

def random_noise_loader(n_samples, dim, dust_const, sig=3):
  sample_a = sig * torch.rand((n_samples, dim))
  sample_a /= torch.unsqueeze(sample_a.sum(dim=1), 1)
  sample_a = sample_a + dust_const
  sample_a /= torch.unsqueeze(sample_a.sum(dim=1), 1)
  sample_b = sig * torch.rand((n_samples, dim))
  sample_b /= torch.unsqueeze(sample_b.sum(dim=1), 1)
  sample_b = sample_b + dust_const
  sample_b /= torch.unsqueeze(sample_b.sum(dim=1), 1)
  sample = torch.cat((sample_a, sample_b), dim=1)
  return sample


def random_shapes_loader(n_samples, dim, dust_const):
  length = int(dim**.5)
  pairs = []
  for i in range(n_samples):
    image1= random_shapes((length, length), max_shapes=10, channel_axis=None)[0]
    image1 = image1 / image1.sum()
    image1 = image1 + dust_const
    image1 = image1 / image1.sum()
    image2= random_shapes((length, length), max_shapes=10, channel_axis=None)[0]
    image2 = image2 / image2.sum()
    image2 = image2 + dust_const
    image2 = image2 /image2.sum()
    pair = np.concatenate((image1.flatten(), image2.flatten()))
    pairs.append(pair)
  pairs = np.array(pairs)
  sample = torch.tensor(pairs)
  return sample

def gen_net_loader(gen_net, n_samples, dim_in):
  sample = prior_sampler(n_samples, dim_in)
  X = gen_net(sample)
  return X


def MNIST_test_loader(MNIST, n_samples):
  rand_mask = torch.randint(low=0, high=len(MNIST), size=(n_samples,2))
  X = MNIST[rand_mask]
  X = torch.flatten(X, start_dim=1)
  return X