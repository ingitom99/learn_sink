import torch
import torchvision
import numpy as np
from skimage.draw import random_shapes
import matplotlib.pyplot as plt
import ot

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

def rando(n_samples, dim, dust_const):
  bernoulli_p_a = torch.rand((n_samples, 1))
  bernoulli_p_a[bernoulli_p_a < 0.05] = 0.05
  bernoulli_p_a[bernoulli_p_a > 0.95] = 0.95
  bernoulli_p_b = torch.rand((n_samples, 1))
  bernoulli_p_b[bernoulli_p_b < 0.05] = 0.05
  bernoulli_p_b[bernoulli_p_b > 0.95] = 0.95
  multiplier_a = torch.randint(1, 4, (n_samples, 1))
  multiplier_b = torch.randint(1, 4, (n_samples, 1))
  sample_a = torch.rand((n_samples, dim))
  mask_a = torch.bernoulli(bernoulli_p_a * torch.ones_like(sample_a))
  sample_a = (sample_a * mask_a)**multiplier_a
  sample_a /= torch.unsqueeze(sample_a.sum(dim=1), 1)
  sample_a = sample_a + dust_const
  sample_a /= torch.unsqueeze(sample_a.sum(dim=1), 1)
  sample_b = torch.rand((n_samples, dim))
  mask_b = torch.bernoulli(bernoulli_p_b * torch.ones_like(sample_b))
  sample_b = (sample_b * mask_b)**multiplier_b
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

def MNIST_test_loader(MNIST, n_samples):
  rand_mask = torch.randint(low=0, high=len(MNIST), size=(n_samples,2))
  X = MNIST[rand_mask]
  X = torch.flatten(X, start_dim=1)
  return X

def get_MNIST():
  mnist_testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=None)
  mnist_testset = torch.flatten(mnist_testset.data.double().to(device), start_dim=1)
  MNIST_TEST = (mnist_testset / torch.unsqueeze(mnist_testset.sum(dim=1), 1))
  MNIST_TEST = MNIST_TEST + dust_const
  MNIST_TEST = MNIST_TEST / torch.unsqueeze(MNIST_TEST.sum(dim=1), 1)
  return MNIST_TEST

def get_OMNI():
  dataset = torchvision.datasets.Omniglot(root="./data", download=True, transform=torchvision.transforms.ToTensor())
  OMNIGLOT = torch.ones((len(dataset), 28**2))
  transformer = torchvision.transforms.Resize((28,28))
  for i in range(len(dataset)):
    img = 1 - transformer(dataset[i][0]).reshape(-1)
    OMNIGLOT[i] = img
  OMNIGLOT_TEST = (OMNIGLOT / torch.unsqueeze(OMNIGLOT.sum(dim=1), 1))
  OMNIGLOT_TEST = OMNIGLOT_TEST + dust_const
  OMNIGLOT_TEST = OMNIGLOT_TEST / torch.unsqueeze(OMNIGLOT_TEST.sum(dim=1), 1)
  return OMNIGLOT_TEST

def test_warmstart(X, C, dim, reg, pred_net, puma):
  emds = []
  for x in X:
    emd_mu = x[:dim] / x[:dim].sum()
    emd_nu = x[dim:] / x[dim:].sum()
    emd = ot.emd2(emd_mu, emd_nu, C)
    emds.append(emd)
  emds = torch.tensor(emds)
  K = torch.exp(C/-reg)
  V = torch.exp(puma(X))
  V_ones = torch.ones_like(puma(X))
  MU = X[:, :dim]
  NU = X[:, dim:]
  rel_err_means = []
  rel_err_means_ones = []
  for i in tqdm(range(400)):
    dists = []
    U = MU / (K @ V.T).T
    V = NU / (K.T @ U.T).T
    for u, v in zip(U, V):
      G = torch.diag(u)@K@torch.diag(v)
      dist = torch.trace(C.T@G)
      dists.append(dist)
    dists = torch.tensor(dists)
    rel_errs = torch.abs(emds - dists) / emds
    rel_err_means.append(rel_errs.mean().item())
    dists_ones = []
    U_ones = MU / (K @ V_ones.T).T
    V_ones = NU / (K.T @ U_ones.T).T
    for u, v in zip(U_ones, V_ones):
      G = torch.diag(u)@K@torch.diag(v)
      dist = torch.trace(C.T@G)
      dists_ones.append(dist)
    dists_ones = torch.tensor(dists_ones)
    rel_errs_ones = torch.abs(emds - dists_ones) / emds
    rel_err_means_ones.append(rel_errs_ones.mean().item())

  rel_err_means = torch.tensor(rel_err_means)
  rel_err_means_ones = torch.tensor(rel_err_means_ones)
  plt.figure()
  plt.title("Rel Err: Predicted Distance versus emd2 (Random Noise)")
  plt.xlabel('# Sink Iters')
  plt.ylabel('Rel Err')
  plt.plot(rel_err_means, label="predicted V0")
  plt.plot(rel_err_means_ones, label="ones V0")
  plt.legend()
  plt.show()
  return None