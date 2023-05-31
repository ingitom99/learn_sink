"""
Auxiliary functions for testing the performance of the predictive network.
"""

# Imports
import torch
import ot 
from sinkhorn_algos import sink_vec
from utils import plot_XPT
import matplotlib.pyplot as plt
from tqdm import tqdm


# Functions
def test_pred_loss(loss_function, X, pred_net, C, dim, reg, plot=True, maxiter=5000):
  """
  Test the performance of the predictive network with respect to a given loss function.

  Inputs:
    loss_function: function
      Loss function for comparing predictions and targets
    X: torch tensor of shape (n_samples, 2*dim)
      Pairs of probability distributions
    pred_net: torch nn.Module
      Predictive network
    C: torch tensor of shape (dim, dim)
      Cost matrix
    dim: int
      Dimension of the probability distributions
    reg: float
      Regularization parameter
    plot: bool
      Whether to plot an example of the distributions and the respective prediction-target pair
    
  Returns:
    loss: float
      Average loss over the samples
  """
  P = pred_net(X)
  with torch.no_grad():
    V0 = torch.ones_like(X[:, :dim])
    V = sink_vec(X[:, :dim], X[:, dim:], C, reg, V0, maxiter)
    V = torch.log(V)
    V = V - torch.unsqueeze(V.mean(dim=1), 1).repeat(1, dim)
  T = V
  loss = loss_function(P, T)
  if (plot == True):
    plot_XPT(X, P, T)
  return loss.item()

def test_pred_dist(X, pred_net, C, reg, dim, title, plot=True):
  """
  Test the performance of the predictive network with respect to the predicted distance versus the Wasserstein distance.
  """
  emds = []
  for x in X:
    emd_mu = x[:dim] / x[:dim].sum()
    emd_nu = x[dim:] / x[dim:].sum()
    emd = ot.emd2(emd_mu, emd_nu, C)
    emds.append(emd)
  emds = torch.tensor(emds)
  K = torch.exp(C/-reg)
  V = torch.exp(pred_net(X))
  MU = X[:, :dim]
  NU = X[:, dim:]
  dists = []
  U = MU / (K @ V.T).T
  V = NU / (K.T @ U.T).T
  for u, v in zip(U, V):
    G = torch.diag(u)@K@torch.diag(v)
    dist = torch.trace(C.T@G)
    dists.append(dist)
  dists = torch.tensor(dists)
  rel_errs = torch.abs(emds - dists) / emds
  if (plot == True):
    plt.figure()
    plt.title(f'Pred Net Distance vs emd2 ({title})')
    plt.xlabel('Sample')
    plt.ylabel('Distance')
    plt.plot(emds, label='emd2')
    plt.plot(dists, label='predicted')
    plt.grid()
    plt.legend()
    plt.show()
  return rel_errs.mean().item()
  
def test_warmstart(X, C, dim, reg, pred_net, title, path):
  """
  Compare the performance of the Sinkhorn algorithm as iterations increase in approximating the Wasserstein distance with
  predicted initial scale factor (v0) versus vector of ones as initial scale factor.

  Inputs:
    X: torch tensor of shape (n_samples, 2*dim)
      Pairs of probability distributions
    C: torch tensor of shape (dim, dim)
      Cost matrix
    dim: int
      Dimension of the probability distributions
    reg: float
      Regularization parameter
    pred_net: torch nn.Module
      Predictive network
    title: str
      Title of the plot
    path: str
      Path to save the plot
    
  Returns:
    None
  """
  # Collecting the Wasserstein distances
  emds = []
  for x in X:
    emd_mu = x[:dim] / x[:dim].sum()
    emd_nu = x[dim:] / x[dim:].sum()
    emd = ot.emd2(emd_mu, emd_nu, C)
    emds.append(emd)
  emds = torch.tensor(emds)

  # Initiliazing Sinkhorn algorithm
  K = torch.exp(C/-reg)
  V_pred = torch.exp(pred_net(X))
  V_ones = torch.ones_like(pred_net(X))
  MU = X[:, :dim]
  NU = X[:, dim:]
  rel_err_means = []
  rel_err_means_ones = []

  # Looping over 1000 iterations of Sinkhorn algorithm
  for i in tqdm(range(1000)):

    # Performing a step of Sinkhorn algorithm for predicted V0
    U_pred = MU / (K @ V_pred.T).T
    V_pred = NU / (K.T @ U_pred.T).T
  
    # Calculating the Sinkhorn distances for predicted V0
    dists_pred = []
    for u, v in zip(U_pred, V_pred):
      G = torch.diag(u)@K@torch.diag(v)
      dist_pred = torch.trace(C.T@G)
      dists.append(dist_pred)
    dists_pred = torch.tensor(dists_pred)
    rel_errs = torch.abs(emds - dists_pred) / emds
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
  plt.title(f"{title}")
  plt.xlabel('# Sinkhorn Iterations')
  plt.ylabel('Relative Error on Wasserstein Distance')
  plt.grid()
  plt.yticks(torch.arange(0, 1.0001, 0.05))
  plt.axhline(y=rel_err_means[0], color='r', linestyle="dashed", label='pred net 0th rel err')
  plt.plot(rel_err_means, label="V0: pred net")
  plt.plot(rel_err_means_ones, label="V0: ones")
  plt.legend()
  plt.savefig(path)
  return None

def sink2_vs_emd2(X, C, dim, reg):
  emds = []
  for x in X:
    emd_mu = x[:dim] / x[:dim].sum()
    emd_nu = x[dim:] / x[dim:].sum()
    emd = ot.emd2(emd_mu, emd_nu, C)
    emds.append(emd)
  emds = torch.tensor(emds)
  sinks = []
  for x in X:
    sink_mu = x[:dim] / x[:dim].sum()
    sink_nu = x[dim:] / x[dim:].sum()
    sink = ot.sinkhorn2(sink_mu, sink_nu, C, reg, numItermax=3000)
    sinks.append(sink)
  sinks = torch.tensor(sinks)
  rel_errs = torch.abs(emds - sinks) / emds
  plt.figure()
  plt.title(f'sinkhorn2 vs emd2')
  plt.xlabel('Sample')
  plt.ylabel('Distance')
  plt.plot(emds, label='emd2')
  plt.plot(sinks, label='predicted')
  plt.grid()
  plt.legend()
  plt.show()
  return rel_errs.mean().item()