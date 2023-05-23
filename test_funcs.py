# Imports
import torch
import ot 
from sinkhorn_algos import sink_vec
from utils import plot_XPT
import matplotlib.pyplot as plt
from tqdm import tqdm 

def test_pred_loss(loss_function, X, pred_net, C, dim, reg, plot=True, maxiter=5000):
  P = pred_net(X)
  T = torch.log(sink_vec(X[:, :dim], X[:, dim:], C, reg, maxiter, V0=None))
  T = T - torch.unsqueeze(T.mean(dim=1), 1).repeat(1, dim)
  loss = loss_function(P, T)
  if (plot == True):
    plot_XPT(X, P, T)
  return loss.item()

def test_pred_edm2(X, pred_net, C, reg, dim, title, plot=True):
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
    plt.title(f'Predicted Distance vs emd2 ({title})')
    plt.plot(emds, label='emd2')
    plt.plot(dists, label='predicted')
    plt.legend()
    plt.show()
  return rel_errs.mean().item()
  
def test_warmstart(X, C, dim, reg, pred_net, title):
  emds = []
  for x in X:
    emd_mu = x[:dim] / x[:dim].sum()
    emd_nu = x[dim:] / x[dim:].sum()
    emd = ot.emd2(emd_mu, emd_nu, C)
    emds.append(emd)
  emds = torch.tensor(emds)
  K = torch.exp(C/-reg)
  V = torch.exp(pred_net(X))
  V_ones = torch.ones_like(pred_net(X))
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
  plt.title(f"Rel Err: Predicted Distance versus emd2, {title}, reg: {reg}")
  plt.xlabel('# Sink Iters')
  plt.ylabel('Rel Err')
  plt.plot(rel_err_means, label="predicted V0")
  plt.plot(rel_err_means_ones, label="ones V0")
  plt.legend()
  plt.show()
  return None