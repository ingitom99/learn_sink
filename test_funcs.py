# Imports
import torch
import ot 
from sinkhorn_algos import sink_vec
from utils import plot_XPT

def test_pred_loss(loss_function, X, pred_net, C, dim, reg, plot=True, maxiter=5000):
  P = pred_net(X)
  T = torch.log(sink_vec(X[:, :dim], X[:, dim:], C, reg, maxiter, V0=None))
  T = T - torch.unsqueeze(T.mean(dim=1), 1).repeat(1, dim)
  loss = loss_function(P, T)
  if (plot == True):
    plot_XPT(X, P, T)
  return loss.item()

def test_pred_edm2(X, pred_net, C, reg, dim):
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
  return rel_errs.mean().item()