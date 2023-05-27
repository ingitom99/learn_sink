"""
Various implementations of the Sinkhorn algorithm for computing approximate solutions to the entropic regularized optimal transport problem.
"""

# Imports
import torch

# Functions
def sink(mu, nu, C, reg, v0, maxiter):
  """
  Sinkhorn algorithm for computing approximate solutions to the entropic regularized optimal transport problem.

  Inputs:
    mu: torch tensor of shape (n,)
      First probability distribution
    nu: torch tensor of shape (n,)
      Second probability distribution
    C: torch tensor of shape (n,n)
      Cost matrix
    reg: float
      Regularization parameter
    v0: torch tensor of shape (n,)
      Initial guess for scaling factor v
    maxiter: int
      Maximum number of iterations
  
  Outputs:
    u: torch tensor of shape (n,)
      First Sinkhorn scaling factor
    v: torch tensor of shape (n,)
      Second Sinkhorn scaling factor
    G: torch tensor of shape (n,n)
      Transport matrix
    dist: float
      Sinkhorn distance (Approx. Wasserstein distance)
  """
  K = torch.exp(-C/reg)
  v = v0
  for i in range(maxiter):
    u = mu / (K @ v)
    v = nu / (K.T @ u)
  G = torch.diag(u)@K@torch.diag(v)    
  dist = torch.trace(C.T@G)
  return u, v, G, dist

def sink_vec(MU, NU, C, reg, V0, maxiter):
  """
  Vectorized Sinkhorn algorithm for computing approximate solutions to the entropic regularized optimal transport problem.

  Inputs:
    MU: torch tensor of shape (n_samples, n)
      First probability distributions
    NU: torch tensor of shape (n_samples, n)
      Second probability distributions
    C: torch tensor of shape (n,n)
      Cost matrix
    reg: float
      Regularization parameter
    V0: torch tensor of shape (n_samples, n)
      Initial guess for scaling factor v
    maxiter: int
      Maximum number of iterations
  
  Outputs:
    V: torch tensor of shape (n_samples, n)
      Second Sinkhorn scaling factors
  """
  K = torch.exp(-C/reg)
  V = V0
  for i in range(maxiter):
    U = MU / (K @ V.T).T
    V = NU / (K.T @ U.T).T
  return V