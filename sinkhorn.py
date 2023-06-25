"""
sinkhorn.py
-----------

Implementation(s) of the Sinkhorn algorithm for computing approximate solutions
to the entropic regularized optimal transport problem.
"""

# Imports
import torch

# Functions
def sink(mu : torch.Tensor, nu : torch.Tensor, C : torch.Tensor, eps : float,
         v0 : torch.Tensor, maxiter : int) -> tuple[torch.Tensor, torch.Tensor,
                                                    torch.Tensor, float]: 

    """
    The standard Sinkhorn algorithm!

    Parameters
    ----------
    mu : (dim,) torch.Tensor
        First probability distribution.
    nu : (dim,) torch.Tensor
        Second probability distribution.
    C : (dim, dim) torch.Tensor
        Cost matrix.
    eps : float
        Regularization parameter.
    v0 : (dim,) torch.Tensor
        Initial guess for scaling factor v.
    maxiter : int
        Maximum number of iterations.

    Returns
    -------
    u : (dim,) torch.Tensor
        1st Scaling factor.
    v : (dim,) torch.Tensor
        2nd Scaling factor.
    G : (dim, dim) torch.Tensor
        Optimal transport plan.
    dist : float
        Optimal transport distance.
    """

    K = torch.exp(-C/eps)
    v = v0

    for _ in range(maxiter):
        u = mu / (K @ v)
        v = nu / (K.T @ u)

    G = torch.diag(u)@K@torch.diag(v)    
    dist = torch.trace(C.T@G)

    return u, v, G, dist

def sink_vec(MU : torch.Tensor, NU : torch.Tensor, C : torch.Tensor,
             eps : float, V0 : torch.Tensor, n_iters : int) -> torch.Tensor:
    
    """
    A vectorized version of the Sinkhorn algorithm to create scaling factors
    to be used for generating targets.

    Parameters
    ----------
    MU : (n_samples, dim) torch.Tensor
        First probability distributions.
    NU : (n_samples, dim) torch.Tensor
        Second probability distributions.
    C : (dim, dim) torch.Tensor
        Cost matrix.
    eps : float
        Regularization parameter.
    V0 : (n_samples, dim) torch.Tensor
        Initial guess for scaling factors V.
    n_iters : int
        Maximum number of iterations.
    
    Returns
    -------
    V : (n_samples, dim) torch.Tensor
        2nd Scaling factor.
    """

    K = torch.exp(-C/eps)
    V = V0
    
    for _ in range(n_iters):
        U = MU / (K @ V.T).T
        V = NU / (K.T @ U.T).T

    return U, V