"""
sinkhorn.py

Implementations the Sinkhorn algorithm for computing approximate solutions to
the entropic regularized optimal transport problem.
"""

# Imports
import torch

# Functions
def sink(mu : torch.Tensor, nu : torch.Tensor, C : torch.Tensor, eps : float,
         v0 : torch.Tensor, maxiter : int) -> tuple[torch.Tensor, torch.Tensor,
                                                    torch.Tensor, float]: 

    """
    The Sinkhorn algorithm!

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
        Optimal transport matrix.
    dist : float
        Optimal transport distance.
    """

    K = torch.exp(-C/eps)
    v = v0

    for i in range(maxiter):
        u = mu / (K @ v)
        v = nu / (K.T @ u)

    G = torch.diag(u)@K@torch.diag(v)    
    dist = torch.trace(C.T@G)

    return u, v, G, dist

def sink_vec(MU : torch.Tensor, NU : torch.Tensor, C : torch.Tensor,
             eps : torch.Tensor, V0 : torch.Tensor,
             n_iters : int) -> torch.Tensor:
    
    """
    A vectorized version of the Sinkhorn algorithm for creating targets.

    Parameters
    ----------
    MU : (n_samples, dim) torch.Tensor
        First probability distributions.
    NU : (n_samples, dim) torch.Tensor
        Second probability distributions.
    C : (dim, dim) torch.Tensor
        Cost matrix.
    eps : (n_samples, 1) torch.Tensor
        Regularization parameters.
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
    
    for i in range(n_iters):
        U = MU / (K @ V.T).T
        V = NU / (K.T @ U.T).T

    return U, V

def sink_var_eps(MU : torch.Tensor, NU : torch.Tensor, C : torch.Tensor,
                 eps : torch.Tensor, V0 : torch.Tensor,
                 n_iters : int) -> torch.Tensor:
    """
    An implementation of the Sinkhorn algorithm for variable regularization parameters.

    Parameters
    ----------
    MU : (n_samples, dim) torch.Tensor
        First probability distributions.
    NU : (n_samples, dim) torch.Tensor
        Second probability distributions.
    C : (dim, dim) torch.Tensor
        Cost matrix.
    eps : (n_samples,) torch.Tensor
        Regularization parameters.
    V0 : (n_samples, dim) torch.Tensor
        Initial guesses for scaling factors V.
    n_iters : int
        Maximum number of iterations.

    Returns
    -------
    U : (n_samples, dim) torch.Tensor
        1st Scaling factors.
    V : (n_samples, dim) torch.Tensor
        2nd Scaling factors.
    """
    U = torch.zeros_like(MU)
    V = torch.zeros_like(NU)

    for mu, nu, e, v0 in zip(MU, NU, eps, V0):

        K = torch.exp(-C/e)
        v = v0

        for i in range(n_iters):
            u = mu / (K @ v)
            v = nu / (K.T @ u)

        U[i] = u
        V[i] = v

    return U, V

def sink_var_eps_vec(MU: torch.Tensor, NU: torch.Tensor, C: torch.Tensor,
                 eps: torch.Tensor, V0: torch.Tensor,
                 n_iters: int) -> torch.Tensor:
    """
    A vectorized implementation of the Sinkhorn algorithm for variable regularization parameters.

    Parameters
    ----------
    MU : (n_samples, dim) torch.Tensor
        First probability distributions.
    NU : (n_samples, dim) torch.Tensor
        Second probability distributions.
    C : (dim, dim) torch.Tensor
        Cost matrix.
    eps : (n_samples,) torch.Tensor
        Regularization parameters.
    V0 : (n_samples, dim) torch.Tensor
        Initial guesses for scaling factors V.
    n_iters : int
        Maximum number of iterations.

    Returns
    -------
    U : (n_samples, dim) torch.Tensor
        1st Scaling factors.
    V : (n_samples, dim) torch.Tensor
        2nd Scaling factors.
    """
    K = torch.exp(-C.unsqueeze(0) / eps.unsqueeze(1))  # Broadcasting for K calculation

    U = MU.new_empty(MU.shape)
    V = NU.new_empty(NU.shape)

    v = V0.clone()

    for _ in range(n_iters):
        u = MU / (K @ v.unsqueeze(2)).squeeze(2)
        v = NU / (K.transpose(1, 2) @ u.unsqueeze(2)).squeeze(2)

    U = u
    V = v

    return U, V


