"""
loss.py
-------

Function(s) for computing loss values.
"""

import torch
from src.nets import GenNet

def hilb_proj_loss(U: torch.Tensor, V: torch.Tensor) -> float:

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
        Mean loss value.
    """
    
    diff = U - V
    spectrum = torch.max(diff, dim=1)[0] - torch.min(diff, dim=1)[0]
    loss = spectrum.mean()

    return loss

def mse_loss(U: torch.Tensor, V: torch.Tensor) -> float:

    """
    Compute the mean squared error between pairs of vectors.

    Parameters
    ----------
    U : (n_samples, dim) torch.Tensor
        First set of vectors.
    V : (n_samples, dim) torch.Tensor
        Second set of vectors.
    
    Returns
    -------
    loss : float
        Mean loss value.
    """

    loss = ((U - V) ** 2).mean()

    return loss

def approximate_matrix_norm(W: torch.tensor, nb_iterations: int):

    """
    Approximates the 2-norm of a matrix using the power method.

    Parameters
    ----------
    W: matrix to compute 2-norm of.
    nb_iterations: number of iterations of the power method.

    Returns
    -------
    Approximate 2-norm of `W`.

    """

    b = torch.rand(W.shape[0], dtype=W.dtype, device=W.device)
    for _ in range(nb_iterations):
        b = torch.matmul(W, torch.matmul(W.T, b))
        b = b / torch.linalg.vector_norm(b)
    norm = torch.sqrt(
        (torch.matmul(b, torch.matmul(W, torch.matmul(W.T, b))) / torch.matmul(b, b))
    )
    return norm

def weight_reg(gen_net : GenNet, reg_coeff : float) -> float:
    
    """
    Compute the weight regularisation value of a generator network
    to be added to a loss value.

    Parameters
    ----------
    gen_net : GenNet
        Generator network.
    reg_coeff : float
        Regularisation coefficient.
    
    Returns
    -------
    reg_val : float
        Regularisation value.
    """
    
    reg_val = 0
    for layer in gen_net.layers:
            weights = layer[0].weight
            reg_contrib = approximate_matrix_norm(weights, 1) - 1
            if reg_contrib > 0:
                reg_val = reg_val + torch.max(reg_contrib)
    reg_val = reg_val * reg_coeff
    return reg_val