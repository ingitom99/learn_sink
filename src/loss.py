"""
loss.py
-------

Function(s) for computing loss values.
"""

import torch
from src.nets import GenNet
from src.data_funcs import approximate_matrix_norm

def hilb_proj_loss(
    U: torch.Tensor,
    V: torch.Tensor,
    gen_net: GenNet,
    loss_reg: float,
    toggle_reg: bool
) -> float:

    """
    Compute the mean Hilbert projective loss between pairs of vectors.

    Parameters
    ----------
    U : (n_samples, dim) torch.Tensor
        First set of vectors.
    V : (n_samples, dim) torch.Tensor
        Second set of vectors.
    gen_net: generator network.
    loss_reg: regularizing coefficient.
    toggle_reg: toggles weight regularization on or off.
    
    Returns
    -------
    loss : float
        Mean loss value.
    """
    def weight_reg(weights, reg_coeff):
        return reg_coeff * torch.max(0.0, approximate_matrix_norm(weights, 1) - 1)
    diff = U - V
    spectrum = torch.max(diff, dim=1)[0] - torch.min(diff, dim=1)[0]
    loss = spectrum.mean()
    if toggle_reg:
        for layer in gen_net.layers:
            loss += weight_reg(layer[0].weight, loss_reg)

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