"""
loss.py
-------

Function(s) for computing loss values.
"""

import torch

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
    spectrum : float
        Loss values for each pair of data points.
    """

    diff = U - V
    spectrum = torch.max(diff, dim=1)[0] - torch.min(diff, dim=1)[0]

    return spectrum