"""
cost.py
-------

Function(s) for generating cost matrices for optimal transport problems.
"""

import torch
import itertools
import numpy as np

def l2_cost(n : int) -> torch.Tensor:

    """
    Generate a square euclidean cost matrix on unit grid with n points per
    dimension.

    Parameters
    ----------
    n : int
    

    partition = np.linspace(0, 1, n)
    couples = np.array(np.meshgrid(partition, partition)).T.reshape(-1, 2)
    x = np.array(list(itertools.product(couples, repeat=2)))
    a = x[:, 0]
    b = x[:, 1]
    C = np.linalg.norm(a - b, axis=1) ** 2
    return torch.tensor(C.reshape((n**2, -1)))
