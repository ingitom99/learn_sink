"""
cost.py
-------

Function(s) for generating cost matrices for optimal transport problems.
"""

import torch
import itertools
import numpy as np

def get_cloud(n : int):

    """
    Generate a point cloud representing a 2D grid of n points per dimension.

    Parameters
    ----------
    n : int
        Number of points per dimension.

    Returns
    -------
    cloud : (n**2, 2) torch.Tensor
        Point cloud.
    """

    partition = np.linspace(0, 1, n)
    cloud = np.stack(np.meshgrid(partition, partition), axis=-1).reshape(-1, 2)

    return cloud

def get_cost(n : int) -> torch.Tensor:

    """
    Generate a square euclidean cost matrix on 2D unit length grid with n points
    per dimension.

    Parameters
    ----------
    n : int
        Number of points per dimension.

    Returns
    -------
    C : (n**2, n**2) torch.Tensor
        Euclidean cost matrix.
    """

    cloud = get_cloud(n)
    x = np.array(list(itertools.product(cloud, repeat=2)))
    a = x[:, 0]
    b = x[:, 1]
    C = torch.tensor(np.linalg.norm(a - b, axis=1) ** 2).reshape(n**2, n**2)
    return C
