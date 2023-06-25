"""
cost.py

Function(s) for generating cost matrices for optimal transport problems.
"""

import torch

def l2_cost_mat(width : int, height : int, normed: bool = True
                   ) -> torch.Tensor:

    """
    Create an L2 distance cost matrix.

    Parameters
    ----------
    width : int
        The width of the cost matrix.
    height : int
        The height of the cost matrix.
    normed : bool, optional
        Whether to normalize the cost matrix or not. The default is True.

    Returns
    -------
    C : (n, n) torch.Tensor
        The cost matrix.
    """

    n = width * height
    C = torch.zeros([n, n])

    for a in range(n):
        for b in range(n):
            ax = a // width
            ay = a % width
            bx = b // width
            by = b % width
            C[a][b] = ((ax - bx)**2 + (ay - by)**2)*.5

    if normed:
        C = C / C.max()

    return C
