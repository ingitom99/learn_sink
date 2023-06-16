# Write a script that given a probability vector of length dim
# it randomly samples n more probability vectors of length dim close to it.

import torch

def mutate(x, n, dim, sigma):
    """
    Given a probability vector of length dim, randomly sample n more probability
    vectors of length dim close to it.
    
    Parameters
    ----------
    x : torch.Tensor
        Probability vector of length dim.
    n : int
        Number of samples to generate.
    dim : int
        Dimension of each probability distribution in the data.
    sigma : float
        Standard deviation of the normal distribution to sample from.
    
    Returns
    -------
    torch.Tensor
        Tensor of shape (n, dim) containing the mutated samples.
    """
    X = x.repeat(n, 1)
    X_mutated = X + sigma * torch.randn(n, dim) 

    return X_mutated

