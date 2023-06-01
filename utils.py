"""
Utility functions for this project.
"""

import torch
import numpy as np
from skimage.draw import random_shapes
import matplotlib.pyplot as plt
  
def hilb_proj_loss(U, V):

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
        Loss value.
    """

    diff = U - V
    spectrum = torch.max(diff, dim=1)[0] - torch.min(diff, dim=1)[0]
    loss = spectrum.mean()

    return loss

def plot_XPT(X : torch.Tensor, P : torch.Tensor, T : torch.Tensor, dim : int
             ) -> None:

    """
    Plot and show a pair of probability distributions formatted as images
    followed by the corresponding target and prediction.

    Parameters
    ----------
    X : (2 * dim) torch.Tensor
        Pair of probability distributions.
    P : (dim) torch.Tensor  
        Prediction.
    T : (dim) torch.Tensor
        Target.
    dim : int
        Dimension of the probability distributions.
    """

    plt.figure()
    plt.title('Mu')
    plt.imshow(X[:784].cpu().detach().numpy().reshape(28, 28), cmap='magma')
    plt.colorbar()
    plt.show()
    plt.figure()
    plt.title('Nu')
    plt.imshow(X[784:].cpu().detach().numpy().reshape(28, 28), cmap='magma')
    plt.colorbar()
    plt.show()
    plt.figure()
    plt.title('T')
    plt.imshow(T.cpu().detach().numpy().reshape(28, 28), cmap='magma')
    plt.colorbar()
    plt.show()
    plt.figure()
    plt.title('P')
    plt.imshow(P.cpu().detach().numpy().reshape(28, 28), cmap='magma')
    plt.colorbar()
    plt.show()

    return None
    
def prior_sampler(n_samples : int, dim : int) -> torch.Tensor:

    """
    Sample from the prior distribution.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    dim : int
        Dimension of the samples.

    Returns
    -------
    samples : (n_samples, 2 * dim) torch.Tensor
        Samples from the prior distribution.
    """

    samples = torch.randn((n_samples, 2 * dim))

    return samples

def testset_sampler(test_set : torch.Tensor, n_samples : int) -> torch.Tensor:

    """
    Randomly sample from a given test set.

    Parameters
    ----------
    test_set : (n_test_samples, 2 * dim) torch.Tensor
        Test set.
    n_samples : int
        Number of samples.
    
    Returns
    -------
    test_sample : (n_samples, 2 * dim) torch.Tensor
        Random sample from the test set.
    """

    rand_mask = torch.randint(low=0, high=len(test_set), size=(n_samples,2))
    test_sample = test_set[rand_mask]
    test_sample = torch.flatten(test_sample, start_dim=1)

    return test_sample