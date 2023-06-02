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

def test_set_sampler(test_set : torch.Tensor, n_samples : int) -> torch.Tensor:

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

    rand_perm = torch.randperm(test_set.size(0))
    rand_mask = rand_perm[:n_samples]
    test_sample = test_set[rand_mask]
    test_sample = torch.flatten(test_sample, start_dim=1)

    return test_sample

def plot_train_losses(losses_train : list, path: str = None) -> None:

    """
    Plot the training losses.

    Parameters
    ----------
    losses_train : list
        List of training losses.
    """
    log_losses = torch.log(torch.tensor(losses_train))
    plt.figure()
    plt.plot(log_losses)
    plt.title('Log Train Losses')
    plt.xlabel('# minibatches')
    plt.ylabel('log loss')
    plt.grid()

    if path:
        plt.savefig(f'{path}')
    else:
        plt.show()
    
    return None

def plot_test_losses(losses_test : dict[str, list], path: str = None) -> None:

    plt.figure()

    for key in losses_test.keys():
        log_data = torch.log(torch.tensor(losses_test[key]))
        plt.plot(log_data, label=key)

    plt.title('Log Test Losses')
    plt.xlabel('# test phases')
    plt.ylabel('log loss')
    plt.grid()
    plt.legend()

    if path:
        plt.savefig(f'{path}')
    else:
        plt.show()

    return None



def plot_test_rel_errs(rel_errs : dict[str, list], path: str = None) -> None:

    plt.figure()
    for key in rel_errs.keys():
        data = rel_errs[key]
        plt.plot(data, label=key)
    plt.title(' Rel Error: PredNet Dist VS ot.emd2')
    plt.xlabel('# test phases')
    plt.ylabel('rel err')
    plt.yticks(torch.arange(0, 1.0001, 0.05))
    plt.grid()
    plt.legend()
    
    if path:
        plt.savefig(f'{path}')
    else:
        plt.show()

    return None