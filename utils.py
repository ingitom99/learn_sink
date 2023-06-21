"""
Utility functions for this project.
"""

import torch
import torch.nn.functional as F
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

def preprocessor(dataset, length, constant):
    # Resize the dataset
    resized_dataset = F.interpolate(dataset.unsqueeze(1), size=(length, length),
                                    mode='bilinear', align_corners=False).squeeze(1)

    # Flatten the dataset
    flattened_dataset = resized_dataset.view(-1, length**2)

    # Normalize the dataset to sum to one in the second dimension
    normalized_dataset = flattened_dataset / flattened_dataset.sum(dim=1, keepdim=True)

    # Add a small constant value to each element
    processed_dataset = normalized_dataset + constant

    # Normalize the dataset again to sum to one in the second dimension
    processed_dataset /= processed_dataset.sum(dim=1, keepdim=True)

    return processed_dataset

def get_pred_dists(P, X, eps, C, dim):
    dists = []
    for p, x, e in zip(P, X, eps):
        mu = x[:dim] / x[:dim].sum()
        nu = x[dim:] / x[dim:].sum()
        K = torch.exp(-C/e)
        v = torch.exp(p)
        u = mu / (K @ v)
        v = nu / (K.T @ u)
        G = torch.diag(u)@K@torch.diag(v)    
        dist = torch.trace(C.T@G)
        dists.append(dist)
    dists = torch.tensor(dists)
    return dists

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

def plot_train_losses(train_losses : dict, path: str = None) -> None:

    """
    Plot the training losses.

    Parameters
    ----------
    train_losses : list
        Dictionary of generative and predictive training losses.
    """
    
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(train_losses['gen'])
    plt.title('Generative Training Loss')
    plt.xlabel('# training phases')
    plt.ylabel('loss')
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(train_losses['pred'])
    plt.title('Predictive Training Loss')
    plt.xlabel('# training phases')
    plt.ylabel('loss')
    plt.grid()
    plt.tight_layout()

    if path:
        plt.savefig(f'{path}')
    else:
        plt.show()

    return None

def plot_test_rel_errs_emd(rel_errs : dict[str, list], path: str = None) -> None:

    plt.figure()
    for key in rel_errs.keys():
        data = rel_errs[key]
        plt.plot(data, label=key)
    plt.title(' Rel Error: PredNet Dist VS ot.emd2 (small eps)')
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

def plot_test_rel_errs_sink(rel_errs : dict[str, list], path: str = None) -> None:
    
        plt.figure()
        for key in rel_errs.keys():
            data = rel_errs[key]
            plt.plot(data, label=key)
        plt.title(' Rel Error: PredNet Dist VS ot.sinkhorn2 (variable eps)')
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