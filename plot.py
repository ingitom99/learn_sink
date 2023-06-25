import torch
import matplotlib.pyplot as plt


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
    length = int(dim**.5)

    plt.figure()
    plt.title('Mu')
    plt.imshow(X[:dim].cpu().detach().numpy().reshape(length, length), cmap='magma')
    plt.colorbar()
    plt.show()
    plt.figure()
    plt.title('Nu')
    plt.imshow(X[dim:].cpu().detach().numpy().reshape(length, length), cmap='magma')
    plt.colorbar()
    plt.show()
    plt.figure()
    plt.title('T')
    plt.imshow(T.cpu().detach().numpy().reshape(length, length), cmap='magma')
    plt.colorbar()
    plt.show()
    plt.figure()
    plt.title('P')
    plt.imshow(P.cpu().detach().numpy().reshape(length, length), cmap='magma')
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

def plot_test_losses(test_losses : dict[str, list], path: str = None) -> None:

    """
    Plot the test losses.

    Parameters
    ----------
    test_losses : list
        Dictionary of test losses.

    Returns
    -------
    None
    """

    plt.figure()

    for key in test_losses.keys():
        plt.plot(test_losses[key], label=key)

    plt.title('Test Losses')
    plt.xlabel('# test phases')
    plt.ylabel('loss')
    plt.grid()
    plt.legend()

    if path:
        plt.savefig(f'{path}')
    else:
        plt.show()

    return None

def plot_test_rel_errs_emd(rel_errs_emd : dict[str, list],
                           path: str = None) -> None:
    
    """
    Plot the test ot.emd2() relative errors.

    Parameters
    ----------
    rel_errs_emd : dict
        Dictionary of test ot.emd2() relative errors.

    Returns
    -------
    None

    """

    plt.figure()
    for key in rel_errs_emd.keys():
        data = rel_errs_emd[key]
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

def plot_test_rel_errs_sink(rel_errs_sink : dict[str, list],
                            path: str = None) -> None:
        
        """
        Plot the test ot.sinkhorn2() relative errors.

        Parameters
        ----------
        rel_errs_sink : dict
            Dictionary of test ot.sinkhorn2() relative errors.

        Returns
        -------
        None

        """
    
        plt.figure()
        for key in rel_errs_sink.keys():
            data = rel_errs_sink[key]
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