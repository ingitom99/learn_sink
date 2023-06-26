"""
plot.py
-------

Function(s) for visualising data, losses, results, etc.
"""

import torch
import matplotlib.pyplot as plt

def plot_XPT(X : torch.Tensor, P : torch.Tensor, T : torch.Tensor, dim : int
             ) -> None:

    """
    Plot and show a pair of probability distributions (MU and NU)formatted as
    images followed by the corresponding target and prediction.

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

    Returns
    -------
    None.
    """
    length = int(dim**.5)

    plt.figure()
    plt.title('Mu')
    plt.imshow(X[:dim].cpu().detach().numpy().reshape(length, length),
               cmap='magma')
    plt.colorbar()
    plt.show()
    plt.figure()
    plt.title('Nu')
    plt.imshow(X[dim:].cpu().detach().numpy().reshape(length, length),
               cmap='magma')
    plt.colorbar()
    plt.show()
    plt.figure()
    plt.title('T')
    plt.imshow(T.cpu().detach().numpy().reshape(length, length),
               cmap='magma')
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
    Plot the training losses for each net side by side

    Parameters
    ----------
    train_losses : list
        Dictionary of generative and predictive training losses.
    path : str, optional
        Path to save the plot to as a png file. If None (default) the plot is
        displayed instead.

    Returns
    -------
    None.
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
    Plot the test losses for each test set on the same plot.

    Parameters
    ----------
    test_losses : dict[str, list]
        Dictionary of test losses.
    path : str, optional
        Path to save the plot to as a png file. If None (default) the plot is
        displayed instead.

    Returns
    -------
    None.
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
    Plot the test relative errors against ot.emd2() for each test set on the
    same plot.

    Parameters
    ----------
    rel_errs_emd : dict[str, list]
        Dictionary of test relative errors against ot.emd2().
    path : str, optional
        Path to save the plot to as a png file. If None (default) the plot is
        displayed instead.

    Returns
    -------
    None.
    """

    plt.figure()
    for key in rel_errs_emd.keys():
        data = rel_errs_emd[key]
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

def plot_test_rel_errs_sink(rel_errs_sink : dict[str, list],
                            path: str = None) -> None:
        
        """
        Plot the test relative errors against ot.sinkhorn2() for each test set
        on the same plot.

        Parameters
        ----------
        rel_errs_sink : dict[str, list]
            Dictionary of test relative errors against ot.sinkhorn2().
        path : str, optional
            Path to save the plot to as a png file. If None (default) the plot
            is displayed instead.

        Returns
        -------
        None.
        """
    
        plt.figure()
        for key in rel_errs_sink.keys():
            data = rel_errs_sink[key]
            plt.plot(data, label=key)
        plt.title(' Rel Error: PredNet Dist VS ot.sinkhorn2')
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

def plot_warmstarts(test_warmstart : dict[str, tuple],
                    folder: str = None) -> None:
    
    """
    Plot the warmstart graphs for each test set on different plots.

    Parameters
    ----------
    test_warmstart : dict[str, tuple]
        Dictionary of warmstart data (predicted V0 and ones V0).
    folder : str, optional
        Folder to save the plots to as png files. If None (default) the plots
        are displayed instead.

    Returns
    -------
    None
    """
      
    for key in test_warmstart.keys():
        plt.figure()
        pred, ones = test_warmstart[key]
        plt.plot(pred, label='predicted V0')
        plt.plot(ones, label='ones V0')
        plt.title(f'Warmstart (sinkhorn vs ot.emd2): {key}')
        plt.xlabel('# iterations')
        plt.ylabel('rel err')
        plt.yticks(torch.arange(0, 1.0001, 0.05))
        plt.grid()
        plt.legend()
    
        if folder:
            path = folder + f'/warmstart_{key}.png'
            plt.savefig(f'{path}')
        else:
            plt.show()

    return None