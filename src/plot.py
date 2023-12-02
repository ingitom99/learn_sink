"""
plot.py
-------

Function(s) for visualising data, losses, results, etc.
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
    plt.title('a')
    plt.imshow(X[:dim].cpu().detach().numpy().reshape(length, length),
               cmap='gray')
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.title('b')
    plt.imshow(X[dim:].cpu().detach().numpy().reshape(length, length),
               cmap='gray')
    plt.colorbar()
    plt.show()

    # plot a and b side by side
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(X[:dim].cpu().detach().numpy().reshape(length, length),
                cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(X[dim:].cpu().detach().numpy().reshape(length, length),
                cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()

    plt.figure()
    plt.title('T')
    plt.imshow(T.cpu().detach().numpy().reshape(length, length),
               cmap='gray')
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.title('P')
    plt.imshow(P.cpu().detach().numpy().reshape(length, length), cmap='gray')
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
    plt.plot(train_losses['gen'], color='#4361EE')
    plt.title('Generative Training Loss')
    plt.xlabel('# Training Iterations')
    plt.ylabel('Loss')
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(train_losses['pred'], color='#4361EE')
    plt.title('Predictive Training Loss')
    plt.xlabel('# Training Iterations')
    plt.ylabel('Loss')
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
    plt.xlabel('# Test Phases')
    plt.ylabel('Loss')
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
        plt.title(r'Relative Error on Entropic Transport Distance')
        plt.xlabel('# Test Phases')
        plt.ylabel(f'Relative Error')
        plt.yticks(torch.arange(0, 1.0001, 0.05))
        plt.grid()
        plt.legend()
        
        if path:
            plt.savefig(f'{path}')
        else:
            plt.show()
    
        return None

def plot_warmstarts_mcv(test_warmstart_mcv : dict[str, tuple],
                    folder: str = None) -> None:
    
    """
    Plot the warmstart (mcv) graphs for each test set on different plots.

    Parameters
    ----------
    test_warmstart_mcv : dict[str, tuple]
        Dictionary of warmstart data (predicted V0, ones V0 and gauss V0).
    folder : str, optional
        Folder to save the plots to as png files. If None (default) the plots
        are displayed instead.

    Returns
    -------
    None.
    """
      
    for key in test_warmstart_mcv.keys():
        plt.figure()
        pred, ones, gauss = test_warmstart_mcv[key]
        plt.plot(pred, label=r'pred', color='#F72585')
        plt.plot(ones, label=r'ones', color='#3F37C9')
        plt.plot(gauss, label=r'gauss', color='#4CC9F0')
        plt.title(f' {key}')
        plt.xlabel('# Sinkhorn Iterations')
        plt.ylabel(f'Marginal Constraint Violation')
        plt.yticks(torch.arange(0, 0.30001, 0.05))
        plt.grid()
        plt.legend()
    
        if folder:
            path = folder + f'/warmstart_mcv_{key}.png'
            plt.savefig(f'{path}')
        else:
            plt.show()

    return None

def plot_warmstarts_sink(test_warmstart_sink : dict[str, tuple],
                    folder: str = None) -> None:
    
    """
    Plot the warmstart (sink) graphs for each test set on different plots.

    Parameters
    ----------
    test_warmstart_sink : dict[str, tuple]
        Dictionary of warmstart data (predicted V0 and ones V0).
    folder : str, optional
        Folder to save the plots to as png files. If None (default) the plots
        are displayed instead.

    Returns
    -------
    None.
    """
      
    for key in test_warmstart_sink.keys():
        plt.figure()
        pred, ones, gauss = test_warmstart_sink[key]
        plt.plot(pred, label=f'pred', color='#F72585')
        plt.plot(ones, label=f'ones', color='#3F37C9')
        plt.plot(gauss, label=f'gauss', color='#4CC9F0')
        plt.title(f'{key}')
        plt.xlabel('# Sinkhorn Iterations')
        plt.ylabel('Relative Error')
        plt.yticks(torch.arange(0, 0.60001, 0.05))
        plt.grid()
        plt.legend()
    
        if folder:
            path = folder + f'/warmstart_sink_{key}.png'
            plt.savefig(f'{path}')
        else:
            plt.show()

    return None

def plot_test_mcvs(test_mcvs : dict[str, list], path: str = None) -> None:

    """
    Plot the test mcvs for each test set on the same plot.

    Parameters
    ----------
    test_mcvs : dict[str, list]
        Dictionary of test mcvs.
    path : str, optional
        Path to save the plot to as a png file. If None (default) the plot is
        displayed instead.

    Returns
    -------
    None.
    """

    plt.figure()
    for key in test_mcvs.keys():
        plt.plot(test_mcvs[key], label=key)
    plt.title(r'MCV During Training')
    plt.xlabel('# Test Phases')
    plt.ylabel('Marginal Constraint Violation')
    plt.grid()
    plt.legend()

    if path:
        plt.savefig(f'{path}')
    else:
        plt.show()

    return None

def plot_warmstart_violins(warmstarts, title, path = None):
    keys = list(warmstarts.keys())

    pred = {}
    ones = {}
    gauss = {}
    for key in warmstarts.keys():
        pred[key], ones[key], gauss[key] = warmstarts[key]
    data_pred = []
    data_ones = []
    data_gauss = []
    plt.figure()
    for key in keys:
        data_pred.append(pred[key])
        data_ones.append(ones[key])
        data_gauss.append(gauss[key])
    labels = []
    def add_label(violin, label):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))
    violin_pred = plt.violinplot(data_pred, showmedians=True,showextrema=False)
    violin_ones = plt.violinplot(data_ones, showmedians=True,showextrema=False)
    violin_gauss = plt.violinplot(data_gauss, showmedians=True,showextrema=False)
    # Make all the violin statistics marks red:
    for partname in violin_pred:
        if partname == 'bodies':
            continue
        vp = violin_pred[partname]
        vp.set_edgecolor('#F72585')
        vp = violin_ones[partname]
        vp.set_edgecolor('#3F37C9')
        vp = violin_gauss[partname]
        vp.set_edgecolor('#4CC9F0')
        
    # Make the violin body blue with a red border:
    for vp in violin_pred['bodies']:
        vp.set_facecolor('#F72585')
        vp.set_edgecolor('#F72585')
        vp.set_alpha(0.5)
        
    # Make the violin body blue with a red border:
    for vp in violin_ones['bodies']:
        vp.set_facecolor('#3F37C9')
        vp.set_edgecolor('#3F37C9')
        vp.set_alpha(0.5)
        
    # Make the violin body blue with a red border:
    for vp in violin_gauss['bodies']:
        vp.set_facecolor('#4CC9F0')
        vp.set_edgecolor('#4CC9F0')
        vp.set_alpha(0.5)
        
    add_label(violin_pred, 'pred')
    add_label(violin_ones, 'ones')
    add_label(violin_gauss, 'gauss')
    
    plt.xticks(torch.arange(1, len(keys) + 1), labels=keys)
    plt.xlim(0.25, len(keys) + 0.75)
    plt.ylabel('Error')
    plt.yticks(torch.arange(0, 1.0001, 0.05))
    plt.legend(*zip(*labels), loc=1)
    plt.grid()
    plt.title(title)
    if path:
        plt.savefig(f'{path}')
    else:
        plt.show()
    return None

# plot 4 datasets on 2x2 grid
def plot_4(warmstart_dict : dict, path : str = None):

    plt.figure()
    for i, key in enumerate(warmstart_dict.keys()):
        plt.subplot(2, 2, i+1)
        pred, ones, gauss = warmstart_dict[key]
        plt.plot(pred, label='pred', color='#F72585')
        plt.plot(ones, label='ones', color='#3F37C9')
        plt.plot(gauss, label='gauss', color='#4CC9F0')
        plt.title(f'{key}')
        plt.legend()

        if i == 0 or i == 1:
            plt.xticks([])

        if i == 1 or i == 3:
            plt.yticks([])

    if path:
        plt.savefig(f'{path}')
    else:
        plt.show()

    return None