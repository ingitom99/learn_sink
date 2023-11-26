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
               cmap='magma')
    plt.colorbar()
    plt.show()
    plt.figure()
    plt.title('b')
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
    plt.xlabel('# Test Phases')
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
        plt.title(r'$E_{SD}$ over Test Phases')
        plt.xlabel('# Test Phases')
        plt.ylabel(r'$E_{SD}$')
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
        plt.plot(pred, label=r'pred $v_0$', color='#F72585')
        plt.plot(ones, label=r'ones $v_0$', color='#3F37C9')
        #plt.plot(gauss, label='gauss', color='#4CC9F0')
        plt.title(f' {key}')
        plt.xlabel('# Sinkhorn iterations')
        plt.ylabel(r'$E_{MCV}$')
        plt.yticks(torch.arange(0, 1.0001, 0.05))
        plt.grid()
        plt.legend()
    
        if folder:
            path = folder + f'/warmstart_mcv_{key}.png'
            plt.savefig(f'{path}')
        else:
            plt.show()

    return None

def plot_warmstarts_emd(test_warmstart_emd : dict[str, tuple],
                    folder: str = None) -> None:
    
    """
    Plot the warmstart (emd) graphs for each test set on different plots.

    Parameters
    ----------
    test_warmstart_emd : dict[str, tuple]
        Dictionary of warmstart data (predicted V0 and ones V0).
    folder : str, optional
        Folder to save the plots to as png files. If None (default) the plots
        are displayed instead.

    Returns
    -------
    None.
    """
      
    for key in test_warmstart_emd.keys():
        plt.figure()
        pred, ones = test_warmstart_emd[key]
        plt.plot(pred, label='predicted V0')
        plt.plot(ones, label='ones V0')
        plt.title(f'Warmstart (emd): {key}')
        plt.xlabel('# iterations')
        plt.ylabel('rel err')
        plt.yticks(torch.arange(0, 1.0001, 0.05))
        plt.grid()
        plt.legend()
    
        if folder:
            path = folder + f'/warmstart_emd_{key}.png'
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
        plt.plot(pred, label=r'pred $v_0$', color='#F72585')
        plt.plot(ones, label=r'ones $v_0$', color='#3F37C9')
        #plt.plot(gauss, label='gauss V0', color='#4CC9F0')
        plt.title(f'{key}')
        plt.xlabel('# Sinkhorn Iterations')
        plt.ylabel(r'$E_{SD}$')
        plt.yticks(torch.arange(0, 1.0001, 0.05))
        plt.grid()
        plt.legend()
    
        if folder:
            path = folder + f'/warmstart_sink_{key}.png'
            plt.savefig(f'{path}')
        else:
            plt.show()

    return None


# plot 4 datasets on 2x2 grid
def plot_4(data, folder = None):

    #plot 4 elements of data on 2x2 subplots
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(data[0].cpu().detach().numpy().reshape(28, 28), cmap='magma')
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(data[1].cpu().detach().numpy().reshape(28, 28), cmap='magma')
    plt.axis('off')
    plt.subplot(2, 2, 3)
    

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
    plt.title(r'$E_{MCV}$ During Training')
    plt.xlabel('# Test Phases')
    plt.ylabel(r'$E_{MCV}$')
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
    add_label(plt.violinplot(data_pred, showmedians=True,showextrema=False, color='#F72585'), 'pred')
    add_label(plt.violinplot(data_ones, showmedians=True,showextrema=False, color='#3F37C9'), 'ones')
    #add_label(plt.violinplot(data_gauss, showmedians=True,showextrema=False, color='#4CC9F0'), 'gauss')
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

def plot_lipschitz_vals(lipschitz_vals, path = None):
    plt.figure()
    plt.plot(lipschitz_vals)
    plt.title('Lipschitz Values Gen Net')
    plt.xlabel('# Lipschitz test phases')
    plt.ylabel('Lipschitz Value')
    plt.grid()
    if path:
        plt.savefig(f'{path}')
    else:
        plt.show()
    return None

# given 20 images of 28x28, plot them in a 4x5 grid
def plot_20_images(images, path = None):
    plt.figure()
    for i in range(20):
        plt.subplot(4, 5, i + 1)
        plt.imshow(images[i].cpu().detach().numpy().reshape(28, 28), cmap='magma')
        plt.axis('off')
    if path:
        plt.savefig(f'{path}')
    else:
        plt.show()
    return None
