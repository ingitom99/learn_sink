"""
Auxiliary functions for testing the performance of the predictive network.
"""

import torch
import ot 
from sinkhorn_algos import sink_vec
from utils import plot_XPT, test_set_sampler, plot_test_losses, plot_test_rel_errs
import matplotlib.pyplot as plt
from tqdm import tqdm
from net import PredNet

def test_pred_loss(pred_net : PredNet, X : torch.Tensor,
                   C : torch.Tensor, eps : float, dim : int, loss_func,
                   n_iters : int, plot : bool) -> float:
    """
    Test the performance of the predictive network with respect to a given
    dataset and loss function.

    Parameters
    ----------
    PredNet : torch.nn.Module
        Predictive network.
    X : (n_samples, 2*dim) torch.Tensor
        Pairs of probability distributions.
    C : (dim, dim) torch.Tensor
        Cost matrix.
    eps : float
        Regularization parameter.
    dim : int
        Dimension of the probability distributions.
    loss_func : function
        Loss function.
    n_iters : int
        Number of Sinkhorn iterations.
    plot : bool
        Whether to plot the results or not.
    
    Returns
    -------
    loss : float
        Loss value.
    """

    
                                      
    P = pred_net(X)

    with torch.no_grad():
        V0 = torch.ones_like(X[:, :dim])
        V = sink_vec(X[:, :dim], X[:, dim:], C, eps, V0, n_iters)[1]
        V = torch.log(V)
    T = V
    T = T - torch.unsqueeze(T.mean(dim=1), 1).repeat(1, dim)
    loss = loss_func(P, T).item()

    if plot:
        plot_XPT(X[0], P[0], T[0], dim)

    return loss

def test_pred_dist(PredNet : torch.nn.Module, X : torch.Tensor,
                   C : torch.Tensor, eps : float, dim : int, plot : bool,
                   title : str) -> float:
    
    """
    Test the performance of the predictive network with respect to the relative
    error on the Wasserstein distance.

    Parameters
    ----------
    PredNet : torch.nn.Module
        Predictive network.
    X : (n_samples, 2*dim) torch.Tensor
        Pairs of probability distributions.
    C : (dim, dim) torch.Tensor
        Cost matrix.
    eps : float
        Regularization parameter.
    dim : int
        Dimension of the probability distributions.
    plot : bool
        Whether to plot the results or not.
    title : str
        Title of the plot.
    
    Returns
    -------
    rel_errs_mean : float
        Mean relative error on the Wasserstein distance.
    """

    # Collecting the Wasserstein distances
    emds = []
    for x in X:
        emd_mu = x[:dim] / x[:dim].sum()
        emd_nu = x[dim:] / x[dim:].sum()
        emd = ot.emd2(emd_mu, emd_nu, C)
        emds.append(emd)
    emds = torch.tensor(emds)


    K = torch.exp(C/-eps)
    V = torch.exp(PredNet(X))

    MU = X[:, :dim]
    NU = X[:, dim:]

    U = MU / (K @ V.T).T
    V = NU / (K.T @ U.T).T

    dists = []

    for u, v in zip(U, V):
        G = torch.diag(u)@K@torch.diag(v)
        dist = torch.trace(C.T@G)
        dists.append(dist)

    dists = torch.tensor(dists)

    rel_errs = torch.abs(emds - dists) / emds
    rel_errs_mean = rel_errs.mean().item()

    if plot:
        plt.figure()
        plt.title(f'Pred Net Distance vs emd2 ({title})')
        plt.xlabel('Sample')
        plt.ylabel('Distance')
        plt.plot(emds, label='emd2')
        plt.plot(dists, label='predicted')
        plt.grid()
        plt.legend()
        plt.show()

    return rel_errs_mean
  
import ot
def test_warmstart(pred_net : PredNet, X : torch.Tensor, C : torch.Tensor,
                   eps : float, dim : int, plot : bool, path : str
                   ) -> tuple[list, list]:
  
    """
    Test the performance of the predictive network as a 'warmstart' for the 
    Sinkhorn algorithm.

    Parameters
    ----------
    pred_net : PredNet
        Predictive network.
    X : (n_samples, 2*dim) torch.Tensor
        Pairs of probability distributions.
    C : (dim, dim) torch.Tensor
        Cost matrix.
    eps : float
        Regularization parameter.
    dim : int
        Dimension of the probability distributions.
    plot : str
        Title of the plot, if None no plot.
    path : str
    
    Returns
    -------
    rel_err_means_pred : list
        Mean relative error on the Wasserstein distance for predicted V0.
    rel_err_means_ones : list
        Mean relative error on the Wasserstein distance for vector of ones V0.
    """

    # Collecting the Wasserstein distances
    emds = []
    for x in X:
        emd_mu = x[:dim] / x[:dim].sum()
        emd_nu = x[dim:] / x[dim:].sum()
        emd = ot.emd2(emd_mu, emd_nu, C)
        emds.append(emd)
    emds = torch.tensor(emds)

    # Initiliazing Sinkhorn algorithm
    K = torch.exp(C/-eps)
    V_pred = torch.exp(pred_net(X))
    V_ones = torch.ones_like(V_pred)
    MU = X[:, :dim]
    NU = X[:, dim:]
    rel_err_means_pred = []
    rel_err_means_ones = []

    # Looping over 1000 iterations of Sinkhorn algorithm
    for i in tqdm(range(1000)):

        # Performing a step of Sinkhorn algorithm for predicted V0
        U_pred = MU / (K @ V_pred.T).T
        V_pred = NU / (K.T @ U_pred.T).T
    
        # Calculating the Sinkhorn distances for predicted V0
        dists_pred = []
        for u, v in zip(U_pred, V_pred):
            G = torch.diag(u)@K@torch.diag(v)
            dist_pred = torch.trace(C.T@G)
            dists_pred.append(dist_pred)
        dists_pred = torch.tensor(dists_pred)
        rel_errs_pred = torch.abs(emds - dists_pred) / emds
        rel_err_means_pred.append(rel_errs_pred.mean().item())
        dists_ones = []
        U_ones = MU / (K @ V_ones.T).T
        V_ones = NU / (K.T @ U_ones.T).T
        for u, v in zip(U_ones, V_ones):
            G = torch.diag(u)@K@torch.diag(v)
            dist = torch.trace(C.T@G)
            dists_ones.append(dist)
        dists_ones = torch.tensor(dists_ones)
        rel_errs_ones = torch.abs(emds - dists_ones) / emds
        rel_err_means_ones.append(rel_errs_ones.mean().item())

    if plot:
        plt.figure()
        plt.title(f"Sinkhorn rel errors with emd2: {plot}")
        plt.xlabel('# Sinkhorn Iterations')
        plt.ylabel('Relative Error on Wasserstein Distance')
        plt.grid()
        plt.yticks(torch.arange(0, 1.0001, 0.05))
        plt.axhline(y=rel_err_means_pred[0], color='r', linestyle="dashed",
                    label='pred net 0th rel err')
        plt.plot(rel_err_means_pred, label="V0: pred net")
        plt.plot(rel_err_means_ones, label="V0: ones")
        plt.legend()
        plt.savefig(path)
    return rel_err_means_pred, rel_err_means_ones

def test_loss(pred_net : PredNet, test_sets: dict, n_samples : int,
              losses_test : dict, device : torch.device, C : torch.Tensor,
              eps : float, dim : int, loss_func : callable, plot : bool):
    """
    """

    for key in test_sets.keys():
        X_test = test_set_sampler(test_sets[key], n_samples).double().to(device)
        loss = test_pred_loss(pred_net, X_test, C, eps, dim, loss_func,
                              2000, True)
        losses_test[key].append(loss)

    if plot:
        plot_test_losses(losses_test)

    return None

def test_rel_err(pred_net : PredNet, test_sets : dict, test_rel_errs : dict,
                 n_samples : int, device : torch.device, C : torch.tensor,
                 eps : float, dim : int, plot : bool) -> None:

    """
    """

    for key in test_sets.keys():
        X_test = test_set_sampler(test_sets[key], n_samples).double().to(device)
        rel_err = test_pred_dist(pred_net, X_test, C, eps, dim, plot, key)
        test_rel_errs[key].append(rel_err)
        print(f"Rel err {key}: {rel_err}")

    if plot:
        plot_test_rel_errs(test_rel_errs, path=None)

    return None

def sink2_vs_emd2(X : torch.Tensor, C : torch.Tensor, eps : float, dim : int
                  ) -> float:
    """
    Compare the relative of the Sinkhorn distance and the Wasserstein distance
    for a given dataset and regularization parameter.

    Parameters
    ----------
    X : (n_samples, 2*dim) torch.Tensor
        Pairs of probability distributions.
    C : (dim, dim) torch.Tensor
        Cost matrix.
    eps : float
        Regularization parameter.
    dim : int
        Dimension of the probability distributions.
    
    Returns
    -------
    rel_errs_mean : float
        Mean relative error of the Sinkhorn distance against the Wasserstein 
        distance.
    """


    emds = []
    for x in X:
        emd_mu = x[:dim] / x[:dim].sum()
        emd_nu = x[dim:] / x[dim:].sum()
        emd = ot.emd2(emd_mu, emd_nu, C)
        emds.append(emd)
    emds = torch.tensor(emds)

    sinks = []
    for x in X:
        sink_mu = x[:dim] / x[:dim].sum()
        sink_nu = x[dim:] / x[dim:].sum()
        sink = ot.sinkhorn2(sink_mu, sink_nu, C, eps, numItermax=3000)
        sinks.append(sink)
    sinks = torch.tensor(sinks)

    rel_errs = torch.abs(emds - sinks) / emds
    rel_errs_mean = rel_errs.mean().item()
    
    return rel_errs_mean