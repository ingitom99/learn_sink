"""
Auxiliary functions for testing the performance of the predictive network.
"""

import torch
import ot 
from sinkhorn import sink_vec
from utils import plot_XPT, test_set_sampler, plot_test_losses, plot_test_rel_errs
import matplotlib.pyplot as plt
from tqdm import tqdm
from nets import PredNet

def test_warmstart(pred_net : PredNet, test_sets : dict, test_emds, C : torch.Tensor,
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

    for key in test_sets.keys():
        X = test_sets[key]
        emds = test_emds[key]

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