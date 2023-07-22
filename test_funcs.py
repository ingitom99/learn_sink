"""
test_funcs.py
-------------

Auxiliary functions for testing the performance of the predictive network.
"""

import torch
from tqdm import tqdm
from nets import PredNet

def get_pred_dists(P : torch.Tensor, X : torch.Tensor, eps : float,
                   C : torch.Tensor, dim : int) -> torch.Tensor:
    
    """
    Get the predicted Sinkhorn distances for a set of pairs of probability
    distributions.

    Parameters
    ----------
    P : (n_samples, dim) torch.Tensor
        Predicted 'V' scaling factors to be used as V0
    X : (n_samples, 2*dim) torch.Tensor
        Pairs of probability distributions.
    eps : float
        Regularization parameter.
    C : (dim, dim) torch.Tensor
        Cost matrix.
    dim : int
        Dimension of the probability distributions.

    Returns
    -------
    dists : (n_samples,) torch.Tensor
        Predicted Sinkhorn distances.
    """

    dists = []
    K = torch.exp(-C/eps)
    for p, x in zip(P, X):
        mu = x[:dim] / x[:dim].sum()
        nu = x[dim:] / x[dim:].sum()
        v = torch.exp(p)
        u = mu / (K @ v)
        v = nu / (K.T @ u)
        G = torch.diag(u)@K@torch.diag(v)    
        dist = torch.trace(C.T@G)
        dists.append(dist)
    dists = torch.tensor(dists)
    return dists
    
def test_warmstart_emd(pred_net : PredNet, test_sets : dict, test_emds,
                   C : torch.Tensor, eps: float,
                   dim : int) -> tuple[list, list]:
  
    """
    Track the performance of the predictive network as a 'warmstart' for the 
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

    test_warmstarts_emd = {}
    with torch.no_grad():
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
            for _ in tqdm(range(1000)):

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

                # Calculating the Sinkhorn distances for ones V0
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
            
            test_warmstarts_emd[key] = (rel_err_means_pred, rel_err_means_ones)

    return test_warmstarts_emd

def MCV(mu : torch.Tensor, nu : torch.Tensor, G : torch.Tensor) -> float:
    """
    Compute the Marginal Constraint Violation (MCV) for a given optimal
    transport plan (G) and its corresponding distributions (mu and nu).

    Parameters
    ----------
    mu : (dim,) torch.Tensor
        First probability distribution.
    nu : (dim,) torch.Tensor
        Second probability distribution.
    G : (dim, dim) torch.Tensor
        Optimal transport plan.

    Returns
    -------
    MCV : float
        Marginal Constraint Violation.
    """
    
    ones = torch.ones_like(mu)
    term_one = ones.T @ G  - nu.T
    term_two = G @ ones - mu
    MCV = (torch.linalg.norm(term_one, ord=1) + torch.linalg.norm(term_two,
                                                                  ord=1))/2
    return MCV

def test_warmstart_MCV(pred_net : PredNet, test_sets : dict, C : torch.Tensor,
                       eps: float, dim : int):
    

    test_warmstarts_MCV = {}
    with torch.no_grad():
        for key in test_sets.keys():
            print(f'Testing warmstart MCV {key}')
            X = test_sets[key]
            # Initiliazing Sinkhorn algorithm
            K = torch.exp(C/-eps)
            MCVs_pred_all = torch.zeros(len(X), 1000)
            MCVs_ones_all = torch.zeros(len(X), 1000)
            V_pred = torch.exp(pred_net(X))
            for (i, x) in tqdm(enumerate(X)):
                mu = x[:dim]
                nu = x[dim:]
                MCVs_pred = []
                MCVs_ones = []
                v_pred = V_pred[i]
                v_ones = torch.ones_like(v_pred)
                for j in range(1000):
                    u_pred = mu / (K @ v_pred)
                    v_pred = nu / (K.T @ u_pred)
                    G_pred = torch.diag(u_pred)@K@torch.diag(v_pred)
                    MCV_pred = MCV(mu, nu, G_pred)
                    print(f'{j}: {MCV_pred.item()}')
                    MCVs_pred.append(MCV_pred)

                    u_ones = mu / (K @ v_ones)
                    v_ones = nu / (K.T @ u_ones)
                    G_ones = torch.diag(u_ones)@K@torch.diag(v_ones)
                    MCV_ones = MCV(mu, nu, G_ones)
                    MCVs_ones.append(MCV_ones)
                MCVs_pred = torch.tensor(MCVs_pred)
                MCVs_ones = torch.tensor(MCVs_ones)
                MCVs_pred_all[i] = MCVs_pred
                MCVs_ones_all[i] = MCVs_ones
            
            test_warmstarts_MCV[key] = (MCVs_pred_all.mean(dim=0),
                                        MCVs_ones_all.mean(dim=0))
        
    return test_warmstarts_MCV