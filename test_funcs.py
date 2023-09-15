"""
test_funcs.py
-------------

Auxiliary functions for testing the performance of the predictive network.
"""

import torch
from tqdm import tqdm
from nets import PredNet
from sinkhorn import MCV
from geometry import get_cloud

from ott.geometry.geometry import Geometry
from ott.problems.linear import linear_problem
from ott.initializers.linear import initializers
from ott.geometry import pointcloud

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
            print(f'Testing warmstart emd {key}')
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

def get_mean_mcv(pred_net : PredNet, X: torch.Tensor, C : torch.Tensor,
                 eps: float, dim : int):
    mcvs = []
    K = torch.exp(C/-eps)
    V_pred = torch.exp(pred_net(X))
    for (i, x) in tqdm(enumerate(X)):
        mu = x[:dim]
        nu = x[dim:]
        v_pred = V_pred[i]
        u_pred = mu / (K @ v_pred)
        v_pred = nu / (K.T @ u_pred)
        G_pred = torch.diag(u_pred)@K@torch.diag(v_pred)
        mcv = MCV(mu, nu, G_pred)
    mcvs.append(mcv)
    mcvs = torch.tensor(mcvs)
    return mcvs.mean().item()

def get_geom(n : int, eps : float) -> Geometry:

    """
    Get a geometry object for the optimal transport problem on a 2D grid.

    Parameters
    ----------
    n : int
        Number of points per dimension.
    eps : float
        Regularisation parameter.

    Returns
    -------
    geom : Geometry
        Geometry object for the optimal transport problem on a 2D grid.
    """

    # Generate a 2D grid of n points per dimension
    cloud = get_cloud(n)
    
    geom = pointcloud.PointCloud(cloud, cloud, epsilon=eps)

    return geom

def get_gauss_init(geom : Geometry, mu : jnp.ndarray, nu : jnp.ndarray) -> jnp.ndarray:
    """
    Get a Gaussian initialisation for the dual vector v.

    Parameters
    ----------
    geom : Geometry
        Geometry of the problem.
    mu : jnp.ndarray
        Source distribution.
    nu : jnp.ndarray
        Target distribution.
    eps : float
        Regularisation parameter.

    Returns
    -------
    v : jnp.ndarray
        Gaussian initialisation for the dual vector v.
    """

    init = initializers.GaussianInitializer()
    prob = linear_problem.LinearProblem(geom, mu, nu)
    u = init.init_dual_a(prob, False)
    return u

def test_warmstart_MCV(pred_net : PredNet, test_sets : dict, C : torch.Tensor,
                       eps: float, dim : int):
    
    length = int(dim**.5)
    geom = get_geom(length, eps)
    initer = initializers.GaussianInitializer()

    test_warmstarts_MCV = {}
    with torch.no_grad():
        for key in test_sets.keys():
            print(f'Testing warmstart MCV {key}')
            X = test_sets[key]
            # Initiliazing Sinkhorn algorithm
            K = torch.exp(C/-eps)
            MCVs_pred_all = torch.zeros(len(X), 1000)
            MCVs_ones_all = torch.zeros(len(X), 1000)
            MCVs_gauss_all = torch.zeros(len(X), 1000)
            V_pred = torch.exp(pred_net(X))
            for (i, x) in tqdm(enumerate(X)):
                mu = x[:dim]
                nu = x[dim:]
                mu_jax = jnp.array(mu)
                nu_jax = jnp.array(nu)
                MCVs_pred = []
                MCVs_ones = []
                MCVs_gauss = []
                v_pred = V_pred[i]
                v_ones = torch.ones_like(v_pred)
                prob = linear_problem.LinearProblem(geom, mu_jax, nu_jax)
                u_gauss = initer.init_dual_a(prob, False)
                for _ in range(1000):

                    u_pred = mu / (K @ v_pred)
                    v_pred = nu / (K.T @ u_pred)
                    G_pred = torch.diag(u_pred)@K@torch.diag(v_pred)
                    MCV_pred = MCV(mu, nu, G_pred)
                    MCVs_pred.append(MCV_pred)

                    u_ones = mu / (K @ v_ones)
                    v_ones = nu / (K.T @ u_ones)
                    G_ones = torch.diag(u_ones)@K@torch.diag(v_ones)
                    MCV_ones = MCV(mu, nu, G_ones)
                    MCVs_ones.append(MCV_ones)
                    
                    v_gauss = nu / (K.T @ u_gauss)
                    u_gauss = mu / (K @ v_gauss)
                    G_gauss = torch.diag(u_gauss)@K@torch.diag(v_gauss)
                    MCV_gauss = MCV(mu, nu, G_gauss)
                    MCVs_gauss.append(MCV_gauss)

                MCVs_pred = torch.tensor(MCVs_pred)
                MCVs_ones = torch.tensor(MCVs_ones)
                MCVs_gauss = torch.tensor(MCVs_gauss)
                MCVs_pred_all[i] = MCVs_pred
                MCVs_ones_all[i] = MCVs_ones
                MCVs_gauss_all[i] = MCVs_gauss
            
            test_warmstarts_MCV[key] = (MCVs_pred_all.mean(dim=0),
                                        MCVs_ones_all.mean(dim=0),
                                        MCVs_gauss_all.mean(dim=0))
        
    return test_warmstarts_MCV

def test_warmstart_sink(pred_net : PredNet, test_sets : dict, test_sinks,
                   C : torch.Tensor, eps: float,
                   dim : int) -> tuple[list, list]:
  
    """
    """
    length = int(dim**.5)
    geom = get_geom(length, eps)
    initer = initializers.GaussianInitializer()

    test_warmstarts_sink = {}
    with torch.no_grad():
        for key in test_sets.keys():
            print(f'Testing warmstart sink {key}')
            X = test_sets[key]
            sinks = test_sinks[key]

            # Initiliazing Sinkhorn algorithm
            K = torch.exp(C/-eps)
            V_pred = torch.exp(pred_net(X))
            V_ones = torch.ones_like(V_pred)
            U_gauss = torch.zeros_like(V_pred)
            MU = X[:, :dim]
            NU = X[:, dim:]

            for (i, x) in enumerate(X):
                mu = x[:dim]
                nu = x[dim:]
                mu_jax = jnp.array(mu)
                nu_jax = jnp.array(nu)
                prob = linear_problem.LinearProblem(geom, mu_jax, nu_jax)
                u_gauss = initer.init_dual_a(prob, False)
                U_gauss[i] = torch.tensor(u_gauss)

            rel_err_means_pred = []
            rel_err_means_ones = []
            rel_err_means_gauss = []

            # Looping over 1000 iterations of Sinkhorn algorithm
            for _ in tqdm(range(1000)):

            
                # Calculating the Sinkhorn distances for predicted V0
                dists_pred = []
                U_pred = MU / (K @ V_pred.T).T
                V_pred = NU / (K.T @ U_pred.T).T
                for u, v in zip(U_pred, V_pred):
                    G = torch.diag(u)@K@torch.diag(v)
                    dist_pred = torch.trace(C.T@G)
                    dists_pred.append(dist_pred)
                dists_pred = torch.tensor(dists_pred)
                rel_errs_pred = torch.abs(sinks - dists_pred) / sinks
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
                rel_errs_ones = torch.abs(sinks - dists_ones) / sinks
                rel_err_means_ones.append(rel_errs_ones.mean().item())

            
            test_warmstarts_sink[key] = (rel_err_means_pred, rel_err_means_ones)

    return test_warmstarts_sink
