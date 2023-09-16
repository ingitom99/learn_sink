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

import numpy as np
import jax.numpy as jnp
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
                    eps: float, dim : int, device : str) -> tuple[list, list]:
    
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
            for i in tqdm(range(len(X))):
                x = X[i]
                mu = x[:dim]
                nu = x[dim:]
                mu_jax = jnp.array(mu.cpu().detach().numpy())
                nu_jax = jnp.array(nu.cpu().detach().numpy())
                MCVs_pred = []
                MCVs_ones = []
                MCVs_gauss = []
                v_pred = V_pred[i]
                v_ones = torch.ones_like(v_pred)
                prob = linear_problem.LinearProblem(geom, mu_jax, nu_jax)
                u_gauss = torch.tensor(np.array(initer.init_dual_a(prob,
                                                    False))).double().to(device)
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

def test_warmstart_sink(pred_net : PredNet, test_sets : dict, test_sinks : dict,
                        C : torch.Tensor, eps: float, dim : int,
                        device : str) -> tuple[list, list]:

    length = int(dim**.5)
    geom = get_geom(length, eps)
    initer = initializers.GaussianInitializer()
    test_warmstarts_sink = {}

    with torch.no_grad():
        for key in test_sets.keys():

            print(f'Testing warmstart sink {key}')
            X = test_sets[key]
            K = torch.exp(C/-eps)
            rel_errs_pred_all = torch.zeros(len(X), 1000)
            rel_errs_ones_all = torch.zeros(len(X), 1000)
            rel_errs_gauss_all = torch.zeros(len(X), 1000)
            V_pred = torch.exp(pred_net(X))

            for i in tqdm(range(len(X))):

                x = X[i]
                sink = test_sinks[key][i]
                mu = x[:dim]
                nu = x[dim:]
                mu_jax = jnp.array(mu.cpu().detach().numpy())
                nu_jax = jnp.array(nu.cpu().detach().numpy())

                rel_errs_pred = []
                rel_errs_ones = []
                rel_errs_gauss = []

                v_pred = V_pred[i]
                v_ones = torch.ones_like(v_pred)
                prob = linear_problem.LinearProblem(geom, mu_jax, nu_jax)
                u_gauss = torch.tensor(np.array(initer.init_dual_a(prob,
                                                False))).double().to(device)

                for _ in range(1000):

                    u_pred = mu / (K @ v_pred)
                    v_pred = nu / (K.T @ u_pred)
                    G_pred = torch.diag(u_pred)@K@torch.diag(v_pred)
                    dist_pred = torch.trace(C.T@G_pred)
                    rel_err_pred = torch.abs(sink - dist_pred) / sink
                    rel_errs_pred.append(rel_err_pred)

                    u_ones = mu / (K @ v_ones)
                    v_ones = nu / (K.T @ u_ones)
                    G_ones = torch.diag(u_ones)@K@torch.diag(v_ones)
                    dist_ones = torch.trace(C.T@G_ones)
                    rel_err_ones = torch.abs(sink - dist_ones) / sink
                    rel_errs_ones.append(rel_err_ones)

                    v_gauss = nu / (K.T @ u_gauss)
                    u_gauss = mu / (K @ v_gauss)
                    G_gauss = torch.diag(u_gauss)@K@torch.diag(v_gauss)
                    dist_gauss = torch.trace(C.T@G_gauss)
                    rel_err_gauss = torch.abs(sink - dist_gauss) / sink
                    rel_errs_gauss.append(rel_err_gauss)

                rel_errs_pred = torch.tensor(rel_errs_pred)
                rel_errs_ones = torch.tensor(rel_errs_ones)
                rel_errs_gauss = torch.tensor(rel_errs_gauss)

                rel_errs_pred_all[i] = rel_errs_pred
                rel_errs_ones_all[i] = rel_errs_ones
                rel_errs_gauss_all[i] = rel_errs_gauss
            
            test_warmstarts_sink[key] = (rel_errs_pred_all.mean(dim=0),
                                        rel_errs_ones_all.mean(dim=0),
                                        rel_errs_gauss_all.mean(dim=0))
        
    return test_warmstarts_sink

def test_warmstart_emd(pred_net : PredNet, test_sets : dict, test_emds : dict,
                        C : torch.Tensor, eps: float, dim : int,
                        device : str) -> tuple[list, list]:

    length = int(dim**.5)
    geom = get_geom(length, eps)
    initer = initializers.GaussianInitializer()
    test_warmstarts_emd = {}

    with torch.no_grad():
        for key in test_sets.keys():

            print(f'Testing warmstart emd {key}')
            X = test_sets[key]
            K = torch.exp(C/-eps)
            rel_errs_pred_all = torch.zeros(len(X), 1000)
            rel_errs_ones_all = torch.zeros(len(X), 1000)
            rel_errs_gauss_all = torch.zeros(len(X), 1000)
            V_pred = torch.exp(pred_net(X))

            for i in tqdm(range(len(X))):

                x = X[i]
                emd = test_emds[key][i]
                mu = x[:dim]
                nu = x[dim:]
                mu_jax = jnp.array(mu.cpu().detach().numpy())
                nu_jax = jnp.array(nu.cpu().detach().numpy())

                rel_errs_pred = []
                rel_errs_ones = []
                rel_errs_gauss = []

                v_pred = V_pred[i]
                v_ones = torch.ones_like(v_pred)
                prob = linear_problem.LinearProblem(geom, mu_jax, nu_jax)
                u_gauss = torch.tensor(np.array(initer.init_dual_a(prob,
                                                False))).double().to(device)

                for _ in range(1000):

                    u_pred = mu / (K @ v_pred)
                    v_pred = nu / (K.T @ u_pred)
                    G_pred = torch.diag(u_pred)@K@torch.diag(v_pred)
                    dist_pred = torch.trace(C.T@G_pred)
                    rel_err_pred = torch.abs(emd - dist_pred) / emd
                    rel_errs_pred.append(rel_err_pred)

                    u_ones = mu / (K @ v_ones)
                    v_ones = nu / (K.T @ u_ones)
                    G_ones = torch.diag(u_ones)@K@torch.diag(v_ones)
                    dist_ones = torch.trace(C.T@G_ones)
                    rel_err_ones = torch.abs(emd - dist_ones) / emd
                    rel_errs_ones.append(rel_err_ones)

                    v_gauss = nu / (K.T @ u_gauss)
                    u_gauss = mu / (K @ v_gauss)
                    G_gauss = torch.diag(u_gauss)@K@torch.diag(v_gauss)
                    dist_gauss = torch.trace(C.T@G_gauss)
                    rel_err_gauss = torch.abs(emd - dist_gauss) / emd
                    rel_errs_gauss.append(rel_err_gauss)

                rel_errs_pred = torch.tensor(rel_errs_pred)
                rel_errs_ones = torch.tensor(rel_errs_ones)
                rel_errs_gauss = torch.tensor(rel_errs_gauss)

                rel_errs_pred_all[i] = rel_errs_pred
                rel_errs_ones_all[i] = rel_errs_ones
                rel_errs_gauss_all[i] = rel_errs_gauss
            
            test_warmstarts_emd[key] = (rel_errs_pred_all.mean(dim=0),
                                        rel_errs_ones_all.mean(dim=0),
                                        rel_errs_gauss_all.mean(dim=0))
        
    return test_warmstarts_emd
    
def test_warmstart_sink_t(t : int, pred_net : PredNet, test_sets : dict, test_sinks : dict,
                        C : torch.Tensor, eps: float, dim : int,
                        device : str) -> tuple[list, list]:

    length = int(dim**.5)
    geom = get_geom(length, eps)
    initer = initializers.GaussianInitializer()

    test_warmstarts_sink_t = {}

    with torch.no_grad():
        for key in test_sets.keys():

            print(f'Testing warmstart sink {key}')
            X = test_sets[key]
            K = torch.exp(C/-eps)
            rel_errs_pred = torch.zeros(len(X))
            rel_errs_ones = torch.zeros(len(X))
            rel_errs_gauss = torch.zeros(len(X))
            V_pred = torch.exp(pred_net(X))

            for i in tqdm(range(len(X))):

                x = X[i]
                sink = test_sinks[key][i]
                mu = x[:dim]
                nu = x[dim:]
                mu_jax = jnp.array(mu.cpu().detach().numpy())
                nu_jax = jnp.array(nu.cpu().detach().numpy())

                v_pred = V_pred[i]
                v_ones = torch.ones_like(v_pred)
                prob = linear_problem.LinearProblem(geom, mu_jax, nu_jax)
                u_gauss = torch.tensor(np.array(initer.init_dual_a(prob,
                                                False))).double().to(device)

                for _ in range(t+1):

                    u_pred = mu / (K @ v_pred)
                    v_pred = nu / (K.T @ u_pred)
                    
                    u_ones = mu / (K @ v_ones)
                    v_ones = nu / (K.T @ u_ones)

                    v_gauss = nu / (K.T @ u_gauss)
                    u_gauss = mu / (K @ v_gauss)                
                
                G_pred = torch.diag(u_pred)@K@torch.diag(v_pred)
                dist_pred = torch.trace(C.T@G_pred)
                rel_err_pred = torch.abs(sink - dist_pred) / sink
                rel_errs_pred[i] = rel_err_pred

                G_ones = torch.diag(u_ones)@K@torch.diag(v_ones)
                dist_ones = torch.trace(C.T@G_ones)
                rel_err_ones = torch.abs(sink - dist_ones) / sink
                rel_errs_ones[i] = rel_err_ones 

                G_gauss = torch.diag(u_gauss)@K@torch.diag(v_gauss)
                dist_gauss = torch.trace(C.T@G_gauss)
                rel_err_gauss = torch.abs(sink - dist_gauss) / sink
                rel_errs_gauss[i] = rel_err_gauss
            
            test_warmstarts_sink_t[key] = (rel_errs_pred,
                                        rel_errs_ones,
                                        rel_errs_gauss)
        
    return test_warmstarts_sink_t


def test_warmstart_MCV_t(t : int, pred_net : PredNet, test_sets : dict, C : torch.Tensor,
                    eps: float, dim : int, device : str) -> tuple[list, list]:
    
    length = int(dim**.5)
    geom = get_geom(length, eps)
    initer = initializers.GaussianInitializer()

    test_warmstarts_MCV_t = {}
    with torch.no_grad():
        for key in test_sets.keys():
            print(f'Testing warmstart MCV {key}')
            X = test_sets[key]
            # Initiliazing Sinkhorn algorithm
            K = torch.exp(C/-eps)
            MCVs_pred = torch.zeros(len(X))
            MCVs_ones = torch.zeros(len(X))
            MCVs_gauss = torch.zeros(len(X))
            V_pred = torch.exp(pred_net(X))
            for i in tqdm(range(len(X))):
                x = X[i]
                mu = x[:dim]
                nu = x[dim:]
                mu_jax = jnp.array(mu.cpu().detach().numpy())
                nu_jax = jnp.array(nu.cpu().detach().numpy())
                v_pred = V_pred[i]
                v_ones = torch.ones_like(v_pred)
                prob = linear_problem.LinearProblem(geom, mu_jax, nu_jax)
                u_gauss = torch.tensor(np.array(initer.init_dual_a(prob,
                                                    False))).double().to(device)
                for _ in range(t+1):

                    u_pred = mu / (K @ v_pred)
                    v_pred = nu / (K.T @ u_pred)

                    u_ones = mu / (K @ v_ones)
                    v_ones = nu / (K.T @ u_ones)

                    v_gauss = nu / (K.T @ u_gauss)
                    u_gauss = mu / (K @ v_gauss)

                G_pred = torch.diag(u_pred)@K@torch.diag(v_pred)
                MCV_pred = MCV(mu, nu, G_pred)
                MCVs_pred[i] = MCV_pred

                G_ones = torch.diag(u_ones)@K@torch.diag(v_ones)
                MCV_ones = MCV(mu, nu, G_ones)
                MCVs_ones[i] = MCV_ones

                G_gauss = torch.diag(u_gauss)@K@torch.diag(v_gauss)
                MCV_gauss = MCV(mu, nu, G_gauss)
                MCVs_gauss[i] = MCV_gauss
            
            test_warmstarts_MCV_t[key] = (MCVs_pred, MCVs_ones, MCVs_gauss)
        
    return test_warmstarts_MCV_t


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

