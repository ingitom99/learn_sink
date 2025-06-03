"""
test_funcs.py
-------------

Auxiliary functions for testing the performance of the predictive network.
"""

import torch
from tqdm import tqdm
from src.nets import PredNet
from src.sinkhorn import MCV
from src.geometry import get_cloud

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
        #nu = x[dim:] / x[dim:].sum()
        v = torch.exp(p)
        u = mu / (K @ v)
        #v = nu / (K.T @ u)
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
        #v_pred = nu / (K.T @ u_pred)
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
                    eps: float, dim : int,
                    device : str, niter : int) -> tuple[list, list]:
    
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
            MCVs_pred_all = torch.zeros(len(X), niter)
            MCVs_ones_all = torch.zeros(len(X), niter)
            MCVs_gauss_all = torch.zeros(len(X), niter)
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
                for _ in range(niter):

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
                        device : str, niter : int) -> tuple[list, list]:

    length = int(dim**.5)
    geom = get_geom(length, eps)
    initer = initializers.GaussianInitializer()
    test_warmstarts_sink = {}

    with torch.no_grad():
        for key in test_sets.keys():

            print(f'Testing warmstart sink {key}')
            X = test_sets[key]
            K = torch.exp(C/-eps)
            rel_errs_pred_all = torch.zeros(len(X), niter)
            rel_errs_ones_all = torch.zeros(len(X), niter)
            rel_errs_gauss_all = torch.zeros(len(X), niter)
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

                for _ in range(niter):

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
                        device : str, niter : int) -> tuple[list, list]:

    length = int(dim**.5)
    geom = get_geom(length, eps)
    initer = initializers.GaussianInitializer()
    test_warmstarts_emd = {}

    with torch.no_grad():
        for key in test_sets.keys():

            print(f'Testing warmstart emd {key}')
            X = test_sets[key]
            K = torch.exp(C/-eps)
            rel_errs_pred_all = torch.zeros(len(X), niter)
            rel_errs_ones_all = torch.zeros(len(X), niter)
            rel_errs_gauss_all = torch.zeros(len(X), niter)
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

                for _ in range(niter):

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
                if t == 0:
                    u_pred = mu / (K @ v_pred)
                    u_ones = mu / (K @ v_ones)
                    v_gauss = nu / (K.T @ u_gauss)

                else:
                    for _ in range(t):

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
                
                if t == 0:
                    u_pred = mu / (K @ v_pred)
                    u_ones = mu / (K @ v_ones)
                    v_gauss = nu / (K.T @ u_gauss)
                else:
                    for _ in range(t):

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
        #v_pred = nu / (K.T @ u_pred)
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

import time

class Timer:
    def __init__(self):
        self.start_time = None
        self.total_elapsed_time = 0

    def start(self):
        if self.start_time is None:
            self.start_time = time.time()
            print("Timer started.")

    def stop(self):
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            self.total_elapsed_time += elapsed_time
            self.start_time = None

    def restart(self):
        self.start_time = time.time()
        print("Timer restarted.")


def test_comp_time(pred_net : PredNet, test_sets : dict, test_sinks : dict,
                        C : torch.Tensor, eps: float, dim : int,
                        device : str, niter : int) -> tuple[list, list]:

    test_comp_times = {}

    with torch.no_grad():
        for key in test_sets.keys():
                
            n = len(test_sets[key])
            sinks = test_sinks[key]
            C_batched = C.unsqueeze(0).repeat(n, 1, 1)
            print(f'Testing comp time {key}')

            timer_pred = Timer()
            timer_ones = Timer()
            
            X = test_sets[key]
            MU = X[:,:dim]
            NU = X[:,dim:]
            K = torch.exp(C/-eps)
            rel_errs_pred = torch.zeros(len(X), niter)
            rel_errs_ones = torch.zeros(len(X), niter)

            comp_times_pred = torch.zeros(niter)
            comp_times_ones = torch.zeros(niter)

            pred_net.eval()
            warmup = pred_net(X)

            timer_pred.start()
            V_pred = torch.exp(pred_net(X))
            timer_pred.stop()

            timer_ones.start()
            V_ones = torch.ones_like(V_pred)
            timer_ones.stop()

            for i in tqdm(range(niter)):

                timer_pred.start()
                U_pred = MU / (K @ V_pred.T).T
                V_pred = NU / (K.T @ U_pred.T).T
                timer_pred.stop()
                comp_times_pred[i] = timer_pred.total_elapsed_time

                timer_ones.start()
                U_ones = MU / (K @ V_ones.T).T
                V_ones = NU / (K.T @ U_ones.T).T
                timer_ones.stop()
                comp_times_ones[i] = timer_ones.total_elapsed_time

                G_pred = torch.matmul(torch.matmul(torch.diag_embed(U_pred), K), torch.diag_embed(V_pred))
                dists_pred = torch.trace(torch.matmul(C_batched.transpose(1, 2), G_pred))
                rel_errs_pred_i = torch.abs(sinks - dists_pred) / sinks
                rel_errs_pred[:, i] = rel_errs_pred_i

                G_ones = torch.matmul(torch.matmul(torch.diag_embed(U_ones), K), torch.diag_embed(V_ones))
                dists_ones = torch.trace(torch.matmul(C_batched.transpose(1, 2), G_ones))
                rel_errs_ones_i = torch.abs(sinks - dists_ones) / sinks
                rel_errs_ones[:, i] = rel_errs_ones_i

            test_comp_times[key] = {'pred' : (comp_times_pred, rel_errs_pred),
                                    'ones' : (comp_times_ones, rel_errs_ones)}

    return test_comp_times


def test_warmstart_sink(pred_net : PredNet, test_sets : dict, test_sinks : dict,
                        C : torch.Tensor, eps: float, dim : int,
                        device : str, niter : int) -> tuple[list, list]:

    length = int(dim**.5)
    geom = get_geom(length, eps)
    initer = initializers.GaussianInitializer()
    test_warmstarts_sink = {}

    with torch.no_grad():
        for key in test_sets.keys():

            print(f'Testing warmstart sink {key}')
            X = test_sets[key]
            K = torch.exp(C/-eps)
            rel_errs_pred_all = torch.zeros(len(X), niter)
            rel_errs_ones_all = torch.zeros(len(X), niter)
            rel_errs_gauss_all = torch.zeros(len(X), niter)
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

                for _ in range(niter):

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

