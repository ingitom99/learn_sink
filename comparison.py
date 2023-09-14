"""
comparison.py
-------------

Auxiliary functions for comparing the performance of the predictive network to
other state-of-the-art methods.
"""

import torch
from tqdm import tqdm
from nets import PredNet
from sinkhorn import MCV
from geometry import get_cloud
import jax.numpy as jnp

from ott.geometry.geometry import Geometry
from ott.problems.linear import linear_problem
from ott.initializers.linear import initializers
from ott.geometry import pointcloud

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
