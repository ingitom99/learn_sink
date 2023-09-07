import numpy as np
import torch

def get_perm(x):
    x = x.detach().cpu().numpy()
    return np.argsort(x)

def get_perm_C(mu, nu, C):

    sigma_mu = get_perm(mu)
    sigma_nu = get_perm(nu)
    C_perm = C[sigma_mu][:, sigma_nu]
    return C_perm

def dualSort(mu, nu, C, niters=3):
    C_perm = get_perm_C(mu, nu, C)
    n = len(mu)
    f = torch.zeros_like(mu)
    for _ in range(niters):
        for i in range(n):
            vals = torch.ones_like(mu)
            for j in range(n):
                vals[j] = C_perm[i,j] - C_perm[j,j] + f[j]
            f[i] = torch.min(vals)
    return f