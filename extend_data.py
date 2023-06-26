"""
extend_data.py
--------------

Function(s) for extending data and corresponding Sinkhorn scaling factors to 
account for rotational invariance and symmetry of the Sinkhorn algorithm.
"""

import torch

def extend(X : torch.Tensor, U : torch.Tensor, V : torch.Tensor,
           dim : int, nan_mask : torch.Tensor,
           device : torch.device) -> torch.Tensor:

    """
    Extend data and corresponding Sinkhorn scaling factors to account for 
    rotational invariance and symmetry of the Sinkhorn algorithm.

    Parameters
    ----------
    X : (n, 2 * dim) torch.Tensor
        The data.
    U : (n, dim) torch.Tensor
        The 'U' Sinkhorn scaling factors.
    V : (n, dim) torch.Tensor
        The 'V' Sinkhorn scaling factors.
    dim : int
        The dimension of the data.
    nan_mask : (n,) torch.Tensor
        A mask of which scaling factors do not contain NaNs.
    device : torch.device
        The device to put the tensors on.

    Returns
    -------
    X : (4 * n_batch, 2 * dim) torch.Tensor
        The extended data.

    T : (4 * n_batch, dim) torch.Tensor
        The extended, centered 'V' Sinkhorn scaling factors (i.e. the targets).
    """

    n_nonan = nan_mask.sum()
    length = int(dim**0.5)

    T = torch.zeros(4*n_nonan, dim).double().to(device)
    MU = torch.zeros(4*n_nonan, dim).double().to(device)
    NU = torch.zeros(4*n_nonan, dim).double().to(device)

    for flip_i, flip in enumerate([False, True]):
        for rot_i, rot in enumerate([0, 2]):

            mini_i = flip_i*2 + rot_i

            if flip:
                MU_curr = X[:, dim:][nan_mask]
                NU_curr = X[:, :dim][nan_mask]

            else:
                MU_curr = X[:, :dim][nan_mask]
                NU_curr = X[:, dim:][nan_mask]

            if (rot != 0):
                MU_curr = torch.rot90(MU_curr.reshape((-1,length, length)),
                    k=rot, dims=(1, 2)).reshape((-1, dim))
                NU_curr = torch.rot90(NU_curr.reshape(-1,length, length),
                    k=rot, dims=(1, 2)).reshape((-1, dim))
            

            if flip:
                T_curr = U[nan_mask]
            else:
                T_curr = V[nan_mask]
            
            if (rot != 0):
                T_curr = torch.rot90(T_curr.reshape((-1,length, length)),
                    k=rot, dims=(1, 2)).reshape((-1,dim))

            T[mini_i*n_nonan:(mini_i+1)*n_nonan] = T_curr
            MU[mini_i*n_nonan:(mini_i+1)*n_nonan] = MU_curr
            NU[mini_i*n_nonan:(mini_i+1)*n_nonan] = NU_curr

    T = T - torch.unsqueeze(T.mean(dim=1), 1).repeat(1, dim)
    X = torch.cat((MU, NU), dim=1)

    return X, T