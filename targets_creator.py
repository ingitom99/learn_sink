import torch

def get_X_T(X, U, V, n_batch, dim, nan_mask, device, center=True):
    length = int(dim**0.5)
    T = torch.zeros(8*n_batch, dim).double().to(device)
    MU = torch.zeros(8*n_batch, dim).double().to(device)
    NU = torch.zeros(8*n_batch, dim).double().to(device)
    for flip_i, flip in enumerate([False, True]):
        for rot_i, rot in enumerate([0, 1, 2, 3]):
            mini_i = flip_i*4 + rot_i
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

            T[mini_i*n_batch:(mini_i+1)*n_batch] = T_curr
            MU[mini_i*n_batch:(mini_i+1)*n_batch] = MU_curr
            NU[mini_i*n_batch:(mini_i+1)*n_batch] = NU_curr

    if center:
        T = T - torch.unsqueeze(T.mean(dim=1), 1).repeat(1, dim)
    X = torch.cat((MU, NU), dim=1)
    perm = torch.randperm(len(X))
    return X[perm], T[perm]