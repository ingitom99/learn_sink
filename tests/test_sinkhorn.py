import torch
import sys

# Ensure src module is importable
sys.path.append('src')

from sinkhorn import sink, sink_vec


def test_sink_vec_matches_sink():
    torch.manual_seed(0)
    n_samples = 3
    dim = 5

    MU = torch.rand(n_samples, dim)
    NU = torch.rand(n_samples, dim)
    MU = MU / MU.sum(dim=1, keepdim=True)
    NU = NU / NU.sum(dim=1, keepdim=True)

    C = torch.rand(dim, dim)
    eps = 0.1
    V0 = torch.ones_like(MU)
    max_iter = 50

    u_list = []
    v_list = []
    for i in range(n_samples):
        u, v, _, _ = sink(MU[i], NU[i], C, eps, V0[i], max_iter)
        u_list.append(u)
        v_list.append(v)
    U_expected = torch.stack(u_list)
    V_expected = torch.stack(v_list)

    U, V = sink_vec(MU, NU, C, eps, V0, max_iter)

    assert torch.allclose(U, U_expected, atol=1e-6, rtol=1e-5)
    assert torch.allclose(V, V_expected, atol=1e-6, rtol=1e-5)
