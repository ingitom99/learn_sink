import torch

def hilb_proj_loss(U, V):

    """
    Compute the mean Hilbert projective loss between pairs of vectors.

    Parameters
    ----------
    U : (n_samples, dim) torch.Tensor
        First set of vectors.
    V : (n_samples, dim) torch.Tensor
        Second set of vectors.
    
    Returns
    -------
    loss : float
        Loss value.
    """

    diff = U - V
    spectrum = torch.max(diff, dim=1)[0] - torch.min(diff, dim=1)[0]
    loss = spectrum.mean()

    return loss