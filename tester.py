
import torch
from test_funcs import test_pred_loss, test_pred_dist
from utils import test_sampler, plot_test_losses, plot_test_rel_errs

def test_loss(pred_net : torch.nn.module, test_set_dict : dict, n_samples : int,
              losses_test : dict, device : torch.device, C : torch.Tensor,
              eps : float, dim : int, loss_func : function, plot : bool):
    """
    """

    for key in test_set_dict.keys():
        X_test = test_sampler(test_set_dict[key], n_samples).double().to(device)
        loss = test_pred_loss(pred_net, X_test, C, eps, dim, loss_func, 5000, False)
        losses_test[key].append(loss)

    if plot:
        plot_test_losses(losses_test)

    return None

def test_rel_err(pred_net, test_set_dict, n_samples, rel_errs, device, C, eps,
                   dim, plot):

    """
    """

    for key in test_set_dict.keys():

        X_test = test_sampler(test_set_dict[key], n_samples).double().to(device)
        rel_err = test_pred_dist(pred_net, X_test, C, eps, dim, plot, key)
        rel_errs[key].append(rel_err)
        print(f"Rel err {key}: {rel_err}")

    if plot:
        plot_test_rel_errs(rel_errs)

    return None

    


