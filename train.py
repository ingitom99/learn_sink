"""
train.py
-------

The evolutionary algorithm(s) for training the neural network(s).
"""

# Imports
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from test_funcs import test_warmstart, get_pred_dists
from sinkhorn import sink_vec
from plot import plot_warmstarts, plot_train_losses, plot_test_losses, plot_test_rel_errs_emd, plot_XPT
from data_funcs import rand_noise
from net import PredNet

def the_hunt(
        pred_net : PredNet,
        loss_func : callable,
        cost_mat : torch.Tensor,    
        eps : float,
        dust_const : float,
        mutation_sigma : float,
        dim : int,
        device : torch.device,
        test_sets : dict,
        test_emds : dict,
        test_T : dict,
        n_loops : int,
        n_batch : int,
        lr : float,
        lr_fact : float,
        test_iter : int,
        results_folder : str,
        checkpoint : int,
        ) -> tuple[dict, dict, dict]:
    
    """
    The puma sharpens its teeth against the test of mutation!

    Parameters
    ----------
    gen_net : GenNet
        The generative neural network.
    pred_net : PredNet
        The predictive neural network.
    loss_func : callable
        The loss function.
    cost_mat : torch.Tensor
        The cost matrix.
    eps : float
        The entropic regularization parameter.
    dust_const : float
        The dusting constant.
    mutation_sigma : float
        The mutation Gaussian random noise standard deviation.
    dim : int
        The dimension of the data.
    device : torch.device
        The device on which to run the algorithm.
    test_sets : dict
        The test sets.
    test_emds : dict
        The ot.emd2() values for the test sets.
    test_T : dict
        The test set target log-centered Sinkhorn scaling factors.
    n_loops : int
        The number of loops of the training algorithm.
    n_batch : int
        The batch size.
    lr : float
        The initial learning rate.
    lr_fact : float
        The learning rate decay factor.
    test_iter : int
        The number of iterations between testing.
    results_folder : str
        The folder in which to save the results.
    checkpoint : int
        The number of iterations between checkpoints.
    
    Returns
    -------
    train_losses : list
        The training losses.
    test_losses : dict
        The test losses.
    test_rel_errs_emd : dict
        The test relative errors against the ot.emd2() values.
    warmstarts : dict
        The data tracking use of the predictive network as an initializtion
        for the Sinkhorn algorithm.
    """

    # Initializing loss, relative error and warmstart collectors
    train_losses = []
    test_losses = {}
    test_rel_errs_emd = {}
    warmstarts = {}
    for key in test_sets.keys():
        test_losses[key] = []
        test_rel_errs_emd[key] = []

    # Initializing optimizers
    pred_optimizer = torch.optim.SGD(pred_net.parameters(), lr=lr)

    # Initializing learning rate schedulers
    pred_scheduler = torch.optim.lr_scheduler.ExponentialLR(pred_optimizer,
                                                            gamma=lr_fact)
    
    # Initializing the worst point
    x_worst = rand_noise(1, dim, dust_const,
                         True).reshape(-1).double().to(device)

    for i in tqdm(range(n_loops)):

        # Testing predictive neural net
        if ((i+1) % test_iter == 0) or (i == 0):

            print(f'Testing pred net at iter: {i+1}')

            # Setting networks to eval mode
            pred_net.eval()

            for key in test_sets.keys():

                X_test = test_sets[key]
                T = test_T[key]
                emd = test_emds[key]

                P = pred_net(X_test)
                loss = loss_func(P, T).mean()
                
                pred_dist = get_pred_dists(P, X_test, eps, cost_mat, dim)

                rel_errs_emd = torch.abs(pred_dist - emd) / emd

                test_rel_errs_emd[key].append(rel_errs_emd.mean().item())
                test_losses[key].append(loss.item())
                 
        # Training Section

        # Setting networks to train mode
        pred_net.train()

        # Creating data

        X = x_worst.repeat(n_batch, 1)
        X = X + mutation_sigma * torch.randn(n_batch, 2*dim).double().to(device)
        X = torch.nn.functional.relu(X)
        X_MU = X[:, :dim] / X[:, :dim].sum(dim=1, keepdim=True)
        X_MU = X_MU + dust_const
        X_MU = X_MU / X_MU.sum(dim=1, keepdim=True)
        X_NU = X[:, dim:] / X[:, dim:].sum(dim=1, keepdim=True)
        X_NU = X_NU + dust_const
        X_NU = X_NU / X_NU.sum(dim=1, keepdim=True)
        X = torch.cat((X_MU, X_NU), dim=1)

        with torch.no_grad(): 
            V0 = torch.ones_like(X[:, :dim])
            U, V = sink_vec(X[:, :dim], X[:, dim:], cost_mat,
                            eps, V0, 1000)
            U = torch.log(U)
            V = torch.log(V)
            
            nan_mask = ~(torch.isnan(U).any(dim=1) & torch.isnan(
                V).any(dim=1)).to(device)
            V = V - torch.unsqueeze(V.mean(dim=1), 1).repeat(1, dim)     

        X = X[nan_mask]
        T= V[nan_mask]
        P = pred_net(X)
                
        pred_losses = loss_func(P, T)

        # Find worst point
        arg_worst = pred_losses.argmax()

        # Update x_worst
        x_worst = X[arg_worst]

        # Calculate mean loss
        pred_loss = pred_losses.mean()
        
        # Update train_losses
        train_losses.append(pred_loss.item())

        # Update
        pred_optimizer.zero_grad()
        pred_loss.backward(retain_graph=True)
        pred_optimizer.step()

        # Updating learning rates
        pred_scheduler.step()

        if ((i+1) % test_iter == 0) or (i == 0):

            # Plot an example of the data and predictions from current iter
            plot_XPT(X[0], P[0], T[0], dim)

            # Plotting losses and rel errs
            plot_train_losses(train_losses)
            plot_test_losses(test_losses)
            plot_test_rel_errs_emd(test_rel_errs_emd)

            # print current learning rate
            print(f'pred lr: {pred_optimizer.param_groups[0]["lr"]}')

        # Checkpointing
        if ((i+1) % checkpoint == 0):
        
            print(f'Checkpointing at iter: {i+1}')

            # Testing mode
            pred_net.eval()

            # Saving nets
            torch.save(pred_net.state_dict(), f'{results_folder}/puma.pt')

            # Test warmstart
            warmstarts = test_warmstart(pred_net, test_sets, test_emds,
                                        cost_mat, eps, dim)

            # Plot the results
            plot_train_losses(train_losses,
                              f'{results_folder}/train_losses.png')
            plot_test_losses(test_losses, f'{results_folder}/test_losses.png')
            plot_test_rel_errs_emd(test_rel_errs_emd,
                                   f'{results_folder}/test_rel_errs.png')
            plot_warmstarts(warmstarts, results_folder)
        
        if ((i+2) % test_iter == 0) or (i == n_loops-1):
            plt.close('all')


    return train_losses, test_losses, test_rel_errs_emd, warmstarts