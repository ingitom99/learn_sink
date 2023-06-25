"""
train.py
-------

The algorithm(s) for training the neural network(s).
"""

# Imports
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from test_funcs import test_warmstart, get_pred_dists
from sinkhorn import sink_vec
from plot import plot_warmstarts, plot_train_losses, plot_test_losses, plot_test_rel_errs_emd, plot_test_rel_errs_sink, plot_XPT
from data_funcs import rand_noise
from nets import GenNet, PredNet
from extend_data import extend

def the_hunt(
        gen_net : GenNet,
        pred_net : PredNet,
        loss_func : callable,
        cost_mat : torch.Tensor,    
        eps : float,
        dust_const : float,
        dim_prior : int,
        dim : int,
        device : torch.device,
        test_sets: dict,
        test_emds : dict,
        test_T : dict,
        n_loops : int,
        n_mini_loops_gen : int,
        n_mini_loops_pred : int,
        n_batch : int,
        lr_pred : float,
        lr_gen : float,
        lr_fact_gen : float,
        lr_fact_pred : float,
        learn_gen : bool,
        bootstrapped : bool,
        n_boot : int,
        extend_data : bool,
        test_iter : int,
        results_folder : str,
        checkpoint : int,
        ) -> tuple[dict, dict, dict]:
    
    """
    The puma sharpens its teeth.

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
    dim_prior : int
        The dimension of the prior.
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
        The number of global loops of the training algorithm.
    n_mini_loops_gen : int
        The number of mini loops for the generative neural network.
    n_mini_loops_pred : int
        The number of mini loops for the predictive neural network.
    n_batch : int
        The batch size.
    lr_pred : float
        The learning rate for the predictive neural network.
    lr_gen : float
        The learning rate for the generative neural network.
    lr_fact_gen : float
        The learning rate decay factor for the generative neural network.
    lr_fact_pred : float
        The learning rate decay factor for the predictive neural network.
    learn_gen : bool
        Whether or not to learn and use the generative neural network for data
        generation. If False, a random noise generator is used instead.
    bootstrapped : bool
        Whether or not to use the bootstrapping method for target creation.
    n_boot : int
        The number iterations used to create targets in the bootstrapping
        method.
    extend_data : bool
        Whether or not to extend the data set using rotations and flips.
    test_iter : int
        The number of iterations between testing.
    results_folder : str
        The folder in which to save the results.
    checkpoint : int
        The number of iterations between checkpoints.
    
    Returns
    -------
    train_losses : dict
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
    train_losses = {'gen': [], 'pred': []}
    test_losses = {}
    test_rel_errs_emd = {}
    warmstarts = {}
    for key in test_sets.keys():
        test_losses[key] = []
        test_rel_errs_emd[key] = []

    # Initializing optimizers
    pred_optimizer = torch.optim.SGD(pred_net.parameters(), lr=lr_pred)
    if learn_gen:
        gen_optimizer = torch.optim.SGD(gen_net.parameters(), lr=lr_gen)   

    # Initializing learning rate schedulers
    pred_scheduler = torch.optim.lr_scheduler.ExponentialLR(pred_optimizer,
                                                            gamma=lr_fact_gen)
    if learn_gen:
        gen_scheduler = torch.optim.lr_scheduler.ExponentialLR(gen_optimizer,
                                                               gamma=lr_fact_pred)
    
    for i in tqdm(range(n_loops)):

        # Testing predictive neural net
        if ((i+1) % test_iter == 0) or (i == 0):

            # Setting networks to eval mode
            pred_net.eval()
            if learn_gen:
                gen_net.eval()

            for key in test_sets.keys():

                X_test = test_sets[key]
                T = test_T[key]
                emd = test_emds[key]

                P = pred_net(X_test)
                loss = loss_func(P, T)
                
                pred_dist = get_pred_dists(P, X_test, eps, cost_mat, dim)

                rel_errs_emd = torch.abs(pred_dist - emd) / emd

                test_rel_errs_emd[key].append(rel_errs_emd.mean().item())
                test_losses[key].append(loss.item())

            # Plotting
            plot_train_losses(train_losses)
            plot_test_losses(test_losses)
            plot_test_rel_errs_emd(test_rel_errs_emd)
                 
        # Training Section

        # Setting networks to train mode
        pred_net.train()
        if learn_gen:
            gen_net.train()

        # Training generative neural net
        if learn_gen:
            for _ in range(n_mini_loops_gen):
                
                if extend_data:
                    n_data = n_batch // 4
                else:
                    n_data = n_batch
                prior_sample = torch.randn((n_batch,
                                            2 * dim_prior)).double().to(device)
                X = gen_net(prior_sample) 

                P = pred_net(X)

                with torch.no_grad():
                    if bootstrapped:
                        V0 = torch.exp(P)
                        U, V = sink_vec(X[:, :dim], X[:, dim:], cost_mat,
                                        eps, V0, n_boot)
                        U = torch.log(U)
                        V = torch.log(V)

                    else:
                        V0 = torch.ones_like(X[:, :dim])
                        U, V = sink_vec(X[:, :dim], X[:, dim:],
                                        cost_mat, eps, V0, 1000)
                        U = torch.log(U)
                        V = torch.log(V)

                    nan_mask = ~(torch.isnan(U).any(dim=1) & torch.isnan(
                        V).any(dim=1)).to(device)

                if extend_data:
                    X_gen, T_gen = extend(X, U, V, dim, nan_mask, device)

                else:
                    X_gen = X[nan_mask]
                    V = V - torch.unsqueeze(V.mean(dim=1), 1).repeat(1, dim)
                    T_gen = V[nan_mask]
            
                X = X_gen
                T = T_gen
                P = pred_net(X)

                gen_loss = -loss_func(P, T)
                train_losses['gen'].append(gen_loss.item())

                # Update
                gen_optimizer.zero_grad()
                gen_loss.backward(retain_graph=True)
                gen_optimizer.step()

        # Training predictive neural net
        for _ in range(n_mini_loops_pred):

            if extend_data:
                n_data = n_batch // 4
            else:
                n_data = n_batch

            if learn_gen:
                prior_sample = torch.randn((n_batch,
                                            2 * dim_prior)).double().to(device)
                X = gen_net(prior_sample) 

            else:
                X = rand_noise(n_data, dim, dust_const,
                               True).double().to(device)

            with torch.no_grad(): 
                if bootstrapped:
                    V0 = torch.exp(pred_net(X))
                    U, V = sink_vec(X[:, :dim], X[:, dim:],
                                    cost_mat, eps, V0, n_boot)
                    U = torch.log(U)
                    V = torch.log(V)

                else:
                    V0 = torch.ones_like(X[:, :dim])
                    U, V = sink_vec(X[:, :dim], X[:, dim:], cost_mat,
                                    eps, V0, 1000)
                    U = torch.log(U)
                    V = torch.log(V)
                
                nan_mask = ~(torch.isnan(U).any(dim=1) & torch.isnan(
                    V).any(dim=1)).to(device)

            if extend_data:
                X_pred, T_pred = extend(X, U, V, dim, nan_mask, device)
                
            else:
                X_pred = X[nan_mask]
                V = V - torch.unsqueeze(V.mean(dim=1), 1).repeat(1, dim)
                T_pred = V[nan_mask]

            X = X_pred
            T = T_pred
            P = pred_net(X)
                    
            pred_loss = loss_func(P, T)
            train_losses['pred'].append(pred_loss.item())

            # Update
            pred_optimizer.zero_grad()
            pred_loss.backward(retain_graph=True)
            pred_optimizer.step()

        if ((i+1) % test_iter == 0) or (i == 0):
            plot_XPT(X[0], P[0], T[0], dim)
        # Checkpointing
        if ((i+1) % checkpoint == 0):
        
            print(f'Checkpointing at epoch {i+1}...')

            # Testing mode
            gen_net.eval()
            pred_net.eval()

            # Saving nets
            torch.save(gen_net.state_dict(), f'{results_folder}/deer.pt')
            torch.save(pred_net.state_dict(), f'{results_folder}/puma.pt')

            # Test warmstart
            warmstarts = test_warmstart(pred_net, test_sets, test_emds, cost_mat, eps, dim)

            # Plot the results
            plot_train_losses(train_losses, f'{results_folder}/train_losses.png')
            plot_test_losses(test_losses, f'{results_folder}/test_losses.png')
            plot_test_rel_errs_emd(test_rel_errs_emd,
                                   f'{results_folder}/test_rel_errs.png')
            plot_warmstarts(warmstarts, results_folder)
        
        if ((i+2) % test_iter == 0) or (i == n_loops-1):
            plt.close('all')

        # Updating learning rates
        if learn_gen:
            gen_scheduler.step()
        pred_scheduler.step()

    return train_losses, test_losses, test_rel_errs_emd, warmstarts
