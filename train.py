"""
train.py

Hunting time!
"""

# Imports
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from plot import  plot_XPT, plot_test_losses, plot_train_losses, plot_test_rel_errs_emd, plot_test_rel_errs_sink
from test_funcs import  get_pred_dists
from data_funcs import rand_noise
from nets import GenNet, PredNet
from sinkhorn import sink_var_eps_vec

def the_hunt(
        gen_net : GenNet,
        pred_net : PredNet,
        loss_func : callable,
        cost_mat : torch.Tensor,
        min_eps_var : float,
        max_eps_var : float,
        eps_test_const : torch.Tensor,
        eps_test_var : torch.Tensor,
        dust_const : float,
        dim_prior : int,
        dim : int,
        device : torch.device,
        test_sets: dict,
        test_emd : dict,
        test_sink : dict,
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
        test_iter : int,
        results_folder : str,
        checkpoint : int,
        ) -> tuple[dict, dict, dict]:

    """
    The hunt!

    Parameters
    ----------
    gen_net : GenNet
        The generator network.
    pred_net : PredNet
        The predictive network.
    loss_func : callable
        The loss function.
    cost_mat : torch.Tensor
        The cost matrix.
    min_eps_var : float
        The min regularization parameter size.
    max_eps_var : float
        The max regularization parameter size.
    eps_test_const : torch.Tensor
        The constant regularization parameter for testing.
    eps_test_var : torch.Tensor
        The variable regularization parameter vector for testing.
    dust_const : float
        The dusting parameter.
    dim_prior : int
        The dimension of the prior distribution.
    dim : int
        The dimension of the data.
    device : torch.device
        The device to run the hunt on.
    test_sets : dict
        The test sets.
    test_emd : dict
        The test sets' ot.emd2() values.
    test_sink : dict
        The test sets' ot.sinkhorn2() values.
    test_T : dict
        The test sets' targets for constant regularization parameter.
    n_loops : int
        The number of training loops.
    n_mini_loops_gen : int
        The number of mini loops for the generator.
    n_mini_loops_pred : int
        The number of mini loops for the predictor.
    n_batch : int
        The batch size.
    lr_pred : float
        The learning rate for the predictor.
    lr_gen : float
        The learning rate for the generator.
    lr_fact_gen : float
        The learning rate decay factor for the generator.
    lr_fact_pred : float
        The learning rate decay factor for the predictor.
    learn_gen : bool
        Whether to learn the generator or use random noise.
    bootstrapped : bool
        Whether to use bootstrapping or not.
    n_boot : int
        The number of bootstraps.
    test_iter : int
        The number of iterations between testing.
    results_folder : str
        The folder to save results to.
    checkpoint : int
        The number of iterations between checkpoints.

    Returns
    -------
    train_losses : dict
        The training losses.
    test_rel_errs_emd : dict
        The test relative errors against ot.emd2().
    test_rel_errs_sink : dict
        The test relative errors against ot.sinkhorn2().
    
    """

    # Initializing loss and relative error collectors
    train_losses = {'gen': [], 'pred': []}
    test_losses = {}
    test_rel_errs_emd = {}
    test_rel_errs_sink = {}
    warmstarts = {}
    for key in test_sets.keys():
        test_losses[key] = []
        test_rel_errs_emd[key] = []
        test_rel_errs_sink[key] = []

    # Initializing optimizers
    pred_optimizer = torch.optim.SGD(pred_net.parameters(), lr=lr_pred)
    if learn_gen:
        gen_optimizer = torch.optim.SGD(gen_net.parameters(), lr=lr_gen)

    # Initializing learning rate scheduler
    pred_scheduler = torch.optim.lr_scheduler.ExponentialLR(pred_optimizer,
                                                            gamma=lr_fact_gen)
    if learn_gen:
        gen_scheduler = torch.optim.lr_scheduler.ExponentialLR(gen_optimizer,
                                                            gamma=lr_fact_pred)

    # Batch loop
    for i in tqdm(range(n_loops)):

        # Testing predictive neural net
        if ((i+1) % test_iter == 0) or (i == 0):
            
            # Setting networks to eval mode
            pred_net.eval()
            if learn_gen:
                gen_net.eval()

            for key in test_sets.keys():

                X_test = test_sets[key]
                X_test_eps_const = torch.cat((X_test, eps_test_const), dim=1)
                X_test_eps_var = torch.cat((X_test, eps_test_var), dim=1)

                P_const = pred_net(X_test_eps_const)
                P_var = pred_net(X_test_eps_var)

                pred_dist_const = get_pred_dists(P_const, X_test,
                                                 eps_test_const, cost_mat, dim)
                pred_dist_var = get_pred_dists(P_var, X_test,
                                                  eps_test_var, cost_mat, dim)

                emd = test_emd[key]
                sink = test_sink[key]
                T = test_T[key]
                
                loss = loss_func(P_const, T)

                rel_errs_emd = torch.abs(pred_dist_const - emd) / emd
                rel_errs_sink = torch.abs(pred_dist_var - sink) / sink

                test_rel_errs_emd[key].append(rel_errs_emd.mean().item())
                test_rel_errs_sink[key].append(rel_errs_sink.mean().item())
                test_losses[key].append(loss.item())

            plot_test_rel_errs_emd(test_rel_errs_emd)
            plot_test_rel_errs_sink(test_rel_errs_sink)
            plot_test_losses(test_losses)
            plot_train_losses(train_losses)
            
        # Training Section

        # Setting networks to train mode
        pred_net.train()
        if learn_gen:
            gen_net.train()

         # Training generative neural net
        if learn_gen:
            for loop in range(n_mini_loops_gen):

                prior_sample = torch.randn((n_batch, 2 * dim_prior)
                                           ).double().to(device)
                X = gen_net(prior_sample)

                MU = X[:, :dim]
                NU = X[:, dim:]

                eps = torch.rand(n_batch, 1).double().to(device)
                eps = min_eps_var + (max_eps_var - min_eps_var) * eps

                X_eps = torch.cat((X, eps), dim=1)

                with torch.no_grad():
                    if bootstrapped:
                        V0 = torch.exp(pred_net(X_eps))
                        U, V = sink_var_eps_vec(MU, NU, cost_mat, eps, V0, n_boot)
                        U = torch.log(U)
                        V = torch.log(V)

                    else:
                        V0 = torch.ones_like(MU)
                        U, V = sink_var_eps_vec(MU, NU, cost_mat, eps, V0, 1000)
                        U = torch.log(U)
                        V = torch.log(V)

                nan_mask = ~(torch.isnan(U).any(dim=1) & torch.isnan(V
                                                        ).any(dim=1)).to(device)
                V = V[nan_mask]

                X = X_eps[nan_mask]
                T = V - torch.unsqueeze(V.mean(dim=1), 1).repeat(1, dim)
                P = pred_net(X)

                gen_loss = -loss_func(P, T)
                train_losses['gen'].append(gen_loss.item())

                # Update
                gen_optimizer.zero_grad()
                gen_loss.backward(retain_graph=True)
                gen_optimizer.step()

        # Training predictive neural net
        for loop in range(n_mini_loops_pred):

            if learn_gen:
                prior_sample = torch.randn((n_batch, 2 * dim_prior)
                                           ).double().to(device)
                X = gen_net(prior_sample)

                MU = X[:, :dim]
                NU = X[:, dim:]

            else:
                X = rand_noise(n_batch, dim, dust_const, True
                               ).double().to(device)
                MU = X[:, :dim]
                NU = X[:, dim:]

            eps = torch.rand(n_batch, 1).double().to(device)
            eps = min_eps_var + (max_eps_var - min_eps_var) * eps

            X_eps = torch.cat((X, eps), dim=1)

            with torch.no_grad():

                if bootstrapped:
                    V0 = torch.exp(pred_net(X_eps))
                    U, V = sink_var_eps_vec(MU, NU, cost_mat, eps, V0, n_boot)
                    U = torch.log(U)
                    V = torch.log(V)

                else:
                    V0 = torch.ones_like(MU)
                    U, V = sink_var_eps_vec(MU, NU, cost_mat, eps, V0, 1000)
                    U = torch.log(U)
                    V = torch.log(V)

                nan_mask = ~(torch.isnan(U).any(dim=1) & torch.isnan(V
                                                    ).any(dim=1)).to(device)
                V = V[nan_mask]

            X = X_eps[nan_mask]
            T = V - torch.unsqueeze(V.mean(dim=1), 1).repeat(1, dim)
            P = pred_net(X)

            pred_loss = loss_func(P, T)
            train_losses['pred'].append(pred_loss.item())

            # Update
            pred_optimizer.zero_grad()
            pred_loss.backward(retain_graph=True)
            pred_optimizer.step()

        if ((i+1) % test_iter == 0):
            plot_XPT(X[0, :2*dim], P[0], T[0], dim)

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
            warmstarts = test_warmstarts(pred_net, test_sets, test_emds, cost, eps, dim)

            # Plot the results
            plot_train_losses(train_losses, f'{results_folder}/train_losses.png')
            plot_test_losses(test_losses, f'{results_folder}/test_losses.png')
            plot_test_rel_errs_emd(test_rel_errs_emd, f'{results_folder}/test_rel_errs.png')
            plot_test_rel_errs_sink(test_rel_errs_sink, f'{results_folder}/test_rel_errs_sink.png')
            plot_warmstarts(warmstarts, results_folder)

        # Updating learning rates
        if learn_gen:
            gen_scheduler.step()
        pred_scheduler.step()

        if ((i+2) % test_iter == 0) or (i == n_loops-1):
            plt.close('all')

    return train_losses, test_rel_errs_emd, test_rel_errs_sink