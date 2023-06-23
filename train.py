"""
Hunting time!
"""

# Imports
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import plot_train_losses, plot_test_rel_errs_emd, plot_test_rel_errs_sink, plot_XPT, get_pred_dists
from data_creators import rand_noise
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
        n_loops : int,
        n_mini_loops_gen : int,
        n_mini_loops_pred : int,
        n_batch : int,
        lr_pred : float,
        lr_gen : float,
        lr_factor : float,
        learn_gen : bool,
        bootstrapped : bool,
        boot_no : int,
        test_iter : int,
        results_folder : str,
        checkpoint : int,
        close_plots_iter : int,
        ) -> tuple[dict, dict, dict]:

    """

    """

    # Initializing loss and relative error collectors
    train_losses = {'gen': [], 'pred': []}
    test_rel_errs_emd = {}
    test_rel_errs_sink = {}
    for i in test_sets.keys():
        test_rel_errs_emd[i] = []
        test_rel_errs_sink[i] = []

    # Initializing optimizers
    pred_optimizer = torch.optim.SGD(pred_net.parameters(), lr=lr_pred)
    if learn_gen:
        gen_optimizer = torch.optim.SGD(gen_net.parameters(), lr=lr_gen)

    # Initializing learning rate scheduler
    pred_scheduler = torch.optim.lr_scheduler.ExponentialLR(pred_optimizer,
                                                            gamma=lr_factor)
    if learn_gen:
        gen_scheduler = torch.optim.lr_scheduler.ExponentialLR(gen_optimizer,
                                                               gamma=lr_factor)

    # Batch loop
    for i in tqdm(range(n_loops)):

        # Test Section

        # Setting networks to eval mode
        pred_net.eval()
        if learn_gen:
            gen_net.eval()

        # Testing predictive neural net
        if ((i+1) % test_iter == 0) or (i == 0):
            for j in test_sets.keys():

                X_test = test_sets[j]
                X_test_eps_const = torch.cat((X_test, eps_test_const), dim=1)
                X_test_eps_var = torch.cat((X_test, eps_test_var), dim=1)

                P_const = pred_net(X_test_eps_const)
                P_var = pred_net(X_test_eps_var)

                pred_dist_const = get_pred_dists(P_const, X_test,
                                                 eps_test_const, cost_mat, dim)
                pred_dist_var = get_pred_dists(P_var, X_test,
                                                  eps_test_var, cost_mat, dim)

                emd = test_emd[j]
                sink = test_sink[j]

                rel_errs_emd = torch.abs(pred_dist_const - emd) / emd
                rel_errs_sink = torch.abs(pred_dist_var - sink) / sink

                test_rel_errs_emd[j].append(rel_errs_emd.mean().item())
                test_rel_errs_sink[j].append(rel_errs_sink.mean().item())

            plot_test_rel_errs_emd(test_rel_errs_emd)
            plot_test_rel_errs_sink(test_rel_errs_sink)

            plot_train_losses(train_losses)
            if (i !=0):
                plot_XPT(X[0, :2*dim], P[0], T[0], dim)


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
                        U, V = sink_var_eps_vec(MU, NU, cost_mat, eps, V0, boot_no)
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
                    U, V = sink_var_eps_vec(MU, NU, cost_mat, eps, V0, boot_no)
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


        # Checkpointing
        if ((i+1) % checkpoint == 0):

            # Testing mode
            gen_net.eval()
            pred_net.eval()

            # Saving nets
            torch.save(gen_net.state_dict(), f'{results_folder}/deer.pt')
            torch.save(pred_net.state_dict(), f'{results_folder}/puma.pt')

            # Plot the results
            plot_train_losses(train_losses,
                              f'{results_folder}/train_losses.png')
            plot_test_rel_errs_emd(test_rel_errs_emd,
                               f'{results_folder}/test_rel_errs_emd.png')
            plot_test_rel_errs_sink(test_rel_errs_emd,
                               f'{results_folder}/test_rel_errs_sink.png')

        # Updating learning rates
        if learn_gen:
            gen_scheduler.step()
        pred_scheduler.step()

        if ((i+1) % close_plots_iter == 0):
            plt.close('all')

    return train_losses, test_rel_errs_emd, test_rel_errs_sink