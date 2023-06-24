"""
Hunting time!
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
        cost : torch.Tensor,    
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

    """

    # Initializing loss and relative error collectors
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

    # Initializing learning rate scheduler
    pred_scheduler = torch.optim.lr_scheduler.ExponentialLR(pred_optimizer, gamma=lr_fact_gen)
    if learn_gen:
        gen_scheduler = torch.optim.lr_scheduler.ExponentialLR(gen_optimizer, gamma=lr_fact_pred)
    
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
                T = test_T[key]
                emd = test_emds[key]

                P = pred_net(X_test)
                loss = loss_func(P, T)
                
                pred_dist = get_pred_dists(P, X_test, eps, cost, dim)

                rel_errs_emd = torch.abs(pred_dist - emd) / emd

                test_rel_errs_emd[key].append(rel_errs_emd.mean().item())
                test_losses[key].append(loss.item())

            plot_test_rel_errs_emd(test_rel_errs_emd)
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
                
                if extend_data:
                    n_data = n_batch // 4
                else:
                    n_data = n_batch
                prior_sample = torch.randn((n_batch, 2 * dim_prior)).double().to(device)
                X = gen_net(prior_sample) 

                P = pred_net(X)

                with torch.no_grad():
                    if bootstrapped:
                        V0 = torch.exp(P)
                        U, V = sink_vec(X[:, :dim], X[:, dim:], cost, eps, V0, n_boot)
                        U = torch.log(U)
                        V = torch.log(V)

                    else:
                        V0 = torch.ones_like(X[:, :dim])
                        U, V = sink_vec(X[:, :dim], X[:, dim:],
                                        cost, eps, V0, 1000)
                        U = torch.log(U)
                        V = torch.log(V)

                    nan_mask = ~(torch.isnan(U).any(dim=1) & torch.isnan(V).any(dim=1)).to(device)

                if extend_data:
                    X_gen, T_gen = extend(X, U, V, n_batch, dim, nan_mask, device, center=True)

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
        for loop in range(n_mini_loops_pred):

            if extend_data:
                n_data = n_batch // 4
            else:
                n_data = n_batch

            if learn_gen:
                prior_sample = torch.randn((n_batch, 2 * dim_prior)).double().to(device)
                X = gen_net(prior_sample) 

            else:
                X = rand_noise(n_data, dim, dust_const, True).double().to(device)

            with torch.no_grad(): 
                if bootstrapped:
                    V0 = torch.exp(pred_net(X))
                    U, V = sink_vec(X[:, :dim], X[:, dim:], cost, eps, V0, n_boot)
                    U = torch.log(U)
                    V = torch.log(V)

                else:
                    V0 = torch.ones_like(X[:, :dim])
                    U, V = sink_vec(X[:, :dim], X[:, dim:], cost, eps, V0, 1000)
                    U = torch.log(U)
                    V = torch.log(V)
                
                nan_mask = ~(torch.isnan(U).any(dim=1) & torch.isnan(V).any(dim=1)).to(device)
                n_batch = nan_mask.sum()

            if extend_data:
                X_pred, T_pred = extend(X, U, V, n_batch, dim, nan_mask, device, center=True)
                
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
            warmstarts = test_warmstart(pred_net, test_sets, test_emds, cost, eps, dim)

            # Plot the results
            plot_train_losses(train_losses, f'{results_folder}/train_losses.png')
            plot_test_losses(test_losses, f'{results_folder}/test_losses.png')
            plot_test_rel_errs_emd(test_rel_errs_emd, f'{results_folder}/test_rel_errs.png')
            plot_test_rel_errs_sink(test_rel_errs_sink, f'{results_folder}/test_rel_errs_sink.png')
            plot_warmstarts(warmstarts, results_folder)
        
        if ((i+2) % test_iter == 0) or (i == n_loops-1):
            plt.close('all')

        # Updating learning rates
        if learn_gen:
            gen_scheduler.step()
        pred_scheduler.step()

    return train_losses, test_losses, test_rel_errs_emd, warmstarts