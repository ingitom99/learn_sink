"""
Hunting time!
"""

# Imports
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from test_funcs import test_loss, test_rel_err, test_warmstart, get_pred_dists
from sinkhorn import sink_vec
from utils import prior_sampler, plot_train_losses, plot_test_losses, plot_test_rel_errs_emd, plot_test_rel_errs_sink, plot_XPT, test_set_sampler
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
        test_emd : dict,
        test_sink : dict,
        test_T : dict,
        n_loops : int,
        n_mini_loops_gen : int,
        n_mini_loops_pred : int,
        batch_size : int,
        lr_pred : float,
        lr_gen : float,
        lr_factor : float,
        learn_gen : bool,
        bootstrapped : bool,
        boot_no : int,
        extend_data : bool,
        test_iter : int,
        results_folder : str,
        checkpoint : int,
        n_warmstart_samples : int,
        ) -> tuple[dict, dict, dict]:
    
    """

    """

    # Initializing loss and relative error collectors
    train_losses = {'gen': [], 'pred': []}
    test_losses = {}
    test_rel_errs_emd = {}
    test_rel_errs_sink = {}
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
            for key in test_sets.keys():

                X_test = test_sets[key]
                T = test_T[key]
                emd = test_emd[key]
                sink = test_sink[key]

                P = pred_net(X_test)
                loss = loss_func(P, T)
                
                pred_dist = get_pred_dists(P, X_test, eps, cost_mat, dim)

                rel_errs_emd = torch.abs(pred_dist - emd) / emd
                rel_errs_sink = torch.abs(pred_dist - sink) / sink

                test_rel_errs_emd[key].append(rel_errs_emd.mean().item())
                test_rel_errs_sink[key].append(rel_errs_sink.mean().item())
                test_losses[key].append(loss.item())

            plot_test_rel_errs_emd(test_rel_errs_emd)
            plot_test_rel_errs_sink(test_rel_errs_sink)
            plot_test_losses(test_losses)
            plot_train_losses(train_losses)

            if (i !=0):
                plot_XPT(X[0], P[0], T[0], dim)
         
        # Training Section

        # Setting networks to train mode
        pred_net.train()
        if learn_gen:
            gen_net.train()

         # Training generative neural net
        if learn_gen:
            for loop in range(n_mini_loops_gen):
                
                if extend_data:
                    n_data = batch_size // 4
                else:
                    n_data = batch_size
                prior_sample = torch.randn((n_batch, 2 * dim_prior)).double().to(device)
                X = gen_net(prior_sample) 

                P = pred_net(X)

                with torch.no_grad():
                    if bootstrapped:
                        V0 = torch.exp(P)
                        U, V = sink_vec(X[:, :dim], X[:, dim:], cost_mat, eps, V0, boot_no)
                        U = torch.log(U)
                        V = torch.log(V)

                    else:
                        V0 = torch.ones_like(X[:, :dim])
                        U, V = sink_vec(X[:, :dim], X[:, dim:],
                                        cost_mat, eps, V0, 1000)
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
                n_data = batch_size // 4
            else:
                n_data = batch_size

            if learn_gen:
                sample = prior_sampler(n_data,
                    dim_prior).double().to(device)
                X = gen_net(sample) 

            else:
                X = rand_noise(n_data, dim, dust_const,
                            True).double().to(device)

            with torch.no_grad(): 
                if bootstrapped:
                    V0 = torch.exp(pred_net(X))
                    U, V = sink_vec(X[:, :dim], X[:, dim:],
                                    cost_mat, eps, V0, boot_no)
                    U = torch.log(U)
                    V = torch.log(V)

                else:
                    V0 = torch.ones_like(X[:, :dim])
                    U, V = sink_vec(X[:, :dim], X[:, dim:],
                                    cost_mat, eps, V0, 1000)
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


        # Checkpointing
        if ((i+1) % checkpoint == 0):
            # Testing mode
            gen_net.eval()
            pred_net.eval()
            # Saving nets
            torch.save(gen_net.state_dict(), f'{results_folder}/deer.pt')
            torch.save(pred_net.state_dict(), f'{results_folder}/puma.pt')
            # Plot the results
            plot_train_losses(train_losses, f'{results_folder}/train_losses.png')
            plot_test_losses(test_losses, f'{results_folder}/test_losses.png')
            plot_test_rel_errs_emd(test_rel_errs_emd, f'{results_folder}/test_rel_errs.png')
            plot_test_rel_errs_sink(test_rel_errs_sink, f'{results_folder}/test_rel_errs_sink.png')
            # Test warmstart
            test_warmstart_trials = {}
            for key in test_sets.keys():
                X_test = test_set_sampler(test_sets[key],
                                        n_warmstart_samples).double().to(device)
                test_warmstart_trials[key] = test_warmstart(pred_net, X_test,
                                    cost_mat, eps, dim, key,
                                    f'{results_folder}/warm_start_{key}.png')
        
        if ((i+1) % test_iter == 0) or (i == 0):
            plot_XPT(X[0], P[0], T[0], dim)
        
        if ((i+2) % test_iter == 0) or (i == n_loops-1):
            plt.close('all')

        # Updating learning rates
        if learn_gen:
            gen_scheduler.step()
        pred_scheduler.step()

    return train_losses, test_losses, test_rel_errs_emd, test_rel_errs_sink