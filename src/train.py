"""
train.py
-------

The algorithm(s) for training the neural network(s).
"""

# Imports
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.test_funcs import *
from src.sinkhorn import sink_vec
from src.plot import *
from src.data_funcs import rand_noise
from src.nets import GenNet, PredNet
from src.checkpoint import checkpoint

def the_hunt(
        gen_net : GenNet,
        pred_net : PredNet,
        loss_func : callable,
        loss_reg: float,
        toggle_reg: bool,
        layer_weights_normed: bool,
        cost_mat : torch.Tensor,
        eps : float,
        dust_const : float,
        dim_prior : int,
        dim : int,
        device : torch.device,
        test_sets: dict,
        test_sinks : dict,
        test_T : dict,
        n_loops : int,
        n_mini_loops_gen : int,
        n_mini_loops_pred : int,
        n_batch : int,
        weight_decay_gen : float,
        weight_decay_pred : float,
        lr_pred : float,
        lr_gen : float,
        lr_fact_gen : float,
        lr_fact_pred : float,
        learn_gen : bool,
        bootstrapped : bool,
        n_boot : int,
        test_iter : int,
        plot_test_images : bool,
        display_test_info : bool,
        results_folder : str,
        checkpoint_iter : int,
        ) -> tuple[dict, dict, dict]:

    """
    The puma sharpens its teeth.
    """

    # initializing loss, relative error and warmstart collectors
    train_losses = {'gen': [], 'pred': []}
    test_losses = {}
    test_rel_errs_sink = {}
    test_mcvs = {}
    for key in test_sets.keys():
        test_losses[key] = []
        test_rel_errs_sink[key] = []
        test_mcvs[key] = []

    # initializing optimizers
    pred_optimizer = torch.optim.SGD(pred_net.parameters(),
                                    lr=lr_pred, weight_decay=weight_decay_pred)
    if learn_gen:
        gen_optimizer = torch.optim.SGD(gen_net.parameters(),
                                    lr=lr_gen, weight_decay=weight_decay_gen)

    # initializing learning rate schedulers
    pred_scheduler = torch.optim.lr_scheduler.ExponentialLR(pred_optimizer,
                                                            gamma=lr_fact_pred)
    if learn_gen:
        gen_scheduler = torch.optim.lr_scheduler.ExponentialLR(gen_optimizer,
                                                            gamma=lr_fact_gen)

    # training loop
    for i in tqdm(range(n_loops)):

        # testing predictive neural net
        if ((i+1) % test_iter == 0) or (i == 0):

            print(f'Testing pred net at iter: {i+1}')

            # setting networks to eval mode
            pred_net.eval()
            if learn_gen:
                gen_net.eval()

            # iterating over test sets
            for key in test_sets.keys():

                X_test = test_sets[key]
                T = test_T[key]
                sink = test_sinks[key]
                P = pred_net(X_test)

                loss = loss_func(P, T, gen_net, loss_reg, toggle_reg)
                test_losses[key].append(loss.item())

                pred_dist = get_pred_dists(P, X_test, eps, cost_mat, dim)
                rel_errs_sink = torch.abs(pred_dist - sink) / sink
                test_rel_errs_sink[key].append(rel_errs_sink.mean().item())
                
                test_mcv = get_mean_mcv(pred_net, X_test, cost_mat, eps, dim)
                test_mcvs[key].append(test_mcv)

                if plot_test_images:
                    plot_XPT(X_test[0], P[0], T[0], dim)

        # Setting networks to train mode
        if learn_gen:
            gen_net.train()
        pred_net.train()

        # Training generative neural net
        if learn_gen:
            for _ in range(n_mini_loops_gen):

                gen_optimizer.zero_grad()

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

                X_gen = X[nan_mask]
                V = V - torch.unsqueeze(V.mean(dim=1), 1).repeat(1, dim)
                T_gen = V[nan_mask]

                X = X_gen
                T = T_gen
                P = pred_net(X)

                gen_loss = -loss_func(P, T, gen_net, loss_reg, toggle_reg)
                train_losses['gen'].append(gen_loss.item())
                gen_loss.backward(retain_graph=True)

                # Update
                gen_optimizer.step()


        # Training predictive neural net
        for _ in range(n_mini_loops_pred):
            pred_optimizer.zero_grad()
            if learn_gen:
                prior_sample = torch.randn((n_batch,
                                        2 * dim_prior)).double().to(device)
                
                if layer_weights_normed:
                    for layer in gen_net.layers:
                        layer[0].weight = torch.nn.parameter.Parameter(layer[0].weight / torch.linalg.matrix_norm(layer[0].weight, ord=2))

                X = gen_net(prior_sample)

            else:
                X = rand_noise(n_batch, dim, dust_const,
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
                non_nan_total = nan_mask.sum().item()

            X_pred = X[nan_mask]
            V = V - torch.unsqueeze(V.mean(dim=1), 1).repeat(1, dim)
            T_pred = V[nan_mask]

            X = X_pred
            T = T_pred
            P = pred_net(X)

            pred_loss = loss_func(P, T)
            train_losses['pred'].append(pred_loss.item())
            pred_loss.backward(retain_graph=True)
            pred_optimizer.step()

        if ((i+1) % test_iter == 0) or (i == 0):
            if display_test_info:
                plot_XPT(X[0], P[0], T[0], dim)
                plot_train_losses(train_losses)
                plot_test_losses(test_losses)
                plot_test_rel_errs_sink(test_rel_errs_sink)
                plot_test_mcvs(test_mcvs)

            # print current learning rates
            if learn_gen:
                print(f'gen lr: {gen_optimizer.param_groups[0]["lr"]}')
            print(f'pred lr: {pred_optimizer.param_groups[0]["lr"]}')

            # print non nan percentage
            print(f'non nan percentage: {non_nan_total / n_batch}')

        # Checkpointing
        if ((i+1) % checkpoint_iter == 0):
            print(f'Checkpointing at iter: {i+1}')

            (
            warmstarts_sink,
            warmstarts_mcv,
            warmstarts_sink_0,
            warmstarts_sink_5,
            warmstarts_sink_10,
            warmstarts_mcv_0,
            warmstarts_mcv_5,
            warmstarts_mcv_10
            ) = checkpoint(gen_net, pred_net, test_sets, test_sinks, cost_mat, eps,
                            dim, device, results_folder, train_losses,
                            test_losses, test_rel_errs_sink, test_mcvs)

        if ((i+2) % test_iter == 0) or (i == n_loops-1):
            plt.close('all')
            
        # Updating learning rates
        if learn_gen:
            gen_scheduler.step()
        pred_scheduler.step()

    results = {
        'pred_net': pred_net,
        'gen_net': gen_net,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'test_rel_errs_sink': test_rel_errs_sink,
        'test_mcvs': test_mcvs,
        'warmstarts_sink': warmstarts_sink,
        'warmstarts_mcv': warmstarts_mcv,
        'warmstarts_sink_0': warmstarts_sink_0,
        'warmstarts_sink_5': warmstarts_sink_5,
        'warmstarts_sink_10': warmstarts_sink_10,
        'warmstarts_mcv_0': warmstarts_mcv_0,
        'warmstarts_mcv_5': warmstarts_mcv_5,
        'warmstarts_mcv_10': warmstarts_mcv_10
        }
    
    return results
