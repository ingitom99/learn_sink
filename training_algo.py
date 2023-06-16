"""
Evolutionary hunting algorithm!
"""

import torch
from tqdm import tqdm
from mutation import mutate
from net import PredNet
from sinkhorn import sink_vec
from utils import normed_dusted
from test import test_loss, test_rel_err, test_warmstart
from utils import plot_train_losses, plot_test_losses, plot_test_rel_errs, plot_XPT, test_set_sampler


def the_hunt(
        pred_net : PredNet,
        dim : int,
        cost_mat : torch.Tensor,
        eps : float,
        dust_const : float,
        device : torch.device,
        loss_func : callable,
        n_loops : int,
        n_batch : int,
        mutation_sigma : float,
        lr : float,
        lr_factor : float,
        bootstrapped : bool,
        boot_no : int,
        test_sets : dict,
        test_iter : int,
        n_test_samples : int,
        checkpoint : int,
        n_warmstart_samples : int,
        results_folder: str,
        ):
    
    # Initializing the optimizer
    optimizer = torch.optim.SGD(pred_net.parameters(), lr=lr)

    # Initializing learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_factor)

    # Initializing loss and relative error collectors
    train_losses = {'mean': [], 'worst': []}
    test_losses = {}
    test_rel_errs = {}
    for i in test_sets.keys():
        test_losses[i] = []
        test_rel_errs[i] = []

    # Initializing the worst point
    x_worst = torch.rand(dim)
    x_worst = x_worst / torch.sum(x_worst)
    x_worst = x_worst + dust_const
    x_worst = x_worst / torch.sum(x_worst)


    for i in tqdm(range(n_loops)):

        # Testing Section
        if ((i+1) % test_iter == 0) or (i == 0):

            pred_net.eval()

            test_loss(pred_net, test_sets, n_test_samples, test_losses, device, cost_mat, eps, dim, loss_func, False)
            test_rel_err(pred_net, test_sets, test_rel_errs, n_test_samples, device, cost_mat, eps, dim, False)
            
            plot_train_losses(train_losses, None)
            plot_test_losses(test_losses, None)
            plot_test_rel_errs(test_rel_errs, None)
        
        # Generate data sampled around a point

        X = mutate(x_worst, n_batch, dim, mutation_sigma)
        X = normed_dusted(X, dust_const)

        # Predictions and targets
        P = pred_net(X)

        with torch.no_grad():
            if bootstrapped:
                V0 = torch.exp(P)
                V = sink_vec(X[:, :dim], X[:, dim:], cost_mat, eps, V0, boot_no)[1]
                V = torch.log(V)

            else:
                V0 = torch.ones_like(X[:, :dim])
                V = sink_vec(X[:, :dim], X[:, dim:], cost_mat, eps, V0, 1000)[1]
                V = torch.log(V)

            nan_mask = ~(torch.isnan(U).any(dim=1) & torch.isnan(V).any(dim=1)).to(device)
            n_batch = nan_mask.sum()
        
        X = X[nan_mask]
        P = P[nan_mask]
        T = V[nan_mask]
        
        # Calculate loss
        loss = loss_func(P, T)

        # Find worst point
        arg_worst = loss.argmax()
        train_losses['worst'].append(loss[arg_worst].item())

        # Calculate mean loss
        loss = loss.mean()

        # Update train losses
        train_losses['mean'].append(loss.item())

        # Update pred net
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # Update x_worst
        x_worst = X[arg_worst]

        # Update learning rate
        scheduler.step()

        # Checkpointing
        if ((i+1) % checkpoint == 0):
            # Testing mode
            pred_net.eval()
            # Saving nets
            torch.save(pred_net.state_dict(), f'{results_folder}/puma.pt')
            # Plot the results
            plot_train_losses(train_losses, f'{results_folder}/train_losses.png')
            plot_test_losses(test_losses, f'{results_folder}/test_losses.png')
            plot_test_rel_errs(test_rel_errs, f'{results_folder}/test_rel_errs.png')
            # Test warmstart
            test_warmstart_trials = {}
            for key in test_sets.keys():
                X_test = test_set_sampler(test_sets[key], n_warmstart_samples).double().to(device)
                test_warmstart_trials[key] = test_warmstart(pred_net, X_test, cost_mat, eps, dim, key, f'{results_folder}/warm_start_{key}.png')
        
        if ((i+1) % test_iter == 0) or (i == 0):
            plot_XPT(X[0], P[0], T[0], dim)

    return train_losses, test_losses, test_rel_errs
