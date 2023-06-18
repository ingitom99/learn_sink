"""
Evolutionary hunting algorithm!
"""

import torch
from tqdm import tqdm
from mutation import mutate
from net import PredNet
from sinkhorn import sink_vec
from data_creators import rand_noise
from utils import normed_dusted
from test_funcs import test_loss, test_rel_err, test_warmstart
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
    x_worst = rand_noise(1, dim, dust_const, True).reshape(-1).double().to(device)

    for i in tqdm(range(n_loops)):

        # Testing Section
        if ((i+1) % test_iter == 0): # or (i == 0):

            pred_net.eval()

            test_loss(pred_net, test_sets, n_test_samples, test_losses, device, cost_mat, eps, dim, loss_func, False)
            test_rel_err(pred_net, test_sets, test_rel_errs, n_test_samples, device, cost_mat, eps, dim, False)

            plot_train_losses(train_losses, None)
            plot_test_losses(test_losses, None)
            plot_test_rel_errs(test_rel_errs, None)

        # Generate data sampled around a point

        pred_net.train()

        X = x_worst.repeat(n_batch, 1)
        X = X + mutation_sigma * torch.randn(n_batch, 2*dim).double().to(device)
        X = torch.nn.functional.relu(X)
        X_a = X[:, :dim] / X[:, :dim].sum(dim=1, keepdim=True)
        X_a = X_a + dust_const
        X_a = X_a / X_a.sum(dim=1, keepdim=True)
        X_b = X[:, dim:] / X[:, dim:].sum(dim=1, keepdim=True)
        X_b = X_b + dust_const
        X_b = X_b / X_b.sum(dim=1, keepdim=True)
        X = torch.cat((X_a, X_b), dim=1)

        # Predictions and targets
        P = pred_net(X)

        with torch.no_grad():
            if bootstrapped:
                V0 = torch.exp(P)
                U, V = sink_vec(X[:, :dim], X[:, dim:], cost_mat, eps, V0, boot_no)
                V = torch.log(V)

            else:
                V0 = torch.ones_like(X[:, :dim])
                U, V = sink_vec(X[:, :dim], X[:, dim:], cost_mat, eps, V0, 1000)
                U = torch.log(U)
                V = torch.log(V)

            nan_mask = ~(torch.isnan(U).any(dim=1) & torch.isnan(V).any(dim=1)).to(device)
            #num = nan_mask.sum()

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