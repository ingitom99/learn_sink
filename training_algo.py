"""
Hunting time!
"""

# Imports
import torch
from tqdm import tqdm
from test_funcs import sink_vec, test_loss, test_rel_err, test_warmstart
from utils import prior_sampler, plot_train_losses, plot_test_losses, plot_test_rel_errs, plot_XPT, test_set_sampler
from data_creators import rand_noise_and_shapes
from nets import GenNet, PredNet


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
        n_samples : int,
        batch_size : int,
        minibatch_size: int,
        n_epochs_pred : int,
        n_epochs_gen : int,
        lr_pred : float,
        lr_gen : float,
        lr_factor : float,
        learn_gen : bool,
        bootstrapped : bool,
        boot_no : int,
        test_iter : int,
        n_test_samples : int,
        results_folder : str,
        checkpoint : int,
        n_warmstart_samples : int,
        ) -> tuple[dict, dict, dict]:
    
    """

    """

    # Initializing loss and relative error collectors
    train_losses = {'gen': [], 'pred': []}
    test_losses = {}
    test_rel_errs = {}
    for i in test_sets.keys():
        test_losses[i] = []
        test_rel_errs[i] = []

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
    for i in tqdm(range(n_samples//batch_size)):
       
        # Testing Section
        if ((i+1) % test_iter == 0) or (i == 0):

            # Setting networks to eval mode
            pred_net.eval()

            test_loss(pred_net, test_sets, n_test_samples, test_losses, device,
                      cost_mat, eps, dim, loss_func, False)

            test_rel_err(pred_net, test_sets, test_rel_errs, n_test_samples, 
                         device, cost_mat, eps, dim, False)
            # Plot the results
            plot_train_losses(train_losses, None)
            plot_test_losses(test_losses, None)
            plot_test_rel_errs(test_rel_errs, None)

                    
        # Training Section

        # Setting networks to train mode
        pred_net.train()
        if learn_gen:
            gen_net.train()

         # Training generative neural net
        if learn_gen:
            for e in range(n_epochs_gen):
                # Data creation
                sample = prior_sampler(batch_size,
                                       dim_prior).double().to(device)
                X_e = gen_net(sample)
                for j in range(batch_size//minibatch_size):
                    X_mini = X_e[j*minibatch_size:(j+1)*minibatch_size]     
                    P_mini = pred_net(X_mini)
                    # Target creation
                    with torch.no_grad():
                        if bootstrapped:
                            V0 = torch.exp(P_mini)
                            V = sink_vec(X_mini[:, :dim], X_mini[:, dim:],
                                         cost_mat, eps, V0, boot_no)
                            V = torch.log(V)
                            V = V - torch.unsqueeze(V.mean(dim=1),
                                                    1).repeat(1, dim)
                        else:
                            V0 = torch.ones_like(X_mini[:, :dim])
                            V = sink_vec(X_mini[:, :dim], X_mini[:, dim:],
                                         cost_mat, eps, V0, 1000)
                            V = torch.log(V)
                            V = V - torch.unsqueeze(V.mean(dim=1),
                                                    1).repeat(1, dim)
                    T_mini = V
                    gen_loss = -loss_func(P_mini, T_mini)
                    train_losses['gen'].append(gen_loss.item())

                    # Update
                    gen_optimizer.zero_grad()
                    gen_loss.backward(retain_graph=True)
                    gen_optimizer.step()

        # Training predictive neural net
        # Data creation
        if learn_gen:
            sample = prior_sampler(batch_size, dim_prior).double().to(device)
            X = gen_net(sample)
    
        else:
            X = rand_noise_and_shapes(batch_size, dim, dust_const,
                                          True).double().to(device)
        for e in range(n_epochs_pred):
            perm = torch.randperm(batch_size).to(device)
            X_e = X[perm]
            for j in range(batch_size//minibatch_size):
                X_mini = X_e[j*minibatch_size:(j+1)*minibatch_size]     
                P_mini = pred_net(X_mini)
                # Target creation
                with torch.no_grad():
                    if bootstrapped:
                        V0 = torch.exp(P_mini)
                        V = sink_vec(X_mini[:, :dim], X_mini[:, dim:], cost_mat,
                                     eps, V0, boot_no)
                        V = torch.log(V)
                        V = V - torch.unsqueeze(V.mean(dim=1), 1).repeat(1, dim)
                    else:
                        V0 = torch.ones_like(X_mini[:, :dim])
                        V = sink_vec(X_mini[:, :dim], X_mini[:, dim:], cost_mat,
                                     eps, V0, 1500)
                        V = torch.log(V)
                        V = V - torch.unsqueeze(V.mean(dim=1), 1).repeat(1, dim)
                T_mini = V
                pred_loss = loss_func(P_mini, T_mini)
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
            plot_test_losses(test_losses,
                             f'{results_folder}/test_losses.png')
            plot_test_rel_errs(test_rel_errs,
                               f'{results_folder}/test_rel_errs.png')
            # Test warmstart
            test_warmstart_trials = {}
            for key in test_sets.keys():
                X_test = test_set_sampler(test_sets[key],
                                        n_warmstart_samples).double().to(device)
                test_warmstart_trials[key] = test_warmstart(pred_net, X_test,
                                    cost_mat, eps, dim, key,
                                    f'{results_folder}/warm_start_{key}.png')
        
        if (i%50 == 0):
            plot_XPT(X_mini[0], P_mini[0], T_mini[0], dim)

        # Updating learning rates
        if learn_gen:
            gen_scheduler.step()
        pred_scheduler.step()

        # Printing batch information
        print('------------------------------------------------')
        print(f"Batch: {i+1}")
        print((f"Train loss gen: {gen_loss.item()}"))
        print(f"Train loss pred: {pred_loss.item()}")
        if learn_gen:
            print(f"gen lr: {gen_scheduler.get_last_lr()[0]}")
        print(f"pred lr: {pred_scheduler.get_last_lr()[0]}")

    return train_losses, test_losses, test_rel_errs