"""
Hunting time!
"""

# Imports
import torch
from tqdm import tqdm
from test_funcs import sink_vec, test_loss, test_rel_err, test_warmstart
from utils import prior_sampler, plot_train_losses, plot_test_losses, plot_test_rel_errs, plot_XPT, test_set_sampler
from data_creators import rand_noise
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
        test_iter : int,
        n_test_samples : int,
        results_folder : str,
        checkpoint : int,
        n_warmstart_samples : int,
        ) -> tuple[dict, dict, dict]:
    
    """

    """
    length = int(dim**.5)
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
    for i in tqdm(range(n_loops)):
       
        # Testing Section
        if ((i+1) % test_iter == 0): # or (i == 0):

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
            for loop in range(n_mini_loops_gen):
                # Data creation
                sample = prior_sampler(batch_size,
                        dim_prior).double().to(device)
                X = gen_net(sample)  
                P = pred_net(X)
                # Target creation
                with torch.no_grad():
                    if bootstrapped:
                        V0 = torch.exp(P)
                        U, V = sink_vec(X[:, :dim], X[:, dim:],
                                        cost_mat, eps, V0, boot_no)
                        U = torch.log(U)
                        V = torch.log(V)
                        #V = V - torch.unsqueeze(V.mean(dim=1),
                        #                       1).repeat(1, dim)
                    else:
                        V0 = torch.ones_like(X[:, :dim])
                        U, V = sink_vec(X[:, :dim], X[:, dim:],
                                        cost_mat, eps, V0, 1000)
                        V = torch.log(V)
                        V = V - torch.unsqueeze(V.mean(dim=1), 1).repeat(1, dim)
                    nan_mask = ~(torch.isnan(U).any(dim=1) & torch.isnan(
                        V).any(dim=1))

                T = V[nan_mask]
                P = P[nan_mask]

                gen_loss = -loss_func(P, T)
                train_losses['gen'].append(gen_loss.item())

                # Update
                gen_optimizer.zero_grad()
                gen_loss.backward(retain_graph=True)
                gen_optimizer.step()

        # Training predictive neural net
        for loop in range(n_mini_loops_pred):
            # Data creation
            if learn_gen:
                sample = prior_sampler(batch_size,
                    dim_prior).double().to(device)
                X = gen_net(sample)  
            else:
                X = rand_noise(batch_size, dim, dust_const,
                               True).double().to(device)
            P = pred_net(X)
            # Target creation
            with torch.no_grad():
                if bootstrapped:
                    V0 = torch.exp(P)
                    U, V = sink_vec(X[:, :dim], X[:, dim:],
                                    cost_mat, eps, V0, boot_no)
                    U = torch.log(U)
                    V = torch.log(V)
                    V = V - torch.unsqueeze(V.mean(dim=1), 1).repeat(1, dim)
                else:
                    V0 = torch.ones_like(X[:, :dim])
                    U, V = sink_vec(X[:, :dim], X[:, dim:],
                                    cost_mat, eps, V0, 1000)
                    U = torch.log(U)
                    V = torch.log(V)
            nan_mask = ~(torch.isnan(U).any(dim=1) & torch.isnan(V).any(dim=1))
            for flip in [False, True]:
                for rot in [0, 1, 2, 3]:
                    if flip:
                        MU = X[:, dim:][nan_mask]
                        NU = X[:, :dim][nan_mask]
                    else:
                        MU = X[:, :dim][nan_mask]
                        NU = X[:, dim:][nan_mask]

                    if (rot != 0):
                        MU = torch.rot90(MU.reshape((-1,length, length)),
                            k=rot, dims=(1, 2)).reshape((-1, dim))
                        NU = torch.rot90(NU.reshape(-1,length, length),
                            k=rot, dims=(1, 2)).reshape((-1, dim))
                    
                    X_curr = torch.cat((MU, NU), dim=1)
                    P = pred_net(X_curr)

                    if flip:
                        T = U[nan_mask]
                    else:
                        T = V[nan_mask]
                    
                    if (rot != 0):
                        T = torch.rot90(T.reshape((-1,length, length)),
                            k=rot, dims=(1, 2)).reshape((-1,dim))
                    
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
        
        if ((i+1) % test_iter == 0) or (i == 0):
            plot_XPT(X_curr[0], P[0], T[0], dim)

        # Updating learning rates
        if learn_gen:
            gen_scheduler.step()
        pred_scheduler.step()

    return train_losses, test_losses, test_rel_errs