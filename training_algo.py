"""
Hunting time!
"""

# Imports
import torch
from tqdm import tqdm
from test_funcs import sink_vec
from utils import prior_sampler
from tester import test_loss, test_rel_err
from data_creators import rand_noise_and_shapes
from nets import GenNet, PredNet



def the_hunt(
        gen_net : GenNet,
        pred_net : PredNet,
        loss_func : callable,
        C : torch.Tensor,    
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
        ) -> tuple[dict, dict, dict]:
    
    """

    """

    # Initializing loss and relative error collectors
    train_losses = {'pred': [], 'gen': []}
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
                      C, eps, dim, loss_func, True)
            test_rel_err(pred_net, test_sets, test_rel_errs, n_test_samples, 
                         device, C, eps, dim, True)
                    
        # Training Section

        # Setting networks to train mode
        pred_net.train()
        if learn_gen:
            gen_net.train()

        # Data creation
        if learn_gen:
            sample = prior_sampler(batch_size, dim_prior).double().to(device)
            X = gen_net(sample)

        else:
            X = rand_noise_and_shapes(batch_size, dim,
                                           dust_const).double().to(device)

         # Training generative neural net
        if learn_gen:
            for e in range(n_epochs_gen):
                perm = torch.randperm(batch_size).to(device)
                X_e = X[perm]
                for j in range(batch_size//minibatch_size):
                    X_mini = X_e[j*minibatch_size:(j+1)*minibatch_size]     
                    P_mini = pred_net(X_mini)
                    # Target creation
                    with torch.no_grad():
                        if bootstrapped:
                            V0 = torch.exp(P_mini)
                            V = sink_vec(X_mini[:, :dim], X_mini[:, dim:], C,
                                         eps, V0, boot_no)
                            V = torch.log(V)
                            V = V - torch.unsqueeze(V.mean(dim=1),
                                                    1).repeat(1, dim)
                        else:
                            V0 = torch.ones_like(X_mini[:, :dim])
                            V = sink_vec(X_mini[:, :dim], X_mini[:, dim:],
                                         C, eps, V0, 1000)
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
                        V = sink_vec(X_mini[:, :dim], X_mini[:, dim:], C, eps,
                                    V0, boot_no)
                        V = torch.log(V)
                        V = V - torch.unsqueeze(V.mean(dim=1), 1).repeat(1, dim)
                    else:
                        V0 = torch.ones_like(X_mini[:, :dim])
                        V = sink_vec(X_mini[:, :dim], X_mini[:, dim:], C, eps,
                                    V0, 1500)
                        V = torch.log(V)
                        V = V - torch.unsqueeze(V.mean(dim=1), 1).repeat(1, dim)
                T_mini = V
                pred_loss = loss_func(P_mini, T_mini)
                train_losses['pred'].append(pred_loss.item())

                # Update
                pred_optimizer.zero_grad()
                pred_loss.backward(retain_graph=True)
                pred_optimizer.step()

        # Updating learning rates
        if learn_gen:
            gen_scheduler.step()
        pred_scheduler.step()

        # Printing batch information
        print('------------------------------------------------')
        print(f"Batch: {i+1}")
        print(f"Train loss: {pred_loss.item()}")
        if learn_gen:
            print(f"gen lr: {gen_scheduler.get_last_lr()[0]}")
        print(f"pred lr: {pred_scheduler.get_last_lr()[0]}")

        return train_losses, test_losses, test_rel_errs