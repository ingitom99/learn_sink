# Imports
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import prior_sampler
from sinkhorn_algos import sink_vec
from utils import MNIST_test_loader, random_noise_loader, random_shapes_loader
from test_funcs import test_pred_edm2, test_pred_loss


def the_hunt(gen_net,
             pred_net,
             C,
             reg,
             dim_prior,
             dim,
             loss_function,
             device,
             MNIST,
             dust_const,
             lr_gen=0.01,
             lr_pred=0.1,
             lr_factor=1.0,
             n_samples= 100000,
             batchsize=500,
             minibatch=100,
             epochs=5,
             train_gen=True):
  
  # Loss and error collecting lists.
  train_losses = []
  test_losses_gn = []
  test_losses_mnist = []
  test_losses_rs = []
  test_losses_rn = []
  rel_errs_gn = []
  rel_errs_mnist = []
  rel_errs_rs = []
  rel_errs_rn = []
  
  # Initializing optimizers
  gen_optimizer = torch.optim.SGD(gen_net.parameters(), lr=lr_gen)
  pred_optimizer = torch.optim.SGD(pred_net.parameters(), lr=lr_pred)

  # Initializing learning rate schedulers
  gen_scheduler = torch.optim.lr_scheduler.ExponentialLR(gen_optimizer, gamma=lr_factor)
  pred_scheduler = torch.optim.lr_scheduler.ExponentialLR(pred_optimizer, gamma=lr_factor)

  for i in tqdm(range(n_samples//batchsize)):
    # Testing Section
    if (i % 25 == 0):
      # Testing data
      prior_sample_test = prior_sampler(100, dim_prior).double().to(device)
      X_gn = gen_net(prior_sample_test)
      X_mnist = MNIST_test_loader(MNIST, 100).double().to(device)
      X_rn = random_noise_loader(100, dim, dust_const, sig=3).double().to(device)
      X_rs = random_shapes_loader(100, dim, dust_const).double().to(device)

      # Loss testing
      test_loss_gn = test_pred_loss(loss_function, X_gn, pred_net, C, dim, reg, plot=True, maxiter=5000)
      test_loss_mnist = test_pred_loss(loss_function, X_mnist, pred_net, C, dim, reg, plot=True, maxiter=5000)
      test_loss_rn = test_pred_loss(loss_function, X_rn, pred_net, C, dim, reg, plot=True, maxiter=5000)
      test_loss_rs = test_pred_loss(loss_function, X_rs, pred_net, C, dim, reg, plot=True, maxiter=5000)
      
      test_losses_gn.append(test_loss_gn)
      test_losses_mnist.append(test_loss_mnist)
      test_losses_rs.append(test_loss_rs)
      test_losses_rn.append(test_loss_rn)
 
      # emd2 relative error testing
      rel_err_gn = test_pred_edm2(X_gn[:10], pred_net, C, reg, dim)
      rel_err_mnist = test_pred_edm2(X_mnist[:10], pred_net, C, reg, dim)
      rel_err_rn = test_pred_edm2(X_rn[:10], pred_net, C, reg, dim)
      rel_err_rs = test_pred_edm2(X_rs[:10], pred_net, C, reg, dim)
      
      rel_errs_gn.append(rel_err_gn)
      rel_errs_mnist.append(rel_err_mnist)
      rel_errs_rs.append(rel_err_rs)
      rel_errs_rn.append(rel_err_rn)

      # Displaying loss information
      plt.figure()
      plt.plot(torch.log(torch.tensor(train_losses)))
      plt.title('Log Train Losses')
      plt.show()
      plt.figure()
      plt.plot(torch.log(torch.tensor(test_losses_gn)), label='gn')
      plt.plot(torch.log(torch.tensor(test_losses_mnist)), label='mnist')
      plt.plot(torch.log(torch.tensor(test_losses_rs)), label='rs')
      plt.plot(torch.log(torch.tensor(test_losses_rn)), label='rn')
      plt.legend()
      plt.title('Log Test Losses')
      plt.show()
      plt.figure()
      plt.plot(torch.tensor(rel_errs_gn), label='gn')
      plt.plot(torch.tensor(rel_errs_mnist), label='mnist')
      plt.plot(torch.tensor(rel_errs_rs), label='rs')
      plt.plot(torch.tensor(rel_errs_rn), label='rn')
      plt.legend()
      plt.title('emd2 Rel Errs')
      plt.show()
    
    # Training predictive neural net.
    sample = prior_sampler(batchsize, dim_prior).double().to(device)
    X = gen_net(sample)
    with torch.no_grad():
      T = torch.log(sink_vec(X[:, :dim], X[:, dim:], C, reg, 1000, V0=None))
      T = T - torch.unsqueeze(T.mean(dim=1), 1).repeat(1, dim)
    for e in range(epochs):
      perm = torch.randperm(batchsize).to(device)
      X_e, T_e = X[perm], T[perm]
      for j in range(batchsize//minibatch): 
        pred_optimizer.zero_grad()
        P_curr = pred_net(X_e[j*minibatch:(j+1)*minibatch])
        T_curr = T_e[j*minibatch:(j+1)*minibatch]
        pred_loss = loss_function(P_curr, T_curr)
        train_losses.append(pred_loss.item())
        pred_loss.backward(retain_graph=True)
        pred_optimizer.step()

    # Training generative neural net
    if (train_gen == True):
      for j in range(batchsize//minibatch):
        P = pred_net(X[j*minibatch:(j+1)*minibatch])
        gen_optimizer.zero_grad()
        gen_loss = -loss_function(P, T[j*minibatch:(j+1)*minibatch])
        gen_loss.backward(retain_graph=True)
        gen_optimizer.step()
    
    # Updating learning rates
    pred_scheduler.step()
    gen_scheduler.step()

    # Printing batch information
    print('------------------------------------------------')
    print(f"Train loss: {pred_loss.item()}")
    print(f"pred lr: {pred_scheduler.get_last_lr()[0]}")
    print(f"gen lr: {gen_scheduler.get_last_lr()[0]}")
  
  # Converting loss and error collecting lists to tensors
  train_losses = torch.tensor(train_losses)
  test_losses_gn = torch.tensor(test_losses_gn)
  test_losses_mnist = torch.tensor(test_losses_mnist)
  test_losses_rs = torch.tensor(test_losses_rs)
  test_losses_rn = torch.tensor(test_losses_rn)
  rel_errs_gn = torch.tensor(rel_errs_gn)
  rel_errs_mnist = torch.tensor(rel_errs_mnist)
  rel_errs_rn = torch.tensor(rel_errs_rn)
  rel_errs_rs = torch.tensor(rel_errs_rs)

  return train_losses, test_losses_gn, test_losses_mnist, test_losses_rs, test_losses_rn, rel_errs_gn, rel_errs_mnist, rel_errs_rn, rel_errs_rs