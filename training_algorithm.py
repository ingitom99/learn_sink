"""
training_algorithm.py contains the training algorihtm function for leanring the Sinkhorn centered logarithms of Sinkhorn scale factors.
"""

#Imports
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sinkhorn_algos import sink_vec
from utils import test_sampler, prior_sampler, rando, plot_XPT, random_shapes_loader, rn_rs
from test_funcs import test_pred_dist, test_pred_loss

# Training algorithm
def the_hunt(gen_net,
             pred_net,
             C,
             reg,
             dim_prior,
             dim,
             loss_function,
             device,
             dust_const,
             MNIST,
             OMNIGLOT,
             CIFAR10,
             FLOWERS102,
             lr_gen,
             lr_pred,
             lr_factor,
             n_samples,
             batchsize,
             minibatch,
             epochs,
             test_iter,
             learn_gen
             ):
  """
  The training algorithm is named the hunt in a metaphor for adversarial learning where the predictive network is the hunter and the generative network is the prey.

  Inputs:
    gen_net: generative network
      The generative network for data creation
    pred_net: predictive network
      The predictive network for learning the centered logarithms of Sinkhorn scale factors
    C: cost matrix
      The cost matrix for the optimal transport problem
    reg: regularization parameter
      The regularization parameter for the optimal transport problem
    dim_prior: integer
      The dimension of the prior sample (i.e. the latent space)
    dim: integer
      The dimension of the data
    loss_function: loss function
      The loss function for training
    device: torch.device
      The device to run the training on
    dust_const: float
      The constant to add to the data to prevent the data from containing zeros
    MNIST: torch.utils.data.Dataset
      The processed MNIST dataset for testing
    OMNIGLOT: torch.utils.data.Dataset
      The processed OMNIGLOT dataset for testing
    CIFAR10: torch.utils.data.Dataset
      The processed CIFAR10 dataset for testing
    FLOWERS102: torch.utils.data.Dataset
      The processed FLOWERS102 dataset for testing
    lr_gen: float
      The learning rate for the generative network
    lr_pred: float
      The learning rate for the predictive network
    lr_factor: float
      The learning rate decay factor
    n_samples: integer
      The number of unique samples to train on
    batchsize: integer
      The number of samples per batch
    minibatch: integer
      The number of samples per minibatch
    epochs: integer
      The number of epochs per batch
    test_iter: integer
      The number of batches between testing
    learn_gen: boolean
      Whether or not to train the generative network
  
  Returns:
    train_losses: torch.tensor
      The training losses
    test_losses_rn: torch.tensor
      The test losses for random noise
    test_losses_mnist: torch.tensor
      The test losses for MNIST
    test_losses_omniglot: torch.tensor
      The test losses for OMNIGLOT
    test_losses_cifar: torch.tensor
      The test losses for CIFAR10
    test_losses_flowers: torch.tensor
      The test losses for FLOWERS102
    rel_errs_rn: torch.tensor
      The relative errors for random noise
    rel_errs_mnist: torch.tensor
      The relative errors for MNIST
    rel_errs_omniglot: torch.tensor
      The relative errors for OMNIGLOT
    rel_errs_cifar: torch.tensor
      The relative errors for CIFAR10
    rel_errs_flowers: torch.tensor
      The relative errors for FLOWERS102

  """
  
  # Loss and error collecting lists
  train_losses = []
  test_losses_rn = []
  test_losses_rs = []
  test_losses_rn_rs = []
  test_losses_mnist = []
  test_losses_omniglot = []
  test_losses_cifar = []
  test_losses_flowers = []
  rel_errs_rn = []
  rel_errs_rs = []
  rel_errs_rn_rs = []
  rel_errs_mnist = []
  rel_errs_omniglot = []
  rel_errs_cifar = []
  rel_errs_flowers = []

  # Initializing optimizers
  if (learn_gen == True):
    gen_optimizer = torch.optim.SGD(gen_net.parameters(), lr=lr_gen)
  pred_optimizer = torch.optim.SGD(pred_net.parameters(), lr=lr_pred)

  # Initializing learning rate scheduler
  if (learn_gen == True):
    gen_scheduler = torch.optim.lr_scheduler.ExponentialLR(gen_optimizer, gamma=lr_factor)
  pred_scheduler = torch.optim.lr_scheduler.ExponentialLR(pred_optimizer, gamma=lr_factor)

  for i in tqdm(range(n_samples//batchsize)):

    # Testing Section
    if ((i+1) % test_iter == 0) or (i == 0):

      # Setting networks to eval mode
      pred_net.eval()
      if (learn_gen == True):
        gen_net.eval()

      # Testing data
      if (learn_gen == True):
        sample = prior_sampler(200, dim_prior).double().to(device)
        X_gn = gen_net(sample)
      X_rn = rando(200, dim, dust_const).double().to(device)
      X_rs = random_shapes_loader(200, dim, dust_const).double().to(device)
      X_rn_rs = rn_rs(200, dim, dust_const).double().to(device)
      X_mnist = test_sampler(MNIST, 200).double().to(device)
      X_omniglot = test_sampler(OMNIGLOT, 200).double().to(device)
      X_cifar = test_sampler(CIFAR10, 200).double().to(device)
      X_flowers = test_sampler(FLOWERS102, 200).double().to(device)

      # Loss testing
      if (learn_gen == True):
        test_loss_gn = test_pred_loss(loss_function, X_gn, pred_net, C, dim, reg, plot=True, maxiter=5000)  
      test_loss_rn = test_pred_loss(loss_function, X_rn, pred_net, C, dim, reg, plot=True, maxiter=5000)
      test_loss_rs = test_pred_loss(loss_function, X_rs, pred_net, C, dim, reg, plot=True, maxiter=5000)
      test_loss_rn_rs = test_pred_loss(loss_function, X_rn_rs, pred_net, C, dim, reg, plot=True, maxiter=5000)
      test_loss_mnist = test_pred_loss(loss_function, X_mnist, pred_net, C, dim, reg, plot=True, maxiter=5000)
      test_loss_omniglot = test_pred_loss(loss_function, X_omniglot, pred_net, C, dim, reg, plot=True, maxiter=5000)
      test_loss_cifar = test_pred_loss(loss_function, X_cifar, pred_net, C, dim, reg, plot=True, maxiter=5000)
      test_loss_flowers = test_pred_loss(loss_function, X_flowers, pred_net, C, dim, reg, plot=True, maxiter=5000)
      
      
      test_losses_rn.append(test_loss_rn)
      test_losses_rs.append(test_loss_rs)
      test_losses_rn_rs.append(test_loss_rn_rs)
      test_losses_mnist.append(test_loss_mnist)
      test_losses_omniglot.append(test_loss_omniglot)
      test_losses_cifar.append(test_loss_cifar)
      test_losses_flowers.append(test_loss_flowers)

 
      # emd2 relative error testing

      if (learn_gen == True):
        rel_err_gn = test_pred_dist(X_gn[:50], pred_net, C, reg, dim, 'Gen Net')
      rel_err_rn = test_pred_dist(X_rn[:50], pred_net, C, reg, dim, 'Random Noise')
      rel_err_rs = test_pred_dist(X_rs[:50], pred_net, C, reg, dim, 'Random Shapes')
      rel_err_rn_rs = test_pred_dist(X_rn_rs[:50], pred_net, C, reg, dim, 'Random Noise and Random Shapes')
      rel_err_mnist = test_pred_dist(X_mnist[:50], pred_net, C, reg, dim, 'MNIST')
      rel_err_omniglot = test_pred_dist(X_omniglot[:50], pred_net, C, reg, dim, 'OMNIGLOT')
      rel_err_cifar = test_pred_dist(X_cifar[:50], pred_net, C, reg, dim, 'CIFAR10')
      rel_err_flowers = test_pred_dist(X_flowers[:50], pred_net, C, reg, dim, 'FLOWERS102')

      rel_errs_rn.append(rel_err_rn)
      rel_errs_rs.append(rel_err_rs)
      rel_errs_rn_rs.append(rel_err_rn_rs)
      rel_errs_mnist.append(rel_err_mnist)
      rel_errs_omniglot.append(rel_err_omniglot)
      rel_errs_cifar.append(rel_err_cifar)
      rel_errs_flowers.append(rel_err_flowers)
      


      # Displaying loss and error information
      print(f"Rel err rn: {rel_err_rn}")
      print(f"Rel err rs: {rel_err_rs}")
      print(f"Rel err rn_rs: {rel_err_rn_rs}")
      print(f"Rel err mnist: {rel_err_mnist}")
      print(f"Rel err omniglot: {rel_err_omniglot}")
      print(f"Rel err cifar: {rel_err_cifar}")
      print(f"Rel err flowers: {rel_err_flowers}")

      plt.figure()
      plt.grid()
      plt.plot(torch.log(torch.tensor(train_losses)))
      plt.title('Log Train Losses')
      plt.xlabel('# minibatches')
      plt.ylabel('log loss')
      plt.show()
      plt.figure()
      plt.plot(torch.log(torch.tensor(test_losses_rn)), label='rn')
      plt.plot(torch.log(torch.tensor(test_losses_rs)), label='rs')
      plt.plot(torch.log(torch.tensor(test_losses_rn_rs)), label='rn_rs')
      plt.plot(torch.log(torch.tensor(test_losses_mnist)), label='mnist')
      plt.plot(torch.log(torch.tensor(test_losses_omniglot)), label='omniglot')
      plt.plot(torch.log(torch.tensor(test_losses_cifar)), label='cifar')
      plt.plot(torch.log(torch.tensor(test_losses_flowers)), label='flowers')
      plt.legend()
      plt.grid()
      plt.xlabel('# test phases')
      plt.ylabel('log loss')
      plt.title('Log Test Losses')
      plt.show()
      plt.figure()
      plt.grid()
      plt.plot(torch.tensor(rel_errs_rn), label='rn')
      plt.plot(torch.tensor(rel_errs_rs), label='rs')
      plt.plot(torch.tensor(rel_errs_rn_rs), label='rn_rs')
      plt.plot(torch.tensor(rel_errs_mnist), label='mnist')
      plt.plot(torch.tensor(rel_errs_omniglot), label='omniglot')
      plt.plot(torch.tensor(rel_errs_cifar), label='cifar')
      plt.plot(torch.tensor(rel_errs_flowers), label='flowers')
      plt.legend()
      plt.yticks(torch.arange(0, 1.0001, 0.05))
      plt.title(' Rel Error: Pred Net Dist VS ot.emd2')
      plt.show()
    
    # Training predictive neural net.

    # Setting networks to train mode
    pred_net.train()
    if (learn_gen == True):
      gen_net.train()
      
    # Data creation
    if (learn_gen == True):
      sample = prior_sampler(batchsize, dim_prior).double().to(device)
      X = gen_net(sample)
    else:
      #sample = random_shapes_loader(batchsize, dim, dust_const).double().to(device)
      #sample = rando(batchsize, dim, dust_const).double().to(device)
      sample = rn_rs(batchsize, dim, dust_const).double().to(device)
      X = sample
    
    # Target creation
    with torch.no_grad():
      V0 = torch.ones_like(X[:, :dim])
      V = sink_vec(X[:, :dim], X[:, dim:], C, reg, V0, 1000)
      V = torch.log(V)
      V = V - torch.unsqueeze(V.mean(dim=1), 1).repeat(1, dim)
    T = V

    # Training loop
    for e in range(epochs):
      perm = torch.randperm(batchsize).to(device)
      X_e, T_e = X[perm], T[perm]
      for j in range(batchsize//minibatch):        
        P_mini = pred_net(X_e[j*minibatch:(j+1)*minibatch])
        T_mini = T_e[j*minibatch:(j+1)*minibatch]
        pred_loss = loss_function(P_mini, T_mini)
        train_losses.append(pred_loss.item())

        # Update
        pred_optimizer.zero_grad()
        pred_loss.backward(retain_graph=True)
        pred_optimizer.step()
    
    # Training generative neural net.
    if (learn_gen == True):
      for j in range(batchsize//minibatch):        
        P_mini = pred_net(X[j*minibatch:(j+1)*minibatch])
        T_mini = T[j*minibatch:(j+1)*minibatch]
        gen_loss = -loss_function(P_mini, T_mini)
        train_losses.append(gen_loss.item())

        # Update
        gen_optimizer.zero_grad()
        gen_loss.backward(retain_graph=True)
        gen_optimizer.step()
      
    # Updating learning rates
    if (learn_gen == True):
      gen_scheduler.step()
    pred_scheduler.step()

    # Printing batch information
    print('------------------------------------------------')
    print(f"Batch: {i+1}")
    print(f"Train loss: {pred_loss.item()}")
    if (learn_gen == True):
      print(f"gen lr: {gen_scheduler.get_last_lr()[0]}")
    print(f"pred lr: {pred_scheduler.get_last_lr()[0]}")
  
  # Converting loss and error collecting lists to tensors
  train_losses = torch.tensor(train_losses)
  test_losses_rn = torch.tensor(test_losses_rn)
  test_losses_rs = torch.tensor(test_losses_rs)
  test_losses_rn_rs = torch.tensor(test_losses_rn_rs)
  test_losses_mnist = torch.tensor(test_losses_mnist)
  test_losses_omniglot = torch.tensor(test_losses_omniglot)
  test_losses_cifar = torch.tensor(test_losses_cifar)
  test_losses_flowers = torch.tensor(test_losses_flowers)
  rel_errs_rn = torch.tensor(rel_errs_rn)
  rel_errs_rs = torch.tensor(rel_errs_rs)
  rel_errs_rn_rs = torch.tensor(rel_errs_rn_rs)
  rel_errs_mnist = torch.tensor(rel_errs_mnist)
  rel_errs_omniglot = torch.tensor(rel_errs_omniglot)
  rel_errs_cifar = torch.tensor(rel_errs_cifar)
  rel_errs_flowers = torch.tensor(rel_errs_flowers)

  return (
          train_losses,
          test_losses_rn,
          test_losses_rs,
          test_losses_rn_rs,
          test_losses_mnist,
          test_losses_omniglot,
          test_losses_cifar,
          test_losses_flowers,
          rel_errs_rn,
          rel_errs_rs,
          rel_errs_rn_rs,
          rel_errs_mnist,
          rel_errs_omniglot,
          rel_errs_cifar,
          rel_errs_flowers
  )