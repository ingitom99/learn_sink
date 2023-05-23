import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sinkhorn_algos import sink_vec
from utils import MNIST_test_loader, random_shapes_loader, prior_sampler, rando
from test_funcs import test_pred_edm2, test_pred_loss

def the_hunt(gen_net,
             pred_net,
             C,
             reg,
             dim_in,
             dim,
             loss_function,
             device,
             MNIST,
             OMNIGLOT,
             dust_const,
             lr_gen=0.5,
             lr_pred=0.5,
             lr_factor=0.999,
             n_samples= 1000000,
             batchsize=1000,
             minibatch=200,
             epochs=5,
             test_iter=100,
             learn_gen=True
             ):
  
  # Loss and error collecting lists
  train_losses = []
  test_losses_rn = []
  test_losses_mnist = []
  test_losses_omniglot = []
  rel_errs_rn = []
  rel_errs_mnist = []
  rel_errs_omniglot = []

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
    if (i % test_iter == 0):
    
      # Testing data
      X_rn = rando(200, dim, dust_const).double().to(device)
      X_mnist = MNIST_test_loader(MNIST, 200).double().to(device)
      X_omniglot = MNIST_test_loader(OMNIGLOT, 200).double().to(device)

      # Loss testing
      test_loss_rn = test_pred_loss(loss_function, X_rn, pred_net, C, dim, reg, plot=True, maxiter=5000)
      test_loss_mnist = test_pred_loss(loss_function, X_mnist, pred_net, C, dim, reg, plot=True, maxiter=5000)
      test_loss_omniglot = test_pred_loss(loss_function, X_omniglot, pred_net, C, dim, reg, plot=True, maxiter=5000)
      
      test_losses_rn.append(test_loss_rn)
      test_losses_mnist.append(test_loss_mnist)
      test_losses_omniglot.append(test_loss_omniglot)

 
      # emd2 relative error testing
      rel_err_rn = test_pred_edm2(X_rn[:50], pred_net, C, reg, dim, 'Random Noise')
      rel_err_mnist = test_pred_edm2(X_mnist[:50], pred_net, C, reg, dim, 'MNIST')
      rel_err_omniglot = test_pred_edm2(X_omniglot[:50], pred_net, C, reg, dim, 'OMNIGLOT')

      rel_errs_rn.append(rel_err_rn)
      rel_errs_mnist.append(rel_err_mnist)
      rel_errs_omniglot.append(rel_err_omniglot)


      # Displaying loss and error information
      print(f"Rel err rn: {rel_err_rn}")
      print(f"Rel err mnist: {rel_err_mnist}")
      print(f"Rel err omniglot: {rel_err_omniglot}")
      plt.figure()
      plt.plot(torch.log(torch.tensor(train_losses)))
      plt.title('Log Train Losses')
      plt.show()
      plt.figure()
      plt.plot(torch.log(torch.tensor(test_losses_rn)), label='rn')
      plt.plot(torch.log(torch.tensor(test_losses_mnist)), label='mnist')
      plt.plot(torch.log(torch.tensor(test_losses_omniglot)), label='omniglot')
      plt.legend()
      plt.title('Log Test Losses')
      plt.show()
      plt.figure()
      plt.plot(torch.tensor(rel_errs_rn), label='rn')
      plt.plot(torch.tensor(rel_errs_mnist), label='mnist')
      plt.plot(torch.tensor(rel_errs_omniglot), label='omniglot')
      plt.legend()
      plt.title('Predicted Distance Relative Error Versus ot.emd2')
      plt.show()
    
    # Training predictive neural net.
    
    if (learn_gen == True):
      sample = prior_sampler(batchsize, dim_in).double().to(device)
      #sample = torch.ones((batchsize, 2*dim_in)).double().to(device)
      X = gen_net(sample)
    else:
      sample = rando(batchsize, dim, dust_const).double().to(device)
      X = sample
    with torch.no_grad():
      V = sink_vec(X[:, :dim], X[:, dim:], C, reg, 1000, V0=None)
      V = torch.log(V)
      V = V - torch.unsqueeze(V.mean(dim=1), 1).repeat(1, dim)
    T = V
    for e in range(epochs):
      perm = torch.randperm(batchsize).to(device)
      X_e, T_e = X[perm], T[perm]
      for j in range(batchsize//minibatch):        
        P_mini = pred_net(X_e[j*minibatch:(j+1)*minibatch])
        T_mini = T_e[j*minibatch:(j+1)*minibatch]
        pred_loss = loss_function(P_mini, T_mini)
        train_losses.append(pred_loss.item())
        pred_optimizer.zero_grad()
        pred_loss.backward(retain_graph=True)
        pred_optimizer.step()
      
    if (i % 50 == 0):
      fig, ax = plt.subplots(1, 2)
      ax[0].imshow(X_e[-1, :dim].cpu().detach().numpy().reshape(28,28), cmap='magma')
      ax[1].imshow(X_e[-1, dim:].cpu().detach().numpy().reshape(28,28), cmap='magma')
      ax[0].set_title('MU')
      ax[1].set_title('NU')
      plt.show()
      plt.figure()
      plt.title('T')
      plt.imshow(T_mini[-1].cpu().detach().numpy().reshape(28,28), cmap='magma')
      plt.colorbar()
      plt.show()
      plt.figure()
      plt.title('P')
      plt.imshow(P_mini[-1].cpu().detach().numpy().reshape(28,28), cmap='magma')
      plt.colorbar()
      plt.show()
    
    if (learn_gen == True):
      for j in range(batchsize//minibatch):        
        P_mini = pred_net(X[j*minibatch:(j+1)*minibatch])
        T_mini = T[j*minibatch:(j+1)*minibatch]
        gen_loss = -loss_function(P_mini, T_mini)
        train_losses.append(gen_loss.item())
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
  test_losses_mnist = torch.tensor(test_losses_mnist)
  rel_errs_rn = torch.tensor(rel_errs_rn)
  rel_errs_mnist = torch.tensor(rel_errs_mnist)

  return train_losses, test_losses_rn, test_losses_mnist, test_losses_omniglot, rel_errs_rn, rel_errs_mnist, rel_errs_omniglot