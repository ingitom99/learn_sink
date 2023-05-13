import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import prior_sampler
from sinkhorn_algos import sink_vec

def the_hunt(gen_net,
             pred_net,
             C,
             reg,
             dim_prior,
             dim,
             loss_function,
             device,
             lr_gen=0.01,
             lr_pred=0.1,
             lr_factor=1.0,
             n_samples= 100000,
             batchsize=500,
             minibatch=100,
             epochs=5,
             train_gen=True):
  
  # Loss collecting lists.
  train_losses = []
  test_losses_pn = []
  test_losses_mnist = []
  test_losses_rs = []
  test_losses_rn = []
  
  # Initializing optimizers.
  gen_optimizer = torch.optim.SGD(gen_net.parameters(), lr=lr_gen)
  pred_optimizer = torch.optim.SGD(pred_net.parameters(), lr=lr_pred)

  # Initializing learning rate schedulers.
  pred_scheduler = torch.optim.lr_scheduler.ExponentialLR(pred_optimizer, gamma=lr_factor)
  gen_scheduler = torch.optim.lr_scheduler.ExponentialLR(gen_optimizer, gamma=lr_factor)

  for i in tqdm(range(n_samples//batchsize)):
    # Testing Section
    # if (i % 5 == 0):
      # Updating losses.
      # test_loss_pn = test_puma_pn(loss_function, pred_net, gen_net, dim, reg, 20, plot=False)
      # test_loss_mnist = test_puma_mnist(loss_function, MNIST_TEST, pred_net, dim, reg, 20, plot=False)
      # test_loss_rs = test_puma_rs(loss_function, pred_net, dim, reg, 20, plot=False)
      # test_loss_rn = test_puma_rn(loss_function, pred_net, dim, reg, 20, plot=False)
      # test_losses_pn.append(test_loss_pn.item())
      # test_losses_mnist.append(test_loss_mnist.item())
      # test_losses_rs.append(test_loss_rs.item())
      # test_losses_rn.append(test_loss_rn.item())

      # Displaying loss information.
      # print('------------------------------------------------')
      # print(f"Test loss gen net: {test_loss_pn.item()}")
      # print(f"Test loss mnist: {test_loss_mnist.item()}")
      # print(f"Test loss rand shapes: {test_loss_rs.item()}")
      # print(f"Test loss rand noise: {test_loss_rn.item()}")
      # plt.figure()
      # plt.plot(torch.log(torch.tensor(train_losses)))
      # plt.title('Train Losses')
      # plt.show()
      # plt.figure()
      # plt.plot(torch.log(torch.tensor(test_losses_pn)), label='pn')
      # plt.plot(torch.log(torch.tensor(test_losses_mnist)), label='mnist')
      # plt.plot(torch.log(torch.tensor(test_losses_rs)), label='rs')
      # plt.plot(torch.log(torch.tensor(test_losses_rn)), label='rn')
      # plt.legend()
      # plt.title('Test Losses')
      # plt.show()
    
    # Training predictive neural net.
    sample = prior_sampler(batchsize, dim_prior).double().to(device)

    X = gen_net(sample)
    with torch.no_grad():
      V0 = torch.ones_like(X[:, :dim]).double().to(device)
      T = torch.log(sink_vec(X[:, :dim], X[:, dim:], C, reg, 1000, V0))
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

    # Training generative neural net.
    if (train_gen == True) & (i%5 == 0):
      for j in range(batchsize//minibatch):
        P_gen = pred_net(X[j*minibatch:(j+1)*minibatch])
        gen_optimizer.zero_grad()
        gen_loss = -loss_function(P_gen, T[j*minibatch:(j+1)*minibatch])
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
  
  # COnverting loss collecting lists to tensors.
  train_losses = torch.tensor(train_losses)
  test_losses_pn = torch.tensor(test_losses_pn)
  test_losses_mnist = torch.tensor(test_losses_mnist)
  test_losses_rs = torch.tensor(test_losses_rs)
  test_losses_rn = torch.tensor(test_losses_rn)

  return train_losses, test_losses_pn, test_losses_mnist, test_losses_rs, test_losses_rn