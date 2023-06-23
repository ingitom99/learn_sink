"""
Let the hunt begin!
"""

# Imports
import datetime
import os
import torch
from cost_matrices import l2_cost_mat
from sinkhorn import sink_vec
from train import the_hunt
from nets import GenNet, PredNet
from loss import hilb_proj_loss
from plot import plot_train_losses, plot_test_losses, plot_test_rel_errs_emd, plot_test_rel_errs_sink
from test_funcs import test_warmstart
from data_funcs import preprocessor, test_set_sampler

# Create 'stamp' folder for saving results
current_time = datetime.datetime.now()
formatted_time = current_time.strftime('%m-%d_%H_%M_%S')
stamp_folder_path = './stamp_'+formatted_time
os.mkdir(stamp_folder_path)

# Hyperparameters
length_prior = 7
length = 28
dim_prior = length_prior**2
dim = length**2
dust_const = 1e-6
skip_const = 0.5
width_gen = 4 * dim
width_pred = 6 * dim

# Training Hyperparams
n_loops = 50000
n_mini_loops_gen = 3
n_mini_loops_pred = 3
n_batch = 500
lr_gen = 0.1
lr_pred = 0.1
lr_factor = 0.9998
learn_gen = True
bootstrapped = True
n_boot = 40
extend_data = True
test_iter = 1000
n_test = 200
checkpoint = 10000
n_warmstart = 50

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Initialization of cost matrix
cost_mat = l2_cost_mat(length, length, normed=True).double().to(device)

# Regularization parameter
eps = cost_mat.max() * 4e-4

mnist = torch.load('./data/mnist_tensor.pt')
omniglot = torch.load('./data/omniglot_tensor.pt')
cifar = torch.load('./data/cifar_tensor.pt')
teddies = torch.load('./data/teddies_tensor.pt')

mnist = preprocessor(mnist, length, dust_const)
omniglot = preprocessor(omniglot, length, dust_const)
cifar = preprocessor(cifar, length, dust_const)
teddies = preprocessor(teddies, length, dust_const)

mnist = test_set_sampler(mnist, n_test).double().to(device)
omniglot = test_set_sampler(omniglot, n_test).double().to(device)
cifar = test_set_sampler(cifar, n_test).double().to(device)
teddies = test_set_sampler(teddies, n_test).double().to(device)

# Create test sets dictionary
test_sets = {'mnist': mnist, 'omniglot': omniglot, 'cifar': cifar,
             'teddies': teddies}

# for each test set, create a dictionary of test emds and test sinks
test_emd = {}
test_sink = {}
test_T = {}

for key in test_sets.keys():
    with torch.no_grad():

        X = test_sets[key]

        V0 = torch.ones_like(X[:, :dim])
        V = sink_vec(X[:, :dim], X[:, dim:], cost_mat, eps, V0, 2000)[1]
        V = torch.log(V)
        T = V - torch.unsqueeze(V.mean(dim=1), 1).repeat(1, dim)
        test_T[key] = T
    
        emds = []
        sinks = []
        for x in X:
            mu = x[:dim] / x[:dim].sum()
            nu = x[dim:] / x[dim:].sum()
            emd = ot.emd2(mu, nu, cost_mat)
            emds.append(emd)
            sink = ot.sinkhorn2(mu, nu, cost_mat, eps, numItermax=2000)
            sinks.append(sink)
        emds = torch.tensor(emds)
        sinks = torch.tensor(sinks)
        test_emd[key] = emds
        test_sink[key] = sinks

# Initialization of loss function
loss_func = hilb_proj_loss

# Initialization of nets
deer = GenNet(dim_prior, dim, width_gen, dust_const,
              skip_const).double().to(device)
puma = PredNet(dim, width_pred).double().to(device)

# no. layers in each net
n_layers_gen = len(deer.layers)
n_layers_pred = len(puma.layers)

# Load model state dict
#deer.load_state_dict(torch.load(f'{stamp_folder_path}/deer.pt'))
#puma.load_state_dict(torch.load(f'{stamp_folder_path}/puma.pt'))

# Training mode
deer.train()
puma.train()

# Create txt file in stamp for hyperparams
current_date = datetime.datetime.now().strftime('%d.%m.%Y')
hyperparams = {
    'date': current_date,
    'prior distribution length': length_prior,
    'data length': length,
    'prior distribution dimension': dim_prior,
    'data dimension': dim,
    'regularization parameter': eps,
    'dust constant': dust_const,
    'skip connection constant': skip_const,
    'no. layers gen': n_layers_gen,
    'no. layers pred': n_layers_pred,
    'hidden layer width gen': width_gen,
    'hidden layer width pred': width_pred,
    'device': device,
    'gen net learning rate': lr_gen,
    'pred net learning rate': lr_pred,
    'learning rates scale factor': lr_factor,
    'no. unique data points gen': n_loops*n_mini_loops_gen*n_batch,
    'no. unique data points pred': n_loops*n_mini_loops_pred*n_batch,
    'no. loops' : n_loops,
    'no. mini loops gen' : n_mini_loops_gen,
    'no. mini loops pred' : n_mini_loops_pred,
    'batch size': n_batch,
    'test_iter': test_iter,
    'no. test samples': n_test,
    'learn gen?': learn_gen,
    'bootstrapped?': bootstrapped,
    'no. bootstraps': n_boot,
    'extend data?': extend_data,
    'checkpoint': checkpoint,
    'no warmstart samples': n_warmstart
}

# Define the output file path
output_file = f'{stamp_folder_path}/params.txt'

# Save the hyperparams to the text file
with open(output_file, 'w', encoding='utf-8') as file:
    for key, value in hyperparams.items():
        file.write(f'{key}: {value}\n')


# Run the hunt
train_losses, test_losses, test_rel_errs = the_hunt(
        deer,
        puma,
        loss_func,
        cost_mat,        
        eps,
        dust_const,
        dim_prior,
        dim,
        device,
        test_sets,
        n_loops,
        n_mini_loops_gen,
        n_mini_loops_pred,
        n_batch,
        lr_pred,
        lr_gen,
        lr_factor,
        learn_gen,
        bootstrapped,
        n_boot,
        extend_data,
        test_iter,
        n_test,
        stamp_folder_path,
        checkpoint,
        n_warmstart
        )

# Testing mode
deer.eval()
puma.eval()

# Saving nets
torch.save(deer.state_dict(), f'{stamp_folder_path}/deer.pt')
torch.save(puma.state_dict(), f'{stamp_folder_path}/puma.pt')

# Plot the results
plot_train_losses(train_losses, f'{stamp_folder_path}/train_losses.png')
plot_test_losses(test_losses, f'{stamp_folder_path}/test_losses.png')
plot_test_rel_errs(test_rel_errs, f'{stamp_folder_path}/test_rel_errs.png')

# Test warmstart
test_warmstart_trials = {}
for key in test_sets.keys():
    X_test = test_set_sampler(test_sets[key],
                              n_warmstart).double().to(device)
    test_warmstart_trials[key] = test_warmstart(puma, X_test, cost_mat, eps,
                        dim, key, f'{stamp_folder_path}/warm_start_{key}.png')