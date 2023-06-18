"""
Let the hunt begin!
"""

# Imports
import datetime
import os
import torch
from cost_matrices import l2_cost_mat
from training_algo import the_hunt
from net import PredNet
from utils import hilb_proj_loss, plot_train_losses, plot_test_losses, plot_test_rel_errs, test_set_sampler, preprocessor
from test import test_warmstart

# Create 'stamp' folder for saving results
current_time = datetime.datetime.now()
formatted_time = current_time.strftime('%m-%d_%H_%M_%S')
stamp_folder_path = './stamp_neuro_'+formatted_time
os.mkdir(stamp_folder_path)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Hyperparameters
length = 28
dim = length**2
dust_const = 5e-6
mutation_sigma = 0.1
width_pred = 4 * dim

# Test sets
mnist = torch.load('./data/mnist_tensor.pt')
omniglot = torch.load('./data/omniglot_tensor.pt')
cifar = torch.load('./data/cifar_tensor.pt')
teddies = torch.load('./data/teddies_tensor.pt')

mnist = preprocessor(mnist, length, dust_const)
omniglot = preprocessor(omniglot, length, dust_const)
cifar = preprocessor(cifar, length, dust_const)
teddies = preprocessor(teddies, length, dust_const)

test_sets = {'mnist': mnist, 'omniglot': omniglot, 'cifar': cifar,
             'teddies': teddies}
             
# Initialization of cost matrix
cost_mat = l2_cost_mat(length, length, normed=True).double().to(device)

# Regularization parameter
eps = cost_mat.max() * 4e-4
print(f'Regularization parameter: {eps}')

# Initialization of loss function
loss_func = hilb_proj_loss

# Initialization of nets
puma = PredNet(dim, width_pred).double().to(device)

# no. layers in each net
n_layers_pred = len(puma.layers)

# Load model state dict
#puma.load_state_dict(torch.load(f'{stamp_folder_path}/puma.pt'))

# Training mode
puma.train()

# Training Hyperparams
n_loops = 5000
n_batch = 500
lr = 0.05
lr_factor = 1.0
bootstrapped = True
boot_no = 40
test_iter = 500
n_test_samples = 200
checkpoint = 10000
n_warmstart_samples = 50

# Create txt file in stamp for hyperparams
current_date = datetime.datetime.now().strftime('%d.%m.%Y')
hyperparams = {
    'date': current_date,
    'data length': length,
    'data dimension': dim,
    'regularization parameter': eps,
    'dust constant': dust_const,
    'no. layers pred': n_layers_pred,
    'hidden layer width pred': width_pred,
    'learning rate': lr,
    'learning rates scale factor': lr_factor,
    'no. unique data points in training': n_loops*n_batch,
    'no. loops' : n_loops,
    'batch size': n_batch,
    'test_iter': test_iter,
    'no. test samples': n_test_samples,
    'bootstrapped?': bootstrapped,
    'no. bootstraps': boot_no,
    'checkpoint': checkpoint,
    'no warmstart samples': n_warmstart_samples
}

# Define the output file path
output_file = f'{stamp_folder_path}/params.txt'

# Save the hyperparams to the text file
with open(output_file, 'w') as file:
    for key, value in hyperparams.items():
        file.write(f'{key}: {value}\n')


# Run the hunt
train_losses, test_losses, test_rel_errs = the_hunt(
        puma,
        dim,
        cost_mat,
        eps,
        dust_const,
        device,
        loss_func,
        n_loops,
        n_batch,
        mutation_sigma,
        lr,
        lr_factor,
        bootstrapped,
        boot_no,
        test_sets,
        test_iter,
        n_test_samples,
        checkpoint,
        n_warmstart_samples,
        stamp_folder_path,
        )

# Testing mode
puma.eval()

# Saving nets
torch.save(puma.state_dict(), f'{stamp_folder_path}/puma.pt')

# Plot the results
plot_train_losses(train_losses, f'{stamp_folder_path}/train_losses.png')
plot_test_losses(test_losses, f'{stamp_folder_path}/test_losses.png')
plot_test_rel_errs(test_rel_errs, f'{stamp_folder_path}/test_rel_errs.png')

# Test warmstart
test_warmstart_trials = {}
for key in test_sets.keys():
    X_test = test_set_sampler(test_sets[key], n_warmstart_samples).double().to(device)
    test_warmstart_trials[key] = test_warmstart(puma, X_test, cost_mat, eps, dim, key, f'{stamp_folder_path}/warm_start_{key}.png')
