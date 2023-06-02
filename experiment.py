"""
Let the hunt begin!
"""

# Imports
import datetime
import os
import torch
import matplotlib.pyplot as plt
from cost_matrices import l2_cost_mat
from training_algo import the_hunt
from nets import GenNet, PredNet
from utils import hilb_proj_loss, plot_train_losses, plot_test_losses, plot_test_rel_errs, test_set_sampler
from data_creators import rand_noise, rand_shapes, rand_noise_and_shapes, get_cifar, get_omniglot, get_mnist, get_flowers
from test_funcs import test_warmstart

# Create 'stamp' folder for saving results
current_time = datetime.datetime.now()
formatted_time = current_time.strftime('%m-%d_%H_%M_%S')
stamp_folder_path = './stamp_'+formatted_time
os.mkdir(stamp_folder_path)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Hyperparameters
length_prior = 10
length = 28
dim_prior = length_prior**2
dim = length**2
dust_const = 1e-5
skip_const = 0.2
width = 4 * dim

# Create/download testsets
rn = rand_noise(5000, dim, dust_const)
rs = rand_shapes(5000, dim, dust_const)
rn_rs = rand_noise_and_shapes(5000, dim, dust_const)
mnist = get_mnist(length, dust_const, download=True)
omniglot = get_omniglot(length, dust_const, download=True)  
cifar = get_cifar(length, dust_const, download=True)
flowers = get_flowers(length, dust_const,  download=True)   

# Create test sets dictionary
test_sets = {'rn': rn, 'rs': rs, 'rn_rs': rn_rs, 'mnist': mnist,
             'omniglot': omniglot, 'cifar': cifar, 'flowers': flowers}
             
# Initialization of cost matrix
cost_mat = l2_cost_mat(length, length, normed=True).double().to(device)

# Regularization parameter
eps = cost_mat.max() * 6e-4
print(f'Regularization parameter: {eps}')

# Initialization of loss function
loss_func = hilb_proj_loss

# Initialization of nets
deer = GenNet(dim_prior, dim, dust_const, skip_const).double().to(device)
puma = PredNet(dim, width).double().to(device)

# Load model state dict
#deer.load_state_dict(torch.load(f'{stamp_folder_path}/deer.pt'))
#puma.load_state_dict(torch.load(f'{stamp_folder_path}/puma.pt'))

# Training mode
deer.train()
puma.train()

# Training Hyperparams
n_samples = 1000000
batch_size = 2500
minibatch_size = 500
n_epochs_gen = 10
n_epochs_pred = 10
lr_pred = 0.1
lr_gen = 0.1
lr_factor = 0.999
learn_gen = False
bootstrapped = True
boot_no = 10
test_iter = 100
n_test_samples = 100


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
        n_samples,
        batch_size,
        minibatch_size,
        n_epochs_gen,
        n_epochs_pred,
        lr_pred,
        lr_gen,
        lr_factor,
        learn_gen,
        bootstrapped,
        boot_no,
        test_iter,
        n_test_samples,
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
    'hidden layer width': width,
    'gen net learning rate': lr_gen,
    'pred net learning rate': lr_pred,
    'learning rates scale factor': lr_factor,
    'no. unique data points': n_samples,
    'batch size': batch_size,
    'minibatch size': minibatch_size,
    'gen net training epochs': n_epochs_gen,
    'pred net training epochs': n_epochs_pred,
    'test_iter': test_iter,
    'learn gen?': learn_gen,
    'bootstrapped?': bootstrapped,
    'no. bootstraps': boot_no,
}

# Define the output file path
output_file = f'{stamp_folder_path}/params.txt'

# Save the hyperparams to the text file
with open(output_file, 'w') as file:
    for key, value in hyperparams.items():
        file.write(f'{key}: {value}\n')

# Test warmstart
for key in test_sets.keys():
    X_test = test_set_sampler(test_sets[key], n_test_samples).double().to(device)
    test_warmstart(puma, X_test, n_test_samples, cost_mat, eps, dim, key,
                   device, dust_const, plot=True)