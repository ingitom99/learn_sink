"""
Let the hunt begin!
"""

# Imports
import datetime
import os
import torch
from cost_matrices import l2_cost_mat
from training_algo import the_hunt
from nets import GenNet, PredNet
from utils import hilb_proj_loss, plot_train_losses, plot_test_losses, plot_test_rel_errs, test_set_sampler
from data_creators import rand_noise, get_cifar, get_omniglot, get_mnist, get_flowers, get_lfw
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
length_prior = 14
length = 28
dim_prior = length_prior**2
dim = length**2
dust_const = 1e-5
skip_const = 0.3
width_gen = 4 * dim
width_pred = 4 * dim

# Create/download testsets
rn = rand_noise(5000, dim, dust_const, False)
mnist = get_mnist(length, dust_const, download=True)
omniglot = get_omniglot(length, dust_const, download=True)  
cifar = get_cifar(length, dust_const, download=True)
lfw = get_lfw(length, dust_const, download=True)   

# Create test sets dictionary
test_sets = {'rn': rn, 'mnist': mnist, 'omniglot': omniglot, 'cifar': cifar,
             'lfw': lfw}
             
# Initialization of cost matrix
cost_mat = l2_cost_mat(length, length, normed=True).double().to(device)

# Regularization parameter
eps = cost_mat.max() * 6e-4
print(f'Regularization parameter: {eps}')

# Initialization of loss function
loss_func = hilb_proj_loss

# Initialization of nets
deer = GenNet(dim_prior, dim, width_gen, dust_const,
              skip_const).double().to(device)
puma = PredNet(dim, width_pred).double().to(device)

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
n_epochs_gen = 5
n_epochs_pred = 5
lr_pred = 0.1
lr_gen = 0.1
lr_factor = 1.0
learn_gen = True
bootstrapped = True
boot_no = 10
test_iter = 50
n_test_samples = 100
checkpoint = 200
n_warmstart_samples = 50

# Create txt file in stamp for hyperparams
current_date = datetime.datetime.now().strftime('%d.%m.%Y')
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
    'hidden layer width gen': width_gen,
    'hidden layer width pred': width_pred,
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
        stamp_folder_path,
        checkpoint,
        n_warmstart_samples
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
                              n_test_samples).double().to(device)
    test_warmstart_trials[key] = test_warmstart(puma, X_test, C, eps,
                        dim, key, f'{stamp_folder_path}/warm_start_{key}.png')