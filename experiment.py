"""
experiment.py
-------------

Script for creating, training and testing a puma and a deer and saving the
results.

Let the hunt begin!
"""

# Imports
import datetime
import os
import torch
import ot
from tqdm import tqdm
from cost import l2_cost
from sinkhorn import sink_vec
from train import the_hunt
from net import PredNet
from loss import hilb_proj_loss
from data_funcs import preprocessor, test_set_sampler

# Create 'stamp' folder for saving results
current_time = datetime.datetime.now()
formatted_time = current_time.strftime('%m-%d_%H_%M_%S')
stamp_folder_path = './stamp_neuro_'+formatted_time
os.mkdir(stamp_folder_path)

# Problem hyperparameters
length = 28
dim = length**2
dust_const = 5e-6
width_pred = 6 * dim
mutation_sigma = 0.1

# Training Hyperparams
n_loops = 10000
n_batch = 500
lr = 0.01
lr_fact = 0.999
test_iter = 1000
n_test = 500
checkpoint = 10000

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Initialization of cost matrix
cost_mat = l2_cost(length, length, normed=True).double().to(device)

# Regularization parameter
eps = cost_mat.max() * 4e-4
print(f'Entropic regularization param: {eps}')

# Loading, preprocessing, and sampling for the test sets dictionary
mnist = torch.load('./data/mnist.pt')
omniglot = torch.load('./data/omniglot.pt')
cifar = torch.load('./data/cifar.pt')
lfw = torch.load('./data/lfw.pt')
bear = torch.load('./data/bear.pt')
quickdraw = torch.load('./data/quickdraw.pt')

mnist = preprocessor(mnist, length, dust_const)
omniglot = preprocessor(omniglot, length, dust_const)
cifar = preprocessor(cifar, length, dust_const)
lfw = preprocessor(lfw, length, dust_const)
bear = preprocessor(bear, length, dust_const)
quickdraw = preprocessor(quickdraw, length, dust_const)

mnist = test_set_sampler(mnist, n_test).double().to(device)
omniglot = test_set_sampler(omniglot, n_test).double().to(device)
cifar = test_set_sampler(cifar, n_test).double().to(device)
lfw = test_set_sampler(lfw, n_test).double().to(device)
bear = test_set_sampler(bear, n_test).double().to(device)
quickdraw = test_set_sampler(quickdraw, n_test).double().to(device)

test_sets = {'mnist': mnist, 'omniglot': omniglot, 'cifar': cifar,
             'lfw' : lfw, 'bear': bear, 'quickdraw': quickdraw}

# Creating a dictionary of test emds, and test targets for each test set
test_emds = {}
test_T = {}

print('Computing test emds, sinks, and targets...')
for key in test_sets.keys():
    print(f'{key}:')
    with torch.no_grad():

        X = test_sets[key]

        V0 = torch.ones_like(X[:, :dim])
        V = sink_vec(X[:, :dim], X[:, dim:], cost_mat, eps, V0, 2000)[1]
        V = torch.log(V)
        T = V - torch.unsqueeze(V.mean(dim=1), 1).repeat(1, dim)
        test_T[key] = T
    
        emds = []
        for x in tqdm(X):
            mu = x[:dim] / x[:dim].sum()
            nu = x[dim:] / x[dim:].sum()
            emd = ot.emd2(mu, nu, cost_mat)
            emds.append(emd)
        emds = torch.tensor(emds)
        test_emds[key] = emds

# Initialization of loss function
loss_func = hilb_proj_loss

# Initialization of net
puma = PredNet(dim, width_pred).double().to(device)

# no. layers in net
n_layers_pred = len(puma.layers)

# Load model state dict
#puma.load_state_dict(torch.load(f'{stamp_folder_path}/puma.pt'))

# Training mode
puma.train()

# Get total trainable parameters
total_params_puma = sum(p.numel() for p in puma.parameters() if p.requires_grad)

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
    'total trainable parameters': total_params_puma,
    'device': device,
    'learning rate': lr,
    'learning rate scale factor': lr_fact,
    'no. unique training data points': n_loops*n_batch,
    'no. loops' : n_loops,
    'batch size': n_batch,
    'test_iter': test_iter,
    'no. test samples': n_test,
    'checkpoint': checkpoint,
}

# Print hyperparams
for key, value in hyperparams.items():
    print(f'{key}: {value}')

# Define the output file path
output_file = f'{stamp_folder_path}/params.txt'

# Save the hyperparams to the text file
with open(output_file, 'w', encoding='utf-8') as file:
    for key, value in hyperparams.items():
        file.write(f'{key}: {value}\n')

# Run the hunt
the_hunt(
    puma,
    loss_func,
    cost_mat,    
    eps,
    dust_const,
    mutation_sigma,
    dim,
    device,
    test_sets,
    test_emds,
    test_T,
    n_loops,
    n_batch,
    lr,
    lr_fact,
    test_iter,
    stamp_folder_path,
    checkpoint,
    )

print('The hunt is over. Time to rest.')
