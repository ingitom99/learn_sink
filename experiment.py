"""
Let the hunt begin!
"""

# Imports
import datetime
import os
import torch
import ot
from cost import l2_cost
from train import the_hunt
from nets import GenNet, PredNet
from loss import hilb_proj_loss
from data_funcs import preprocessor, test_set_sampler
from sinkhorn import sink_vec

# Create 'stamp' folder for saving results
current_time = datetime.datetime.now()
formatted_time = current_time.strftime('%m-%d_%H_%M_%S')
stamp_folder_path = './stamp_var_eps_'+formatted_time
os.mkdir(stamp_folder_path)

# Scenario Hyperparams
length_prior = 7
length = 28
dim_prior = length_prior**2
dim = length**2
dust_const = 1e-6
skip_const = 0.5
width_gen = 6 * dim
width_pred = 6 * dim
eps_test_const_val = 4e-4
min_eps_var = 5e-4
max_eps_var = 1e-3

# Training Hyperparams
n_loops = 10000
n_mini_loops_gen = 3
n_mini_loops_pred = 3
n_batch = 200
lr_gen = 0.1
lr_pred = 0.1
lr_fact_gen = 1.0
lr_fact_pred = 1.0
learn_gen = True
bootstrapped = True
n_boot = 40
n_test = 100
test_iter = 1000
checkpoint = 1000000

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Initialization of cost matrix
cost = l2_cost(length, length, normed=True).double().to(device)

# Regularization parameter
eps = cost.max() * 4e-4
print(f'Entropic egularization param: {eps}')

# Loading, preprocessing, and sampling for the test sets dictionary
mnist = torch.load('./data/mnist_tensor.pt')
omniglot = torch.load('./data/omniglot_tensor.pt')
cifar = torch.load('./data/cifar_tensor.pt')
lfw = torch.load('./data/lfw_tensor.pt')
bear = torch.load('./data/bear_tensor.pt')
quickdraw = torch.load('./data/quickdraw_tensor.pt')

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

test_sets = {'mnist': mnist, 'omniglot': omniglot, 'cifar': cifar, 'lfw': lfw,
            'bear': bear, 'quickdraw': quickdraw}

# create n_test x 1 vector of random epsilons
eps_test_var = torch.rand(n_test, 1).double().to(device)
eps_test_var = min_eps_var + (max_eps_var - min_eps_var) * eps_test_var
eps_test_const = eps_test_const_val * torch.ones_like(eps_test_var)

# for each test set, create a dictionary of test emds and test sinks
test_emds = {}
test_sinks = {}
test_T = {}

print('Computing test emds, sinks, and targets...')
for key in test_sets.keys():
    print(f'{key}:')
    with torch.no_grad():

        X = test_sets[key]

        V0 = torch.ones_like(X[:, :dim])
        V = sink_vec(X[:, :dim], X[:, dim:], cost, eps, V0, 2000)[1]
        V = torch.log(V)
        T = V - torch.unsqueeze(V.mean(dim=1), 1).repeat(1, dim)
        test_T[key] = T
    
        emds = []
        sinks = []
        for x, e in zip(X, eps_test_var):
            mu = x[:dim] / x[:dim].sum()
            nu = x[dim:] / x[dim:].sum()
            emd = ot.emd2(mu, nu, cost)
            emds.append(emd)
            sink = ot.sinkhorn2(mu, nu, cost, e, 2000)
            sinks.append(sink)
        emds = torch.tensor(emds)
        sinks = torch.tensor(sinks)
        test_emds[key] = emds
        test_sinks[key] = sinks

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

# Total number of parameters for each net
n_params_gen = sum(p.numel() for p in deer.parameters())
n_params_pred = sum(p.numel() for p in puma.parameters())
print(f'No. parameters in gen net: {n_params_gen}')
print(f'No. parameters in pred net: {n_params_pred}')

# Create txt file in stamp for hyperparams
current_date = datetime.datetime.now().strftime('%d.%m.%Y')
hyperparams = {
    'date': current_date,
    'prior distribution length': length_prior,
    'data length': length,
    'prior distribution dimension': dim_prior,
    'data dimension': dim,
    'small eps constant emd testing': eps_test_const_val,
    'min eps var': min_eps_var,
    'max eps var': max_eps_var,
    'dust constant': dust_const,
    'skip connection constant': skip_const,
    'no. layers gen': n_layers_gen,
    'no. layers pred': n_layers_pred,
    'hidden layer width gen': width_gen,
    'hidden layer width pred': width_pred,
    'No. trainable parameters gen': n_params_gen,
    'No. trainable parameters pred': n_params_pred,
    'device': device,
    'gen net learning rate': lr_gen,
    'pred net learning rate': lr_pred,
    'gen net learning rate factor': lr_fact_gen,
    'pred net learning rate factor': lr_fact_pred,
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
    'checkpoint': checkpoint,
}

# Define the output file path
output_file = f'{stamp_folder_path}/params.txt'

# Save the hyperparams to the text file
with open(output_file, 'w', encoding='utf-8') as file:
    for key, value in hyperparams.items():
        file.write(f'{key}: {value}\n')


# Run the hunt
train_losses, test_losses, test_rel_errs_emd, warmstarts = the_hunt(
        gen_net,
        pred_net,
        loss_func,
        cost,  
        dust_const,
        dim_prior,
        dim,
        device,
        min_eps_var,
        max_eps_var, 
        eps_test_const,
        eps_test_var, 
        test_sets,
        test_emd,
        test_sink,
        test_T,
        n_loops,
        n_mini_loops_gen,
        n_mini_loops_pred,
        n_batch,
        lr_pred,
        lr_gen,
        lr_fact_gen,
        lr_fact_pred,
        learn_gen,
        bootstrapped,
        n_boot,
        extend_data,
        test_iter,
        results_folder,
        checkpoint,
        ) -> tuple[dict, dict, dict]:

print('The hunt is over. Time to rest.')
