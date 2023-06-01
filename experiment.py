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
from utils import hilb_proj_loss
from data_creators import rand_noise, rand_shapes, rand_noise_and_shapes, get_cifar, get_omniglot, get_mnist, get_flowers,  random_shapes_loader, random_noise
from test_funcs import test_warmstart

# Create 'stamp' folder for saving results
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%m-%d_%H_%M_%S")
stamp_folder_path = "./stamp_"+formatted_time
os.mkdir(stamp_folder_path)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Hyperparameters
length_prior = 10
length = 28
dim_prior = length_prior**2
dim = length**2
dust_const = 1e-5
skip_const = 0.2
width = 4

# Download testsets
rn = random_noise(10000, dim, dust_const)
rs = random_shapes_loader(10000, dim, dust_const)
rn_rs = rand_noise_and_shapes(10000, dim, dust_const)
mnist = get_mnist(length, dust_const, download=True)
omniglot = get_omniglot(length, dust_const, download=True)  
cifar = get_cifar(length, dust_const, download=True)
flowers = get_flowers(length, dust_const,  download=True)   

test_sets = {
    'rn': rn,
    'rs': rs,
    'rn_rs': rn_rs,
    'mnist': mnist,
    'omniglot': omniglot,
    'cifar': cifar,
    'flowers': flowers
}
             
# Initialization of cost matrix
cost_mat = l2_cost_mat(length, length, normed=True).double().to(device)

# Regularization constant
eps = cost_mat.max() * 6e-4
print(f"Reg: {eps}")

# Initialization of loss function
loss_func = hilb_proj_loss

# Initialization of nets
deer = GenNet(dim_prior, dim, dust_const, skip_const).double().to(device)
puma = PredNet(dim, width).double().to(device)

# Load model state dict
#deer.load_state_dict(torch.load(f"{stamp_folder_path}/deer.pt"))
#puma.load_state_dict(torch.load(f"{stamp_folder_path}/puma.pt"))

# Training mode
deer.train()
puma.train()

# Training Hyperparams
n_samples = 1000000
batch_size = 2500
minibatch_size = 500
n_epochs_pred = 10
gen_pred_ratio = 1
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
        puma,
        deer,
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
        n_epochs_pred,
        gen_pred_ratio,
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

# Plot the results
plt.figure()
plt.plot(torch.log(train_losses))
plt.grid()
plt.xlabel('# minibatches')
plt.ylabel('log loss')
plt.title('Log Train Losses')
plt.savefig(f"{stamp_folder_path}/log_train_losses.png")
plt.figure()
plt.grid()
plt.plot(torch.log(test_losses_rn), label='rn')
plt.plot(torch.log(test_losses_rs), label='rs')
plt.plot(torch.log(test_losses_mnist), label='mnist')
plt.plot(torch.log(test_losses_omniglot), label='omniglot')
plt.plot(torch.log(test_losses_cifar), label='cifar')
plt.plot(torch.log(test_losses_flowers), label='flowers')
plt.xlabel('# test phases')
plt.ylabel('log loss')
plt.legend()
plt.title('Log Test Losses')
plt.savefig(f"{stamp_folder_path}/log_test_losses.png")
plt.figure()
plt.grid()
plt.plot(rel_errs_rn, label='rn')
plt.plot(rel_errs_rs, label='rs')
plt.plot(rel_errs_mnist, label='mnist')
plt.plot(rel_errs_omniglot, label='omniglot')
plt.plot(rel_errs_cifar, label='cifar')
plt.plot(rel_errs_flowers, label='flowers')
plt.legend()
plt.yticks(torch.arange(0, 1.0001, 0.05))
plt.title(' Rel Error: Pred Net Dist VS ot.emd2')
plt.savefig(f"{stamp_folder_path}/rel_errs.png")

# Saving nets
torch.save(deer.state_dict(), f"{stamp_folder_path}/deer.pt")
torch.save(puma.state_dict(), "{stamp_folder_path}/puma.pt")

# Create txt file in stamp for hyperparams
current_date = datetime.datetime.now().strftime("%d.%m.%Y")
hyperparams = {
    'date': current_date,
    'length_prior': length_prior,
    'length': length,
    'dim_prior': dim_prior,
    'dim': dim,
    'reg': reg,
    'dust_const': dust_const,
    'skip_const': skip_const,
    'width': width,
    'lr_gen': lr_gen,
    'lr_factor': lr_factor,
    'n_samples': n_samples,
    'batchsize': batchsize,
    'minibatch': minibatch,
    'epochs': epochs,
    'test_iter': test_iter,
    'learn_gen': learn_gen
}

# Define the output file path
output_file = f"{stamp_folder_path}/params.txt"

# Save the hyperparams to the text file
with open(output_file, 'w') as file:
    for key, value in hyperparams.items():
        file.write(f"{key}: {value}\n")

# Test warmstart
for key in test_sets.keys():
    X_test = test_sampler(test_sets[key], n_test_samples).double().to(device)
    test_warmstart(puma, X_test, n_test_samples, cost_mat, eps, dim, key,
                   device, dust_const, plot=True)