"""
Let the hunt begin!
"""
# Imports
import datetime
import os
import torch
import matplotlib.pyplot as plt
from cost_matrices import euclidean_cost_matrix
from training_algorithm import the_hunt
from nets import gen_net, pred_net
from utils import hilb_proj_loss, test_sampler, rando, random_shapes_loader
from data_creators import get_cifar, get_omniglot, get_mnist, get_flowers
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
mnist = get_mnist(length, dust_const, download=True)
omniglot = get_omniglot(length, dust_const, download=True)  
cifar = get_cifar(length, dust_const, download=True)
flowers = get_flowers(length, dust_const,  download=True)   


# Initialization of cost matrix
C = euclidean_cost_matrix(length, length, normed=True).double().to(device)

# Regularization constant
eps = C.max() * 6e-4
print(f"Reg: {eps}")

# Initialization of loss function
loss_function = hilb_proj_loss

# Initialization of nets
deer = gen_net(dim_prior, dim, dust_const, skip_const).double().to(device)
puma = pred_net(dim, width).double().to(device)

# Load model state dict
#deer.load_state_dict(torch.load(f"{stamp_folder_path}/deer.pt"))
#puma.load_state_dict(torch.load(f"{stamp_folder_path}/puma.pt"))

# Training mode
deer.train()
puma.train()

# Training Hyperparams
lr_gen=0.5
lr_pred=1.0
lr_factor=0.99
n_samples= 1000000
batchsize=2500
minibatch=500
epochs=10
test_iter=20
learn_gen=False

# Run the hunt
results = the_hunt(deer, 
        puma,
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
        )

# Unpack results
train_losses, test_losses_rn, test_losses_rs, test_losses_mnist, test_losses_omniglot, test_losses_cifar, test_losses_flowers, rel_errs_rn, rel_errs_rs, rel_errs_mnist, rel_errs_omniglot, rel_errs_cifar, rel_errs_flowers = results

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
X_rn = rando(100, dim, dust_const).double().to(device)
X_rs = random_shapes_loader(100, dim, dust_const).double().to(device)
X_mnist = test_sampler(MNIST, 100).double().to(device)
X_omniglot = test_sampler(OMNIGLOT, 100).double().to(device)
X_cifar = test_sampler(CIFAR10, 100).double().to(device)
X_flowers = test_sampler(FLOWERS102, 100).double().to(device)

test_warmstart(X_rn, C, dim, reg, puma, "Random Noise", f"{stamp_folder_path}/warmstart_rn.png")
test_warmstart(X_mnist, C, dim, reg, puma, "MNIST", f"{stamp_folder_path}/warmstart_mnist.png")
test_warmstart(X_omniglot, C, dim, reg, puma, "OMNIGLOT", f"{stamp_folder_path}/warmstart_omniglot.png")
test_warmstart(X_cifar, C, dim, reg, puma, "CIFAR10", f"{stamp_folder_path}/warmstart_cifar.png")
test_warmstart(X_flowers, C, dim, reg, puma, "FLOWERS102", f"{stamp_folder_path}/warmstart_flowers.png")