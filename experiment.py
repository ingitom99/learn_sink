import datetime
import torch
import matplotlib.pyplot as plt
from cost_matrices import euclidean_cost_matrix
from training_algorithm import the_hunt
from nets import gen_net, pred_net
from utils import hilb_proj_loss, test_sampler, rando, random_shapes_loader
from data_creators import get_MNIST, get_OMNIGLOT, get_CIFAR10, get_FLOWERS102
from test_funcs import test_warmstart

# Stamp folder
stamp_folder = "stamp_i"

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Hyperparameters
length_prior = 28
length = 28
dim_prior = length_prior**2
dim = length**2
dust_const = 1e-5
skip_const = 0.3
width = 2

# Download testsets
MNIST = get_MNIST(length, dust_const, download=False).double().to(device)
OMNIGLOT = get_OMNIGLOT(length, dust_const, download=False).double().to(device)
CIFAR10 = get_CIFAR10(length, dust_const, download=False).double().to(device)
FLOWERS102 = get_FLOWERS102(length, dust_const, download=False).double().to(device)

# Initialization of cost matrix
C = euclidean_cost_matrix(length, length, normed=True).double().to(device)

# Regularization constant
reg = C.max() * 5e-4
print(f"Reg: {reg}")

# Initialization of loss function
loss_function = hilb_proj_loss

# Load model state dict
#deer.load_state_dict(torch.load(f"./{stamp_folder}/nets/deer.pt"))
#puma.load_state_dict(torch.load(f"./{stamp_folder}/nets/puma.pt"))

# Initialization of nets
deer = gen_net(dim_prior, dim, dust_const, skip_const).double().to(device)
puma = pred_net(dim, width).double().to(device)

# Training mode
deer.train()
puma.train()

# Training Hyperparams
lr_gen=0.5
lr_pred=2.0
lr_factor=0.99
n_samples= 100000
batchsize=10000
minibatch=1000
epochs=3
test_iter=5
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
plt.title('Log Train Losses')
plt.savefig(f"./{stamp_folder}/log_train_losses.png")
plt.figure()
plt.grid()
plt.plot(torch.log(test_losses_rn), label='rn')
plt.plot(torch.log(test_losses_rs), label='rs')
plt.plot(torch.log(test_losses_mnist), label='mnist')
plt.plot(torch.log(test_losses_omniglot), label='omniglot')
plt.plot(torch.log(test_losses_cifar), label='cifar')
plt.plot(torch.log(test_losses_flowers), label='flowers')
plt.legend()
plt.title('Log Test Losses')
plt.savefig(f"./{stamp_folder}/log_test_losses.png")
plt.figure()
plt.grid()
plt.plot(rel_errs_rn, label='rn')
plt.plot(rel_errs_rs, label='rs')
plt.plot(rel_errs_mnist, label='mnist')
plt.plot(rel_errs_omniglot, label='omniglot')
plt.plot(rel_errs_cifar, label='cifar')
plt.plot(rel_errs_flowers, label='flowers')
plt.legend()
plt.title('Predicted Distance Relative Error Versus ot.emd2')
plt.savefig(f"./{stamp_folder}/rel_errs.png")

# Saving nets
torch.save(deer.state_dict(), "./stamp/deer.pt")
torch.save(puma.state_dict(), "./stamp/puma.pt")

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
output_file = f"./{stamp_folder}/params.txt"

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

test_warmstart(X_rn, C, dim, reg, puma, "Random Noise", f"./{stamp_folder}/warmstart_rn.png")
test_warmstart(X_mnist, C, dim, reg, puma, "MNIST", f"./{stamp_folder}/warmstart_mnist.png")
test_warmstart(X_omniglot, C, dim, reg, puma, "OMNIGLOT", f"./{stamp_folder}/warmstart_omniglot.png")
test_warmstart(X_cifar, C, dim, reg, puma, "CIFAR10", f"./{stamp_folder}/warmstart_cifar.png")
test_warmstart(X_flowers, C, dim, reg, puma, "FLOWERS102", f"./{stamp_folder}/warmstart_flowers.png")