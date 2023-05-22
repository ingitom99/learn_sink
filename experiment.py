# Imports
import torch
from cost_matrices import euclidean_cost_matrix
from training_algorithm import the_hunt
from nets import gen_net, pred_net
from utils import get_MNIST, get_OMNI, hilb_proj_loss, test_warmstart, MNIST_test_loader

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

# Download MNIST
MNIST_TEST = get_MNIST(dust_const).double().to(device)
OMNI_TEST = get_OMNI(dust_const).double().to(device)

# Initialization of cost matrix
C = euclidean_cost_matrix(length, length, normed=True).double().to(device)

# Regularization constant
reg = C.max() * 5e-4
print(f"Reg: {reg}")

# Initialization of loss function
loss_function = hilb_proj_loss

# Initialization of nets
deer = gen_net(dim_prior, dim, dust_const, skip_const).double().to(device)
puma = pred_net(dim).double().to(device)

# Training mode
deer.train()
puma.train()

# Run the hunt
the_hunt(deer,
        puma,
        C,
        reg,
        dim_prior,
        dim,
        loss_function,
        device,
        MNIST_TEST,
        OMNI_TEST,
        dust_const,
        lr_gen=0.5,
        lr_pred=0.5,
        lr_factor=0.999,
        n_samples= 1000000,
        batchsize=1000,
        minibatch=200,
        epochs=5,
        test_iter=100,
        learn_gen=True
        )

# Saving nets
torch.save(deer.state_dict(), "./nets/deer.pt")
torch.save(puma.state_dict(), "./nets/puma.pt")

# Testing mode
deer.eval()
puma.eval()

# Test warmstart
X_rn = rando(100, dim, dust_const).double().to(device)
X_mnist= MNIST_test_loader(MNIST_TEST, 100).double().to(device)
X_omniglot = MNIST_test_loader(OMNI_TEST, 100).double().to(device)

test_warmstart(X_rn, C, dim, reg, puma, 'Random Noise')
test_warmstart(X_mnist, C, dim, reg, puma, 'MNIST')
test_warmstart(X_omniglot, C, dim, reg, puma, 'OMNIGLOT')