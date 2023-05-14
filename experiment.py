import torch
import torchvision.datasets as datasets
from cost_matrices import euclidean_cost_matrix
from training_algorithm import the_hunt
from nets import gen_net, pred_net

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Hyperparameters
prior_l = 10
post_l = 28
dim_prior = prior_l**2
dim = post_l**2
reg = 0.0003
dust_const = 1e-5
skip_const = 0.3

# Download MNIST
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
mnist_testset = torch.flatten(mnist_testset.data.double().to(device), start_dim=1)
MNIST_TEST = (mnist_testset / torch.unsqueeze(mnist_testset.sum(dim=1), 1))
MNIST_TEST = MNIST_TEST + dust_const
MNIST_TEST = MNIST_TEST / torch.unsqueeze(MNIST_TEST.sum(dim=1), 1)

# Initialization
C = euclidean_cost_matrix(post_l, post_l, normed=True).double().to(device)
loss_function = torch.nn.MSELoss()
deer = gen_net(dim_prior, dim, dust_const, skip_const).double().to(device)
puma = pred_net(dim).double().to(device)

train_losses, test_losses_pn, test_losses_mnist, test_losses_rs, test_losses_rn = the_hunt(
    deer,
    puma,
    C,
    reg,
    dim_prior,
    dim,
    loss_function,
    device,
    lr_gen=0.01,
    lr_pred=0.1,
    lr_factor=1.0,
    n_samples= 10000, 
    batchsize=100,
    minibatch=10,
    epochs=5,
    train_gen=True
    )