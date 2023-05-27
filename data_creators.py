"""
Functions to create datasets for testing.
"""

# Imports
import torch
import torchvision

# Functions
def get_MNIST(length, dust_const, download=True):
  """
  Get the MNIST dataset and process it.

  Inputs:
    length: int
      Length of the reshaped images
    dust_const: float
      Constant to add to the images to avoid zero entries
    download: bool
      Whether to download the dataset or not
  """
  mnist_testset = torchvision.datasets.MNIST(root='./data', train=False, download=download, transform=torchvision.transforms.Resize((length, length), antialias=True))
  mnist_testset = torch.flatten(mnist_testset.data, start_dim=1)
  MNIST_TEST = (mnist_testset / torch.unsqueeze(mnist_testset.sum(dim=1), 1))
  MNIST_TEST = MNIST_TEST + dust_const
  MNIST_TEST = MNIST_TEST / torch.unsqueeze(MNIST_TEST.sum(dim=1), 1)
  return MNIST_TEST

def get_OMNIGLOT(length, dust_const, download=True):
  """
  Get the OMNIGLOT dataset and process it.

  Inputs:
    length: int
      Length of the reshaped images
    dust_const: float
      Constant to add to the images to avoid zero entries
    download: bool
      Whether to download the dataset or not
    
  Returns:
    OMNIGLOT: torch tensor of shape (n_samples, length**2)
      Processed OMNIGLOT dataset
  """
  dataset = torchvision.datasets.Omniglot(root="./data", download=download, transform=torchvision.transforms.ToTensor())
  OMNIGLOT = torch.ones((len(dataset), length**2))
  resizer = torchvision.transforms.Resize((length, length), antialias=True)
  for i in range(len(dataset)):
    img = 1 - resizer(dataset[i][0]).reshape(-1)
    OMNIGLOT[i] = img
  OMNIGLOT = (OMNIGLOT / torch.unsqueeze(OMNIGLOT.sum(dim=1), 1))
  OMNIGLOT = OMNIGLOT + dust_const
  OMNIGLOT = OMNIGLOT / torch.unsqueeze(OMNIGLOT.sum(dim=1), 1)
  return OMNIGLOT
  
def get_CIFAR10(length, dust_const, download=True):
  """
  Get the CIFAR10 dataset and process it.

  Inputs:
    length: int
      Length of the reshaped images
    dust_const: float
      Constant to add to the images to avoid zero entries
    download: bool
      Whether to download the dataset or not
  
  Returns:
    CIFAR10: torch tensor of shape (n_samples, length**2)
      Processed CIFAR10 dataset
  """
  dataset = torchvision.datasets.CIFAR10(root="./data", download=download, transform=torchvision.transforms.Grayscale())
  totensor = torchvision.transforms.ToTensor()
  resizer = torchvision.transforms.Resize((length, length), antialias=True)
  CIFAR10 = torch.zeros((len(dataset), length**2))
  for i in range(len(dataset)):
    img = totensor(dataset[i][0])
    img = resizer(img).reshape(-1)
    CIFAR10[i] = img
  CIFAR10 = (CIFAR10 / torch.unsqueeze(CIFAR10.sum(dim=1), 1))
  CIFAR10 = CIFAR10 + dust_const
  CIFAR10 = CIFAR10 / torch.unsqueeze(CIFAR10.sum(dim=1), 1)
  return CIFAR10

def get_FLOWERS102(dust_const, length, download=True):
  """
  Get the FLOWERS102 dataset and process it.

  Inputs:
    length: int
      Length of the reshaped images
    dust_const: float
      Constant to add to the images to avoid zero entries
    download: bool
      Whether to download the dataset or not
  
  Returns:
    FLOWERS102: torch tensor of shape (n_samples, length**2)
      Processed FLOWERS102 dataset
  """
  dataset = torchvision.datasets.Flowers102(root="./data", download=download, transform=torchvision.transforms.Grayscale())
  totensor = torchvision.transforms.ToTensor()
  resizer = torchvision.transforms.Resize((length, length), antialias=True)
  FLOWERS102 = torch.zeros((len(dataset), length**2))
  for i in range(len(dataset)):
    img = totensor(dataset[i][0])
    img = resizer(img).reshape(-1)
    FLOWERS102[i] = img
  FLOWERS102 = (FLOWERS102 / torch.unsqueeze(FLOWERS102.sum(dim=1), 1))
  FLOWERS102 = FLOWERS102 + dust_const
  FLOWERS102 = FLOWERS102 / torch.unsqueeze(FLOWERS102.sum(dim=1), 1)
  return FLOWERS102