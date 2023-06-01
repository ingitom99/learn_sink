"""
Functions to create datasets for testing.
"""

import torch
import torchvision

def get_mnist(length : int, dust_const : float, download: bool = True
              ) -> (torch.Tensor):
    
    """
    Get the MNIST dataset and process it.

    Parameters
    ----------
    length : int
        Length of the reshaped images.
    dust_const : float
        Constant to add to the images to avoid zero entries.
    download : bool, optional
        Whether to download the dataset or not. The default is True.
    
    Returns
    -------
    mnist : (n_samples, length**2) torch.Tensor
        Processed MNIST dataset.
    """

    mnist = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=download,
    transform=torchvision.transforms.Resize((length, length), antialias=True)
    )
    mnist = torch.flatten(mnist.data, start_dim=1)
    mnist = mnist / torch.unsqueeze(mnist.sum(dim=1), 1)
    mnist = mnist  + dust_const
    mnist = mnist / torch.unsqueeze(mnist .sum(dim=1), 1)
    return mnist 

def get_omniglot(length : int, dust_const : float, download: bool=True
                 ) -> torch.Tensor:
    """
    Get the omniglot dataset and process it.

    Parameters
    ----------
    length : int
        Length of the reshaped images.
    dust_const : float
        Constant to add to the images to avoid zero entries.
    download : bool, optional
        Whether to download the dataset or not. The default is True.
    
    Returns
    -------
    omniglot : (n_samples, length**2) torch.Tensor
        Processed omniglot dataset.
    """
    dataset = torchvision.datasets.omniglot(
    root="./data",
    download=download,
    transform=torchvision.transforms.ToTensor()
    )
    omniglot = torch.ones((len(dataset), length**2))
    resizer = torchvision.transforms.Resize((length, length), antialias=True)
    for i, datapoint in enumerate(dataset):
        img = 1 - resizer(datapoint[0]).reshape(-1)
        omniglot[i] = img
    omniglot = (omniglot / torch.unsqueeze(omniglot.sum(dim=1), 1))
    omniglot = omniglot + dust_const
    omniglot = omniglot / torch.unsqueeze(omniglot.sum(dim=1), 1)
    return omniglot
  
def get_cifar(length : int, dust_const : int, download: bool = True
              ) -> torch.Tensor:

    """
    Get the CIFAR10 dataset and process it.

    Parameters
    ----------
    length : int
        Length of the reshaped images.
    dust_const : float
        Constant to add to the images to avoid zero entries.
    download : bool, optional
        Whether to download the dataset or not. The default is True.

    Returns
    -------
    cifar : (n_samples, length**2) torch.Tensor
        Processed CIFAR10 dataset.
    """

    dataset = torchvision.datasets.CIFAR100(
    root="./data", download=download, 
    transform=torchvision.transforms.Grayscale()
    )
    totensor = torchvision.transforms.ToTensor()
    resizer = torchvision.transforms.Resize((length, length), antialias=True)
    cifar = torch.zeros((len(dataset), length**2))
    for i, data_point in enumerate(dataset):
        img = totensor(data_point[0])
        img = resizer(img).reshape(-1)
        cifar[i] = img
    cifar = cifar / torch.unsqueeze(cifar.sum(dim=1), 1)
    cifar = cifar + dust_const
    cifar = cifar / torch.unsqueeze(cifar.sum(dim=1), 1)
    return cifar

def get_flowers(length, dust_const, download=True) -> torch.Tensor:

    """
    Get the FLOWERS102 dataset and process it.

    Parameters
    ----------
    length : int
        Length of the reshaped images.
    dust_const : float
        Constant to add to the images to avoid zero entries.
    download : bool, optional
        Whether to download the dataset or not. The default is True.
    
    Returns
    -------
    flowers : (n_samples, length**2) torch.Tensor
        Processed FLOWERS102 dataset.
    """

    dataset = torchvision.datasets.flowers(
        root="./data",
        download=download,
        transform=torchvision.transforms.Grayscale()
        )
    totensor = torchvision.transforms.ToTensor()
    resizer = torchvision.transforms.Resize((length, length), antialias=True)
    flowers = torch.zeros((len(dataset), length**2))
    for i, datapoint in enumerate(dataset):
        img = totensor(datapoint[0])
        img = resizer(img).reshape(-1)
        flowers[i] = img
    flowers = (flowers / torch.unsqueeze(flowers.sum(dim=1), 1))
    flowers = flowers + dust_const
    flowers = flowers / torch.unsqueeze(flowers.sum(dim=1), 1)
    return flowers