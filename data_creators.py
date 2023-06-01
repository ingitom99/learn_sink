"""
Functions to create datasets for testing.
"""

import torch
import torchvision
import numpy as np
from skimage.draw import random_shapes

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

def rand_noise(n_samples : int, dim : int, dust_const : float) -> torch.Tensor:

    """
    Create a sample of randomly masked, randomly exponentiated, random uniform
    noise.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    dim : int
        Dimension of the samples.
    dust_const : float
        Constant added to the samples to avoid zero values.
    
    Returns
    -------
    sample : (n_samples, 2 * dim) torch.Tensor
        Sample of pairs of probability distributions.
    
    """

    bernoulli_p = torch.rand((n_samples, 1))
    bernoulli_p[bernoulli_p < 0.03] = 0.03

    multiplier = torch.randint(1, 6, (n_samples, 1))

    sample_a = torch.rand((n_samples, dim))
    sample_b = torch.rand((n_samples, dim))
    
    mask_a = torch.bernoulli(bernoulli_p * torch.ones_like(sample_a))
    mask_b = torch.bernoulli(bernoulli_p * torch.ones_like(sample_b))

    sample_a = (sample_a * mask_a)**multiplier
    sample_b = (sample_b * mask_b)**multiplier

    sample_b /= torch.unsqueeze(sample_b.sum(dim=1), 1)
    sample_a /= torch.unsqueeze(sample_a.sum(dim=1), 1)

    sample_a = sample_a + dust_const
    sample_b = sample_b + dust_const

    sample_a /= torch.unsqueeze(sample_a.sum(dim=1), 1)
    sample_b /= torch.unsqueeze(sample_b.sum(dim=1), 1)

    sample = torch.cat((sample_a, sample_b), dim=1)

    return sample

def rand_shapes(n_samples : int, dim : int, dust_const : float) -> torch.Tensor:

    """
    Create a sample of images containing random shapes as pairs of probability
    distributions.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    dim : int
        Dimension of the samples.
    dust_const : float
        Constant added to the samples to avoid zero values.

    Returns
    -------
    sample : (n_samples, 2 * dim) torch.Tensor
        Sample of pairs of probability distributions.
    """

    length = int(dim**.5)
    pairs = []

    for i in range(n_samples):
        image1 = random_shapes((length, length), max_shapes=8, min_shapes=2,
                               min_size=2, max_size=12, channel_axis=None,
                               allow_overlap=True)[0]
        image1 = image1.max() - image1
        image1 = image1 / image1.sum()
        image1 = image1 + dust_const
        image1 = image1 / image1.sum()
        image2= random_shapes((length, length), max_shapes=8, min_shapes=2,
                              min_size=2, max_size=12, channel_axis=None,
                              allow_overlap=True)[0]
        image2 = image2.max() - image2
        image2 = image2 + dust_const
        image2 = image2 /image2.sum()
        pair = np.concatenate((image1.flatten(), image2.flatten()))
        pairs.append(pair)

    pairs = np.array(pairs)
    sample = torch.tensor(pairs)

    return sample

def rand_noise_and_shapes(n_samples : int, dim : int, dust_const : float) -> torch.Tensor:
  
    """
    Generate a data set of pairs of samples of random shapes combined with
    random noise.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    dim : int
        Dimension of the samples.
    dust_const : float
        Constant added to the probability distributions to avoid zero values.
    
    Returns
    -------
    sample : (n_samples, 2 * dim) torch.Tensor
        Sample of pairs of probability distributions.
    """

    rs = rs(n_samples, dim, dust_const)
    rn = rn(n_samples, dim, dust_const)

    rs_rand_fact = torch.rand((n_samples, 1))
    rn_rand_fact = torch.rand((n_samples, 1))

    sample = rs_rand_fact * rs  + rn_rand_fact * rn

    sample_mu = sample[:, :dim]
    sample_nu = sample[:, dim:]

    sample_mu = sample_mu / torch.unsqueeze(sample_mu.sum(dim=1), 1)
    sample_nu = sample_nu / torch.unsqueeze(sample_nu.sum(dim=1), 1)

    sample = torch.cat((sample_mu, sample_nu), dim=1)

    return sample