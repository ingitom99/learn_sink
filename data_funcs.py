"""
data_funcs.py
-------------

Functions to create and process data for training and testing.
"""

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import os
import urllib
from tqdm import tqdm
from skimage.draw import random_shapes


def test_set_sampler(test_set : torch.Tensor, n_samples : int) -> torch.Tensor:

    """
    Randomly sample from a data set to create pairs of samples for testing.

    Parameters
    ----------
    test_set : (n_test_set, dim) torch.Tensor
        Test set.
    n_samples : int
        Number of samples.
    
    Returns
    -------
    test_sample : (n_samples, 2 * dim) torch.Tensor
        Random sample from the test set.
    """

    rand_perm = torch.randperm(test_set.size(0))
    rand_mask_a = rand_perm[:n_samples]
    rand_mask_b = rand_perm[n_samples:2*n_samples]
    test_sample_a = test_set[rand_mask_a]
    test_sample_b = test_set[rand_mask_b]
    test_sample = torch.cat((test_sample_a, test_sample_b), dim=1)

    return test_sample

def preprocessor(dataset : torch.Tensor, length : int, dust_const : float
                    ) -> torch.Tensor:
    
    """
    Preprocess (resize, normalize, dust) a dataset for training.

    Parameters
    ----------
    dataset : (n, dim) torch.Tensor
        Dataset to be preprocessed.
    length : int
        Length of the side of each image in the dataset.
    dust_const : float
        Constant added to the dataset to avoid zero values (dusting).

    Returns
    -------
    processed_dataset : (n, dim) torch.Tensor
        Preprocessed dataset.
    """

    resized_dataset = F.interpolate(dataset.unsqueeze(1), size=(length, length),
                                mode='bilinear', align_corners=False).squeeze(1)

    flattened_dataset = resized_dataset.view(-1, length**2)

    normalized_dataset = flattened_dataset / flattened_dataset.sum(dim=1,
                                                                   keepdim=True)

    processed_dataset = normalized_dataset + dust_const

    processed_dataset =  processed_dataset / processed_dataset.sum(dim=1,
                                                                   keepdim=True)

    return processed_dataset

def rand_noise(n_samples : int, dim : int, dust_const : float,
               pairs : bool) -> torch.Tensor:

    """
    Create a sample of random uniform noise cubed.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    dim : int
        Dimension of the samples.
    dust_const : float
        Constant added to the samples to avoid zero values.
    pairs : bool
        Whether to return a sample of pairs of probability distributions or a
        sample of single probability distributions.
    
    Returns
    -------
    sample : (n_samples, 2 * dim or dim) torch.Tensor
        Sample of pairs of probability distributions.
    
    """

    sample_a = torch.rand((n_samples, dim))**3
    sample_a /= torch.unsqueeze(sample_a.sum(dim=1), 1)
    sample_a = sample_a + dust_const
    sample_a /= torch.unsqueeze(sample_a.sum(dim=1), 1)
    
    if pairs:
        sample_b = torch.rand((n_samples, dim))**3
        sample_b /= torch.unsqueeze(sample_b.sum(dim=1), 1)
        sample_b = sample_b + dust_const
        sample_b /= torch.unsqueeze(sample_b.sum(dim=1), 1)
        sample = torch.cat((sample_a, sample_b), dim=1)
        return sample
    
    else:
        return sample_a

def rand_shapes(n_samples : int, dim : int, dust_const : float,
                pairs : bool) -> torch.Tensor:

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
    pairs : bool
        Whether to return a sample of pairs of probability distributions or a
        sample of single probability distributions.

    Returns
    -------
    sample : (n_samples, 2 * dim or dim) torch.Tensor
        Sample of pairs of probability distributions.
    """

    length = int(dim**.5)
    sample_a = []
    sample_b = []
    for i in range(n_samples):
        image1 = random_shapes((length, length), max_shapes=8, min_shapes=2,
                               min_size=2, max_size=12, channel_axis=None,
                               allow_overlap=True)[0]
        image1 = image1.max() - image1
        image1 = image1 / image1.sum()
        image1 = image1 + dust_const
        image1 = image1 / image1.sum()
        sample_a.append(image1.flatten())
        if pairs:
            image2= random_shapes((length, length), max_shapes=8, min_shapes=2,
                                min_size=2, max_size=12, channel_axis=None,
                                allow_overlap=True)[0]
            image2 = image2.max() - image2
            image2 = image2 + dust_const
            image2 = image2 /image2.sum()
            sample_b.append(image2.flatten())

    sample_a = np.array(sample_a)
    sample_a = torch.tensor(sample_a)

    if pairs:
        sample_b = np.array(sample_b)
        sample_b = torch.tensor(sample_b)
        sample = torch.cat((sample_a, sample_b), dim=1)
        return sample
    else:
        return sample_a
    
                      

def rand_noise_and_shapes(n_samples : int, dim : int, dust_const : float,
                          pairs : bool) -> torch.Tensor:
  
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
    pairs : bool
        Whether to return a sample of pairs of probability distributions or a
        sample of single probability distributions.
    
    Returns
    -------
    sample : (n_samples, 2 * dim or dim) torch.Tensor
        Sample of pairs of probability distributions.
    """
    
    rn = rand_noise(n_samples, dim, dust_const, pairs)
    rs = rand_shapes(n_samples, dim, dust_const, pairs)
    
    rn_rand_fact = torch.rand((n_samples, 1))
    rs_rand_fact = torch.rand((n_samples, 1))

    sample = rn_rand_fact * rn + rs_rand_fact * rs

    if pairs:
        sample_mu = sample[:, :dim]
        sample_nu = sample[:, dim:]
        sample_mu = sample_mu / torch.unsqueeze(sample_mu.sum(dim=1), 1)
        sample_nu = sample_nu / torch.unsqueeze(sample_nu.sum(dim=1), 1)
        sample = torch.cat((sample_mu, sample_nu), dim=1)
        return sample
    else:
        sample = sample / torch.unsqueeze(sample.sum(dim=1), 1)
        return sample

def get_mnist(n_samples : int, path : str) -> None:

    """
    Download and save a set of MNIST images as a pytorch tensor in a '.pt' file.

    Parameters
    ----------
    n_samples : int
        Number of samples from the MNIST dataset.
    path : str
        Path to save the dataset.

    Returns
    -------
    None
    """
    
    dataset = torchvision.datasets.MNIST(root="./data", download=True,
                                transform=torchvision.transforms.ToTensor())
    mnist = torch.zeros((len(dataset), 28, 28))
    for i, datapoint in enumerate(dataset):
        image = datapoint[0]
        mnist[i] = image
    rand_perm = torch.randperm(len(mnist))
    mnist_save = mnist[rand_perm][:n_samples]
    torch.save(mnist_save, path)
    return None
    

def get_omniglot(n_samples : int, path : str) -> None:

    """
    Download and save a set of Omniglot images as a pytorch tensor in a '.pt'
    file.

    Parameters
    ----------
    n_samples : int
        Number of samples from the Omniglot dataset.
    path : str
        Path to save the dataset.

    Returns
    -------
    None
    """

    dataset = torchvision.datasets.Omniglot(root="./data", download=True,
                                transform=torchvision.transforms.ToTensor())
    
    omniglot = torch.zeros((len(dataset), 105, 105))
    for i, datapoint in enumerate(dataset):
        image = datapoint[0]
        omniglot[i] = image
    
    rand_perm = torch.randperm(len(omniglot))
    omniglot_save = omniglot[rand_perm][:n_samples]

    torch.save(omniglot_save, path)

    return None
    
def get_cifar(n_samples : int, path : str) -> None:

    """
    Download and save a set of CIFAR10 images as a pytorch tensor in a '.pt'
    file.

    Parameters
    ----------
    n_samples : int
        Number of samples from the CIFAR10 dataset.
    path : str
        Path to save the dataset.

    Returns
    -------
    None
    """

    dataset = torchvision.datasets.CIFAR10(root="./data", download=True,
                                transform=torchvision.transforms.Grayscale())
    transformer = torchvision.transforms.ToTensor()
    cifar = torch.zeros((len(dataset), 32, 32))
    for i, datapoint in enumerate(dataset):
        image = transformer(datapoint[0])
        cifar[i] = image
    rand_perm = torch.randperm(len(cifar))
    cifar_save = cifar[rand_perm][:n_samples]
    torch.save(cifar_save, path)

    return None

def get_lfw(n_samples : int, path : str) -> None:

    """
    Download and save a set of LFW images as a pytorch tensor in a '.pt'
    file.

    Parameters
    ----------
    n_samples : int
        Number of samples from the LFW dataset.
    path : str
        Path to save the dataset.

    Returns
    -------
    None
    """

    dataset = torchvision.datasets.LFWPeople(root="./data", download=True, 
                                transform=torchvision.transforms.Grayscale())
    transformer = torchvision.transforms.ToTensor()
    lfw = torch.zeros((len(dataset), 250, 250))

    for i, datapoint in enumerate(dataset):
        image = transformer(datapoint[0])
        lfw[i] = image
    
    rand_perm = torch.randperm(len(lfw))
    lfw_save = lfw[rand_perm][:n_samples]
    torch.save(lfw_save, path)
    return None 

def get_quickdraw(n_samples : int, root_np : str, path_torch : str,
                  class_name : str) -> None:
    
    """
    Download and save a set of Quickdraw images of a specified class as a
    pytorch tensor in a '.pt' file using an intermediary numpy array and file.

    Parameters
    ----------
    n_samples : int
        Number of samples from the Quickdraw dataset.
    root_np : str
        Path to folder to save the numpy array.
    path_torch : str
        Path to save the pytorch tensor.
    class_name : str
        Name of the class of images to download.

    Returns
    -------
    None
    """

    # Create directory if it does not exist
    if not os.path.exists(root_np):
        os.makedirs(root_np)

    # Define class-specific URL and filename
    url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{class_name}.npy"
    filename = os.path.join(root_np, f"{class_name}.npy")

    # Download the dataset file
    urllib.request.urlretrieve(url, filename)

    # Replace spaces in class name with underscores
    class_name = class_name.replace(' ', '_')
    filename = os.path.join(root_np, f"{class_name}.npy")

    # Load numpy array and convert to tensor
    data_array = np.load(filename)
    dataset = torch.from_numpy(data_array).float()

    # Concatenate tensors along the first dimension
    dataset = dataset.reshape(-1, 28, 28)

    rand_perm = torch.randperm(len(dataset))
    dataset = dataset[rand_perm][:n_samples]

    torch.save(dataset, path_torch)

    return None

def get_quickdraw_class_names():

    """
    Get the list of class names for the Quickdraw dataset.

    Parameters
    ----------
    None

    Returns
    -------
    class_names : list
        List of class names.
    """

    url = "https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt"
    response = urllib.request.urlopen(url)

    class_names = []
    for line in response:
        class_name = line.decode('utf-8').strip()
        class_names.append(class_name)

    return class_names

def get_quickdraw_multi(n_samples : int, n_classes : int, root_np : str,
                        path_torch : str) -> None:

    """
    Download and save a set of Quickdraw images from a specified number of
    random classes as a pytorch tensor in a '.pt' file using intermediary numpy
    arrays and files.

    WARNING: SLOWWWW!

    Parameters
    ----------
    n_samples : int
        Number of samples from the Quickdraw dataset.
    n_classes : int
        Number of random classes to use.
    root_np : str
        Path to folder to save the numpy arrays.
    path_torch : str
        Path to save the pytorch tensor.

    Returns
    -------
    None
    """
    
    datasets = []

    class_names = get_quickdraw_class_names()

    rand_mask = np.random.choice(len(class_names), n_classes, replace=False)

    class_names = np.array(class_names)[rand_mask]

    n_samples_class = n_samples // n_classes

    for class_name in tqdm(class_names):

        # if class_name is two words, replace space with %20
        if ' ' in class_name:
            class_name = class_name.replace(' ', '%20')

        # Create directory if it does not exist
        if not os.path.exists(root_np):
            os.makedirs(root_np)

        # Define class-specific URL and filename
        url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{class_name}.npy"
        filename = os.path.join(root_np, f"{class_name}.npy")

        # Download the dataset file
        urllib.request.urlretrieve(url, filename)

        # Replace spaces in class name with underscores
        class_name = class_name.replace(' ', '_')
        filename = os.path.join(root_np, f"{class_name}.npy")

        # Load numpy array and convert to tensor
        data_array = np.load(filename)
        dataset = torch.from_numpy(data_array).float()

        # Concatenate tensors along the first dimension
        dataset = dataset.reshape(-1, 28, 28)

        rand_perm = torch.randperm(len(dataset))
        dataset = dataset[rand_perm][:n_samples_class]

        datasets.append(dataset)
    
    dataset = torch.cat(datasets, dim=0)

    rand_perm = torch.randperm(len(dataset))
    dataset = dataset[rand_perm][:n_samples]

    torch.save(dataset, path_torch)

    return None