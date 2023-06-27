"""
make_data.py
------------

This script will download and save data for testing.
"""

import torch
from data_funcs import get_mnist, get_omniglot, get_cifar, get_lfw, get_quickdraw

n_samples = 10000

# mnist
path_mnist = './data/mnist.pt'
get_mnist(n_samples, path_mnist)

# omniglot
path_omniglot = './data/omniglot.pt'
get_omniglot(n_samples, path_omniglot)

# lfw
path_lfw = './data/lfw.pt'
get_lfw(n_samples, path_lfw)

# cifar
path_cifar = './data/cifar.pt'
get_cifar(n_samples, path_cifar)

# teddies
root_np = './quickdraw'
path_bear = '/content/gdrive/MyDrive/learn_sink_stuff_IT_VL/data/bear.pt'
class_name = 'bear'
get_quickdraw(n_samples, root_np, path_bear, class_name)