"""
net.py
-------

Pytorch neural network class for the predictive networks.
"""

import torch.nn as nn

class PredNet(nn.Module):

    """
    Predictive network class.
    """

    def __init__(self, dim, width):

        """
        Parameters
        ----------
        dim : int
            Dimension of each probability distribution in the data.
        width : int
            Width of the predictive network hidden layers.
        """

        super(PredNet, self).__init__()
        self.dim = dim
        self.width = width
        self.l_1 = nn.Sequential(nn.Linear(2*dim, width), nn.BatchNorm1d(width),
                                nn.ELU())
        self.l_2 = nn.Sequential(nn.Linear(width, width), nn.BatchNorm1d(width),
                                 nn.ELU())
        self.l_3 = nn.Sequential(nn.Linear(width, width), nn.BatchNorm1d(width),
                                 nn.ELU())
        self.l_4 = nn.Sequential(nn.Linear(width, dim))
        self.layers = [self.l_1, self.l_2, self.l_3, self.l_4]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    