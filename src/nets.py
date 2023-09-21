"""
nets.py
-------

Pytorch neural network classes for the generative and predictive networks.
"""
    
import torch
import torch.nn as nn
import torchvision

class GenNet(nn.Module):

    """
    Data generating network class.
    """

    def __init__(self, dim_prior, dim, width, dust_const, skip_const):

        """
        Parameters
        ----------
        dim_prior : int
            Dimension of the prior distribution.
        dim : int
            Dimension of each probability distribution in the data.
        dust_const : float
            Constant to add to the images to avoid zero entries.
        skip_const : float
            Constant to control the strength of the skip connection.
        """

        super(GenNet, self).__init__()
        self.dim_prior = dim_prior
        self.dim = dim
        self.width = width
        self.dust_const = dust_const
        self.skip_const = skip_const
        self.length_prior = int(self.dim_prior**.5)
        self.length = int(self.dim**.5)
        self.l_1 = nn.Sequential(nn.Linear(2*dim_prior, width),
                                 nn.BatchNorm1d(width), nn.ELU())
        self.l_2 = nn.Sequential(nn.Linear(width, width),
                                 nn.BatchNorm1d(width), nn.ELU())
        self.l_3 = nn.Sequential(nn.Linear(width, 2*dim), nn.Sigmoid())
        self.layers = [self.l_1, self.l_2, self.l_3]

    def forward(self, x):

        # Creating a reshaped copy of the input to use as a skip connection
        x_0 = x.reshape(2, x.size(0), self.length_prior,
                                            self.length_prior)
        transform = torchvision.transforms.Resize(
            (self.length, self.length),
            antialias=True
            )
        x_0 = torch.cat((transform(x_0[0]).reshape(x.size(0), self.dim),
                            transform(x_0[1]).reshape(x.size(0), self.dim)), 1)

        # Forward pass
        for layer in self.layers:
            x = layer(x)

        x = x + self.skip_const * x_0
        x = nn.functional.relu(x)

        x_a = x[:, :self.dim]
        x_b = x[:, self.dim:]
        x_a = x_a / torch.unsqueeze(x_a.sum(dim=1), 1)
        x_b = x_b / torch.unsqueeze(x_b.sum(dim=1), 1)
        x_a = x_a + self.dust_const
        x_b = x_b + self.dust_const
        x_a = x_a / torch.unsqueeze(x_a.sum(dim=1), 1)
        x_b = x_b / torch.unsqueeze(x_b.sum(dim=1), 1)
        x = torch.cat((x_a, x_b), dim=1)
        
        return x

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
        self.l_3 = nn.Sequential(nn.Linear(width, dim))
        self.layers = [self.l_1, self.l_2, self.l_3, self.l_3]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

# write a function to do pca algorithm