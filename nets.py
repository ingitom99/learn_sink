"""
Pytorch neural network classes for the generative and predictive networks.
"""
    
import torch
import torch.nn as nn

class GenNet(nn.Module):

    """
    Data generating network class.
    """

    def __init__(self, dim_prior, dim, dust_const, skip_const):

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
        self.dust_const = dust_const
        self.skip_const = skip_const
        self.length_prior = int(self.dim_prior**.5)
        self.length = int(self.dim**.5)
        self.l_1 = nn.Sequential(nn.Linear(2*dim_prior, 6*dim),
                                 nn.BatchNorm1d(6*dim), nn.ELU())
        self.l_2 = nn.Sequential(nn.Linear(6*dim, 6*dim),
                                 nn.BatchNorm1d(6*dim), nn.ELU())
        self.l_3 = nn.Sequential(nn.Linear(6*dim, 2*dim), nn.Sigmoid())
        self.layers = [self.l_1, self.l_2, self.l_3]

    def forward(self, x):

        # Creating a reshaped copy of the input to use as a skip connection
        x_0 = x.detach().clone().reshape(2, x.size(0), self.length_prior,
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

        # Normalization
        x_a = x[:, :self.dim]
        x_b = x[:, self.dim:]
        x_a = x_a / torch.unsqueeze(x_a.sum(dim=1), 1)
        x_b = x_b / torch.unsqueeze(x_b.sum(dim=1), 1)

        # Adding skip connection
        x_a = x_a + self.skip_const * nn.functional.relu(x_0[:, :self.dim])
        x_b = x_b + self.skip_const * nn.functional.relu(x_0[:, self.dim:])

        # Normalization and dusting
        x_a = x_a / torch.unsqueeze(x_a.sum(dim=1), 1)
        x_b = x_b / torch.unsqueeze(x_b.sum(dim=1), 1)
        x_a = x_a + self.dust_const
        x_b = x_b + self.dust_const
        x_a = x_a / torch.unsqueeze(x_a.sum(dim=1), 1)
        x_b = x_b / torch.unsqueeze(x_b.sum(dim=1), 1)
        x = torch.cat((x_a, x_b), dim=1)
        return x

# Predictive network class
class pred_net(nn.Module):

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

        super(pred_net, self).__init__()
        self.dim = dim
        self.width = width
        self.l_1 = nn.Sequential(nn.Linear(2*dim, width), nn.BatchNorm1d(width),
                                nn.ELU())
        self.l_2 = nn.Sequential(nn.Linear(width, width), nn.BatchNorm1d(width),
                                 nn.ELU())
        self.l_3 = nn.Sequential(nn.Linear(width, dim))
        self.layers = [self.l_1, self.l_2, self.l_3]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x