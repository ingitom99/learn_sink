import torch
import torch.nn as nn
from torchvision.transforms import Resize

class pred_net(nn.Module):

  def __init__(self, dim):
    super(pred_net, self).__init__()
    self.dim = dim
    self.l1 = nn.Sequential(nn.Linear(2*dim, 4*dim), nn.BatchNorm1d(4*dim), nn.ELU())
    self.l2 = nn.Sequential(nn.Linear(4*dim, 4*dim), nn.BatchNorm1d(4*dim), nn.ELU())
    self.l3 = nn.Sequential(nn.Linear(4*dim, dim))
    self.layers = [self.l1, self.l2, self.l3]

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
  
class gen_net(nn.Module):
  
  def __init__(self, dim_in, dim_out, dust_const, skip_const):
    super(gen_net, self).__init__()
    self.dim_in = dim_in
    self.dim_out = dim_out
    self.dust_const = dust_const
    self.skip_const = skip_const
    self.length_in = int(self.dim_in**.5)
    self.length_out = int(self.dim_out**.5)
    self.l1 = nn.Sequential(nn.Linear(2*dim_in, 2*dim_out),nn.BatchNorm1d(2*dim_out), nn.Sigmoid())
    self.layers = [self.l1]

  def forward(self, x):
    x_0 = x.detach().clone().reshape(2, x.size(0), self.length_in, self.length_in)
    transform = Resize((self.length_out, self.length_out), antialias=True)
    x_0 = torch.cat((transform(x_0[0]).reshape(x.size(0), self.dim_out), transform(x_0[1]).reshape(x.size(0), self.dim_out)), 1)
    for layer in self.layers:
      x = layer(x)

    
    x_a = x[:, :self.dim_out]
    x_b = x[:, self.dim_out:]
    x_a = x_a / torch.unsqueeze(x_a.sum(dim=1), 1)
    x_b = x_b / torch.unsqueeze(x_b.sum(dim=1), 1)
    x_a = x_a + self.skip_const * nn.functional.relu(x_0[:, :dim])
    x_b = x_b + self.skip_const * nn.functional.relu(x_0[:, dim:])
    x_a = x_a / torch.unsqueeze(x_a.sum(dim=1), 1)
    x_b = x_b / torch.unsqueeze(x_b.sum(dim=1), 1)
    x_a = x_a + self.dust_const
    x_b = x_b + self.dust_const
    x_a = x_a / torch.unsqueeze(x_a.sum(dim=1), 1)
    x_b = x_b / torch.unsqueeze(x_b.sum(dim=1), 1)
    x = torch.cat((x_a, x_b), dim=1)
    return x