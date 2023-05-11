import torch
import matplotlib.pyplot as plt

def hilb_proj_loss(u, v):
  log_u = torch.log(u)
  log_v = torch.log(v)
  diff = log_u - log_v
  spectrum = torch.max(diff, dim=1)[0] - torch.min(diff, dim=1)[0]
  loss = spectrum.mean()
  return loss

def plot_XPT(X, P, T):
  fig, ax = plt.subplots(2, 2)
  ax[0][0].set_title('Mu')
  ax[0][1].set_title('Nu')
  ax[1][0].set_title('P')
  ax[1][1].set_title('T')
  ax[0][0].imshow(X[0, :784].cpu().detach().numpy().reshape(28, 28), cmap='magma')
  ax[0][1].imshow(X[0, 784:].cpu().detach().numpy().reshape(28,28), cmap='magma')
  ax[1][0].imshow(P[0].cpu().detach().numpy().reshape(28, 28), cmap='magma')
  ax[1][1].imshow(T[0].cpu().detach().numpy().reshape(28,28), cmap='magma')
  plt.show()

  
def prior_sampler(n_samples, dim):
  sample = torch.randn((n_samples, 2 * dim))
  return sample

def random_noise(n_samples, dim, dust_const):
  sample_a = 4 * torch.rand((n_samples, dim)).double().to(device)
  sample_a /= torch.unsqueeze(sample_a.sum(dim=1), 1)
  sample_a = sample_a + dust_const
  sample_a /= torch.unsqueeze(sample_a.sum(dim=1), 1)
  sample_b = 4 * torch.rand((n_samples, dim)).double().to(device)**2
  sample_b /= torch.unsqueeze(sample_b.sum(dim=1), 1)
  sample_b = sample_b + dust_const
  sample_b /= torch.unsqueeze(sample_b.sum(dim=1), 1)
  sample = torch.cat((sample_a, sample_b), dim=1)
  return sample