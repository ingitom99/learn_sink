# Imports
import torch

def euclidean_cost_matrix(width : int, height : int , normed: bool=True) -> torch.Tensor:
  """
  Returns a cost matrix for the euclidean distance between pixels in an image of size width x height
  """
  n = width*height
  M = torch.zeros([n, n])

  for a in range(n):
    for b in range(n):
      ax = a // width
      ay = a % width
      bx = b // width
      by = b % width
      M[a][b] = ((ax - bx)**2 + (ay - by)**2)*.5

  if (normed == True):
    M = M / M.max()
    
  return M
