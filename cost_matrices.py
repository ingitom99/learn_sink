import torch

def euclidean_cost_matrix(width, height, normed=True):
  n = width*height
  M = torch.zeros([n, n])
  for a in range(n):
    for b in range(n):
      ax = a // width
      ay = a % width
      bx = b // width
      by = b % width
      M[a][b] = (ax - bx)**2 + (ay - by)**2
  if (normed == True):
    M = M / M.max()
  return M
