import torch

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sink_vec(mu, nu, C, reg, maxiter, V0=None):
  K = torch.exp(-C/reg)
  if (V0 == None):
    V0 = torch.ones_like(mu).double().to(device)
  v = V0
  for i in range(maxiter):
    u = mu / (K @ v.T).T
    v = nu / (K.T @ u.T).T
  return v

def sink_safe(MU, NU, C, reg, maxiter, V0):
  K = torch.exp(-C/reg)
  V_final = torch.zeros_like(MU).double().to(device)
  for i, (mu, nu, v0) in enumerate(zip(MU, NU, V0)):
    v_prev = v0
    for iter in range(maxiter):
      u_new = mu / (K @ v_prev)
      v_new = nu / (K.T @ u_new)
      if (torch.isnan(u_new).sum() > 0):
        print('U IS NAN!')
        V_final[i] = v_prev
        break
      if (torch.isnan(v_new).sum() > 0):
        print('V IS NAN!')
        V_final[i] = v_prev
        break
      u_prev = u_new
      v_prev = v_new
      if (iter == (maxiter-1)):

        V_final[i] = v_prev
  return V_final

def sink_dist(mu, nu, C, reg, maxiter, v0):
  K = torch.exp(-C/reg)
  v = v0
  for i in range(maxiter):
    u = mu / (K @ v)
    v = nu / (K.T @ u)
  G = torch.diag(u)@K@torch.diag(v)    
  dist = torch.trace(C.T@G)
  return dist