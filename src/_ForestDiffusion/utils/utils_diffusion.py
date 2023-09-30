import numpy as np
from ForestDiffusion.utils.diffusion import VPSDE

# Build the dataset of x(t) at multiple values of t 
def build_data_xt(x0, x1, n_t=101, diffusion_type='flow', eps=1e-3, sde=None):
  b, c = x1.shape

  # Expand x0, x1
  x0 = np.expand_dims(x0, axis=0) # [1, b, c]
  x1 = np.expand_dims(x1, axis=0) # [1, b, c]

  # t and expand
  t = np.linspace(eps, 1, num=n_t)
  t_expand = np.expand_dims(t, axis=(1,2)) # [t, 1, 1]

  if diffusion_type == 'vp': # Forward diffusion from x0 to x1
    mean, std = sde.marginal_prob(x1, t_expand)
    x_t = mean + std*x0
  else: # Interpolation between x0 and x1
    x_t = t_expand * x1 + (1 - t_expand) * x0 # [t, b, c]
  x_t = x_t.reshape(-1,c) # [t*b, c]

  X = x_t

  # Output to predict
  if diffusion_type == 'vp':
    alpha_, sigma_ = sde.marginal_prob_coef(x1, t_expand)
    y = x0.reshape(b, c)
  else:
    y = x1.reshape(b, c) - x0.reshape(b, c) # [b, c]

  return X, y

#### Below is for Flow-Matching Sampling ####

# Euler solver
def euler_solve(y0, my_model, N=101):
  h = 1 / (N-1)
  y = y0
  t = 0
  # from t=0 to t=1
  for i in range(N-1):
    y = y + h*my_model(t=t, y=y)
    t = t + h
  return y
