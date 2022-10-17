import numpy as np

def cross_entropy_error(y,t):
  delta = 1e-7  #発散防止
  return -np.sum(t*np.log(y+delta))


def sum_squared_error(y, t):
  return 0.5 * np.sum((y-t)**2)
