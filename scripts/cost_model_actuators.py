import numpy as np

class CostModelActuators(object):
  def __init__(self, weights, nx):
    self.weights = weights
    self.nx = nx

  def calc(self, x, u):
    return 0.5*np.sum(u*u*self.weights)

  def calcDiff(self, x, u):
    Lx = np.zeros(self.nx)
    Lu = self.weights*u
    Lxx = np.zeros((self.nx, self.nx))
    Lux = np.zeros((u.shape[0], self.nx))
    Luu = np.zeros((u.shape[0], u.shape[0]))
    np.fill_diagonal(Luu, self.weights)
    return Lx, Lu, Lxx, Lux, Luu
