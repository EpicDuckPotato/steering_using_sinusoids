import numpy as np

class CostModelSum(object):
  def __init__(self, nx):
    self.cost_models = []
    self.weights = []
    self.nx = nx

  def add_cost(self, cost_model, weight):
    self.cost_models.append(cost_model)
    self.weights.append(weight)

  def calc(self, x, u):
    return sum([weight*model.calc(x, u) for weight, model in zip(self.weights, self.cost_models)])

  def calcDiff(self, x, u):
    Lx = np.zeros(self.nx) 
    Lu = np.zeros_like(u)
    Lxx = np.zeros((self.nx, self.nx))
    # Lxx = np.eye(self.nx) # The line above would be the correct Hessian, but adding this diagonal term improves convergence
    Lux = np.zeros((u.shape[0], self.nx))
    Luu = np.zeros((u.shape[0], u.shape[0]))

    for weight, model in zip(self.weights, self.cost_models):
      Lxi, Lui, Lxxi, Luxi, Luui = model.calcDiff(x, u)
      Lx += weight*Lxi
      Lu += weight*Lui
      Lxx += weight*Lxxi
      Lux += weight*Luxi
      Luu += weight*Luui

    return Lx, Lu, Lxx, Lux, Luu
