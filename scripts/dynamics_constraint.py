import numpy as np

class DynamicsConstraint(object):
  def __init__(self, nx, nu, naux, dynamics, dynamics_deriv, statediff, statediff_deriv, step, dt):
    self.nx = nx
    self.nu = nu
    self.naux = naux
    self.vars_per_step = self.nx + self.nu + self.naux
    self.dynamics = dynamics
    self.dynamics_deriv = dynamics_deriv
    self.statediff = statediff
    self.statediff_deriv = statediff_deriv
    self.step = step
    self.x0start = self.vars_per_step*step
    self.x0end = self.x0start + self.nx
    self.u0start = self.x0end
    self.u0end = self.u0start + self.nu
    self.x1start = self.vars_per_step*(step + 1)
    self.x1end = self.vars_per_step*(step + 1) + self.nx
    self.size = self.nx
    self.lb = np.zeros(self.size)
    self.ub = np.zeros(self.size)
    self.dt = dt

  def get_c(self, v):
    x0 = v[self.x0start:self.x0end]
    u0 = v[self.u0start:self.u0end]
    x1 = v[self.x1start:self.x1end]
    x1_pred = self.dynamics(x0, u0, self.dt)

    # Defect should equal zero
    return self.statediff(x1_pred, x1)

  def get_jacobian_structure(self, startrow):
    rows = []
    cols = []
    for row in range(self.nx):
      for col in range(self.nx + self.nu): # Dependence on current x and u
        rows.append(startrow + row)
        cols.append(self.x0start + col)
      for col in range(self.nx): # Dependence on next x
        rows.append(startrow + row)
        cols.append(self.x1start + col)

    return np.array(rows), np.array(cols)

  def get_jacobian(self, v):
    x0 = v[self.x0start:self.x0end]
    u0 = v[self.u0start:self.u0end]
    x1 = v[self.x1start:self.x1end]
    x1_pred = self.dynamics(x0, u0, self.dt)

    A, B = self.dynamics_deriv(x0, u0, self.dt)

    Jx1_pred, Jx1 = self.statediff_deriv(x1_pred, x1)

    dc_dx0 = Jx1_pred@A
    dc_du0 = Jx1_pred@B
    dc_dx1 = Jx1

    cat = np.concatenate

    return cat([cat((dc_dx0[row], dc_du0[row], dc_dx1[row])) for row in range(self.nx)])

  # Gauss-Newton Hessian approximation
  def get_hessian_structure(self):
    return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
  
  def get_hessian(self, v, lagrange, startrow):
    return np.zeros(0)
