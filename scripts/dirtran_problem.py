import numpy as np
import cyipopt
import scipy.sparse

class DirtranProblem(object):
  # steps is the length of the trajectory we're optimizing, i.e. it's the number of (x, u) pairs in the trajectory.
  # costs is a list of length steps where each element is a CostModel (see cost_model_sum for how to construct
  # one of these).
  # constraints is a list of any length (element i doesn't necessarily correspond to step i).
  # See dynamics_constraint.py for how to construct one of these.
  # init_xs and init_us are also length steps + 1, providing the initial guess.
  # xflb and xfub are optional constraints on the terminal state
  def __init__(self, steps, dt, nx, nu, x0lb, x0ub, xlb, xub, ulb, uub, costs, constraints, init_xs, init_us, xflb=None, xfub=None):
    self.steps = steps
    self.dt = dt
    self.nx = nx
    self.nu = nu
    self.vars_per_step = self.nx + self.nu
    self.costs = costs
    self.constraint_models = constraints
    self.init_xs = init_xs
    self.init_us = init_us

    self.num_constraints = sum([constraint.size for constraint in self.constraint_models])

    self.cl = np.concatenate([model.lb for model in self.constraint_models])
    self.cu = np.concatenate([model.ub for model in self.constraint_models])

    self.num_decision_vars = self.get_num_decision_vars()

    self.lb = -10000*np.ones(self.num_decision_vars)
    self.ub = 10000*np.ones(self.num_decision_vars)

    for step in range(self.steps + 1):
      # State constraints
      self.lb[self.vars_per_step*step:self.vars_per_step*step + self.nx] = xlb
      self.ub[self.vars_per_step*step:self.vars_per_step*step + self.nx] = xub

      # Actuator constraints
      self.lb[self.vars_per_step*step + self.nx:self.vars_per_step*(step + 1)] = ulb
      self.ub[self.vars_per_step*step + self.nx:self.vars_per_step*(step + 1)] = uub

    # Initial constraint
    self.lb[:self.nx] = x0lb
    self.ub[:self.nx] = x0ub

    # Terminal constraint
    if xflb is not None:
      self.lb[-self.vars_per_step:-self.nu] = xflb
    if xfub is not None:
      self.ub[-self.vars_per_step:-self.nu] = xfub

  def solve(self):
    warm_start = np.zeros(self.num_decision_vars)

    for step in range(self.steps + 1):
      xstart = self.vars_per_step*step
      ustart = xstart + self.nx
      uend = ustart + self.nu
      warm_start[xstart:ustart] = self.init_xs[step]
      warm_start[ustart:uend] = self.init_us[step]

    nlp = cyipopt.Problem(
       n=self.num_decision_vars,
       m=self.num_constraints,
       problem_obj=self,
       lb=self.lb,
       ub=self.ub,
       cl=self.cl,
       cu=self.cu,
    )
    #nlp.addOption(b'print_level', 0)
    nlp.addOption(b'warm_start_init_point', 'yes')
    nlp.addOption(b'tol', 1.0)
    '''
    nlp.addOption(b'tol', 1.0)
    nlp.addOption(b'compl_inf_tol', 1e-2)
    nlp.addOption(b'warm_start_init_point', 'yes')
    nlp.addOption(b'mu_init', 1e-4)
    '''

    '''
    # TODO: take out
    v = warm_start
    c = self.constraints(v)
    Jrows, Jcols = self.jacobianstructure()
    Jvals = self.jacobian(v)
    print(len(Jrows), len(Jcols), len(Jvals))
    J = scipy.sparse.coo_array((Jvals, (Jrows, Jcols)), shape=(self.num_constraints, self.num_decision_vars)).todense()
    Jfd = np.zeros((self.num_constraints, self.num_decision_vars))
    vplus = np.copy(v)
    eps = 1e-7
    # for i in range(self.num_decision_vars - self.nu - self.nx, self.num_decision_vars):
    # for i in range(self.num_decision_vars):
    for i in range(self.nx + self.nu):
      vplus[i] += eps
      cplus = self.constraints(vplus)
      Jfd[:, i] = (cplus - c)/eps
      print(np.linalg.norm(J[:, i] - Jfd[:, i], ord=np.inf))
      vplus[i] = v[i]

    quit()
    '''

    '''
    print(np.linalg.norm(self.constraints(warm_start), ord=np.Inf))
    quit()
    '''
    '''
    self.constraints(warm_start)
    quit()
    '''

    soln, info = nlp.solve(warm_start)
    '''
    if info['status'] != 0:
      return [], [], False
    '''

    xs = []
    us = []
    for step in range(self.steps + 1):
      xs.append(soln[self.vars_per_step*step:self.vars_per_step*step + self.nx])
      us.append(soln[self.vars_per_step*step + self.nx:self.vars_per_step*(step + 1)])

    return xs, us, True

  def get_num_decision_vars(self):
    return (self.nx + self.nu)*(self.steps + 1)
    
  def get_num_constraints(self):
    return sum([constraint.size for constraint in self.constraint_models])

  def obj_step(self, v, step):
    xstart = self.vars_per_step*step
    ustart = xstart + self.nx
    uend = ustart + self.nu
    x = v[xstart:ustart]
    u = v[ustart:uend]
    return self.costs[step].calc(x, u)

  def objective(self, v):
    return sum([self.obj_step(v, step) for step in range(self.steps + 1)])*self.dt

  def grad_step(self, v, step):
    xstart = self.vars_per_step*step
    ustart = xstart + self.nx
    uend = ustart + self.nu
    x = v[xstart:ustart]
    u = v[ustart:uend]
    Lx, Lu, _, _, _ = self.costs[step].calcDiff(x, u)
    return np.concatenate((Lx, Lu))

  def gradient(self, v):
    """Returns the gradient of the objective with respect to v."""
    return np.concatenate([self.grad_step(v, step) for step in range(self.steps + 1)])*self.dt

  def constraints(self, v):
    """Returns the constraints."""
    # return np.concatenate([constraint.get_c(v) for constraint in self.constraint_models])
    c = np.concatenate([constraint.get_c(v) for constraint in self.constraint_models])
    return c

  def jacobian(self, v):
    """Returns the Jacobian of the constraints with respect to v."""
    return np.concatenate([constraint.get_jacobian(v) for constraint in self.constraint_models])

  def jacobianstructure(self):
    """Returns the row and column indices for non-zero vales of the
    Jacobian."""
    all_rows = []
    all_cols = []
    startrow = 0
    for constraint in self.constraint_models:
      rows, cols = constraint.get_jacobian_structure(startrow) 
      all_rows.append(rows)
      all_cols.append(cols)
      startrow += constraint.size

    return np.concatenate(all_rows), np.concatenate(all_cols)

  def hessianstructure(self):
    """Returns the row and column indices for non-zero vales of the
    Hessian."""
    rows = []
    cols = []
    # Cost Hessian
    for step in range(self.steps + 1):
      for row in range(self.nx + self.nu):
        for col in range(row + 1): # Go until row + 1 to keep it lower triangular
          rows.append(self.vars_per_step*step + row)
          cols.append(self.vars_per_step*step + col)

    all_rows = [np.array(rows)]
    all_cols = [np.array(cols)]

    for constraint in self.constraint_models:
      rows, cols = constraint.get_hessian_structure() 
      all_rows.append(rows)
      all_cols.append(cols)

    return np.concatenate(all_rows), np.concatenate(all_cols)

  def cost_hessian_step(self, v, step):
    xstart = self.vars_per_step*step
    ustart = xstart + self.nx
    uend = xstart + self.nx + self.nu
    x = v[xstart:ustart]
    u = v[ustart:uend]
    _, _, Lxx, Lux, Luu = self.costs[step].calcDiff(x, u)

    cat = np.concatenate
    xrows = cat([Lxx[row][:row + 1] for row in range(self.nx)])
    urows = cat([cat((Lux[row], Luu[row][:row + 1])) for row in range(self.nu)])

    return cat((xrows, urows))

  def hessian(self, v, lagrange, obj_factor):
    """Returns the non-zero values of the Hessian."""
    cost_hessian = obj_factor*np.concatenate([self.cost_hessian_step(v, step) for step in range(self.steps + 1)])*self.dt

    startrow = 0
    constraint_hessians = []
    for constraint in self.constraint_models:
      constraint_hessians.append(constraint.get_hessian(v, lagrange, startrow))
      startrow += constraint.size

    return np.concatenate([cost_hessian] + constraint_hessians)


  def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                   d_norm, regularization_size, alpha_du, alpha_pr,
                   ls_trials):
    """Prints information at every Ipopt iteration."""
    '''
    if iter_count > 15:
      return False
    '''

    return 

    msg = "Objective value at iteration #{:d} is - {:g}"

    #print(msg.format(iter_count, obj_value))
