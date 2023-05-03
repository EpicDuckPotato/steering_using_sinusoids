#!/usr/bin/env python3

import numpy as np
import rospy
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation as R
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState
from dynamics_constraint import DynamicsConstraint
from dirtran_problem import DirtranProblem
from cost_model_actuators import CostModelActuators
from cost_model_sum import CostModelSum
import time
import rospkg

# ROS stuff
rospy.init_node('car_node', anonymous=True)

br = TransformBroadcaster()
car_pose_msg = TransformStamped()
car_pose_msg.header.frame_id = 'world'
car_pose_msg.child_frame_id = 'car'
car_pose_msg.transform.rotation.w = 1

joints_pub = rospy.Publisher('/car/joint_state', JointState, queue_size=1)
joints_msg = JointState()
joints_msg.header.frame_id = 'world'
joints_msg.name.append('car_to_front_wheelbase')
joints_msg.position.append(0)
joints_msg.velocity.append(0)
joints_msg.effort.append(0)

# Car length and width (remember to change these if you change the URDF)
w = 0.5
l = 1.0

# Time step and number of time intervals in the trajectory (the number of states
# in the trajectory is therefore steps + 1)
dt = 0.01
steps = 999

# Number of states and controls
nx = 4
nu = 2

# Initial robot state
x_0 = -5
y_0 = 1
phi_0 = 1 # Front wheel angle wrt theta
theta_0 = 0.05 # Rear wheel angle

# Desired robot state
x_d = 0
y_d = 0.5
phi_d = 0
theta_d = 0

def dynamics(x, u, dt):
  # Unpack state and control
  x, y, phi, theta = x
  u1, u2 = u

  # Integrate dynamics
  xdot = np.cos(theta)*u1
  ydot = np.sin(theta)*u1
  phidot = u2
  thetadot = 1/l*np.tan(phi)*u1

  x += xdot*dt
  y += ydot*dt
  phi += phidot*dt
  theta += thetadot*dt

  return np.array([x, y, phi, theta])

def dynamics_deriv(x, u, dt):
  # Unpack state and control
  x, y, phi, theta = x
  u1, u2 = u

  c = np.cos(theta)
  s = np.sin(theta)

  # Deriv of next state wrt prev state
  A = np.eye(nx)
  A[0, 3] += -s*u1*dt
  A[1, 3] += c*u1*dt
  A[3, 2] = 1/(l*np.cos(phi)**2)*u1*dt

  # Deriv of next state wrt control
  B = np.zeros((nx, nu))
  B[0, 0] = c*dt
  B[1, 0] = s*dt
  B[2, 1] = dt
  B[3, 0] = 1/l*np.tan(phi)*dt

  return A, B

def xdiff(x1, x2):
  return x2 - x1

def xdiff_deriv(x1, x2):
  return -np.eye(nx), np.eye(nx)

# Trajectory optimization. Here x refers to the total state of size nx
x0lb = np.array([x_0, y_0, phi_0, theta_0])
x0ub = np.copy(x0lb)
xlb = -100*np.ones(nx)
xub = 100*np.ones(nx)
ulb = -100*np.ones(nu)
uub = 100*np.ones(nu)
costs = []
constraints = []
init_xs = []
init_us = []
xflb = np.array([x_d, y_d, phi_d, theta_d])
xfub = np.copy(xflb)

for step in range(steps):
  costs.append(CostModelActuators(1, nx))
  constraints.append(DynamicsConstraint(nx, nu, 0, dynamics, dynamics_deriv, xdiff, xdiff_deriv, step, dt))
  alpha = step/steps
  # init_xs.append((1 - alpha)*x0lb + alpha*xflb) # Initialize with straight-line
  init_xs.append(x0lb) # Initialize with start state
  init_us.append(np.zeros(nu))

costs.append(CostModelSum(nx))

# init_xs.append(np.copy(xflb)) # Initialize with straight-line
init_xs.append(np.copy(x0lb)) # Initialize with start state
init_us.append(np.zeros(nu)) 

bt = time.perf_counter()
problem = DirtranProblem(steps, dt, nx, nu, x0lb, x0ub, xlb, xub, ulb, uub, costs, constraints, init_xs, init_us, xflb, xfub)
xs, us, status = problem.solve()
at = time.perf_counter()
print('Optimized trajectory in %f seconds' %(at - bt))
x_trj = [x[0] for x in xs]
y_trj = [x[1] for x in xs]
phi_trj = [x[2] for x in xs]
theta_trj = [x[3] for x in xs]

control_cost = sum([0.5*u@u*dt for u in us])
print('Total control cost: %f' %(control_cost))

np.save('car_trj_trajopt.npy', np.stack((x_trj, y_trj, phi_trj, theta_trj), 1))

rospack = rospkg.RosPack()
rospath = rospack.get_path('steering_using_sinusoids')
np.save(rospath + '/scripts/car_trj_trajopt.npy', np.stack((x_trj, y_trj, phi_trj, theta_trj), 1))

# Visualize results
start_time = rospy.Time.now()
i = 0
rate = rospy.Rate(1/dt)
while not rospy.is_shutdown():
  x = x_trj[i]
  y = y_trj[i]
  phi = phi_trj[i]
  theta = theta_trj[i]
  if i < len(x_trj) - 1 and (rospy.Time.now() - start_time).to_sec() > 2:
    i += 1

  # Publish visualization
  joints_msg.position[0] = phi
  joints_msg.header.stamp = rospy.Time.now()
  joints_pub.publish(joints_msg)

  car_pose_msg.transform.translation.x = x
  car_pose_msg.transform.translation.y = y
  car_pose_msg.transform.translation.z = 0.75
  car_pose_msg.transform.rotation.x = 0
  car_pose_msg.transform.rotation.y = 0
  car_pose_msg.transform.rotation.z = np.sin(theta/2)
  car_pose_msg.transform.rotation.w = np.cos(theta/2)
  car_pose_msg.header.stamp = rospy.Time.now()
  br.sendTransform(car_pose_msg)

  rate.sleep()
