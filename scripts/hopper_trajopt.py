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

g = 9.8 # Gravity
mb = 1 # Body mass
ml = 5 # Leg mass
l_0 = 0 # Initial leg extension (0 extension corresponds to leg length 1)
phi_0 = 0 # Initial leg angle, radians
theta_0 = 0 # Initial body angle, radians
xdot_0 = 0 # Initial x velocity
zdot_0 = 5 # Initial z velocity
x_0 = 0 # Initial horizontal position
z_0 = 1 # Initial height
T = 2*zdot_0/g # Time of flight
w = 2*np.pi/T

l_d = l_0
phi_d = phi_0
theta_d = np.pi/4

# ROS stuff
rospy.init_node('hopper_node', anonymous=True)

body_pub = rospy.Publisher('/hopper/body', Marker, queue_size=1)
body_marker = Marker()
body_marker.type = Marker.CUBE
body_marker.action = Marker.ADD
body_marker.pose.position.x = 0
body_marker.pose.position.y = 0
body_marker.pose.position.z = 0
body_marker.pose.orientation.x = 0
body_marker.pose.orientation.y = 0
body_marker.pose.orientation.z = 0
body_marker.pose.orientation.w = 1
body_marker.scale.x = 0.5
body_marker.scale.y = 0.25
body_marker.scale.z = 0.25
body_marker.color.r = 0.5
body_marker.color.g = 0.5
body_marker.color.b = 0.5
body_marker.color.a = 1
body_marker.header.frame_id = "world"
body_marker.ns = "hopper_node"

leg_pub = rospy.Publisher('/hopper/leg', Marker, queue_size=1)
leg_marker = Marker()
leg_marker.type = Marker.CYLINDER
leg_marker.action = Marker.ADD
leg_marker.pose.position.x = 0
leg_marker.pose.position.y = 0
leg_marker.pose.position.z = 0
leg_marker.pose.orientation.x = 0
leg_marker.pose.orientation.y = 0
leg_marker.pose.orientation.z = 0
leg_marker.pose.orientation.w = 1
leg_marker.scale.x = 0.2
leg_marker.scale.y = 0.2
leg_marker.scale.z = 1.0
leg_marker.color.r = 0.5
leg_marker.color.g = 0.5
leg_marker.color.b = 0.5
leg_marker.color.a = 1
leg_marker.header.frame_id = "world"
leg_marker.ns = "hopper_node"

br = TransformBroadcaster()
body_pose_msg = TransformStamped()
body_pose_msg.header.frame_id = 'world'
body_pose_msg.child_frame_id = 'body'
body_pose_msg.transform.rotation.w = 1

# Time step and number of time intervals in the trajectory (the number of states
# in the trajectory is therefore steps + 1)
dt = 0.01
steps = int(np.ceil(T/dt))

# Number of states and controls
nx = 3
nu = 2

def dynamics(x, u, dt):
  # Unpack state and control
  phi, l, theta = x
  u1, u2 = u

  # Integrate dynamics
  phi += u1*dt
  l += u2*dt
  theta += -ml*(l + 1)**2/(1 + ml*(l + 1)**2)*u1*dt

  return np.array([phi, l, theta])

def dynamics_deriv(x, u, dt):
  # Unpack state and control
  phi, l, theta = x
  u1, u2 = u

  # Deriv of next state wrt prev state
  A = np.eye(nx)
  A[2, 1] = -2*ml*(l + 1)/(1 + ml*(l + 1)**2)**2*u1*dt

  # Deriv of next state wrt control
  B = np.zeros((nx, nu))
  B[0, 0] = dt
  B[1, 1] = dt
  B[2, 0] = -ml*(l + 1)**2/(1 + ml*(l + 1)**2)*dt

  return A, B

def xdiff(x1, x2):
  return x2 - x1

def xdiff_deriv(x1, x2):
  return -np.eye(nx), np.eye(nx)

# Trajectory optimization. Here x refers to the total state of size nx
x0lb = np.array([phi_0, l_0, theta_0])
x0ub = np.copy(x0lb)
xlb = -100*np.ones(nx)
xub = 100*np.ones(nx)
ulb = -100*np.ones(nu)
uub = 100*np.ones(nu)
costs = []
constraints = []
init_xs = []
init_us = []
xflb = np.array([phi_d, l_d, theta_d])
xfub = np.copy(xflb)

a1, a2 = -4.290043105194483, 5

x = np.copy(x0lb)

for step in range(steps):
  costs.append(CostModelActuators(1, nx))
  constraints.append(DynamicsConstraint(nx, nu, 0, dynamics, dynamics_deriv, xdiff, xdiff_deriv, step, dt))
  alpha = step/steps

  # Initialize with straight-line
  init_xs.append((1 - alpha)*x0lb + alpha*xflb) 
  init_us.append(np.zeros(nu))

  # Initialize with start state
  '''
  init_xs.append(x0lb) 
  init_us.append(np.zeros(nu))
  '''

  # Initialize using sinusoids
  '''
  init_xs.append(np.copy(x))
  t = step*dt
  init_us.append(np.array([a1*np.sin(w*t), a2*np.cos(w*t)]))
  x = dynamics(init_xs[-1], init_us[-1], dt)
  '''

costs.append(CostModelSum(nx))

# Initialize with straight-line
init_xs.append(np.copy(xflb)) 
init_us.append(np.zeros(nu)) 

# Initialize with start state
'''
init_xs.append(np.copy(x0lb)) 
init_us.append(np.zeros(nu)) 
'''

# Initialize using sinusoids
'''
init_xs.append(np.copy(x))
init_us.append(np.zeros(nu)) 
'''

problem = DirtranProblem(steps, dt, nx, nu, x0lb, x0ub, xlb, xub, ulb, uub, costs, constraints, init_xs, init_us, xflb, xfub)
bt = time.perf_counter()
xs, us, status = problem.solve()
at = time.perf_counter()
print('Optimized trajectory in %f seconds' %(at - bt))
phi_trj = [x[0] for x in xs]
l_trj = [x[1] for x in xs]
theta_trj = [x[2] for x in xs]

control_cost = sum([0.5*u@u*dt for u in us])
print('Total control cost: %f' %(control_cost))

# Visualize results
start_time = rospy.Time.now()
i = 0
x = x_0
z = z_0
xdot = xdot_0
zdot = zdot_0
rate = rospy.Rate(1/dt)
while not rospy.is_shutdown():
  phi = phi_trj[i]
  l = l_trj[i]
  theta = theta_trj[i]

  if i < len(phi_trj) - 1 and (rospy.Time.now() - start_time).to_sec() > 2:
    i += 1

    x += xdot*dt
    z += (zdot - 0.5*g*dt)*dt
    zdot += -g*dt

  # Publish visualization
  body_pos = np.array([x, 0., z])
  body_marker.pose.position.x = body_pos[0]
  body_marker.pose.position.z = body_pos[2]
  body_quat = R.from_rotvec([0, -theta, 0]).as_quat()
  body_marker.pose.orientation.x = body_quat[0]
  body_marker.pose.orientation.y = body_quat[1]
  body_marker.pose.orientation.z = body_quat[2]
  body_marker.pose.orientation.w = body_quat[3]

  leg_R = R.from_rotvec([0., -phi, 0.])
  leg_quat = leg_R.as_quat()
  leg_marker.pose.orientation.x = leg_quat[0]
  leg_marker.pose.orientation.y = leg_quat[1]
  leg_marker.pose.orientation.z = leg_quat[2]
  leg_marker.pose.orientation.w = leg_quat[3]
  leg_length = 1 + l
  leg_marker.scale.z = leg_length
  leg_center = body_pos - leg_length/2*leg_R.as_matrix()[:, 2]
  leg_marker.pose.position.x = leg_center[0]
  leg_marker.pose.position.y = leg_center[1]
  leg_marker.pose.position.z = leg_center[2]

  body_marker.header.stamp = rospy.Time.now()
  leg_marker.header.stamp = rospy.Time.now()
  body_pub.publish(body_marker)
  leg_pub.publish(leg_marker)

  body_pose_msg.transform.translation.x = body_pos[0]
  body_pose_msg.transform.translation.y = body_pos[1]
  body_pose_msg.transform.translation.z = body_pos[2]
  body_pose_msg.transform.rotation.x = body_quat[0]
  body_pose_msg.transform.rotation.y = body_quat[1]
  body_pose_msg.transform.rotation.z = body_quat[2]
  body_pose_msg.transform.rotation.w = body_quat[3]
  body_pose_msg.header.stamp = rospy.Time.now()
  br.sendTransform(body_pose_msg)
  
  rate.sleep()
