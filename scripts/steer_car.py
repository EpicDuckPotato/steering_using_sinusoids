#!/usr/bin/env python3

import numpy as np
import rospy
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation as R
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState
from scipy.integrate import quad

rospy.init_node('hopper_node', anonymous=True)

# Car length and width (remember to change these if you change the URDF)
w = 0.5
l = 1.0

def integrate_dynamics(x, y, phi, theta, u1, u2, dt):
  # Integrate dynamics
  xdot = np.cos(theta)*u1
  ydot = np.sin(theta)*u1
  phidot = u2
  thetadot = 1/l*np.tan(phi)*u1

  x += xdot*dt
  y += ydot*dt
  phi += phidot*dt
  theta += thetadot*dt

  return x, y, phi, theta

def v_to_u(theta, v1, v2):
  u1 = v1/np.cos(theta)
  u2 = v2
  return u1, u2

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

dt = 0.01

# Initialize robot state
x_0 = -5
y_0 = 1
phi_0 = 1 # Front wheel angle wrt theta
theta_0 = 0.05 # Rear wheel angle

# Desired robot state
x_d = 0
y_d = 0.5
phi_d = 0
theta_d = 0

# Initialize state and time
x = x_0
y = y_0
phi = phi_0 
theta = theta_0

# Store computed trajectory
x_trj = [x]
y_trj = [y]
phi_trj = [phi]
theta_trj = [theta]

# In stage 0, we steer x and phi to their desired values.
# In stage 1, we steer theta to its desired value.
# In stage 2, we steer y to its desired value

##### STAGE 0 ####
delta_t0 = 4 # Desired duration of stage 0
t = 0 # Set t = 0 at the beginning of each stage for clarity of thought

rel_rate_phi_x = 8 # Steer phi faster than x to avoid singularity
v1_stage0 = (x_d - x_0)/delta_t0
v2_stage0 = rel_rate_phi_x*(phi_d - phi_0)/delta_t0

# Integrate to get the state at the end of the stage
while t < delta_t0:
  v1 = v1_stage0

  if t < delta_t0/rel_rate_phi_x:
    v2 = v2_stage0
  else:
    v2 = 0

  u1, u2 = v_to_u(theta, v1, v2)
  x, y, phi, theta = integrate_dynamics(x, y, phi, theta, u1, u2, dt)
  x_trj.append(x)
  y_trj.append(y)
  phi_trj.append(phi)
  theta_trj.append(theta)

  t += dt

##### STAGE 1 ####
delta_t1 = 2 # Desired duration of stage 1
w_stage1 = 2*np.pi/delta_t1
t = 0 # Set t = 0 at the beginning of each stage for clarity of thought

alpha = np.sin(theta)
alpha_d = np.sin(theta_d)

# Pick an arbitrary a2 (amplitude of sinusoid for v2 = phidot)
a2_stage1 = np.pi/4

# Numerically compute the first Fourier coefficient of the Fourier series for the
# mutliplier for v1 in the dynamics of alphadot
def integrand(t):
  return 1/l*np.tan(a2_stage1/w_stage1*np.sin(w_stage1*t) + phi)*np.sin(w_stage1*t)
beta1_stage1 = w_stage1/np.pi*quad(integrand, 0, delta_t1)[0]
a1_stage1 = (alpha_d - alpha)*w_stage1/(np.pi*beta1_stage1)

# Integrate to get the state at the end of the stage
while t < delta_t1:
  v1 = a1_stage1*np.sin(w_stage1*t)
  v2 = a2_stage1*np.cos(w_stage1*t)

  u1, u2 = v_to_u(theta, v1, v2)
  x, y, phi, theta = integrate_dynamics(x, y, phi, theta, u1, u2, dt)
  x_trj.append(x)
  y_trj.append(y)
  phi_trj.append(phi)
  theta_trj.append(theta)

  t += dt

##### STAGE 2 ####
delta_t2 = 4 # Desired duration of stage 2
w_stage2 = 2*np.pi/delta_t2
t = 0 # Set t = 0 at the beginning of each stage for clarity of thought

alpha = np.sin(theta)

# Pick an arbitrary a2 (amplitude of sinusoid for v2 = phidot)
a2_stage2 = -np.pi/4

# Now we have to binary search for a1
a1_low = 0
a1_high = 1

class inner_integrand(object):
  def __init__(self, a1):
    self.a1 = a1

  def __call__(self, tau):
    phi_t = a2_stage2/(2*w_stage2)*np.sin(2*w_stage2*tau) + phi
    return 1/l*np.tan(phi_t)*self.a1*np.sin(w_stage2*tau)

class outer_integrand(object):
  def __init__(self, a1):
    self.a1 = a1
    self.inner_integrand = inner_integrand(a1)

  def __call__(self, t):
    alpha_t = quad(self.inner_integrand, 0, t)[0] + alpha
    return alpha_t/np.sqrt(1 - alpha_t**2)*np.sin(w_stage2*t)

# Find an upper bound
got_upper_bound = False
while not got_upper_bound:
  a1 = a1_high
  beta1 = w_stage2/np.pi*quad(outer_integrand(a1), 0, delta_t2)[0]
  y_final = y + np.pi*a1*beta1/w_stage2 
  if y_final < y_d:
    got_upper_bound = True
  else:
    a1_low = a1_high
    a1_high *= 2

# Binary search within bounds
max_iter = 10
tol = 0.01
for i in range(max_iter):
  a1 = (a1_low + a1_high)/2
  beta1 = w_stage2/np.pi*quad(outer_integrand(a1), 0, delta_t2)[0]
  y_final = y + np.pi*a1*beta1/w_stage2 

  y_err = np.abs(y_final - y_d)
  if y_err < tol:
    print('Found a good a1 value')
    break

  if y_final < y_d:
    a1_high = a1
  else:
    a1_low = a1

a1_stage2 = a1

# Integrate to get the state at the end of the stage
dt = 0.01
while t < delta_t2:
  v1 = a1_stage2*np.sin(w_stage2*t)
  v2 = a2_stage2*np.cos(2*w_stage2*t)

  u1, u2 = v_to_u(theta, v1, v2)
  x, y, phi, theta = integrate_dynamics(x, y, phi, theta, u1, u2, dt)
  x_trj.append(x)
  y_trj.append(y)
  phi_trj.append(phi)
  theta_trj.append(theta)

  t += dt

# Visualize results
i = 0
rate = rospy.Rate(1/dt)
while not rospy.is_shutdown():
  x = x_trj[i]
  y = y_trj[i]
  phi = phi_trj[i]
  theta = theta_trj[i]
  if i < len(x_trj) - 1:
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
