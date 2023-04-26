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
  xdot = u1
  ydot = np.tan(theta)*u1
  phidot = u2
  thetadot = 1/l*np.tan(phi)/np.cos(theta)*u1

  x += xdot*dt
  y += ydot*dt
  phi += phidot*dt
  theta += thetadot*dt

  return x, y, phi, theta

def v_to_u(phi, theta, v1, v2):
  u1 = v1
  u2 = -2/l*np.sin(phi)**2*np.tan(theta)*v1 + l*np.cos(theta)**2*np.cos(phi)**2*v2
  return u1, u2

def convert_to_chained(x, y, phi, theta):
  xi1 = x
  xi2 = 1/(l*np.cos(theta)**2)*np.tan(phi)
  xi3 = np.tan(theta)
  xi4 = y
  return xi1, xi2, xi3, xi4

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

# Initial and desired robot state in chained coordinates
xi1_0, xi2_0, xi3_0, xi4_0 = convert_to_chained(x_0, y_0, phi_0, theta_0)
xi1_d, xi2_d, xi3_d, xi4_d = convert_to_chained(x_d, y_d, phi_d, theta_d)

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

# In stage 0, we steer xi1 and xi2
# In stage 1, we steer xi3
# In stage 2, we steer xi4

##### STAGE 0 ####
delta_t0 = 4 # Desired duration of stage 0
t = 0 # Set t = 0 at the beginning of each stage for clarity of thought

rel_rate_xi2_xi1 = 8
v1_stage0 = (xi1_d - xi1_0)/delta_t0
v2_stage0 = rel_rate_xi2_xi1*(xi2_d - xi2_0)/delta_t0

# Integrate to get the state at the end of the stage
while t < delta_t0:
  v1 = v1_stage0

  if t < delta_t0/rel_rate_xi2_xi1:
    v2 = v2_stage0
  else:
    v2 = 0

  u1, u2 = v_to_u(phi, theta, v1, v2)
  x, y, phi, theta = integrate_dynamics(x, y, phi, theta, u1, u2, dt)
  x_trj.append(x)
  y_trj.append(y)
  phi_trj.append(phi)
  theta_trj.append(theta)

  xi1, xi2, xi3, xi4 = convert_to_chained(x, y, phi, theta)
  t += dt

##### STAGE 1 ####
k = 1
delta_t1 = 2 # Desired duration of stage 1
w_stage1 = 2*np.pi/delta_t1
t = 0 # Set t = 0 at the beginning of each stage for clarity of thought

xi1, xi2, xi3, xi4 = convert_to_chained(x, y, phi, theta)

# Pick an arbitrary b (amplitude of sinusoid for v2)
b_stage1 = -np.pi/4
a_stage1 = ((xi3_d - xi3)*np.math.factorial(k)*w_stage1**(k + 1)/(2*np.pi*b_stage1))**(1/k)*2

# Integrate to get the state at the end of the stage
while t < delta_t1:
  v1 = a_stage1*np.sin(w_stage1*t)
  v2 = b_stage1*np.cos(w_stage1*k*t)

  u1, u2 = v_to_u(phi, theta, v1, v2)
  x, y, phi, theta = integrate_dynamics(x, y, phi, theta, u1, u2, dt)
  x_trj.append(x)
  y_trj.append(y)
  phi_trj.append(phi)
  theta_trj.append(theta)

  t += dt

##### STAGE 2 ####
k = 2
delta_t2 = 4 # Desired duration of stage 2
w_stage2 = 2*np.pi/delta_t2
t = 0 # Set t = 0 at the beginning of each stage for clarity of thought

xi1, xi2, xi3, xi4 = convert_to_chained(x, y, phi, theta)

# Pick an arbitrary b (amplitude of sinusoid for v2)
b_stage2 = -np.pi/4
a_stage2 = ((xi4_d - xi4)*np.math.factorial(k)*w_stage2**(k + 1)/(2*np.pi*b_stage2))**(1/k)*2

# Integrate to get the state at the end of the stage
while t < delta_t2:
  v1 = a_stage2*np.sin(w_stage2*t)
  v2 = b_stage2*np.cos(w_stage2*k*t)

  u1, u2 = v_to_u(phi, theta, v1, v2)
  x, y, phi, theta = integrate_dynamics(x, y, phi, theta, u1, u2, dt)
  x_trj.append(x)
  y_trj.append(y)
  phi_trj.append(phi)
  theta_trj.append(theta)

  t += dt

print(y)
quit()

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
