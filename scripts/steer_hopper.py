#!/usr/bin/env python3

import numpy as np
import rospy
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation as R

g = 9.8 # Gravity
mb = 1 # Body mass
ml = 1 # Leg mass
l0 = 0 # Initial leg extension (0 extension corresponds to leg length 1)
phi0 = 0 # Initial leg angle, radians
theta0 = 0 # Initial body angle, radians
xdot0 = 0 # Initial x velocity
zdot0 = 5 # Initial z velocity
x0 = 0 # Initial horizontal position
z0 = 1 # Initial height
T = 2*zdot0/g # Time of flight

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
body_marker.color.r = 0
body_marker.color.g = 1
body_marker.color.b = 0
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
leg_marker.color.r = 0
leg_marker.color.g = 0
leg_marker.color.b = 1
leg_marker.color.a = 1
leg_marker.header.frame_id = "world"
leg_marker.ns = "hopper_node"

dt = 0.02
rate = rospy.Rate(1/dt)

phi = phi0
l = l0
theta = theta0
x = x0
z = z0
xdot = xdot0
zdot = zdot0

t = 0

start_time = rospy.Time.now()
while not rospy.is_shutdown():
  u1 = 0
  u2 = 0

  if (rospy.Time.now() - start_time).to_sec() > 2 and t < T:
    # Integrate dynamics
    phi += u1*dt
    l += u2*dt
    theta += ml*(l + 1)**2/(1 + ml*(l + 1)**2)*u1*dt

    x += xdot*dt
    z += (zdot - 0.5*g*dt)*dt
    zdot += -g*dt

    t += dt

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
  
  rate.sleep()
