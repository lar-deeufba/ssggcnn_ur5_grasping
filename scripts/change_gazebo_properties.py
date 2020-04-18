#!/usr/bin/env python
from gazebo_msgs.msg import ODEPhysics
from geometry_msgs.msg import Vector3
from gazebo_msgs.srv import SetPhysicsProperties
from std_msgs.msg import Float64
import rospy

set_gravity = rospy.ServiceProxy('/gazebo/set_physics_properties', SetPhysicsProperties)

time_step = Float64(0.004) # default: 0.001
max_update_rate = Float64(250.0) # default: 1000.0
gravity = Vector3()
gravity.x = 0.0
gravity.y = 0.0
gravity.z = -9.8
ode_config = ODEPhysics()
ode_config.auto_disable_bodies = False
ode_config.sor_pgs_precon_iters = 0
ode_config.sor_pgs_iters = 50
ode_config.sor_pgs_w = 1.3
ode_config.sor_pgs_rms_error_tol = 0.0
ode_config.contact_surface_layer = 0.001
ode_config.contact_max_correcting_vel = 50.0 # default: 100.0
ode_config.cfm = 0.0
ode_config.erp = 0.2
ode_config.max_contacts = 20
print "Changing gazebo properties."
set_gravity(time_step.data, max_update_rate.data, gravity, ode_config)
