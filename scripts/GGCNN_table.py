#!/usr/bin/python

import rospy
import actionlib
import numpy as np
import argparse
import copy
from copy import deepcopy
import rosservice
import sys
import re

from std_msgs.msg import Float64MultiArray, Float32MultiArray
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import WrenchStamped, Pose, Point, Quaternion
from controller_manager_msgs.srv import SwitchController

# Gazebo
from gazebo_msgs.msg import ModelState, ModelStates, ContactsState, ContactState, LinkState
from gazebo_msgs.srv import GetModelState, GetLinkState

from tf import TransformListener, TransformerROS, TransformBroadcaster
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# Inverse kinematics
from trac_ik_python.trac_ik import IK

# Robotiq
# import roslib; roslib.load_manifest('robotiq_2f_gripper_control')
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output  as outputMsg

import cv2
import matplotlib.pyplot as plt 
from cv_bridge import CvBridge
# -----------

CLOSE_GRIPPER_VEL = 0.05
MAX_GRIPPER_CLOSE_INIT = 0.25 # Maximum angle that the gripper should be started using velocity command
GRIPPER_INIT = True # Tells the node that the gripper was started
PICKING = False # Tells the node that the object must follow the gripper

def parse_args():
	parser = argparse.ArgumentParser(description='AAPF_Orientation')
	parser.add_argument('--gazebo', action='store_true', help='Set the parameters related to the simulated enviroonment in Gazebo')
	args = parser.parse_args()
	return args

class vel_control(object):
	def __init__(self, args, joint_values = None):
		rospy.init_node('command_GGCNN_ur5')

		self.args = args
		self.joint_values_home = joint_values

		# Topic used to publish vel commands
		self.pub_vel = rospy.Publisher('/joint_group_vel_controller/command', Float64MultiArray,  queue_size=1)

		self.joint_vels_gripper = Float64MultiArray()
		self.pub_vel_gripper = rospy.Publisher('/gripper_controller_vel/command', Float64MultiArray,  queue_size=1)

		# Used to perform TF transformations
		self.tf = TransformListener()
		self.br = TransformBroadcaster()

		# Used to change the controller
		self.controller_switch = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)

		# actionClient used to send joint positions
		self.client = actionlib.SimpleActionClient('pos_based_pos_traj_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
		print "Waiting for server (pos_based_pos_traj_controller)..."
		self.client.wait_for_server()
		print "Connected to server (pos_based_pos_traj_controller)"
		
		# Used by the Quintic Traj Planner
		self.initial_traj_duration = 5.0
		self.final_traj_duration = 500.0

		# Gazebo topics
		if self.args.gazebo:
			# For picking
			self.pub_model_position = rospy.Publisher('/gazebo/set_link_state', LinkState, queue_size=10)
			self.model_coordinates = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
			rospy.Subscriber('gazebo/model_states', ModelStates, self.get_model_state_callback, queue_size=10)
			
			# Subscriber used to read joint values
			rospy.Subscriber('/joint_states', JointState, self.ur5_actual_position_callback, queue_size=2)
			rospy.sleep(1.0)
			
			# USED FOR COLLISION DETECTION
			self.finger_links = ['robotiq_85_right_finger_tip_link', 'robotiq_85_left_finger_tip_link']
			# LEFT GRIPPER
			self.string = ""
			rospy.Subscriber('/left_finger_bumper_vals', ContactsState, self.monitor_contacts_left_finger_callback) # ContactState
			self.left_collision = False
			self.contactState_left = ContactState()
			# RIGHT GRIPPER
			rospy.Subscriber('/right_finger_bumper_vals', ContactsState, self.monitor_contacts_right_finger_callback) # ContactState
			self.right_collision = False
			self.contactState_right = ContactState()

			self.client_gripper = actionlib.SimpleActionClient('gripper_controller_pos/follow_joint_trajectory', FollowJointTrajectoryAction)
			print "Waiting for server (gripper_controller_pos)..."
			self.client_gripper.wait_for_server()
			print "Connected to server (gripper_controller_pos)"
			
		# GGCNN
		self.joint_values_ggcnn = []
		self.posCB = []
		self.ori = []
		self.grasp_cartesian_pose = []
		self.gripper_angle_grasp = 0.0
		self.final_orientation = 0.0
		if self.args.gazebo:
			self.offset_x = 0.0
			self.offset_y = 0.0
			self.offset_z = 0.019
		else:
			self.offset_x = -0.03 # 0.002
			self.offset_y = 0.02 # -0.05
			self.offset_z = 0.058 # 0.013

		rospy.Subscriber('ggcnn/out/command', Float32MultiArray, self.ggcnn_command_callback, queue_size=1)
				
		# Standard attributes used to send joint position commands
		self.joint_vels = Float64MultiArray()
		self.goal = FollowJointTrajectoryGoal()
		self.goal.trajectory = JointTrajectory()
		self.goal.trajectory.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint',
											'elbow_joint', 'wrist_1_joint', 'wrist_2_joint',
											'wrist_3_joint']
		self.initial_time = 4

		# Robotiq control
		self.pub_gripper_command = rospy.Publisher('Robotiq2FGripperRobotOutput', outputMsg.Robotiq2FGripper_robot_output, queue_size=1)
		self.d = None # msg received from GGCN
		self.gripper_max_width = 0.14

		# Denavit-Hartenberg parameters of UR5
		# The order of the parameters is d1, SO, EO, a2, a3, d4, d45, d5, d6
		self.ur5_param = (0.089159, 0.13585, -0.1197, 0.425, 0.39225, 0.10915, 0.093, 0.09465, 0.0823 + 0.15)

	def turn_velocity_controller_on(self):
		self.controller_switch(['joint_group_vel_controller'], ['pos_based_pos_traj_controller'], 1)

	def turn_position_controller_on(self):
		self.controller_switch(['pos_based_pos_traj_controller'], ['joint_group_vel_controller'], 1)

	def turn_gripper_velocity_controller_on(self):
		self.controller_switch(['gripper_controller_vel'], ['gripper_controller_pos'], 1)

	def turn_gripper_position_controller_on(self):
		self.controller_switch(['gripper_controller_pos'], ['gripper_controller_vel'], 1)

	def monitor_contacts_left_finger_callback(self, msg):
		if msg.states:
			# print(msg)
			self.left_collision = True
			string = msg.states[0].collision1_name
			string_collision = re.findall(r'::(.+?)::',string)[0]
			# print("Left String_collision: ", string_collision)
			if string_collision in self.finger_links:
				string = msg.states[0].collision2_name
				# print("Left Real string (object): ", string)
				self.string = re.findall(r'::(.+?)::', string)[0]
				# print("Left before: ", self.string)
			else:
				self.string = string_collision
				# print("Left in else: ", string_collision)
		else:
			self.left_collision = False

	def monitor_contacts_right_finger_callback(self, msg):
		if msg.states:
			self.right_collision = True
			string = msg.states[0].collision1_name
			string_collision = re.findall(r'::(.+?)::',string)[0]
			# print("Right String_collision: ", string_collision)
			if string_collision in self.finger_links:
				string = msg.states[0].collision2_name
				# print("Right Real string (object): ", string)
				self.string = re.findall(r'::(.+?)::',string)[0]
				# print("Right before: ", self.string)
			else:
				self.string = string_collision
				# print("Right in else: ", self.string)
		else:
			self.right_collision = False

	"""
	The joint states published by /joint_staes of the UR5 robot are in wrong order.
	/joint_states topic normally publishes the joint in the following order:
	[elbow_joint, shoulder_lift_joint, shoulder_pan_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint]
	But the correct order of the joints that must be sent to the robot is:
	['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
	"""
	def ur5_actual_position_callback(self, joint_values_from_ur5):
		if self.args.gazebo:
			self.th3, self.robotic, self.th2, self.th1, self.th4, self.th5, self.th6 = joint_values_from_ur5.position
			# print("Robotic angle: ", self.robotic)
		else:
			self.th3, self.th2, self.th1, self.th4, self.th5, self.th6 = joint_values_from_ur5.position
		
		self.actual_position = [self.th1, self.th2, self.th3, self.th4, self.th5, self.th6]

	def get_model_state_callback(self, msg):
		self.object_picking()

	"""
	GGCNN Command Subscriber Callback
	"""
	def ggcnn_command_callback(self, msg):
		# self.tf.waitForTransform("base_link", "object_detected", rospy.Time.now(), rospy.Duration(4.0))
		object_pose, object_ori = self.tf.lookupTransform("base_link", "object_detected", rospy.Time(0))
		self.d = list(msg.data)

		object_pose[0] += self.offset_x
		object_pose[1] += self.offset_y
		object_pose[2] += self.offset_z
		
		self.posCB = object_pose
		self.ori = self.d[3]

		self.br.sendTransform((object_pose[0], 
                               object_pose[1],
                               object_pose[2]), 
                               quaternion_from_euler(0.0, 0.0, self.ori),
                               rospy.Time.now(),
                               "object_link",
                               "base_link")
		
	def get_link_position_picking(self):
		link_name = self.string
		# print("Link name: ", link_name)
		model_coordinates = self.model_coordinates(self.string, 'wrist_3_link')
		self.model_pose_picking = model_coordinates.link_state.pose

	def reset_link_position_picking(self):
		self.string = ""

	def object_picking(self):
		global PICKING

		if PICKING:
			angle = quaternion_from_euler(1.57, 0.0, 0.0)
			object_picking = LinkState()
			object_picking.link_name = self.string
			object_picking.pose = Pose(self.model_pose_picking.position, self.model_pose_picking.orientation)
			object_picking.reference_frame = "wrist_3_link"
			self.pub_model_position.publish(object_picking)            

	def get_ik(self, pose):
		"""Get the inverse kinematics 
		
		Get the inverse kinematics of the UR5 robot using track_IK package giving a desired intial pose
		
		Arguments:
			pose {list} -- A pose representing x, y and z

		Returns:
			sol {list} -- Joint angles or None if track_ik is not able to find a valid solution
		"""
		# camera_support_angle_offset = 0.25
		camera_support_angle_offset = 0.0
		
		q = quaternion_from_euler(0.0, -3.14 + camera_support_angle_offset, 0.0)
		# Joint order:
		# ('shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link', 'tool0')            
		ik_solver = IK("base_link", "grasping_link", solve_type="Distance")
		sol = ik_solver.get_ik([0.2201039360819781, -1.573845095552878, -1.521853400505349, -1.6151347051274518, 1.5704492904506875, 0.0], 
				pose[0], pose[1], pose[2], q[0], q[1], q[2], q[3])
		if sol is not None:
			sol = list(sol)
			sol[-1] = 0.0
			
		return sol

	"""
	Quintic polynomial trajectory
	"""
	def traj_planner(self, cart_pos, grasp_step = 'move', way_points_number = 10):
		"""
		pp - Position points
		vp - Velocity points
		"""
		
		if grasp_step == 'pregrasp':
			print("POSCB: ", self.posCB)
			self.grasp_cartesian_pose = deepcopy(self.posCB)
			self.grasp_cartesian_pose[-1] += 0.1
			joint_pos = self.get_ik(self.grasp_cartesian_pose)
			joint_pos[-1] = self.ori
			self.final_orientation = deepcopy(self.ori)
			self.gripper_angle_grasp = deepcopy(self.d[-2])
		elif grasp_step == 'grasp':
			self.grasp_cartesian_pose[-1] -= 0.1
			joint_pos = self.get_ik(self.grasp_cartesian_pose)
			joint_pos[-1] = self.final_orientation
		elif grasp_step == 'move':
			joint_pos = self.get_ik(cart_pos)
			joint_pos[-1] = 0.0

		v0 = a0 = vf = af = 0
		t0 = self.initial_traj_duration
		tf = (t0 + self.final_traj_duration) / way_points_number # tf by way point
		t = tf / 10 # for each movement
		ta = tf / 10 # to complete each movement
		a = [0.0]*6
		pos_points, vel_points, acc_points = [0.0]*6, [0.0]*6, [0.0]*6
		
		goal = FollowJointTrajectoryGoal()
		goal.trajectory = JointTrajectory()
		goal.trajectory.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint',
									   'elbow_joint', 'wrist_1_joint', 'wrist_2_joint',
									   'wrist_3_joint']

		for i in range(6):
			q0 = self.actual_position[i]
			qf = joint_pos[i]

			b = np.array([q0,v0,a0,qf,vf,af]).transpose()
			m = np.array([[1, t0, t0**2,   t0**3,    t0**4,    t0**5],
						  [0,  1,  2*t0, 3*t0**2,  4*t0**3,  5*t0**4],
						  [0,  0,     2,    6*t0, 12*t0**2, 20*t0**3],
						  [1, tf, tf**2,   tf**3,    tf**4,    tf**5],
						  [0,  1,  2*tf, 3*tf**2,  4*tf**3,  5*tf**4],
						  [0,  0,     2,    6*tf, 12*tf**2, 20*tf**3]])
			a[i] = np.linalg.inv(m).dot(b)

		for i in range(way_points_number):
			for j in range(6):
				pos_points[j] =   a[j][0] +   a[j][1]*t +    a[j][2]*t**2 +    a[j][3]*t**3 +   a[j][4]*t**4 + a[j][5]*t**5
				vel_points[j] =   a[j][1] + 2*a[j][2]*t +  3*a[j][3]*t**2 +  4*a[j][4]*t**3 + 5*a[j][5]*t**4
				acc_points[j] = 2*a[j][2] + 6*a[j][3]*t + 12*a[j][4]*t**2 + 20*a[j][5]*t**3

			goal.trajectory.points.append(JointTrajectoryPoint(positions = pos_points,
															   velocities = vel_points,
															   accelerations = acc_points,
															   time_from_start = rospy.Duration(t))) #default 0.1*i + 5
			t += ta

		self.client.send_goal(goal)

	def genCommand(self, char, command, pos = None):
		"""Update the command according to the character entered by the user."""    

		if char == 'a':
			# command = outputMsg.Robotiq2FGripper_robot_output();
			command.rACT = 1 # Gripper activation
			command.rGTO = 1 # Go to position request
			command.rSP  = 255 # Speed
			command.rFR  = 150 # Force

		if char == 'r':
			command.rACT = 0

		if char == 'c':
			command.rACT = 1
			command.rGTO = 1
			command.rATR = 0
			command.rPR = 255
			command.rSP = 40
			command.rFR = 150
			
		# @param pos Gripper width in meters. [0, 0.087]
		if char == 'p':
			command.rACT = 1
			command.rGTO = 1
			command.rATR = 0
			command.rPR = int(np.clip((13.-230.)/self.gripper_max_width * self.ori + 230., 0, 255))
			command.rSP = 40
			command.rFR = 150

		if char == 'o':
			command.rACT = 1
			command.rGTO = 1
			command.rATR = 0
			command.rPR = 0
			command.rSP = 40
			command.rFR = 150

		return command

	def command_gripper(self, action):
		command = outputMsg.Robotiq2FGripper_robot_output();
		command = self.genCommand(action, command)
		self.pub_gripper_command.publish(command)  

	def gripper_send_position_goal(self, position = 0.3, action = 'move'):
		if action == 'pre_grasp_angle':
			max_distance = 0.085
			angular_coeff = 0.11
			K = 1.3
			angle = (max_distance - self.gripper_angle_grasp) / angular_coeff * K
			position = angle
			print("Angle: ", angle)

		self.turn_gripper_position_controller_on()
		goal = FollowJointTrajectoryGoal()
		goal.trajectory = JointTrajectory()
		goal.trajectory.joint_names = ['robotiq_85_left_knuckle_joint']
		goal.trajectory.points.append(JointTrajectoryPoint(positions = [position],
														   velocities = [0.4],
														   accelerations = [0.0],
														   time_from_start = rospy.Duration(0.2)))
		self.client_gripper.send_goal(goal)


	"""
	Control the gripper by using velocity controller
	"""
	def gripper_vel_control(self, action):
		global GRIPPER_INIT
		global CLOSE_GRIPPER_VEL

		self.turn_gripper_velocity_controller_on()		

		rate = rospy.Rate(100)

		# Translating this while into English: 
		# While rospy is not shutdown and gripper is still running and the angle of the gripper is not greater than 0.7
		# and colliisions of borth gripper occur (the collision must occur to both grippers tip in order to stop them.
		# That's why we use OR
		while not rospy.is_shutdown() and GRIPPER_INIT and not self.robotic > 0.78 and not self.left_collision and not self.right_collision:
			print(self.robotic)
			self.joint_vels_gripper.data = np.array([CLOSE_GRIPPER_VEL])
			self.pub_vel_gripper.publish(self.joint_vels_gripper)
			rate.sleep()		

		# stops the robot after the goal is reached
		rospy.loginfo("Gripper stopped!")
		self.joint_vels_gripper.data = np.array([0.0]) # stop gripper
		self.pub_vel_gripper.publish(self.joint_vels_gripper)

	"""
	Send the HOME position to the robot
	self.client.wait_for_result() won't work well in Gazebo.
	Instead, a while loop has been created to ensure that the robot reaches the
	goal even after the failure.
	"""
	def home_pos(self):
		global GRIPPER_INIT

		# self.turn_position_controller_on()
		# rospy.sleep(0.1)

		self.joint_vels_gripper.data = np.array([0.0])
		self.pub_vel_gripper.publish(self.joint_vels_gripper)

		# First point is current position
		try:
			self.goal.trajectory.points = [(JointTrajectoryPoint(positions=self.joint_values_home, velocities=[0]*6, time_from_start=rospy.Duration(self.initial_traj_duration)))]
			self.client.send_goal(self.goal)
			self.client.wait_for_result()
		except KeyboardInterrupt:
			self.client.cancel_goal()
			raise
		except:
			raise

		print "\n==== Goal reached!"

	
def main():
	global GRIPPER_INIT, PICKING

	arg = parse_args()

	# Turn position controller ON
	ur5_vel = vel_control(arg)

	# ur5_vel.turn_position_controller_on()
	point_init_home = [-0.37, 0.11, 0.15] # [-0.35, 0.03, 0.05] - behind box - #[-0.40, 0.0, 0.15] - up
	joint_values_home = ur5_vel.get_ik(point_init_home)
	ur5_vel.joint_values_home = joint_values_home

	# Send the robot to the custom HOME position
	raw_input("==== Press enter to 'home' the robot!")
	rospy.on_shutdown(ur5_vel.home_pos)
	ur5_vel.traj_planner(point_init_home)

	# Remove all objects from the scene and press enter
	raw_input("==== Press enter to move the robot to the 'depth cam shot' position!")
	point_init = [-0.37, 0.11, 0.05]
	ur5_vel.traj_planner(point_init)
		
	raw_input("==== Press enter to init to gripper!")
	PICKING = False
	if arg.gazebo:
		rospy.loginfo("Starting the gripper in Gazebo! Please wait...")
		# ur5_vel.gripper_vel_control('open')
		ur5_vel.gripper_send_position_goal(0.4)
		rospy.loginfo("Gripper started!")   
	else:
		rospy.loginfo("Starting the real gripper! Please wait...")
		ur5_vel.command_gripper('r')
		rospy.sleep(0.5)
		ur5_vel.command_gripper('a')
		ur5_vel.command_gripper('o')
	
	GRIPPER_INIT = True

	while not rospy.is_shutdown():

		raw_input("==== Press enter to move to the pre grasp position!")
		ur5_vel.traj_planner(point_init, 'pregrasp')

		# It closes the gripper before approaching the object
		# It prevents the gripper to collide with other objects when grasping
		raw_input("==== Press enter to close the gripper to a pre-grasp angle!")
		if arg.gazebo:
			# ur5_vel.gripper_vel_control('close_to_angle')
			ur5_vel.gripper_send_position_goal(action='pre_grasp_angle')
		else:
			ur5_vel.command_gripper('p')

		raw_input("==== Press enter to move the robot to the goal position given by GGCNN!")
		ur5_vel.traj_planner([], 'grasp')

		raw_input("==== Press enter to close the gripper!")
		if arg.gazebo:
			ur5_vel.gripper_vel_control('grasp_object')
			ur5_vel.get_link_position_picking()
		else:
			raw_input("==== Press enter to close the gripper!")
			ur5_vel.command_gripper('c')

		raw_input("==== Press enter to move the object upwards!")
		PICKING = True # Attach object
		ur5_vel.traj_planner([-0.4, 0.0, 0.15])

		raw_input("==== Press enter to move the object to the bin!")
		ur5_vel.traj_planner([-0.2, -0.3, 0.15])

		raw_input("==== Press enter to open the gripper!")
		PICKING = False # Detach object
		if arg.gazebo:
			ur5_vel.gripper_send_position_goal(0.3)
			ur5_vel.reset_link_position_picking()
		else:
			ur5_vel.command_gripper('o')

		raw_input("==== Press enter to move to the initial position!")
		ur5_vel.traj_planner(point_init)

if __name__ == '__main__':
	try:
		main()
	except rospy.ROSInterruptException:
		print "Program interrupted before completion"
