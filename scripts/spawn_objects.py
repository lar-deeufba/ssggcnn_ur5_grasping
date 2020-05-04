#!/usr/bin/env python
import rospy
import tf
import rospkg
from gazebo_msgs.srv import SpawnModel, GetModelState, GetLinkState
import time
from geometry_msgs.msg import *
from gazebo_msgs.msg import ModelState, ModelStates
import os
from os.path import expanduser
from pathlib import Path
from tf import TransformListener
from tf.transformations import quaternion_from_euler

class Moving():
	def __init__(self, model_name, Spawning1, y_pose, x_pose, z_pose, oriFinal, path):
		self.pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=1)
		self.model_name = model_name
		self.rate = rospy.Rate(10)
		self.path = path
		self.x_model_pose = x_pose
		self.y_model_pose = y_pose
		self.z_model_pose = z_pose
		self.Spawning1 = Spawning1
		self.orientation = oriFinal

	def spawning(self,):
		with open(self.path) as f:
			product_xml = f.read()
		item_name = "product_{0}_0".format(0)
		print("Spawning model:%s", self.model_name)
		# X and Y positions are somewhat in an incorrect order in Gazebo
		item_pose = Pose(Point(x=self.y_model_pose, y=self.x_model_pose,z=self.z_model_pose),
						 Quaternion(self.orientation[0], self.orientation[1], self.orientation[2], self.orientation[3]))
		self.Spawning1(self.model_name, product_xml, "", item_pose, "world")

def uncluttered_objects():
	rospack = rospkg.RosPack()
	rospy.init_node('spawn_model')
	Spawning1 = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)
	rospy.wait_for_service("gazebo/spawn_sdf_model")
	model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
	
	Home = rospack.get_path('ssggcnn_ur5_grasping')

	bico_dosador = Home + '/models/bico_dosador/model.sdf'
	little_bin_box = Home + '/models/little_bin_box/model.sdf'
	part1 = Home + '/models/part1/model.sdf'
	part2 = Home + '/models/part2/model.sdf'
	unob_1 = Home + '/models/unob_1/model.sdf'
	unob_2 = Home + '/models/unob_2/model.sdf'
	unob_3 = Home + '/models/unob_3/model.sdf'

	# The spawn need some time to wait the UR5 to show in Gazebo
	object_coordinates = model_coordinates("robot", "")
	z_position = object_coordinates.pose.position.z
	y_position = object_coordinates.pose.position.y
	x_position = object_coordinates.pose.position.x

	ptFinal = [-0.09, -0.47, -0.004] # all together in the bin
	oriFinal = quaternion_from_euler(0.0, 0.0, 0.0)
	moving4 = Moving("part1", Spawning1, x_position + ptFinal[0], y_position + ptFinal[1], z_position + ptFinal[2], oriFinal, part1)
	moving4.spawning()    

	ptFinal = [-0.043, -0.46, -0.002] # all together in the bin
	oriFinal = quaternion_from_euler(0.0, 0.0, -1.57)
	moving4 = Moving("part2", Spawning1, x_position + ptFinal[0], y_position + ptFinal[1], z_position + ptFinal[2], oriFinal, part2)
	moving4.spawning()

	ptFinal = [-0.05, -0.5, -0.005] # all together in the bin
	oriFinal = quaternion_from_euler(0.0, 0.0, 0.0)
	moving4 = Moving("bico_dosador", Spawning1, x_position + ptFinal[0], y_position + ptFinal[1], z_position + ptFinal[2], oriFinal, bico_dosador)
	moving4.spawning()

	ptFinal = [0.16, -0.45, 0.0] # all together in the bin
	oriFinal = quaternion_from_euler(0.0, 0.0, 0.0)
	moving4 = Moving("little_bin_box", Spawning1, x_position + ptFinal[0], y_position + ptFinal[1], z_position + ptFinal[2], oriFinal, little_bin_box)
	moving4.spawning()

	ptFinal = [-0.058, -0.41, -0.001] # all together in the bin
	oriFinal = quaternion_from_euler(0.0, 0.0, 0.0)
	moving4 = Moving("unob_1", Spawning1, x_position + ptFinal[0], y_position + ptFinal[1], z_position + ptFinal[2], oriFinal, unob_1)
	moving4.spawning()

	ptFinal = [-0.13, -0.41, -0.002] # all together in the bin
	oriFinal = quaternion_from_euler(0.0, 0.0, 0.0)
	moving4 = Moving("unob_2", Spawning1, x_position + ptFinal[0], y_position + ptFinal[1], z_position + ptFinal[2], oriFinal, unob_2)
	moving4.spawning()

	ptFinal = [-0.141, -0.45, -0.002] # all together in the bin
	oriFinal = quaternion_from_euler(0.0, 0.0, 0.0)
	moving4 = Moving("unob_3", Spawning1, x_position + ptFinal[0], y_position + ptFinal[1], z_position + ptFinal[2], oriFinal, unob_3)
	moving4.spawning()

if __name__ == '__main__':
	uncluttered_objects()
