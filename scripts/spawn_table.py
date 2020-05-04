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

def spawn_table():
	rospack = rospkg.RosPack()
	rospy.init_node('spawn_model')
	Spawning1 = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)
	rospy.wait_for_service("gazebo/spawn_sdf_model")
	model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
	
	Home = rospack.get_path('ssggcnn_ur5_grasping')

	path_table = Home + '/models/table/model.sdf'

	# The spawn need some time to wait the UR5 to show in Gazebo
	rospy.sleep(1.0)
	object_coordinates = model_coordinates("robot", "")
	z_position = object_coordinates.pose.position.z
	y_position = object_coordinates.pose.position.y
	x_position = object_coordinates.pose.position.x


	# This is the position of the object spawned in gazebo relative to the base_link
	ptFinal = [0.0, -0.4, 0.00]
	oriFinal = quaternion_from_euler(0.0, 0.0, 1.57)

	moving1 = Moving("table", Spawning1, ptFinal[0], ptFinal[1], ptFinal[2], oriFinal, path_table)
	moving1.spawning()

if __name__ == '__main__':
	spawn_table()