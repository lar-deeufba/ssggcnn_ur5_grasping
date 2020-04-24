#!/usr/bin/env python
# Code available in
# https://github.com/neobotix/neo_simulation
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

    # Not used yet
    def moving_goal(self):
        obstacle = ModelState()
        ptFinal, oriFinal = tf.lookupTransform("base_link", "ar_marker_0", rospy.Time(0))
        obstacle.model_name = "custom_box"
        obstacle.pose = model.pose[i]
        obstacle.twist = Twist()
        obstacle.twist.linear.y = 1.3
        obstacle.twist.angular.z = 0
        self.pub_model.publish(obstacle)

def main():
    rospack = rospkg.RosPack()
    rospy.init_node('spawn_model')
    Spawning1 = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)
    rospy.wait_for_service("gazebo/spawn_sdf_model")
    model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    
    Home = rospack.get_path('real_time_grasp')

    path_table = Home + '/models/table/model.sdf'
    path_box = Home + '/models/box/model.sdf'
    bico_dosador = Home + '/models/bico_dosador/model.sdf'
    little_bin_box = Home + '/models/little_bin_box/model.sdf'
    part1 = Home + '/models/part1/model.sdf'
    part2 = Home + '/models/part2/model.sdf'
    part3 = Home + '/models/part3/model.sdf'
    moldura_final = Home + '/models/moldura_final/model.sdf'
    marker_bot = Home + '/models/markerbot/model.sdf'

    # The spawn need some time to wait the UR5 to show in Gazebo
    rospy.sleep(1.0)
    object_coordinates = model_coordinates("robot", "")
    z_position = object_coordinates.pose.position.z
    y_position = object_coordinates.pose.position.y
    x_position = object_coordinates.pose.position.x

    print "X, Y, Z: ", x_position, y_position, z_position

    # This is the position of the object spawned in gazebo relative to the base_link
    ptFinal = [0.0, -0.4, 0.00]
    oriFinal = quaternion_from_euler(0.0, 0.0, 1.57)

    rospy.sleep(0.1)
    
    moving1 = Moving("table", Spawning1, ptFinal[0], ptFinal[1], ptFinal[2], oriFinal, path_table)
    moving1.spawning()

    # rospy.sleep(0.1)

    # ptFinal = [0.0, -0.45, -0.005]
    # oriFinal = quaternion_from_euler(0.0, 0.0, 0.0)
    # moving4 = Moving("little_bin_box", Spawning1, x_position + ptFinal[0], y_position + ptFinal[1], z_position + ptFinal[2], oriFinal, little_bin_box)
    # moving4.spawning()

    rospy.sleep(0.3)

    # ptFinal = [0.0, -0.95, 0.16] # on the markerbot
    # ptFinal = [0.0, -0.45, 0.1] # on the table
    ptFinal = [-0.07, -0.45, 0.1] # all together in the bin
    oriFinal = quaternion_from_euler(0.0, 0.0, 0.0)
    moving4 = Moving("part1", Spawning1, x_position + ptFinal[0], y_position + ptFinal[1], z_position + ptFinal[2], oriFinal, part1)
    moving4.spawning()    

    rospy.sleep(0.3)

    # ptFinal = [-0.01, -0.6, 0.1]
    ptFinal = [-0.14, -0.45, 0.1] # all together in the bin
    oriFinal = quaternion_from_euler(0.0, 0.0, 0.0)
    moving4 = Moving("bico_dosador", Spawning1, x_position + ptFinal[0], y_position + ptFinal[1], z_position + ptFinal[2], oriFinal, bico_dosador)
    moving4.spawning()

    rospy.sleep(0.1)

    # ptFinal = [-0.01, -0.45, 0.1]
    ptFinal = [-0.07, -0.50, 0.1] # all together in the bin
    oriFinal = quaternion_from_euler(0.0, 0.0, -1.57)
    moving4 = Moving("part2", Spawning1, x_position + ptFinal[0], y_position + ptFinal[1], z_position + ptFinal[2], oriFinal, part2)
    moving4.spawning()

    rospy.sleep(0.1)

    # ptFinal = [0.01, -0.65, 0.1]
    ptFinal = [-0.14, -0.50, 0.1] # all together in the bin
    oriFinal = quaternion_from_euler(0.0, 0.0, 0.0)
    moving4 = Moving("part3", Spawning1, x_position + ptFinal[0], y_position + ptFinal[1], z_position + ptFinal[2], oriFinal, part3)
    moving4.spawning()
    
    # rospy.sleep(0.1)

    # ptFinal = [0.0, -0.55, 0.0]
    # oriFinal = quaternion_from_euler(0.0, 0.0, 0.0)
    # moving4 = Moving("moldura_final", Spawning1, x_position + ptFinal[0], y_position + ptFinal[1], z_position + ptFinal[2], oriFinal, moldura_final)
    # moving4.spawning()

if __name__ == '__main__':
    main()
