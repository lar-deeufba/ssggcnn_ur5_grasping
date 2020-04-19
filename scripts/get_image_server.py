#!/usr/bin/env python

from real_time_grasp.srv import GetImage, GetImageResponse
import rospy
from sensor_msgs.msg import Image

global IMAGE

def depth_callback(msg):
	global IMAGE
	IMAGE = msg

def handle_get_image(req):
	global IMAGE
	rospy.Subscriber("camera/depth/image_raw", Image, depth_callback, queue_size=10)
    return GetImageResponse(IMAGE)

def get_image_server():
    rospy.init_node('get_image_server')
    s = rospy.Service('get_depth_image', Image, handle_get_image)
    rospy.spin()

if __name__ == "__main__":
    get_image_server()
