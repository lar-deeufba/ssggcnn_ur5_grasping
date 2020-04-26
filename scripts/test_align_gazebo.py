#!/usr/bin/python

import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import cv2
from cv_bridge import CvBridge
from cv_bridge import CvBridge, CvBridgeError

class test_aligment(object):
	def __init__(self):
		
		self.depth_image = None
		self.color_image = None
		self.bridge = CvBridge()

		rospy.Subscriber("/camera/depth/image_raw", Image, self.get_depth_callback, queue_size=10)
		rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=10)

		# publica o aligned
		self.cmd_pub = rospy.Publisher('ggcnn/out/command', Float32MultiArray, queue_size=1)

	def get_depth_callback(self, msg):
		self.depth_image = msg
		
	def image_callback(self, msg):
		self.color_image = msg

	def show_both_images(self):
		# Depth Image format 16UC1
		# C1 means one channel image
		# 16U means 16 (unsigned) bits per channel
		depth_image = self.depth_image
		depth_image.encoding = "mono16"
		depth_image = self.bridge.imgmsg_to_cv2(depth_image) # 16UC1
		depth_image = cv2.normalize(depth_image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
		
		color_image = self.color_image
		color_image.encoding = 'bgr8'
		color_image = self.bridge.imgmsg_to_cv2(color_image)
		color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
		print("Color: ", color_image.dtype)
		print(color_image.max())
		
		added_image = cv2.addWeighted(depth_image,0.8,color_image,0.2,0)

		cv2.imshow('image',added_image)
		k=cv2.waitKey(10) # refresh each 10ms

	def concatenated_images(self):
		numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)


def main():
	rospy.init_node("test_aligment")
	test = test_aligment()
	rospy.sleep(1.0)

	while not rospy.is_shutdown():
		test.show_both_images()
	
if __name__ == '__main__':
	try:
		main()
				
	except rospy.ROSInterruptException:
		print "Finished before completion"
