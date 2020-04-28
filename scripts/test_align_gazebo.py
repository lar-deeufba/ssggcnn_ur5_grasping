#!/usr/bin/python

import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Int32MultiArray
import cv2
from cv_bridge import CvBridge
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class test_aligment(object):
	def __init__(self):
		
		self.depth_image = None
		self.color_image = None
		self.ssd_bb_color_image = None
		self.ggcnn_ssd_depth_image = None
		self.points_vec = []
		self.bridge = CvBridge()
		self.center_calibrated_point = np.array([312, 240]) # x, y
		self.actual_depth_image = None

		rospy.Subscriber("/camera/depth/image_raw", Image, self.get_depth_callback, queue_size=10)
		rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=10)
		rospy.Subscriber("/ssd/img/bouding_box", Image, self.ssd_bb_color_callback, queue_size=10)
		rospy.Subscriber("/ggcnn/img/grasp_depth_with_square", Image, self.grasp_depth_with_square_callback, queue_size=10)
		rospy.Subscriber("/ggcnn/img/depth_shot_with_copied_img", Image, self.depth_shot_with_copied_image_callback, queue_size=10)
		# rospy.Subscriber("/ggcnn/img/depth_ssd_square", Image, self.ggcnn_ssd_callback, queue_size=10)
		rospy.Subscriber('sdd_points_array', Int32MultiArray, self.bounding_boxes_callback, queue_size=10)

	def get_depth_callback(self, msg):
		self.depth_image = self.transform_depth_img(msg)

	def depth_shot_with_copied_image_callback(self, msg):
		self.depth_shot_with_copied_image = self.transform_depth_img(msg)
		
	def image_callback(self, msg):
		self.color_image = self.transform_color_img(msg, 'bgr8')

	def grasp_depth_with_square_callback(self, msg):
		self.grasp_depth_with_square_image = self.transform_depth_img(msg)

	def ssd_bb_color_callback(self, msg):
		if msg is not None:
			self.ssd_bb_color_image = self.transform_color_img(msg)

	def ggcnn_ssd_callback(self, msg):
		if msg is not None:
			self.ggcnn_ssd_depth_image = self.transform_depth_img(msg)

	def transform_color_img(self, img, encoding='rgb8'):
		color_image = img
		color_image.encoding = 'rgb8'
		color_image = self.bridge.imgmsg_to_cv2(color_image, encoding)
		# color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
		return color_image

	def transform_depth_img(self, img):
		depth_image = img
		depth_image.encoding = "mono16"
		depth_image = self.bridge.imgmsg_to_cv2(depth_image) # 16UC1
		depth_image = cv2.normalize(depth_image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
		depth_image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2RGB)
		return depth_image

	def resize(self, img, scale_percent):
		width = int(img.shape[1] * scale_percent / 100)
		height = int(img.shape[0] * scale_percent / 100)
		dsize = (width, height)
		return cv2.resize(img, dsize)
			
	def bounding_boxes_callback(self, msg):
		center_calibrated_point = self.center_calibrated_point
		print("msg: ", msg)
		box_number = len(msg.data) / 4
		if box_number != 0:
			depth = self.depth_image
			actual_depth_image = depth.copy()
			
			box_points = list(msg.data)
			i, index_inf, index_sup = 0, 0, 4
			points_vec = []
			K = 0.2
			while i < box_number:
				points_from_box = box_points[index_inf: index_sup]
				
				center = ((points_from_box[0] + points_from_box[2])/2, (points_from_box[1] + points_from_box[3])/2)

				# dist = np.linalg.norm(center - center_calibrated_point)
				dist = [int(center[0] - center_calibrated_point[0]), int(center[1] - center_calibrated_point[1])]

				offset = 10

				final_distance = [int(dist[0]*K), int(dist[1]*K)]

				start_point = (points_from_box[0], points_from_box[1])
				end_point = (points_from_box[2], points_from_box[3])
				actual_depth_image = cv2.rectangle(actual_depth_image, start_point, end_point, (200, 0, 0), 1)

				start_point_2 = (points_from_box[0] + final_distance[0] - offset, points_from_box[1] + final_distance[1] - offset)
				end_point_2 = (points_from_box[2] + final_distance[0] + offset, points_from_box[3] + final_distance[1] + offset)
				actual_depth_image = cv2.rectangle(actual_depth_image, start_point_2, end_point_2, (200, 0, 0), 3)

				new_points = [start_point_2[0], start_point_2[1], end_point_2[0], end_point_2[1]]
				print("New points: ", new_points)

				# actual_depth_image = cv2.circle(actual_depth_image, tuple(center_calibrated_point), 3, (200, 0, 0), -1) 
				# actual_depth_image = cv2.circle(actual_depth_image, center, 3, (200, 0, 0), -1) 
				index_inf += 4
				index_sup += 4
				i += 1

				points_vec.append(new_points)

			self.points_vec = points_vec
			self.actual_depth_image = actual_depth_image

	def show_depth_image_test(self, string):
		if string == 'depth_with_square':
			image = self.actual_depth_image
		elif string == 'ssd':
			image = self.ssd_bb_color_image
		elif string == 'grasp_depth_with_square':
			image = self.grasp_depth_with_square_image

		if image is not None:
			# self.depth_pub_ssd_square.publish(self.bridge.cv2_to_imgmsg(image))
			cv2.imshow(string,image)
			k=cv2.waitKey(10) # refresh each 10ms

	def concatenated_images(self):
		numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)

	def show_all_images(self):				
		ssd_image = self.ssd_bb_color_image
		depth_raw_image = self.depth_image
		depth_filtered_image = self.grasp_depth_with_square_image

		if ssd_image is not None:
			percentage_of_reduction = 35
			ssd_image = self.resize(ssd_image, percentage_of_reduction)
			depth_raw_image = self.resize(depth_raw_image, percentage_of_reduction)
			depth_filtered_image = self.resize(depth_filtered_image, percentage_of_reduction)
			numpy_horizontal_concat = np.concatenate((ssd_image, depth_raw_image, depth_filtered_image), axis=1)
			cv2.imshow('image', numpy_horizontal_concat)
			k=cv2.waitKey(10) # refresh each 10ms

	def show_color_and_depth(self):
		grasp_with_square = self.grasp_depth_with_square_image
		color_image = self.color_image
		percentage_of_reduction = 35
		grasp_with_square = self.resize(grasp_with_square, percentage_of_reduction)
		color_image = self.resize(color_image, percentage_of_reduction)
		numpy_horizontal_concat = np.concatenate((color_image, grasp_with_square), axis=1)
		cv2.imshow('image', numpy_horizontal_concat)
		k=cv2.waitKey(10) # refresh each 10ms

def main():
	rospy.init_node("test_aligment")
	test = test_aligment()
	rospy.sleep(1.0)
	# rospy.spin()
	while not rospy.is_shutdown():
		# test.show_depth_image_test('depth_with_square')
		# test.show_depth_image_test('ssd')
		# test.show_depth_image_test('grasp_depth_with_square')
		test.show_all_images()
		# test.show_color_and_depth()
	
if __name__ == '__main__':
	try:
		main()
				
	except rospy.ROSInterruptException:
		print "Finished before completion"
