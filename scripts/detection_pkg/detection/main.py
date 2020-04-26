#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import os
from os.path import join as pjoin
import numpy as np
import cv2

import mxnet as mx
from mxnet import nd
from gluoncv import model_zoo
from gluoncv.data.transforms import presets

import config as cfg
import transforms
from bboxes import BboxList
import rospy
from geometry_msgs.msg import Point

from Kinect import Kinect
from std_msgs.msg import String, Int32MultiArray
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from cv_bridge import CvBridge, CvBridgeError
import rospkg

def filter_predictions(ids, scores, bboxes, threshold=0.0):
	"""filter and resize predictions"""
	idx = scores.squeeze().asnumpy() > threshold
	fscores = scores.squeeze().asnumpy()[idx]
	fids = ids.squeeze().asnumpy()[idx]
	fbboxes = bboxes.squeeze().asnumpy()[idx]
	return fids, fscores, fbboxes


class Detector:
	def __init__(self, model_path, model='ssd512', ctx='cpu', classes='normal'):

		
		#dataset_root = pjoin(cfg.dataset_folder, dataset)
		# with open(pjoin(dataset_root, 'classes.txt'), 'r') as f:
		#     classes = [line.strip() for line in f.readlines()]
		#     classes = [line for line in classes if line]
		
		if classes == 'normal':
			self.classes = cfg.CLASSES
		elif classes == 'grasp':
			self.classes = cfg.CLASSES_GRASP
		else:
			raise ValueError('Wrong classes, valid args: normal, grasp.')
		if ctx == 'cpu':
			ctx = mx.cpu()
		elif ctx == 'gpu':
			ctx = mx.gpu()
		else:
			raise ValueError('Invalid context.')
		self.ctx = ctx
		self.short, self.width, self.height = None, None, None
		if model.lower() == 'ssd512':
			model_name = 'ssd_512_resnet50_v1_coco'
			self.width, self.height = 512, 512
			self.transform = transforms.SSDDefaultTransform(self.width, self.height)
		elif model.lower() == 'ssd300':
			model_name = 'ssd_300_vgg16_atrous_coco'
			self.width, self.height = 300, 300
			self.transform = transforms.SSDDefaultTransform(self.width, self.height)
		elif (model.lower() == 'yolo416') or (model.lower() == 'yolo416'):
			model_name = 'yolo3_darknet53_coco'
			self.width, self.height = 416, 416
		elif (model.lower() == 'yolo608') or (model.lower() == 'yolo416'):
			model_name = 'yolo3_darknet53_coco'
			self.width, self.height = 608, 608
			self.transform = transforms.SSDDefaultTransform(self.width, self.height)
		elif (model.lower() == 'frcnn') or (model.lower() == 'faster_rcnn'):
			model_name = 'faster_rcnn_resnet50_v1b_coco'
			self.short = 600
			self.transform = transforms.FasterRCNNDefaultTransform(short=600)
		elif model.lower() == 'ssd512_mobile':
			model_name = 'ssd_512_mobilenet1.0_coco'
			self.width, self.height = 512, 512
			self.transform = transforms.SSDDefaultTransform(self.width, self.height)
		else:
			raise ValueError('Invalid model `{}`.'.format(model.lower()))

		net = model_zoo.get_model(model_name, pretrained=False, ctx=ctx)
		net.initialize(force_reinit=True, ctx=ctx)
		net.reset_class(classes=self.classes)
		net.load_parameters(model_path, ctx=ctx)
		self.net = net


	@classmethod
	def list_datasets(cls):
		return cls.model_data.keys()


	@classmethod
	def list_models(cls, dataset):
		try:
			models = cls.model_data[dataset]
		except KeyError:
			raise ValueError('Dataset {} does not exist, avaliable datasets {}'.format(dataset, cls.model_data.keys()))
		return models


	def detect(self, img, threshold=0.5, mantain_scale=True):
		""" 
		Detects Bounding Boxes in a image.
		Inputs
		------
		img: input image as a numpy array
		threshold: detection threshold
		mantain_sacale: if true return bounding boxes in the original image coordinates
		
		Outputs
		-------
		bbox_list: a bounding box list object containing all filtered predictions
		timg: transformed image
		"""
		# TODO: improve this check to work in all cases
		if np.max(img) < 1.1:
			img = img * 255
		
		in_height, in_width = img.shape[:2]

		timg = self.transform(mx.nd.array(img))
		t_height, t_width = timg.shape[1:]

		width_ratio = in_width / t_width
		height_ratio = in_height / t_height

		timg = self.transform(mx.nd.array(img))
		ids, scores, bboxes = self.net(timg.expand_dims(axis=0).as_in_context(self.ctx))
		fids, fscores, fbboxes = filter_predictions(ids, scores, bboxes, 
			threshold=threshold)
		if mantain_scale:
			rep = np.repeat(
				np.array([[width_ratio, height_ratio, width_ratio, height_ratio]]),
				fbboxes.shape[0], axis=0)
			rscaled_bboxes = fbboxes * rep
			out_img = img
		else:
			rscaled_bboxes = fbboxes
			out_img = timg
	
		box_list = BboxList.from_arrays(fids, fscores, rscaled_bboxes, self.classes, th=threshold)

		return box_list, timg


rospy.init_node('talker', anonymous=True)
cam=Kinect()
rospack=rospkg.RosPack()
path=rospack.get_path("real_time_grasp")
det=Detector(path + "/scripts/detection_pkg/model.params")
pub1 = rospy.Publisher('point1', Point, queue_size=10)
pub2 = rospy.Publisher('point2', Point, queue_size=10)
arraypub = rospy.Publisher('sdd_points_array', Int32MultiArray, queue_size=10)
rospy.Subscriber("/camera/color/image_raw", Image, cam.set_image)
rate = rospy.Rate(10) # 10hz
points_to_send = Int32MultiArray()
while not rospy.is_shutdown():
	#im=cv2.imread("cam.png")
	#im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
	
	if cam.has_image > 0:
		im=cam.image
		[caixas,timag]= det.detect(im)
		size = len(caixas)
		if size != 0:
			i = 0
			points_to_send_list = []
			print("size: ", size)
			while i < size:
				print(caixas[i].class_name)# == "test"
				print(caixas[i])
				caixas[i].class_name = ""
				img=caixas.draw(im)
				cv2.circle(img,(int(caixas[i].x1),int(caixas[i].y1)), 2, (0,0,255), -1)
				cv2.circle(img,(int(caixas[i].x2),int(caixas[i].y2)), 2, (0,0,255), -1)
				cv2.imshow('image',img)
				cv2.waitKey(1)
				#print(caixas)
				point1= Point()
				point2= Point()
				point1.x=int(caixas[i].x1)
				points_to_send_list.append(point1.x)
				point1.y=int(caixas[i].y1)
				points_to_send_list.append(point1.y)
				point2.x=int(caixas[i].x2)
				points_to_send_list.append(point2.x)
				point2.y=int(caixas[i].y2)
				points_to_send_list.append(point2.y)
				#print(point1)
				#print(point2)
				pub1.publish(point1)
				pub2.publish(point2)
				# i=size
				i+=1
			points_to_send.data = points_to_send_list # assign the array with the value you want to send
			print(points_to_send.data)
			arraypub.publish(points_to_send)
			points_to_send.data = []
		else:
			print("No obj found")
	
	rate.sleep()





