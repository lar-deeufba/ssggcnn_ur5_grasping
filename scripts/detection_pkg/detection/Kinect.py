#!/usr/bin/env python
import numpy as np
import cv2
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from cv_bridge import CvBridge, CvBridgeError

class Kinect:
    def __init__(self):
        self.bridge = CvBridge()
        self.image=Image()
        self.image_depth=Image()
        self.cloud=PointCloud2()
        self.has_image=0
        self.has_depth=0
        self.has_cloud=0
        
    def set_image(self,data):
        try:
            self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.has_image=1
        except CvBridgeError as e:
            print(e)
            
    def set_image_depth(self,data):
        try:
            self.image = self.bridge.imgmsg_to_cv2(data, "mono16")
            self.has_depth=1
        except CvBridgeError as e:
            print(e)

    def set_point(self,data):
        self.cloud.header=data.header
    	self.cloud.height=data.height
    	self.cloud.width=data.width
    	self.cloud.is_bigendian=data.is_bigendian
    	self.cloud.point_step=data.point_step
    	self.cloud.row_step=data.row_step
    	self.cloud.is_dense=data.is_dense
        self.cloud.data= data.data;
        self.cloud.fields = data.fields	
        self.has_cloud=0
    
	
    def show_image(self):
        if self.has_image == 1:
            cv2.imshow('image',self.image)
            k=cv2.waitKey(1)

    def show_image_depth(self):
        if self.has_depth == 1:
            cv2.imshow('image3D',self.image_depth)
            k=cv2.waitKey(1)
    

