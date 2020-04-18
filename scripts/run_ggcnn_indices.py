#! /usr/bin/env python
# Copyright (c) 2018, Douglas Morrison, ARC Centre of Excellence for Robotic Vision (ACRV), Queensland University of Technology
# All rights reserved.

import time

import numpy as np
import rospkg
import tensorflow as tf
from keras.models import load_model
from tf import TransformBroadcaster, TransformListener

import cv2
import scipy.ndimage as ndimage
from skimage.draw import circle
from skimage.feature import peak_local_max

import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo, JointState
from std_msgs.msg import Float32MultiArray

from tf.transformations import quaternion_from_euler, euler_from_quaternion

from copy import deepcopy, copy
import argparse

class TimeIt:
    def __init__(self, s):
        self.s = s
        self.t0 = None
        self.t1 = None
        self.print_output = False

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, t, value, traceback):
        self.t1 = time.time()
        if self.print_output:
            print('%s: %s' % (self.s, self.t1 - self.t0))

def parse_args():
    parser = argparse.ArgumentParser(description='GGCN and SSD grasping')
    parser.add_argument('--real', action='store_true', help='Consider the real intel realsense')
    args = parser.parse_args()
    return args

class ssgg_grasping(object):
    def __init__(self, args):
        rospy.init_node('ggcnn_ssd_detection')

        self.args = args

        self.bridge = CvBridge()

        # Load the Network.
        rospack = rospkg.RosPack()
        Home = rospack.get_path('real-time-grasp')
        MODEL_FILE = Home + '/data/epoch_29_model.hdf5'
        with tf.device('/device:GPU:0'):
            self.model = load_model(MODEL_FILE)

        # TF pkg
        self.transf = TransformListener()
        self.br = TransformBroadcaster()

        # Load GGCN parameters
        self.crop_size = rospy.get_param("/GGCNN/crop_size")
        self.FOV = rospy.get_param("/GGCNN/FOV")
        self.camera_topic_info = rospy.get_param("/GGCNN/camera_topic_info")
        if self.args.real:
            self.camera_topic = rospy.get_param("/GGCNN/camera_topic_realsense")
        else:
            self.camera_topic = rospy.get_param("/GGCNN/camera_topic")

        # Output publishers.
        self.grasp_pub = rospy.Publisher('ggcnn/img/grasp', Image, queue_size=1)
        self.depth_pub = rospy.Publisher('ggcnn/img/depth', Image, queue_size=1)
        self.depth_with_square = rospy.Publisher('ggcnn/img/depth_with_square', Image, queue_size=1)
        self.ang_pub = rospy.Publisher('ggcnn/img/ang', Image, queue_size=1)
        self.cmd_pub = rospy.Publisher('ggcnn/out/command', Float32MultiArray, queue_size=1)

        # Initialise some var
        self.depth = None
        self.depth_crop = None
        self.depth_copy = None
        self.depth_message = None
        self.pred_out = None
        self.points_out = None
        self.height_res = []
        self.width_res = []
        self.depth_nan = None
        self.grasp_img = None
        self.cos_out = None
        self.sin_out = None
        self.ang_out = None
        self.width_out = None
        self.ang = 0.0
        self.width = 0.0
        self.grasping_point = []

        # Subscribers
        rospy.Subscriber(self.camera_topic, Image, self.depth_callback, queue_size=10)

        # Initialise some globals.
        self.max_pixel_index = np.array([150, 150])
        self.max_pixel_reescaled = np.array([150, 150])

        # Tensorflow graph to allow use in callback.
        self.graph = tf.get_default_graph()

        # Get the camera parameters
        camera_info_msg = rospy.wait_for_message(self.camera_topic_info, CameraInfo)
        K = camera_info_msg.K
        self.fx = K[0]
        self.cx = K[2]
        self.fy = K[4]
        self.cy = K[5]

    def depth_callback(self, depth_message):
        self.depth_message = depth_message
        # Depth is the depth image converted into open cv format
        # Real realsense resolution 480x640
        depth = self.bridge.imgmsg_to_cv2(depth_message)

        self.depth_copy = deepcopy(depth)

        # 'TEST DEPTH'
        # depth_test_circle = depth.copy()

        self.height_res, self.width_res = depth.shape

        # It crops a 300x300 resolution square at the top of the depth image
        # depth[0:300, 170:470]
        depth_crop = depth[0 : self.crop_size, (self.width_res - self.crop_size)//2 : (self.width_res - self.crop_size)//2 + self.crop_size]

        # Creates a deep copy of the depth_crop image
        depth_crop = depth_crop.copy()

        # Returns the positions represented by nan values
        depth_nan = np.isnan(depth_crop)
        self.depth_nan = depth_nan.copy()
         
        # Substitute nan values by zero
        depth_crop[self.depth_nan] = 0
        self.depth_crop = depth_crop
                            
    def inpaint(self):
        # open cv inpainting does weird things at the border.
        # default border is a constant color
        depth_crop = cv2.copyMakeBorder(self.depth_crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT)

        # se o numero que esta no vetor acima for 0, retorna o numero 1 na mesma posicao (como se fosse True)
        # se depth_crop == 0, retorna 1 como inteiro.
        # Ou seja, copia os pixels pretos da imagem e a posicao deles
        mask = (depth_crop == 0).astype(np.uint8)

        # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        depth_scale = np.abs(depth_crop).max()

        # Normalize
        depth_crop = depth_crop.astype(np.float32)/depth_scale  # Has to be float32, 64 not supported

        # Substitute mask values by near values. See opencv doc for more detail
        depth_crop = cv2.inpaint(depth_crop, mask, 1, cv2.INPAINT_NS)

        # Back to original size and value range.
        depth_crop = depth_crop[1:-1, 1:-1]

        # reescale image
        self.depth_crop = depth_crop * depth_scale
        
    # def resize_img(self):    
        # self.depth_crop = cv2.resize(self.depth_crop, (self.crop_size, self.crop_size), cv2.INTER_AREA)

    def inference(self):        
        # Convert depth_crop values to meters
        # self.depth_crop = self.depth_crop

        # values smaller than -1 become -1, and values larger than 1 become 1.
        self.depth_crop = np.clip((self.depth_crop - self.depth_crop.mean())/1000.0, -1, 1)

        with self.graph.as_default():
            self.pred_out = self.model.predict(self.depth_crop.reshape((1, self.crop_size, self.crop_size, 1)))

        points_out = self.pred_out[0].squeeze()
        points_out[self.depth_nan] = 0
        self.points_out = points_out

    def filter_network_outputs(self):
        cos_out = self.pred_out[1].squeeze()
        sin_out = self.pred_out[2].squeeze()
        ang_out = np.arctan2(sin_out, cos_out) / 2.0
        width_out = self.pred_out[3].squeeze() * 150.0  # Scaled 0-150:0-1

        # Filter the outputs
        # The filters are applied to augment the chances of getting a good grasp pose
        points_out = ndimage.filters.gaussian_filter(self.points_out, 5.0)
        self.points_out = np.clip(points_out, 0.0, 1.0-1e-3)
        self.ang_out = ndimage.filters.gaussian_filter(ang_out, 2.0)
        self.width_out = ndimage.filters.gaussian_filter(width_out, 1.0)

    def get_control_parameters(self):
        """
        Get the following parameters:
            max_pixel_index: max pixel identified in the depth_crop image (300x300)
            ang: End effector orientation (planar)
            width_px: Gripper width in pixel
            g_width and width_m = Gripper width in meters calculated in different ways
            grasping_point = End effector pose
        """
        link_pose, _ = self.transf.lookupTransform("base_link", "grasping_link", rospy.Time(0))
        ROBOT_Z = link_pose[2]

        # Track the global max.
        # max_pixel correponds to the position of the max value in points_out
        max_pixel = np.array(np.unravel_index(np.argmax(self.points_out), self.points_out.shape))

        # Return max_pixel posiiton as an int
        self.max_pixel_index = max_pixel.astype(np.int)

        self.ang = self.ang_out[self.max_pixel_index[0], self.max_pixel_index[1]]
        self.width_px = self.width_out[self.max_pixel_index[0], self.max_pixel_index[1]]

        crop_size_width = float(self.crop_size)
        self.g_width = 2.0 * (ROBOT_Z + 0.24) * np.tan(self.FOV / self.height_res * self.width_px / 2.0 / 180.0 * np.pi) #* 0.37

        reescaled_height = int(self.max_pixel_index[0])
        reescaled_width = int((self.width_res - self.crop_size) // 2 + self.max_pixel_index[1])
        self.max_pixel_reescaled = [reescaled_height, reescaled_width]
        point_depth = self.depth_copy[self.max_pixel_reescaled[0], self.max_pixel_reescaled[1]]

        # Get the grip width in meters
        width_m = self.width_out / crop_size_width * 2.0 * point_depth * np.tan(self.FOV * crop_size_width / self.height_res / 2.0 / 180.0 * np.pi) / 1000 #* 0.37
        self.width_m = abs(width_m[self.max_pixel_index[0], self.max_pixel_index[1]])
                    
        if not np.isnan(point_depth):
            # These magic numbers are my camera intrinsic parameters.
            x = (self.max_pixel_reescaled[1] - self.cx)/(self.fx) * point_depth
            y = (self.max_pixel_reescaled[0] - self.cy)/(self.fy) * point_depth
            self.grasping_point = [x, y, point_depth]

    def publish_grasping_rectangle_image(self):
        """
        Show the depth image with a rectangle representing the grasp
        Image resolution: 300x300
        """
        rectangle_ang = self.ang - np.pi/2
        
        vetx = [-(self.width_px/2), (self.width_px/2), (self.width_px/2), -(self.width_px/2), -(self.width/2)]
        vety = [10, 10, -10, -10, 10]
        
        X = [ int((vetx[i] * np.cos(self.ang) - vety[i] * np.sin(self.ang)) + self.max_pixel_index[0]) for i in range(len(vetx))]
        Y = [ int((vety[i] * np.cos(self.ang) + vetx[i] * np.sin(self.ang)) + self.max_pixel_index[1]) for i in range(len(vetx))]
        
        rr1, cc1 = circle(self.max_pixel_index[0], self.max_pixel_index[1], 5)
        depth_crop_copy = self.depth_crop.copy()
        depth_crop_copy[rr1, cc1] = 0.2
        cv2.line(depth_crop_copy, (Y[0],X[0]), (Y[1],X[1]), (0, 0, 0), 2)
        cv2.line(depth_crop_copy, (Y[1],X[1]), (Y[2],X[2]), (0.2, 0.2, 0.2), 2)
        cv2.line(depth_crop_copy, (Y[2],X[2]), (Y[3],X[3]), (0, 0, 0), 2)
        cv2.line(depth_crop_copy, (Y[3],X[3]), (Y[4],X[4]), (0.2, 0.2, 0.2), 2)

        self.depth_with_square.publish(self.bridge.cv2_to_imgmsg(depth_crop_copy))
        

    def get_grasp_image(self):
        # Draw grasp markers on the points_out and publish it. (for visualisation)
        # points_out was used in gaussian_filter for last
        grasp_img = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
        # Draw the red area in the image
        grasp_img[:,:,2] = (self.points_out * 255.0)

        # draw the circle at the green point
        rr, cc = circle(self.max_pixel_index[0], self.max_pixel_index[1], 5)
        grasp_img[rr, cc, 0] = 0
        grasp_img[rr, cc, 1] = 255
        grasp_img[rr, cc, 2] = 0
        self.grasp_img = grasp_img

    def publish_images(self):
        #Publish the output images (not used for control, only visualisation)
        grasp_img = self.bridge.cv2_to_imgmsg(self.grasp_img, 'bgr8')
        grasp_img.header = self.depth_message.header
        self.grasp_pub.publish(grasp_img)

        self.depth_pub.publish(self.bridge.cv2_to_imgmsg(self.depth_crop))
        self.ang_pub.publish(self.bridge.cv2_to_imgmsg(self.ang_out))

    def publish_data_to_robot(self):
        # -1 is multiplied by cmd_msg.data[3] because the object_detected frame is inverted
        if self.args.real:
            offset_x = -0.03 # 0.002
            offset_y = 0.02 # -0.05
            offset_z = 0.058 # 0.013
        else:
            offset_x = 0.005
            offset_y = 0.0
            offset_z = 0.02

        # Output the best grasp pose relative to camera.
        cmd_msg = Float32MultiArray()
        cmd_msg.data = [self.grasping_point[0]/1000.0, self.grasping_point[1]/1000.0, self.grasping_point[2]/1000.0, self.ang, self.width_m, self.g_width]
        self.cmd_pub.publish(cmd_msg)
        
        # The transformation between object_detected and base_link can be done way better with TF2
        self.br.sendTransform((0.0, 0.0, 0.0), quaternion_from_euler(0.0, 0.0, 0.0),
                         rospy.Time.now(),
                         "camera_depth_optical_frame_rotated",
                         "camera_depth_optical_frame")

        self.br.sendTransform((cmd_msg.data[0], 
                               cmd_msg.data[1], 
                               cmd_msg.data[2]), 
                               quaternion_from_euler(0.0, 0.0, -1*cmd_msg.data[3]),
                               rospy.Time.now(),
                               "object_detected",
                               "camera_depth_optical_frame_rotated")

        
        # self.transf.waitForTransform("base_link", "object_detected", rospy.Time(), rospy.Duration(0.1))
        # object_pose, object_ori = self.transf.lookupTransform("base_link", "object_detected", rospy.Time(0))
        # object_ori = euler_from_quaternion(object_ori)

        # self.br.sendTransform((object_pose[0] + offset_x, 
                               # object_pose[1] + offset_y,
                               # object_pose[2] + offset_z), 
                               # quaternion_from_euler(0.0, 0.0, object_ori[2]),
                               # rospy.Time.now(),
                               # "object_link",
                               # "base_link")

def main():
    args = parse_args()
    grasp_detection = ssgg_grasping(args)
    rospy.sleep(0.2)

    rate = rospy.Rate(120)
    while not rospy.is_shutdown():
        print "running"
        grasp_detection.inpaint()
        grasp_detection.inference()
        grasp_detection.filter_network_outputs()
        grasp_detection.get_control_parameters()
        grasp_detection.publish_grasping_rectangle_image()
        grasp_detection.get_grasp_image()
        grasp_detection.publish_images()
        grasp_detection.publish_data_to_robot()
        
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        print "Program interrupted before completion"
