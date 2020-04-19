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
from std_msgs.msg import Float32MultiArray, Int32MultiArray
from real_time_grasp.srv import *

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
        Home = rospack.get_path('real_time_grasp')
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
        self.depth_pub_ssd_square = rospy.Publisher('ggcnn/img/depth_ssd_square', Image, queue_size=1)
        self.depth_pub_copied_img = rospy.Publisher('ggcnn/img/depth_shot_with_copied_img', Image, queue_size=1)
        self.depth_pub_shot = rospy.Publisher('ggcnn/img/depth_shot', Image, queue_size=1)
        self.depth_with_square = rospy.Publisher('ggcnn/img/grasp_depth_with_square', Image, queue_size=1)
        self.img_with_square = rospy.Publisher('ggcnn/img/grasp_img_with_square', Image, queue_size=1)
        self.ang_pub = rospy.Publisher('ggcnn/img/ang', Image, queue_size=1)
        self.cmd_pub = rospy.Publisher('ggcnn/out/command', Float32MultiArray, queue_size=1)

        # Initialise some var
        # self.depth = None
        self.color_img = None
        self.depth_crop = None
        self.depth_copy_for_point_depth = None
        self.depth_copy_for_rectangle = None
        self.depth_message = None
        self.depth_message_ggcnn = None
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
        self.width_px = 0.0
        self.width_m = 0.0
        self.g_width = 0.0
        self.grasping_point = []
        self.depth_image_shot = None
        self.points_vec = []
        self.depth_image_shot_copy = None
        self.depth_image_shot_with_object_copied = None

        self.depth_offset_px = 10

        # Initialise some globals.
        self.max_pixel = np.array([150, 150])
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

        # Subscribers
        rospy.Subscriber(self.camera_topic, Image, self.get_depth_callback, queue_size=10)
        rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=10)
        rospy.Subscriber('sdd_points_array', Int32MultiArray, self.bounding_boxes_callback, queue_size=10)
        # rospy.Subscriber(self.camera_topic, Image, self.depth_callback, queue_size=10)
    
    def get_depth_callback(self, depth_message):
        self.depth_message = depth_message
        
    def bounding_boxes_callback(self, msg):
        # print("msg: ", msg)
        box_number = len(msg.data) / 4
        if box_number != 0:
            depth = self.bridge.imgmsg_to_cv2(self.depth_message)
            actual_depth_image = depth.copy()
            
            box_points = list(msg.data)
            # print(box_points)
            i, index_inf, index_sup = 0, 0, 4
            offset = self.depth_offset_px # it is necessary because depth and color images are not always fully aligned
            points_vec = []
            
            while i < box_number:
                points = box_points[index_inf: index_sup]
                points_vec.append(points)

                start_point = (points[0] - offset, points[1] - offset)
                end_point = (points[2] + offset, points[3] + offset)
                color = (255, 0, 0)
                thickness = 2
                actual_depth_image = cv2.rectangle(actual_depth_image, start_point, end_point, color, thickness)
                
                index_inf += 4
                index_sup += 4
                i += 1

            self.points_vec = points_vec
            points_vec = []
            self.depth_pub_ssd_square.publish(self.bridge.cv2_to_imgmsg(actual_depth_image))

    def image_callback(self, color_msg):
        color_img = self.bridge.imgmsg_to_cv2(color_msg)
        height_res, width_res, _ = color_img.shape
        offset_height = 0# -12
        offset_width = 0# 12
        color_img = color_img[0 + offset_height: self.crop_size + offset_height, (width_res - self.crop_size)//2 + offset_width : (width_res - self.crop_size)//2 + self.crop_size + offset_width]
        self.color_img = color_img

    def get_depth_image_shot(self):
        self.depth_image_shot = rospy.wait_for_message("camera/depth/image_raw", Image)
        # depth_image_shot = self.depth_image_shot
        # depth = self.bridge.imgmsg_to_cv2(depth_image_shot)
        # depth_image_shot_copy = depth.copy()
        # height_res, width_res = depth_image_shot_copy.shape
        # self.depth_image_shot_copy = depth_image_shot_copy[0 : self.crop_size,
        # (width_res - self.crop_size)//2 : (width_res - self.crop_size)//2 + self.crop_size]

    def copy_obj_to_depth_img(self):
        points = self.points_vec
        depth_image_shot = deepcopy(self.depth_image_shot)
        depth_image_shot = self.bridge.imgmsg_to_cv2(depth_image_shot)
        depth_image_shot_copy = depth_image_shot.copy()

        depth_message = self.depth_message
        depth_message = self.bridge.imgmsg_to_cv2(depth_message)
        depth_message_copy = depth_message.copy()

        offset = self.depth_offset_px
        number_of_boxes = len(points)
        i = 0
        while i < number_of_boxes:
            depth_image_shot_copy[points[i][1] - offset : points[i][3] + offset, points[i][0] - offset : points[i][2] + offset] \
             = depth_message_copy[points[i][1] - offset : points[i][3] + offset, points[i][0] - offset : points[i][2] + offset] 
            i += 1

        depth_image_shot = self.bridge.cv2_to_imgmsg(depth_image_shot_copy)
        depth_image_shot.header = self.depth_message.header
        self.depth_image_shot_with_object_copied = depth_image_shot

        self.depth_pub_copied_img.publish(depth_image_shot)
        self.depth_pub_shot.publish(self.depth_image_shot)

    # vai ter que se inscrever na copia da imagem (shot + square)
    def depth_process_ggcnn(self):
        depth_message = self.depth_image_shot_with_object_copied

        print "Aqui 0"

        # INPUT
        depth = self.bridge.imgmsg_to_cv2(depth_message)
        
        depth_copy_for_point_depth = depth.copy()

        height_res, width_res = depth.shape
        # It crops a 300x300 resolution square at the top of the depth image - depth[0:300, 170:470]
        depth_crop = depth[0 : self.crop_size, 
                           (width_res - self.crop_size)//2 : (width_res - self.crop_size)//2 + self.crop_size]
        # Creates a deep copy of the depth_crop image
        depth_crop = depth_crop.copy()
        # Returns the positions represented by nan values
        depth_nan = np.isnan(depth_crop)
        depth_nan = depth_nan.copy()
        # Substitute nan values by zero
        depth_crop[depth_nan] = 0

        # INPAINT PROCESS
        depth_crop = cv2.copyMakeBorder(depth_crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        # se o numero que esta no vetor acima for 0, retorna o numero 1 na mesma posicao (como se fosse True)
        # se depth_crop == 0, retorna 1 como inteiro.
        # Ou seja, copia os pixels pretos da imagem e a posicao deles
        mask = (depth_crop == 0).astype(np.uint8)
        # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        depth_scale = np.abs(depth_crop).max()
        # Normalize
        depth_crop = depth_crop.astype(np.float32) / depth_scale  # Has to be float32, 64 not supported
        # Substitute mask values by near values. See opencv doc for more detail
        depth_crop = cv2.inpaint(depth_crop, mask, 1, cv2.INPAINT_NS)
        # Back to original size and value range.
        depth_crop = depth_crop[1:-1, 1:-1]
        # reescale image
        depth_crop = depth_crop * depth_scale

        # INFERENCE PROCESS
        depth_crop = depth_crop/1000.0
        # values smaller than -1 become -1, and values larger than 1 become 1.
        depth_crop = np.clip((depth_crop - depth_crop.mean()), -1, 1)
        with self.graph.as_default():
            pred_out = self.model.predict(depth_crop.reshape((1, self.crop_size, self.crop_size, 1)))
        points_out = pred_out[0].squeeze()
        cos_out = pred_out[1].squeeze()
        sin_out = pred_out[2].squeeze()
        width_out = pred_out[3].squeeze() * 150.0  # Scaled 0-150:0-1
        points_out[depth_nan] = 0
        ang_out = np.arctan2(sin_out, cos_out) / 2.0

        # FILTERING PROCESS
        # The filters are applied to augment the chances of getting a good grasp pose
        points_out = ndimage.filters.gaussian_filter(points_out, 5.0)
        points_out = np.clip(points_out, 0.0, 1.0-1e-3)
        ang_out = ndimage.filters.gaussian_filter(ang_out, 2.0)
        width_out = ndimage.filters.gaussian_filter(width_out, 1.0)

        # CONTROL PROCESS
        link_pose, _ = self.transf.lookupTransform("base_link", "grasping_link", rospy.Time(0))
        ROBOT_Z = link_pose[2]
        # Track the global max.
        # max_pixel correponds to the position of the max value in points_out
        max_pixel = np.array(np.unravel_index(np.argmax(points_out), points_out.shape))
        # Return max_pixel posiiton as an int (300x300)
        max_pixel = max_pixel.astype(np.int)
        ang = ang_out[max_pixel[0], max_pixel[1]]
        width_px = width_out[max_pixel[0], max_pixel[1]]
        reescaled_height = int(max_pixel[0])
        reescaled_width = int((width_res - self.crop_size) // 2 + max_pixel[1])
        max_pixel_reescaled = [reescaled_height, reescaled_width]
        point_depth = depth_copy_for_point_depth[max_pixel_reescaled[0], max_pixel_reescaled[1]]

        # GRASP WIDTH PROCESS
        g_width = 2.0 * (ROBOT_Z + 0.24) * np.tan(self.FOV / height_res * width_px / 2.0 / 180.0 * np.pi) #* 0.37
        crop_size_width = float(self.crop_size)
        width_m = width_out / crop_size_width * 2.0 * point_depth * np.tan(self.FOV * crop_size_width / height_res / 2.0 / 180.0 * np.pi) / 1000 #* 0.37
        width_m = abs(width_m[max_pixel[0], max_pixel[1]])
                    
        if not np.isnan(point_depth):
            # These magic numbers are my camera intrinsic parameters.
            x = (max_pixel_reescaled[1] - self.cx)/(self.fx) * point_depth
            y = (max_pixel_reescaled[0] - self.cy)/(self.fy) * point_depth
            grasping_point = [x, y, point_depth]

        # OUTPUT
        self.depth_message_ggcnn = depth_message
        self.points_out = points_out
        self.depth_crop = depth_crop
        self.ang = ang
        self.ang_out = ang_out
        self.width_px = width_px
        self.max_pixel = max_pixel
        self.max_pixel_reescaled = max_pixel_reescaled
        self.g_width = g_width
        self.width_m = width_m
        self.point_depth = point_depth
        self.grasping_point = grasping_point

    def publish_grasping_rectangle_image(self):
        """
        Show the depth image with a rectangle representing the grasp
        Image resolution: 300x300
        """
        print "Aqui 1"
        depth_crop = self.depth_crop
        color_img = self.color_img 
        if depth_crop is not None and color_img is not None:
            ang = self.ang
            width_px = self.width_px
            max_pixel = self.max_pixel

            rectangle_ang = ang #- np.pi/2
            
            vetx = [-(width_px/2), (width_px/2), (width_px/2), -(width_px/2)]
            vety = [10, 10, -10, -10]
            
            X = [ int((     vetx[i]*np.cos(rectangle_ang) + vety[i]*np.sin(rectangle_ang)) + self.max_pixel[1]) for i in range(len(vetx))]
            Y = [ int((-1 * vetx[i]*np.sin(rectangle_ang) + vety[i]*np.cos(rectangle_ang)) + self.max_pixel[0]) for i in range(len(vetx))]
     
            rr1, cc1 = circle(max_pixel[0], max_pixel[1], 5)
            depth_crop_copy = depth_crop.copy()
            depth_crop_copy[rr1, cc1] = 0.2
            cv2.line(depth_crop_copy, (X[0],Y[0]), (X[1],Y[1]), (0, 0, 0), 2)
            cv2.line(depth_crop_copy, (X[1],Y[1]), (X[2],Y[2]), (0.2, 0.2, 0.2), 2)
            cv2.line(depth_crop_copy, (X[2],Y[2]), (X[3],Y[3]), (0, 0, 0), 2)
            cv2.line(depth_crop_copy, (X[3],Y[3]), (X[0],Y[0]), (0.2, 0.2, 0.2), 2)
            self.depth_with_square.publish(self.bridge.cv2_to_imgmsg(depth_crop_copy))#, encoding="rgb8"))

            img_crop_copy = color_img.copy()
            img_crop_copy[rr1, cc1] = 0.2
            cv2.line(img_crop_copy, (X[0],Y[0]), (X[1],Y[1]), (0, 0, 0), 2)
            cv2.line(img_crop_copy, (X[1],Y[1]), (X[2],Y[2]), (0.2, 0.2, 0.2), 2)
            cv2.line(img_crop_copy, (X[2],Y[2]), (X[3],Y[3]), (0, 0, 0), 2)
            cv2.line(img_crop_copy, (X[3],Y[3]), (X[0],Y[0]), (0.2, 0.2, 0.2), 2)
            self.img_with_square.publish(self.bridge.cv2_to_imgmsg(img_crop_copy, encoding="rgb8"))

    def get_grasp_image(self):
        """
        Show the detected grasp regions of the image
        """
        points_out = self.points_out

        print "Aqui 2"
        if points_out is not None:
            max_pixel = self.max_pixel

            # Draw grasp markers on the points_out and publish it. (for visualisation)
            # points_out was used in gaussian_filter for last
            grasp_img = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
            # Draw the red area in the image
            grasp_img[:,:,2] = (points_out * 255.0)
            # draw the circle at the green point
            rr, cc = circle(max_pixel[0], max_pixel[1], 5)
            grasp_img[rr, cc, 0] = 0
            grasp_img[rr, cc, 1] = 255
            grasp_img[rr, cc, 2] = 0

            self.grasp_img = grasp_img

    def publish_images(self):
        grasp_img = self.grasp_img
        depth_message = self.depth_message_ggcnn
        ang_out = self.ang_out
        depth_crop = self.depth_crop

        print "Aqui 3"
        if grasp_img is not None:
            #Publish the output images (not used for control, only visualisation)
            grasp_img = self.bridge.cv2_to_imgmsg(grasp_img, 'bgr8')
            grasp_img.header = depth_message.header
            
            self.grasp_pub.publish(grasp_img)
            self.depth_pub.publish(self.bridge.cv2_to_imgmsg(depth_crop))
            self.ang_pub.publish(self.bridge.cv2_to_imgmsg(ang_out))
            
    def publish_data_to_robot(self):
        grasping_point = self.grasping_point
        ang = self.ang
        width_m = self.width_m
        g_width = self.g_width

        print "Aqui 4"
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
        cmd_msg.data = [grasping_point[0]/1000.0, grasping_point[1]/1000.0, grasping_point[2]/1000.0, -1*ang, width_m, g_width]
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

        self.transf.waitForTransform("base_link", "object_detected", rospy.Time(), rospy.Duration(0.1))
        object_pose, object_ori = self.transf.lookupTransform("base_link", "object_detected", rospy.Time(0))
        object_ori = euler_from_quaternion(object_ori)
        self.br.sendTransform((object_pose[0] + offset_x, 
                               object_pose[1] + offset_y,
                               object_pose[2] + offset_z), 
                               quaternion_from_euler(0.0, 0.0, object_ori[2]),
                               rospy.Time.now(),
                               "object_link",
                               "base_link")

def main():
    args = parse_args()
    grasp_detection = ssgg_grasping(args)
    rospy.sleep(3.0)

    raw_input("Move the objects out of the camera view and move the robot to the pre-grasp position.")
    grasp_detection.get_depth_image_shot()
    
    raw_input("Press enter to start the GGCNN")
    rate = rospy.Rate(120)
    while not rospy.is_shutdown():
        # rospy.spin()
        print "Running"
        grasp_detection.copy_obj_to_depth_img()
        grasp_detection.depth_process_ggcnn()
        grasp_detection.publish_grasping_rectangle_image()
        grasp_detection.get_grasp_image()
        grasp_detection.publish_images()
        # grasp_detection.publish_data_to_robot()        
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        print "Program interrupted before completion"