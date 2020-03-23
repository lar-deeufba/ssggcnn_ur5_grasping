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

import copy

import argparse

parser = argparse.ArgumentParser(description='GGCNN')
# store_false assumes that variable is already true and is only set to false if is given in command terminal
parser.add_argument('--real', action='store_true', help='Consider the real intel realsense')
args = parser.parse_args()

bridge = CvBridge()

# Load the Network.
rospack = rospkg.RosPack()
Home = rospack.get_path('real-time-grasp')
MODEL_FILE = Home + '/data/epoch_29_model.hdf5'
with tf.device('/device:GPU:0'):
    model = load_model(MODEL_FILE)

rospy.init_node('ggcnn_detection')
transf = TransformListener()
br = TransformBroadcaster()

# Load GGCN parameters
crop_size = rospy.get_param("/GGCNN/crop_size")
FOV = rospy.get_param("/GGCNN/FOV")
camera_topic_info = rospy.get_param("/GGCNN/camera_topic_info")
if args.real:
    camera_topic = rospy.get_param("/GGCNN/camera_topic_realsense")
else:
    camera_topic = rospy.get_param("/GGCNN/camera_topic")

# Output publishers.
grasp_pub = rospy.Publisher('ggcnn/img/grasp', Image, queue_size=1)
grasp_plain_pub = rospy.Publisher('ggcnn/img/grasp_plain', Image, queue_size=1)
depth_pub = rospy.Publisher('ggcnn/img/depth', Image, queue_size=1)
depth_with_square = rospy.Publisher('ggcnn/img/depth_with_square', Image, queue_size=1)
depth_test = rospy.Publisher('ggcnn/img/depth_test', Image, queue_size=1)
ang_pub = rospy.Publisher('ggcnn/img/ang', Image, queue_size=1)
cmd_pub = rospy.Publisher('ggcnn/out/command', Float32MultiArray, queue_size=1)

# Initialise some globals.
prev_mp = np.array([150, 150])

# Tensorflow graph to allow use in callback.
graph = tf.get_default_graph()

# Get the camera parameters
camera_info_msg = rospy.wait_for_message(camera_topic_info, CameraInfo)
K = camera_info_msg.K
fx = K[0]
cx = K[2]
fy = K[4]
cy = K[5]

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

def depth_callback(depth_message):
    global model
    global graph
    global prev_mp
    global ROBOT_Z
    global fx, cx, fy, cy
    global transf
    global crop_size, args

    # The EOF position should be tracked in real time by depth_callback
    link_pose, _ = transf.lookupTransform("base_link", "grasping_link", rospy.Time(0))
    ROBOT_Z = link_pose[2]

    # Each with is used to calculate the processing time
    with TimeIt('Crop'):

        # Depth is the depth image converted into open cv format
        # Real realsense resolution 480x640
        depth = bridge.imgmsg_to_cv2(depth_message)

        'TEST DEPTH'
        depth_test_circle = depth.copy()
        
        height_res, width_res = depth.shape
        # print("Height_res: ", height_res)
        # print("Width_res: ", width_res)
        
        # It crops a 300x300 resolution square at the top of the depth image
        # depth[0:300, 170:470]
        depth_crop = depth[0 : crop_size, (width_res - crop_size)//2 : (width_res - crop_size)//2 + crop_size]
        
        # Creates a deep copy of the depth_crop image
        depth_crop = depth_crop.copy()

        # Returns the positions represented by nan values
        depth_nan = np.isnan(depth_crop)
        depth_nan = depth_nan.copy()

        # Substitute nan values by zero
        depth_crop[depth_nan] = 0
                        
    with TimeIt('Inpaint'):
        # open cv inpainting does weird things at the border.
        # default border is a constant color
        depth_crop = cv2.copyMakeBorder(depth_crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT)

        # se o numero que esta no vetor acima for 0, retorna o numero 1 na mesma posicao (como se fosse True)
        # se depth_crop == 0, retorna 1 como inteiro.
        # Ou seja, copia os pixels pretos da imagem e a posicao deles
        mask = (depth_crop == 0).astype(np.uint8)

        # from mvp repo
        # kernel = np.ones((3, 3),np.uint8)
        # mask = cv2.dilate(mask, kernel, iterations=1)
        # depth_crop[mask==1] = 0

        # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        # Copia o maior valor para depois realizar a escala
        depth_scale = np.abs(depth_crop).max()

        # Normalize
        depth_crop = depth_crop.astype(np.float32)/depth_scale  # Has to be float32, 64 not supported

        # aplica a mascara no depth_crop
        # a mascara ja possui os valores dos pixels pretos (informacoes falhas do sensor kinect)
        depth_crop = cv2.inpaint(depth_crop, mask, 1, cv2.INPAINT_NS)

        # Back to original size and value range.
        # Retira as bordas da imagem, pegando os pixels de dentro
        depth_crop = depth_crop[1:-1, 1:-1]

        # depois da escala a imagem fica totalmente branca
        # pode ser que a imagem fique branca por causa da transformacao p/ float32
        # o formato era uint16 antes da transformacao
        # perde alguma informacao?
        depth_crop = depth_crop * depth_scale
        
    # with TimeIt('Calculate Depth'):
        # Figure out roughly the depth in mm of the part between the grippers for collision avoidance.
        # depth_center = depth_crop[100:141, 130:171].flatten()
        # depth_center.sort()
        # depth_center = depth_center[:10].mean() * 1000.0

    with TimeIt('Resizing'):
            # Resize
        depth_crop = cv2.resize(depth_crop, (crop_size, crop_size), cv2.INTER_AREA)

    with TimeIt('Inference'):
        # Convert depth_crop values to meters
        depth_crop = depth_crop/1000.0

        # values smaller than -1 become -1, and values larger than 1 become 1.
        depth_crop = np.clip((depth_crop - depth_crop.mean()), -1, 1)

        with graph.as_default():
            pred_out = model.predict(depth_crop.reshape((1, crop_size, crop_size, 1)))

        points_out = pred_out[0].squeeze()
        points_out[depth_nan] = 0

    with TimeIt('Trig'):
        cos_out = pred_out[1].squeeze()
        sin_out = pred_out[2].squeeze()
        ang_out = np.arctan2(sin_out, cos_out)/2.0

        width_out = pred_out[3].squeeze() * 150.0  # Scaled 0-150:0-1

    with TimeIt('Filter'):
        # Filter the outputs
        # The filters are applied to augment the chances of getting a good grasp pose
        points_out = ndimage.filters.gaussian_filter(points_out, 5.0)
        ang_out = ndimage.filters.gaussian_filter(ang_out, 2.0)
        width_out = ndimage.filters.gaussian_filter(width_out, 1.0)

        points_out = np.clip(points_out, 0.0, 1.0-1e-3)

    with TimeIt('Control'):
        # Calculate the best pose from the camera intrinsics.
        maxes = None

        # Use ALWAYS_MAX = True for the open-loop solution.
        ALWAYS_MAX = True

        # print("ROBOT_Z: ", ROBOT_Z)

        if ROBOT_Z > 0.34 or ALWAYS_MAX:  # > 0.34 initialises the max tracking when the robot is reset.
            # Track the global max.
            # max_pixel correponds to the position of the max value in points_out
            max_pixel = np.array(np.unravel_index(np.argmax(points_out), points_out.shape))
            
            # Return max_pixel posiiton as an int
            prev_mp = max_pixel.astype(np.int)
        else:
            # Calculate a set of local maxes.  Choose the one that is closes to the previous one.
            maxes = peak_local_max(points_out, min_distance=10, threshold_abs=0.1, num_peaks=3)
            if maxes.shape[0] == 0:
                return
            max_pixel = maxes[np.argmin(np.linalg.norm(maxes - prev_mp, axis=1))]

            # Keep a global copy for next iteration.
            prev_mp = (max_pixel * 0.25 + prev_mp * 0.75).astype(np.int)

        ang = ang_out[max_pixel[0], max_pixel[1]]
        # rotates the grasp rectangle (-1.57 rad) so that the end effector grasps the object correctly
        ang = ang - np.pi/2
        width = width_out[max_pixel[0], max_pixel[1]]

        # It preserves the max pixel relative to the 300x300 resolution
        max_pixel_detected = max_pixel.copy()

        # Get max pixel based on the original resolution of the image
        # Convert max_pixel back to uncropped/resized image coordinates in order to do the camera transform.
        # max_pixel = ((np.array(max_pixel) / 300.0 * crop_size) + np.array([(height_res - crop_size) // 2, (width_res - crop_size) // 2]))
        # max_pixel = np.round(max_pixel).astype(np.int)
        reescaled_height = int(prev_mp[0])
        reescaled_width = int((width_res - crop_size) // 2 + prev_mp[1])
        max_pixel = [reescaled_height, reescaled_width]

        'Depth crop with square'
        vetx = [-(width/2), (width/2), (width/2), -(width/2), -(width/2)]
        vety = [10, 10, -10, -10, 10]    
        X = [ int((vetx[i] * np.cos(ang) - vety[i] * np.sin(ang)) + max_pixel_detected[0]) for i in range(len(vetx))]
        Y = [ int((vety[i] * np.cos(ang) + vetx[i] * np.sin(ang)) + max_pixel_detected[1]) for i in range(len(vetx))]
        rr1, cc1 = circle(max_pixel_detected[0], max_pixel_detected[1], 5)
        depth_crop_copy = depth_crop.copy()
        depth_crop_copy[rr1, cc1] = 0.2
        cv2.line(depth_crop_copy, (Y[0],X[0]), (Y[1],X[1]), (0, 0, 0), 2)
        cv2.line(depth_crop_copy, (Y[1],X[1]), (Y[2],X[2]), (0.2, 0.2, 0.2), 2)
        cv2.line(depth_crop_copy, (Y[2],X[2]), (Y[3],X[3]), (0, 0, 0), 2)
        cv2.line(depth_crop_copy, (Y[3],X[3]), (Y[4],X[4]), (0.2, 0.2, 0.2), 2)
        depth_with_square.publish(bridge.cv2_to_imgmsg(depth_crop_copy))
        'Depth crop with square'

        'Max pixel test topic'
        # Get the max_pixel height
        # Offset is used to sync the object position and cam feedback
        # offset_mm = 100 if args.real else 45
        rr, cc = circle(max_pixel[0], max_pixel[1], 15)
        point_depth = depth_test_circle[max_pixel[0], max_pixel[1]]
        depth_test_circle[rr, cc] = 0
        depth_test_circle = bridge.cv2_to_imgmsg(depth_test_circle)
        depth_test.publish(depth_test_circle)
        'Max pixel test topic'

        ang = ang + np.pi/2
        crop_size_width = float(crop_size)
        # print("Width [pixels]: %.6s" % (width))
        g_width = 2.0 * (ROBOT_Z + 0.24) * np.tan(FOV / height_res * width / 2.0 / 180.0 * np.pi) #* 0.37
                    
        # Get the grip width in meters
        width_m = width_out / crop_size_width * 2.0 * point_depth * np.tan(FOV * crop_size_width / height_res / 2.0 / 180.0 * np.pi) / 1000 #* 0.37
        width_m = abs(width_m[max_pixel_detected[0], max_pixel_detected[1]])
        print("Width_m: ", width_m)
        
        # These magic numbers are my camera intrinsic parameters.
        x = (max_pixel[1] - cx)/(fx) * point_depth
        y = (max_pixel[0] - cy)/(fy) * point_depth
        z = point_depth
        print("Point_depth: ", z)

        if np.isnan(z):
            return

    with TimeIt('Draw'):
        # Draw grasp markers on the points_out and publish it. (for visualisation)
        # points_out was used in gaussian_filter for last
        grasp_img = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        # Draw the red area in the image
        grasp_img[:,:,2] = (points_out * 255.0)

        # grasp plain does not have the green point
        grasp_img_plain = grasp_img.copy()

        # draw the circle at the green point
        rr, cc = circle(prev_mp[0], prev_mp[1], 5)
        grasp_img[rr, cc, 0] = 0
        grasp_img[rr, cc, 1] = 255
        grasp_img[rr, cc, 2] = 0

    with TimeIt('Publish'):
        # Publish the output images (not used for control, only visualisation)
        grasp_img = bridge.cv2_to_imgmsg(grasp_img, 'bgr8')
        grasp_img.header = depth_message.header
        grasp_pub.publish(grasp_img)

        grasp_img_plain = bridge.cv2_to_imgmsg(grasp_img_plain, 'bgr8')
        grasp_img_plain.header = depth_message.header
        grasp_plain_pub.publish(grasp_img_plain)

        depth_pub.publish(bridge.cv2_to_imgmsg(depth_crop))
        ang_pub.publish(bridge.cv2_to_imgmsg(ang_out))

        # -1 is multiplied by cmd_msg.data[3] because the object_detected frame is inverted
        if args.real:
            offset_x = -0.03 # 0.002
            offset_y = 0.02 # -0.05
            offset_z = 0.058 # 0.013
            # Fixed frame related to the depth optical frame
            angle_offset = 0.0
            angle_deviation = 1
        else:
            offset_x = 0.01
            offset_y = 0.01
            offset_z = 0.0
            angle_offset = 0.0
            angle_deviation = 1 #np.cos(angle_offset)

        # Output the best grasp pose relative to camera.
        cmd_msg = Float32MultiArray()
        cmd_msg.data = [x/1000.0, y/1000.0, z/1000.0, ang, width_m, g_width]
        cmd_pub.publish(cmd_msg)
        # print("x: %.6s | y: %.6s | z: %.6s" % (cmd_msg.data[0], cmd_msg.data[1], cmd_msg.data[2]))        

        # The transformation between object_detected and base_link can be done way better with TF2
        br.sendTransform((0.0, 0.0, 0.0), quaternion_from_euler(angle_offset, 0.0, 0.0),
                         rospy.Time.now(),
                         "camera_depth_optical_frame_rotated",
                         "camera_depth_optical_frame")

        br.sendTransform((cmd_msg.data[0], 
                          cmd_msg.data[1], 
                          cmd_msg.data[2]), 
                          quaternion_from_euler(0.0, 0.0, -1*cmd_msg.data[3]),
                          rospy.Time.now(),
                          "object_detected",
                          "camera_depth_optical_frame_rotated")

        object_pose, object_ori = transf.lookupTransform("base_link", "object_detected", rospy.Time(0))
        object_ori = euler_from_quaternion(object_ori)

        br.sendTransform((object_pose[0] + offset_x, 
                          object_pose[1] + offset_y, 
                          (object_pose[2] + offset_z)*angle_deviation), 
                          quaternion_from_euler(0.0, 0.0, object_ori[2]),
                          rospy.Time.now(),
                          "object_link",
                          "base_link")

depth_sub = rospy.Subscriber(camera_topic, Image, depth_callback, queue_size=1)

while not rospy.is_shutdown():
    rospy.spin()