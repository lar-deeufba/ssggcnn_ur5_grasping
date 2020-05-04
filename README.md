<p align="center">
<a href="https://www.youtube.com/watch?v=_uEvq0K7n-Q" target="_blank">
<img src="https://user-images.githubusercontent.com/28100951/80933477-49bfad80-8d9a-11ea-888c-16b51b39562e.png" width="600">
</p>
</a>

------------

<a id="top"></a>
### Contents
1. [Description](#1.0)
2. [Required packages - Kinetic Version](#2.0)
3. [Run GGCNN in Gazebo and RVIZ](#3.0)
4. [Connecting with the real UR5](#4.0)

------------

<a name="1.0"></a>
### 1.0 - Description

This paper proposes a two-step cascaded system with the [Generative Grasping Convolutional Neural Network (GG-CNN)](https://github.com/dougsm/ggcnn_kinova_grasping) and the [Single Shot Multibox Detector architecture (SSD)](https://github.com/czrcbl/train_detection) to perform grasping in a vision-based object recognition system. We call the proposed method as Single Shot Generative Grasping Neural Network (SSGG-CNN). The GG-CNN is a powerful object-independent grasping synthesis method well-known for the outstanding performance in open-loop and closed-loop systems using a pixel-wise grasp quality prediction. However, it is not capable of distinguishing between manipulable objects and fixed objects in the workspace. In order to mitigate this problem, the SSD was adopted to perform object detection. It allows the grasping system to perform the grasp only in manipulable objects identified by the SSD. We found an average success rate of 85% over 20 grasps attempts considering open-loop with static and uncluttered objects randomly organized on a planar surface. 

The [GGCNN](https://github.com/dougsm/ggcnn_kinova_grasping) was created by *[Douglas Morrison](http://dougsm.com), [Peter Corke](http://petercorke.com), [Jürgen Leitner](http://juxi.net)* (2018).

<a name="2.0"></a>
### 2.0 - Required packages - Kinetic Version

This code was developed with Python 2.7 on Ubuntu 16.04 with ROS Kinetic.

- [Realsense Gazebo Plugin](https://github.com/pal-robotics/realsense_gazebo_plugin)
- [Realsense-ros](https://github.com/IntelRealSense/realsense-ros) Release version 2.2.11
- [Librealsense](https://github.com/IntelRealSense/librealsense) Release version 2.31.0 - Install from source
- [Moveit Kinetic](https://moveit.ros.org/install/)
- [Moveit Python](https://github.com/mikeferguson/moveit_python)
- [Robotiq Gripper](https://github.com/crigroup/robotiq)
- [Universal Robot](https://github.com/ros-industrial/universal_robot)
- [ur_modern_driver](https://github.com/ros-industrial/ur_modern_driver)
- [Gluoncv](https://github.com/dmlc/gluon-cv)
- [Opencv](https://github.com/opencv/opencv)
- [Mxnet](https://mxnet.apache.org/) Install Mxnet for your CUDA version.

> **_NOTE:_**  This package should be placed into your src folder. Please open an issue if you find any problem related to this package.

#### Easy install

In order to install all the required packages easily, create a new catkin workspace
```bash
mkdir -p ~/catkin_ws_new/src
```

Clone this repository into the src folder
```bash
cd ~/catkin_ws_new/src
git clone https://github.com/lar-deeufba/real_time_grasp
```

Run the install.sh file
```bash
cd ~/catkin_ws_new/src/real_time_grasp/install
sudo chmod +x ./install.sh
./install.sh
```

#### This repository also need the SSD512 implementation created by [czrcbl](https://github.com/czrcbl). Please follow the next procedures provided by the author.

Install bboxes before continuing. You can install directly from `github`:
```bash
pip install git+https://github.com/czrcbl/bboxes
```

Or you can clone the repository and install on editable mode:
```bash
git clone https://github.com/czrcbl/bboxes
cd bboxes
git install -e .
```

Download the [model2.params](https://drive.google.com/file/d/1NamkTraRxDBBKDzN5p5D1lCBShqOHp36/view?usp=sharing) in the following link and move it to the `detection_pkg` folder.

<a name="3.0"></a>
### 3.0 - Run GGCNN and SSD512 in Gazebo and RVIZ

Launch Gazebo first:
obs: The robot may not start correctly due to a hack method used to set initial joint positions in Gazebo as mentioned in this [issue](https://github.com/ros-simulation/gazebo_ros_pkgs/issues/93#). If it happens, try to restart Gazebo.
`bico` is a parameter that loads a 3D part in order to test the proposed method. We are not allowed to share this STL file since it is part of a private project. Therefore you will not be able to run the SSD512 net with the pre-trained `bico` part.
```bash
roslaunch real_time_grasp gazebo_ur5.launch bico:=true
```

Run this node and let the UR5 robot move to the front of the 3D printer. Then launch the next node (SSD512 node). 
The arm will only perform the grasp after the GGCNN node is running.
```bash
rosrun real_time_grasp command_GGCNN_ur5.py --gazebo
```

Start the SSD512 detection node. This node is responsible for detecting the object to be filtered.
```bash
roslaunch real_time_grasp detection.launch
```

Launch RVIZ if you want to see the frame (object_detected) corresponding to the object detected by GGCNN and the point cloud.
In order to see the point cloud, please add pointcloud2 into the RVIZ and select the correct topic:
```bash
roslaunch real_time_grasp rviz_ur5.launch
```

Run the GGCNN. This node will publish a frame corresponding to the object detected by the GGCNN.
```bash
rosrun real_time_grasp SSD512_GGCNN.py
```

Running the following command will speed up the Gazebo simulation a little bit :)
```bash
rosrun real_time_grasp change_gazebo_properties.py
```

You might want to see the grasp or any other image. In order to do that, you can use the rqt_image_view.
```bash
rosrun rqt_image_view
```

<a name="4.0"></a>
### 4.0 - Connecting with the real UR5

Use the following command in order to connect with the real UR5.
If you are using velocity control, do not use bring_up. Use ur5_ros_control instead.

```
roslaunch real_time_grasp ur5_ros_control.launch robot_ip:=192.168.131.13
```

Launch the real Intel Realsense D435
```
roslaunch real_time_grasp rs_d435_camera.launch
```

Launch the gripper control node
```
rosrun robotiq_2f_gripper_control Robotiq2FGripperRtuNode.py /dev/ttyUSB0
```

Launch the GG-CNN node
```
rosrun real_time_grasp run_ggcnn_ur5.py --real
```

Launch the main node of the Intel Realsense D435
```
rosrun real_time_grasp command_GGCNN_ur5.py
```

If you want to visualize the depth cloud, you can launch RVIZ
```
roslaunch real_time_grasp rviz_ur5.launch
```
