
## Single Shot Generative Grasping Convolutional Neural Network (SSGG-CNN)

<p align="center">
<a href="https://www.youtube.com/watch?v=_uEvq0K7n-Q" target="_blank">
<img src="https://user-images.githubusercontent.com/28100951/80933477-49bfad80-8d9a-11ea-888c-16b51b39562e.png" width="600">
</p>
</a>

------------

<a id="top"></a>
### Contents
1. [Authors](#1.0)
2. [Description](#2.0)
3. [Required packages - Kinetic Version](#3.0)
4. [Run GGCNN in Gazebo and RVIZ](#4.0)
5. [Connecting with the real UR5](#5.0)

------------
<a name="1.0"></a>
### 1.0 - Authors

- Caio Viturino* - [[Lattes](http://lattes.cnpq.br/4355017524299952)] [[Linkedin](https://www.linkedin.com/in/engcaiobarros/)] - engcaiobarros@gmail.com
- Kleber de Lima Santana Filho - [[Lattes](http://lattes.cnpq.br/3942046874020315)] [[Linkedin](https://www.linkedin.com/in/engkleberfilho/)] - engkleberf@gmail.com
- Daniel M. de Oliveira* - danielmoura@ufba.br
- Cézar Bieniek Lemos* - cezarcbl@protonmail.com
- André Gustavo Scolari Conceição* - [[Lattes](http://lattes.cnpq.br/6840685961007897)] - andre.gustavo@ufba.br

*LaR - Laboratório de Robótica, Departamento de Engenharia Elétrica e de Computação, Universidade Federal da Bahia, Salvador, Brasil

<a name="2.0"></a>
### 2.0 - Description

This paper proposes a two-step cascaded system with the [Generative Grasping Convolutional Neural Network (GG-CNN)](https://github.com/dougsm/ggcnn_kinova_grasping) and the [Single Shot Multibox Detector architecture (SSD)](https://github.com/czrcbl/train_detection) to perform grasping in a vision-based object recognition system. We call the proposed method as Single Shot Generative Grasping Neural Network (SSGG-CNN). The GG-CNN is a powerful object-independent grasping synthesis method well-known for the outstanding performance in open-loop and closed-loop systems using a pixel-wise grasp quality prediction. However, it is not capable of distinguishing between manipulable objects and fixed objects in the workspace. In order to mitigate this problem, the SSD was adopted to perform object detection. It allows the grasping system to perform the grasp only in manipulable objects identified by the SSD. We found an average success rate of 85% over 20 grasps attempts considering open-loop with static and uncluttered objects randomly organized on a planar surface. 

The [GGCNN](https://github.com/dougsm/ggcnn_kinova_grasping) was created by *[Douglas Morrison](http://dougsm.com), [Peter Corke](http://petercorke.com), [Jürgen Leitner](http://juxi.net)* (2018).

<a name="3.0"></a>
### 3.0 - Required packages - Kinetic Version

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
git clone https://github.com/lar-deeufba/ssggcnn_ur5_grasping
```

Run the install.sh file
```bash
cd ~/catkin_ws_new/src/ssggcnn_ur5_grasping/install
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

<a name="4.0"></a>
### 4.0 - Run SSGG-CNN in Gazebo
Please follow each following steps:

#### 4.1 - Launch Gazebo:
```bash
roslaunch ssggcnn_ur5_grasping gazebo_ur5.launch
```

#### 4.2 - Run the UR5 control node 
```bash
rosrun ssggcnn_ur5_grasping ur5_open_loop.py --gazebo
```
Press enter until the following message appears and jump to the next step:
"==== Press enter to move the robot to the 'depth cam shot' position!
"

#### 4.3 - Change the Gazebo properties (OPTIONAL)
It will speed up your Gazebo simulation a little bit :)
```bash
rosrun ssggcnn_ur5_grasping change_gazebo_properties.py
```

#### 4.4 - Spawn the objects in the workspace
```bash
rosrun ssggcnn_ur5_grasping spawn_objects.py
```

#### 4.5 - Run the SSD node
```bash
rosrun ssggcnn_ur5_grasping main.py
```

#### 4.6 - Run the GG-CNN node
```bash
rosrun ssggcnn_ur5_grasping run_ggcnn.py --ssggcnn
```

#### 4.7 - UR5 control node
After running the GG-CNN node you are able to move the robot and perform the grasp.
Press enter to complete each related task specified in ur5_open_loop.py

#### 4.8 - Visualize the images published by the GG-CNN
You might want to see the grasp or any other image. In order to do that, you can use the rqt_image_view.
```bash
rosrun rqt_image_view
```

#### 4.9 - Visualize depth cloud in RVIZ
If you want to visualize the data being published by the Intel Realsense D435 please run the following node:
```bash
rosrun ssggcnn_ur5_grasping rviz_ur5.launch
```
