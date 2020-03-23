------------

<a id="top"></a>
### Contents
1. [Description](#1.0)
2. [Required packages - Kinetic Version](#2.0)
3. [Run GGCNN in Gazebo and RVIZ](#3.0)
4. [Sending commands through the action server](#4.0)
5. [Connecting with real UR5](#5.0)
6. [Meetings minutes](#6.0)
7. [To do](#7.0)

------------

<a name="1.0"></a>
### 1.0 Description

This repository was created in order to develop an improved version of the [GGCNN]((https://github.com/dougsm/ggcnn_kinova_grasping)) Grasp Method created by Doug Morrison (2018).

> **_NOTE:_**  This package should be placed into your src folder

<a name="2.0"></a>
### 2.0 Required packages - Kinetic Version

> **_NOTE:_** Please access the grasp_project repository in order to check the required packages in the '2.0 Required packages' section. The same packages are used in this repository. You do not need to clone grasp_project repository.

- [grasp_project](https://github.com/caiobarrosv/grasp_project) - Created by Caio Viturino

#### Easy install

In order to install all the required packages easily, create a catkin workspace folder and then a src inside it.
```bash
mkdir -p ~/catkin_ws_new/src
```

Clone this repository into the src folder
```bash
cd ~/catkin_ws_new/src
git clone https://github.com/lar-deeufba/real-time-grasp
```

Run the install.sh file
```bash
cd ~/catkin_ws_new/src/real-time-grasp/install
sudo chmod +x ./install.sh
./install.sh #without sudo
```

<a name="3.0"></a>
### 3.0 Run GGCNN in Gazebo and RVIZ

Launch Gazebo first:
obs: The robot may not start correctly due to a hack method used to set initial joint positions in gazebo as mentioned in this [issue](https://github.com/ros-simulation/gazebo_ros_pkgs/issues/93#). If it happens, try to restart gazebo.
```bash
roslaunch real-time-grasp gazebo_ur5.launch
```

Launch RVIZ if you want to see the frame (object_detected) corresponding to the object detected by GGCNN and the point cloud.
In order to see the point cloud, please add pointcloud2 into the RVIZ and select the correct topic:
```bash
roslaunch real-time-grasp rviz_ur5.launch
```

Run the GGCNN. This node will publish a frame corresponding to the object detected by the GGCNN.
```bash
rosrun real-time-grasp run_ggcnn_ur5.py
```

Running this node will move the robot to the position published by the run_ggcnn_ur5.py node.
```bash
rosrun real-time-grasp command_GGCNN_ur5.py --gazebo
```

You might want to see the grasp or any other image. In order to do that, you can use the rqt_image_view.
```bash
rosrun rqt_image_view
```

<a name="4.0"></a>
### 4.0 Sending commands through the action server

If you want to test the position controller sending commands directly to the /'controller_command'/command topic use
the following:

```bash
rostopic pub -1 /pos_based_pos_traj_controller/command trajectory_msgs/JointTrajectory "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
joint_names: ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
points:
  - positions: [1.57, 0, 0, 0, 0, 0]
    time_from_start: {secs: 1, nsecs: 0}"
```

<a name="5.0"></a>
### 5.0 Connecting with real UR5

Use the following command in order to connect with real UR5.
If you are using velocity control, do not use bring_up. Use ur5_ros_control instead.

```
roslaunch real-time-grasp ur5_ros_control.launch robot_ip:=192.168.131.13
```

Launch the real Intel Realsense D435
```
roslaunch real-time-grasp rs_d435_camera.launch
```

Launch the gripper control node
```
rosrun robotiq_2f_gripper_control Robotiq2FGripperRtuNode.py /dev/ttyUSB0
```

Launch the ggcnn node
```
rosrun real-time-grasp run_ggcnn_ur5.py --real
```

Launch the main node of the Intel Realsense D435
```
rosrun real-time-grasp command_GGCNN_ur5.py
```

If you want to visualize the depth or point cloud, you can launch RVIZ
```
roslaunch real-time-grasp rviz_ur5.launch
```

Firstly check the machine IP. The IP configured on the robot must have the last digit different.

```bash
ifconfig
```

Disable firewall

```bash
sudo ufw disable
```

Set up a static IP on UR5 according to the following figure

![config](https://user-images.githubusercontent.com/28100951/71323978-2ca7d380-24b8-11ea-954c-940b009cfd93.jpg)

Set up a connection on Ubuntu according to the following figure

![config_ethernet2](https://user-images.githubusercontent.com/28100951/71323962-fe29f880-24b7-11ea-86dc-756729932de4.jpg)

<a name="6.0"></a>
### 6.0 Meetings minutes
#### Meeting - 25/11/2019
Topics covered:
1. Preferably use devices already in the lab, such as UR5, Intel Realsense and Gripper 2-Fingers Robotiq
2. Check how to use neural networks to predict the position of objects. Thus, the proposed method would be robust against camera limitations regarding the proximity of the object, that is, even if there is no depth information, the neural network would use past data to predict where the object is at the given moment.
3. Search for grasping applications.
4. Translate the thesis into English

<a name="7.0"></a>
### 7.0 To do
#### March/20
- [x] Test realsense post-processing to enhance depth images - librealsense/examples/post-processing
- [x] Record a rosbag file of the realsense depth cam
- [x] Set the right position for the object detected frame
- [x] Test the goal position using UR5
- [x] Implement Robotiq gripper and force control

#### April/20
- [] Update realsense-ros to the new version