cmake_minimum_required(VERSION 2.8.3)
project(ssggcnn_ur5_grasping)

find_package(catkin REQUIRED COMPONENTS
  pcl_conversions
  pcl_ros
  gazebo_ros
  roscpp
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
)

find_package(OpenCV REQUIRED)

#find_package(gazebo REQUIRED)

## Generate services in the 'srv' folder
#add_service_files(
#  FILES
#)

catkin_package(CATKIN_DEPENDS message_runtime)


###########
## Build ##
###########

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y -g -O3")