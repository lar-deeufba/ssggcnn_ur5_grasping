cmake_minimum_required(VERSION 2.8.3)
project(real_time_grasp)

find_package(catkin REQUIRED COMPONENTS
  pcl_conversions
  pcl_ros
  message_generation
  gazebo_ros
  roscpp
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
)

find_package(OpenCV REQUIRED)

#find_package(gazebo REQUIRED)

catkin_package(CATKIN_DEPENDS message_runtime)

add_service_files(
  FILES
  GetImage.srv
)

###########
## Build ##
###########

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
)

add_executable(main scripts/pcl_sim/main.cpp)
target_link_libraries(main ${catkin_LIBRARIES} ${OpenCV_LIBS})
add_dependencies(main pcl_sim_generate_messages_cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y -g -O3")