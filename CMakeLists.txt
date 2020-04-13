cmake_minimum_required(VERSION 2.8.3)
project(real-time-grasp)

find_package(catkin REQUIRED COMPONENTS
  pcl_conversions
  pcl_ros
  message_generation
  roscpp
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
)

find_package( OpenCV REQUIRED )

catkin_package(
 CATKIN_DEPENDS message_runtime
)

###########
## Build ##
###########

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(main scripts/pcl_sim/main.cpp)
target_link_libraries(main ${catkin_LIBRARIES} ${OpenCV_LIBS})
add_dependencies(main pcl_sim_generate_messages_cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y -g -O3")