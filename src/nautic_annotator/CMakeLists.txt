cmake_minimum_required(VERSION 3.0.2)
project(nautic_annotator LANGUAGES CXX)

# ------------------------- Build options -------------------------
# Use C++17 for std::filesystem
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release if not set (catkin usually sets this via env/args)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ------------------------- Dependencies --------------------------
find_package(PCL 1.10 REQUIRED COMPONENTS common io filters kdtree segmentation)
find_package(OpenCV REQUIRED) # installed with ros-noetic
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  pcl_ros
  pcl_conversions
  cv_bridge
)

catkin_package(
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs pcl_ros pcl_conversions cv_bridge
  # INCLUDE_DIRS include          # uncomment if you add headers in include/
  # LIBRARIES nautic_annotator
)

add_definitions(${PCL_DEFINITIONS})

add_executable(${PROJECT_NAME}_node src/nautic_annotator_node_v2.cpp)
target_include_directories(${PROJECT_NAME}_node
  PRIVATE
    ${catkin_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
)

# Link stdc++fs for older GCCs that still require it (pre-9)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
  target_link_libraries(${PROJECT_NAME}_node stdc++fs)
endif()

add_dependencies(${PROJECT_NAME}_node ${catkin_EXPORTED_TARGETS})

target_compile_options(${PROJECT_NAME}_node
  PRIVATE
    -Wall -Wextra
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(${PROJECT_NAME}_node PRIVATE -O3)
endif()

install(TARGETS ${PROJECT_NAME}_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
