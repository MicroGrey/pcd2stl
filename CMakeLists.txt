cmake_minimum_required(VERSION 3.0.2)
project(pcd2stl)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PCL REQUIRED COMPONENTS visualization)
find_package(VTK REQUIRED)

add_executable(pcd2stl src/pcd2stl.cpp)

target_link_libraries(pcd2stl 
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES} 
  ${VTK_LIBRARIES}
  pcl_surface
  pcl_features )