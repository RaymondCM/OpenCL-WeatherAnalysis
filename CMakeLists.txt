#Root CMake file. Declared shared dynamic libs here
cmake_minimum_required(VERSION 3.6.0)
project(AssignmentOne)

#Use C++11
set (CMAKE_CXX_STANDARD 11)

#Add root to be used in the source code
add_definitions(-DPROJECT_ROOT="${CMAKE_SOURCE_DIR}")  

#Find OpenCL dependancy for Parallel
find_package(OpenCL REQUIRED)
MESSAGE( STATUS "OpenCL_INCLUDE_DIR: " ${OpenCL_INCLUDE_DIR})
MESSAGE( STATUS "OpenCL_LIBRARIES: " ${OpenCL_LIBRARIES})

#Add all source files
add_executable(AssignmentOne "main.cpp" "Utils.hpp")

#Include target specific include directories
target_include_directories(AssignmentOne PUBLIC ${OpenCL_INCLUDE_DIR})

        
#Link library files
target_link_libraries(AssignmentOne ${OpenCL_LIBRARIES} )