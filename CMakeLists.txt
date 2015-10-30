# University of Central Florida Lunar Knights!

# got info for this stuff from here:
# http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_gcc_cmake/linux_gcc_cmake.html
cmake_minimum_required(VERSION 2.8)
# got this line from StackOverflow, using Modern C++ practices as much as possible
set(CMAKE_CXX_STANDARD 11)
project( LKtelemetry )
find_package( OpenCV REQUIRED )
add_executable( LKtelemetry main.cc )
target_link_libraries( LKtelemetry ${OpenCV_LIBS} )
