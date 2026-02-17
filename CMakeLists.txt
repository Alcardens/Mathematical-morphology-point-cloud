cmake_minimum_required(VERSION 3.29)
project(Mathematical-morphology-point-cloud)

set(CMAKE_CXX_STANDARD 20)

find_package(CGAL REQUIRED)
add_executable(Mathematical-morphology-point-cloud src/dilation.cpp
        src/grouping.cpp
        src/grouping.h)
target_link_libraries(Mathematical-morphology-point-cloud CGAL::CGAL)