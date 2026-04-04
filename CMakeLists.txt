cmake_minimum_required(VERSION 3.29)
project(Mathematical-morphology-point-cloud)

set(CMAKE_CXX_STANDARD 20)
option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

find_package(CGAL REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(LASlib CONFIG REQUIRED)

add_executable(Mathematical-morphology-point-cloud
        src/main.cpp
        src/point_cloud_dilation.cpp
        src/gpu_spatial_hash.cpp
        src/grouping.cpp
        src/point_cloud_erosion.cpp
        src/point_cloud_addition.cpp
        src/density_estimation.cpp
        src/point_cloud_dilation_density.cpp)

target_compile_definitions(Mathematical-morphology-point-cloud PRIVATE
        CGAL_LINKED_WITH_LASLIB
        NOMINMAX)

target_link_libraries(Mathematical-morphology-point-cloud
        OpenGL::GL
        GLEW::GLEW
        CGAL::CGAL
        glfw
        LASlib)