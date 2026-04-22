cmake_minimum_required(VERSION 3.29)
project(Mathematical-morphology-point-cloud)

set(CMAKE_CXX_STANDARD 20)
option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

find_package(CGAL       REQUIRED)
find_package(OpenGL     REQUIRED)
find_package(GLEW       REQUIRED)
find_package(glfw3      REQUIRED)
find_package(LASlib     CONFIG REQUIRED)
find_package(TBB        REQUIRED)
find_package(pybind11   CONFIG REQUIRED)

set(MORPHOLOGY_SOURCES
        src/point_cloud_dilation.cpp
        src/gpu_spatial_hash.cpp
        src/grouping.cpp
        src/point_cloud_erosion.cpp
        src/point_cloud_addition.cpp
        src/density_estimation.cpp
        src/point_cloud_dilation_density.cpp
        src/point_cloud_intersection.cpp
        src/point_cloud_erosion_density.cpp
        src/point_cloud_dilation_orientation.cpp
        src/point_cloud_erosion_orientation.cpp
        src/point_cloud_erosion_orientation_brute.cpp)

set(MORPHOLOGY_LIBS
        OpenGL::GL GLEW::GLEW CGAL::CGAL glfw
        LASlib TBB::tbb)

set(MORPHOLOGY_DEFS
        CGAL_LINKED_WITH_LASLIB NOMINMAX)

# Executable
add_executable(Mathematical-morphology-point-cloud
        src/main.cpp ${MORPHOLOGY_SOURCES})
target_link_libraries(Mathematical-morphology-point-cloud PRIVATE ${MORPHOLOGY_LIBS})
target_compile_definitions(Mathematical-morphology-point-cloud PRIVATE ${MORPHOLOGY_DEFS})

# Python module
pybind11_add_module(morphology src/bindings.cpp ${MORPHOLOGY_SOURCES})
target_link_libraries(morphology PRIVATE ${MORPHOLOGY_LIBS})
target_compile_definitions(morphology PRIVATE ${MORPHOLOGY_DEFS})