cmake_minimum_required(VERSION 3.24)

#set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_CUDA_ARCHITECTURES 61)


project(GPU_memory_access_visualization LANGUAGES CUDA)

# define header only target
add_library(CUDA_MAV_HEADER INTERFACE)
target_include_directories(CUDA_MAV_HEADER INTERFACE ${PROJECT_SOURCE_DIR}/include)

# the visualization function is no standalone application, therefore we test it with some examples
add_subdirectory(examples/basic)
