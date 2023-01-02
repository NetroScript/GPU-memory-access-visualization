cmake_minimum_required(VERSION 3.24)

#set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_CUDA_ARCHITECTURES 61)


project(GPU_memory_access_visualization LANGUAGES CUDA)



include_directories(.)
include_directories(src)


add_executable(GPU_memory_access_visualization
        src/cuda_mav.cu src/cuda_mav.h)
