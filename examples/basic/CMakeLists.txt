cmake_minimum_required(VERSION 3.24)

set(_TARGET_NORMAL normal)

add_executable(${_TARGET_NORMAL})
target_sources(${_TARGET_NORMAL}
   PRIVATE
   normal.cu)
set_target_properties(${_TARGET_NORMAL} PROPERTIES
  CUDA_CXX_STANDARD 17
)

# add include path to the cuda_mav.cuh header
target_link_libraries(${_TARGET_NORMAL} PRIVATE CUDA_MAV_HEADER)

set(_TARGET_ANNOTATED annotated)

add_executable(${_TARGET_ANNOTATED})
target_sources(${_TARGET_ANNOTATED}
   PRIVATE
   annotated.cu)
set_target_properties(${_TARGET_ANNOTATED} PROPERTIES
  CUDA_CXX_STANDARD 17
)

set(_TARGET_RANDOM_2D_MATRIX random2dMatrix)

add_executable(${_TARGET_RANDOM_2D_MATRIX})
target_sources(${_TARGET_RANDOM_2D_MATRIX}
   PRIVATE
   random2dMatrix.cu)
set_target_properties(${_TARGET_RANDOM_2D_MATRIX} PROPERTIES
  CUDA_CXX_STANDARD 17
)
