cmake_minimum_required(VERSION 3.24)

option(ENABLE_CUDA_DEVICE_DEBUG "Enable generating CUDA device code debug information." OFF)

set(_TARGET_NORMAL normal)

add_executable(${_TARGET_NORMAL})
target_sources(${_TARGET_NORMAL}
   PRIVATE
   normal.cu)
set_target_properties(${_TARGET_NORMAL} PROPERTIES
  CUDA_CXX_STANDARD 17
)
if(ENABLE_CUDA_DEVICE_DEBUG)
  target_compile_options(${_TARGET_NORMAL} PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
endif()

# add include path to the cuda_mav.cuh header
target_link_libraries(${_TARGET_NORMAL} PRIVATE CUDA_MAV_HEADER)

set(_TARGET_RANDOM_2D_MATRIX random2dMatrix)

add_executable(${_TARGET_RANDOM_2D_MATRIX})
target_sources(${_TARGET_RANDOM_2D_MATRIX}
   PRIVATE
   random2dMatrix.cu)
set_target_properties(${_TARGET_RANDOM_2D_MATRIX} PROPERTIES
  CUDA_CXX_STANDARD 17
)

set(_TARGET_REDUCE reduce)

add_executable(${_TARGET_REDUCE})
target_sources(${_TARGET_REDUCE}
   PRIVATE
   reduce.cu)
set_target_properties(${_TARGET_REDUCE} PROPERTIES
  CXX_STANDARD 17
  CUDA_CXX_STANDARD 17
)
if(ENABLE_CUDA_DEVICE_DEBUG)
  target_compile_options(${_TARGET_ANNOTATED} PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
endif()
