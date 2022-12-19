cmake_minimum_required(VERSION 3.18)
project(ranges LANGUAGES CUDA)

set(_TARGET_NORMAL normal)

add_executable(${_TARGET_NORMAL})
target_sources(${_TARGET_NORMAL}
   PRIVATE
   normal.cu)
set_target_properties(${_TARGET_NORMAL} PROPERTIES
  CUDA_CXX_STANDARD 17
)


set(_TARGET_ANNOTATED annotated)

add_executable(${_TARGET_ANNOTATED})
target_sources(${_TARGET_ANNOTATED}
   PRIVATE
   annotated.cu)
set_target_properties(${_TARGET_ANNOTATED} PROPERTIES
  CUDA_CXX_STANDARD 17
)
