cmake_minimum_required(VERSION 2.8)
project(ParallelStatic)
find_package(OpenCV REQUIRED)
find_package(MPI REQUIRED)
add_executable(ParallelStatic ParallelStatic.cpp)
target_link_libraries(ParallelStatic ${OpenCV_LIBS})
include_directories(${MPI_INCLUDE_PATH})
target_link_libraries(ParallelStatic ${MPI_LIBRARIES})
if(MPI_COMPILE_FLAGS)
  set_target_properties(ParallelStatic PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(ParallelStatic PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
