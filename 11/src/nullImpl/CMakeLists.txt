cmake_minimum_required(VERSION 2.8)

set(PROJECT_NAME "nullImpl")
project(${PROJECT_NAME})

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
include_directories (${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include)
# include_directories (${CMAKE_CURRENT_SOURCE_DIR}/../include)

# Configure built shared libraries in top-level lib directory
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../lib)

message(${CMAKE_SOURCE_DIR})

include_directories(${CMAKE_SOURCE_DIR}/../src/include)
include_directories(${CMAKE_SOURCE_DIR}/../src/include/opencv)
include_directories(${CMAKE_SOURCE_DIR}/../src/include/opencv2)

link_directories(${CMAKE_SOURCE_DIR}/lib)

# aux_source_directory(${CMAKE_SOURCE_DIR}/src/nullImpl SRC)
aux_source_directory(${CMAKE_SOURCE_DIR} SRC)

# add_definitions(-D__cplusplus)

# Build the shared libraries
# add_library (frvt_11_via_001 SHARED ${SRC})

# target_link_libraries (frvt_11_via_001
# ${CMAKE_SOURCE_DIR}/lib/libncnn.a
# opencv_core
# opencv_highgui
# opencv_imgproc
# )

add_executable (${PROJECT_NAME} ${SRC})

find_package(OpenCV 3.4.6 REQUIRED)

target_link_libraries (${PROJECT_NAME} PRIVATE
${CMAKE_SOURCE_DIR}/../../lib/libncnn.so
# ${CMAKE_SOURCE_DIR}/../../lib/libncnn.a
# ${CMAKE_SOURCE_DIR}/../../lib/libGenericCodeGen.a
# ${CMAKE_SOURCE_DIR}/../../lib/libglslang.a
# ${CMAKE_SOURCE_DIR}/../../lib/libMachineIndependent.a
# ${CMAKE_SOURCE_DIR}/../../lib/libOGLCompiler.a
# ${CMAKE_SOURCE_DIR}/../../lib/libOSDependent.a
# ${CMAKE_SOURCE_DIR}/../../lib/libSPIRV.a
opencv_core
opencv_highgui
opencv_imgproc
opencv_imgcodecs
-lpthread -lm
-lgomp
)
