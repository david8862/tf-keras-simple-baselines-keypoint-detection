cmake_minimum_required(VERSION 3.5)
project(simplebaselineKeypoint LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CMAKE_BUILD_TYPE "Release")

#### specify the compiler flag
SET(CMAKE_CXX_FLAGS  "-std=c++14 -O2")

set(SIMPLEBASELINE_KEYPOINT_SRC
        simplebaselineKeypoint.cpp)

#set(MNN_ROOT_PATH /mnt/d/Projects/MNN)
link_directories("${MNN_ROOT_PATH}/build/")

add_executable(simplebaselineKeypoint ${SIMPLEBASELINE_KEYPOINT_SRC})

target_include_directories(simplebaselineKeypoint PRIVATE ${MNN_ROOT_PATH}/include/ ${MNN_ROOT_PATH}/3rd_party/imageHelper/)
target_link_libraries(simplebaselineKeypoint PRIVATE -lMNN -lstdc++ -lpthread)
#target_link_libraries(simplebaselineKeypoint PRIVATE libMNN.a -Wl,--whole-archive -Wl,--no-whole-archive -lstdc++ -lpthread)
