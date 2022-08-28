cmake_minimum_required(VERSION 3.14.7)

project(tflite_build)

include(FetchContent)
FetchContent_Declare(tensorflow_lite_micro URL "https://github.com/tensorflow/tflite-micro/archive/refs/heads/main.zip")
FetchContent_MakeAvailable(tensorflow_lite_micro)
FetchContent_GetProperties(tensorflow_lite_micro)

set(FLATBUFFERS_INSTALL FALSE CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATC FALSE CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_TESTS FALSE CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATHASH FALSE CACHE BOOL "" FORCE)
FetchContent_Declare(flatbuffers URL "https://github.com/google/flatbuffers/archive/refs/tags/v2.0.6.zip")
FetchContent_MakeAvailable(flatbuffers)

add_executable(run_inference
  main.cpp
  ${tensorflow_lite_micro_SOURCE_DIR}/tensorflow/lite/micro/all_ops_resolver.cc)

target_include_directories(run_inference
  PRIVATE
    ${tensorflow_lite_micro_SOURCE_DIR})

target_link_libraries(run_inference
  PRIVATE
    flatbuffers)
