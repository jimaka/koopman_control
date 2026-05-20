cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(koopman_control LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ONNXRUNTIME_ROOT)
    set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../koopman_mpc/third_party/onnxruntime")
endif()

set(ONNXRUNTIME_INCLUDE "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")

find_package(yaml-cpp REQUIRED)

set(KOOPMAN_CONTROL_SOURCES
    src/koopman_onnx_model.cpp
    src/mpc_controller.cpp
    src/mpc_config_loader.cpp
    src/motion_bridge.cpp
)

add_library(koopman_control ${KOOPMAN_CONTROL_SOURCES})
target_include_directories(koopman_control
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${ONNXRUNTIME_INCLUDE})
target_link_libraries(koopman_control PUBLIC "${ONNXRUNTIME_LIB}")

target_link_libraries(koopman_control PUBLIC yaml-cpp)

if(UNIX AND NOT APPLE)
    set_target_properties(koopman_control PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
        INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib")
endif()

install(TARGETS koopman_control
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib)
install(DIRECTORY include/koopman_control
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp")
