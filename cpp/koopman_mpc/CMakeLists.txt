cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(koopman_mpc_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../koopman_control
                 ${CMAKE_CURRENT_BINARY_DIR}/koopman_control)

add_executable(koopman_mpc_cpp src/main.cpp)
target_link_libraries(koopman_mpc_cpp PRIVATE koopman_control)
set_property(TARGET koopman_mpc_cpp PROPERTY CXX_STANDARD 17)

add_executable(verify_rollout tools/verify_rollout.cpp)
target_link_libraries(verify_rollout PRIVATE koopman_control)
set_property(TARGET verify_rollout PROPERTY CXX_STANDARD 17)

if(UNIX AND NOT APPLE)
    set(ORT_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/lib")
    set_target_properties(koopman_mpc_cpp verify_rollout PROPERTIES
        BUILD_RPATH "${ORT_LIB_DIR}"
        INSTALL_RPATH "${ORT_LIB_DIR}")
endif()
