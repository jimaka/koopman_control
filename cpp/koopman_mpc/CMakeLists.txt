cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(koopman_mpc_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ONNX Runtime（由 build.sh 下载到 third_party/onnxruntime）
if(NOT DEFINED ONNXRUNTIME_ROOT)
    set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime")
endif()

set(ONNXRUNTIME_INCLUDE "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")

if(NOT EXISTS "${ONNXRUNTIME_INCLUDE}/onnxruntime_cxx_api.h")
    message(FATAL_ERROR
        "ONNX Runtime not found at ${ONNXRUNTIME_ROOT}. "
        "Run: bash cpp/koopman_mpc/build.sh")
endif()

set(KOOPMAN_MPC_SOURCES
    src/koopman_onnx_model.cpp
    src/mpc_controller.cpp
)

add_library(koopman_mpc_lib ${KOOPMAN_MPC_SOURCES})
target_include_directories(koopman_mpc_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ONNXRUNTIME_INCLUDE})
target_link_libraries(koopman_mpc_lib PUBLIC "${ONNXRUNTIME_LIB}")

add_executable(koopman_mpc_cpp src/main.cpp)
target_link_libraries(koopman_mpc_cpp PRIVATE koopman_mpc_lib)
set_property(TARGET koopman_mpc_cpp PROPERTY CXX_STANDARD 17)

add_executable(verify_rollout tools/verify_rollout.cpp)
target_link_libraries(verify_rollout PRIVATE koopman_mpc_lib)
set_property(TARGET verify_rollout PROPERTY CXX_STANDARD 17)

if(UNIX AND NOT APPLE)
    set_target_properties(koopman_mpc_cpp verify_rollout PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
        INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib")
endif()
