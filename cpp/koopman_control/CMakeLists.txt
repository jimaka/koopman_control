# =============================================================================
# Koopman MPC 控制库 CMake 构建脚本
# =============================================================================

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(koopman_control LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 是否启用 ONNX（闭环仿真 plant / demo）。OFF 时编译「仅 MPC（OSQP）」核心，
# 不依赖 ONNX Runtime，适合只跑 MPC 优化、规避 ORT 下载。
option(KOOPMAN_ENABLE_ONNX "Build with ONNX Runtime plant (simulate/demo)" ON)

if(NOT DEFINED ONNXRUNTIME_ROOT)
    set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../koopman_mpc/third_party/onnxruntime")
endif()

set(ONNXRUNTIME_INCLUDE "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")

find_package(yaml-cpp REQUIRED)

include(FetchContent)
set(OSQP_BUILD_SHARED_LIB OFF CACHE BOOL "" FORCE)
set(OSQP_BUILD_STATIC_LIB ON CACHE BOOL "" FORCE)
set(OSQP_ENABLE_MATLAB OFF CACHE BOOL "" FORCE)
set(OSQP_ENABLE_R_LANGUAGE OFF CACHE BOOL "" FORCE)
set(OSQP_ENABLE_INTERRUPT OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    osqp
    GIT_REPOSITORY https://github.com/osqp/osqp.git
    GIT_TAG v0.6.3
)
FetchContent_MakeAvailable(osqp)

# MPC 核心源（不含 ONNX）
set(KOOPMAN_CONTROL_SOURCES
    src/mpc_controller.cpp
    src/mpc_config_loader.cpp
    src/motion_bridge.cpp
    src/koopman_latent_model.cpp
    src/koopman_encode.cpp
    src/koopman_decoder.cpp
    src/pose_linearize.cpp
    src/latent_mpc_qp.cpp
)
if(KOOPMAN_ENABLE_ONNX)
    list(APPEND KOOPMAN_CONTROL_SOURCES src/koopman_onnx_model.cpp)
endif()

option(KOOPMAN_BUILD_QP_VERIFY "Build verify_latent_qp tool" ON)

add_library(koopman_control ${KOOPMAN_CONTROL_SOURCES})
target_include_directories(koopman_control
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${osqp_SOURCE_DIR}/include)
target_link_libraries(koopman_control PUBLIC yaml-cpp osqpstatic)

if(KOOPMAN_ENABLE_ONNX)
    target_compile_definitions(koopman_control PUBLIC KOOPMAN_ENABLE_ONNX=1)
    target_include_directories(koopman_control PUBLIC ${ONNXRUNTIME_INCLUDE})
    target_link_libraries(koopman_control PUBLIC "${ONNXRUNTIME_LIB}")
    if(UNIX AND NOT APPLE)
        set_target_properties(koopman_control PROPERTIES
            BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
            INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib")
    endif()
else()
    target_compile_definitions(koopman_control PUBLIC KOOPMAN_ENABLE_ONNX=0)
    message(STATUS "koopman_control: ONNX disabled (MPC-only build)")
endif()

install(TARGETS koopman_control
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib)
install(DIRECTORY include/koopman_control
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp")

if(KOOPMAN_BUILD_QP_VERIFY)
    add_executable(verify_latent_qp tools/verify_latent_qp.cpp)
    target_link_libraries(verify_latent_qp PRIVATE koopman_control)

    add_executable(verify_pose_linearize tools/verify_pose_linearize.cpp)
    target_link_libraries(verify_pose_linearize PRIVATE koopman_control)
endif()
