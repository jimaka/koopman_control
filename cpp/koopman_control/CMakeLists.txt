# =============================================================================
# Koopman MPC 控制库 CMake 构建脚本
# 产物：静态/动态库 koopman_control，供 koopman_mpc demo 与 ship_control 链接
# =============================================================================

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(koopman_control LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ONNX Runtime 默认路径（可由 -DONNXRUNTIME_ROOT= 覆盖）
if(NOT DEFINED ONNXRUNTIME_ROOT)
    set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../koopman_mpc/third_party/onnxruntime")
endif()

set(ONNXRUNTIME_INCLUDE "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")

find_package(yaml-cpp REQUIRED)

set(KOOPMAN_CONTROL_SOURCES
    src/koopman_onnx_model.cpp   # ONNX rollout 推理
    src/mpc_controller.cpp       # MPC 优化与仿真
    src/mpc_config_loader.cpp    # YAML 配置加载
    src/motion_bridge.cpp        # motion.cpp 桥接
    src/koopman_latent_model.cpp # v4 潜空间矩阵 / condensed 预测
    src/koopman_encode.cpp       # v4 dict16 encoder
    src/latent_mpc_qp.cpp        # Tier-1 潜空间 QP
    src/mpc_controller_qp.cpp    # QP-MPC 控制器
)

option(KOOPMAN_BUILD_QP_VERIFY "Build verify_latent_qp tool" ON)

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

if(KOOPMAN_BUILD_QP_VERIFY)
    add_executable(verify_latent_qp tools/verify_latent_qp.cpp)
    target_link_libraries(verify_latent_qp PRIVATE koopman_control)
endif()
