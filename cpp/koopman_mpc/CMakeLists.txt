cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(koopman_mpc_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# LibTorch（pip install torch 后可用）
find_package(Torch REQUIRED)

set(KOOPMAN_MPC_SOURCES
    src/koopman_torch_model.cpp
    src/mpc_controller.cpp
)

add_library(koopman_mpc_lib ${KOOPMAN_MPC_SOURCES})
target_include_directories(koopman_mpc_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(koopman_mpc_lib PUBLIC "${TORCH_LIBRARIES}")
target_compile_definitions(koopman_mpc_lib PUBLIC ${TORCH_CXX_FLAGS})
set_property(TARGET koopman_mpc_lib PROPERTY CXX_STANDARD 17)

add_executable(koopman_mpc_cpp src/main.cpp)
target_link_libraries(koopman_mpc_cpp PRIVATE koopman_mpc_lib)
set_property(TARGET koopman_mpc_cpp PROPERTY CXX_STANDARD 17)

add_executable(verify_rollout tools/verify_rollout.cpp)
target_link_libraries(verify_rollout PRIVATE koopman_mpc_lib)
set_property(TARGET verify_rollout PROPERTY CXX_STANDARD 17)

# LibTorch 需要与系统 ABI 一致
if (UNIX AND NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${TORCH_CXX_FLAGS}")
endif()
