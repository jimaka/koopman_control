#pragma once

#include <array>

namespace koopman_control {

struct MpcConfig {
    /** 与 ONNX rollout 步数一致（v4 20s 模型为 200） */
    int horizon = 200;
    /** 仅优化前 N 步控制量，后续步零阶保持 */
    int opt_control_steps = 40;
    float dt = 0.1f;
    float w_xy = 10.f;
    float w_yaw = 5.f;
    float w_vel = 0.5f;
    float w_u = 1e-4f;
    float w_du = 0.05f;
    int opt_iters = 15;
    float opt_lr = 0.05f;
    std::array<float, 4> u_min{-100.f, -35.f, -100.f, -35.f};
    std::array<float, 4> u_max{100.f, 35.f, 100.f, 35.f};
};

}  // namespace koopman_control
