#pragma once

#include <array>
#include <string>

namespace koopman_mpc {

struct MpcConfig {
    int horizon = 20;
    float dt = 0.1f;
    float w_xy = 10.f;
    float w_yaw = 5.f;
    float w_vel = 0.5f;
    float w_u = 1e-4f;
    float w_du = 0.05f;
    int opt_iters = 40;
    float opt_lr = 0.08f;
    std::array<float, 4> u_min{-100.f, -35.f, -100.f, -35.f};
    std::array<float, 4> u_max{100.f, 35.f, 100.f, 35.f};
};

}  // namespace koopman_mpc
