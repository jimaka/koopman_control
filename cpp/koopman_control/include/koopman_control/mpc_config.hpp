#pragma once

#include <array>

namespace koopman_control {

struct MpcConfig {
    int horizon = 20;
    int opt_control_steps = 2;
    float dt = 1.0f;
    int control_hold_steps = 1;

    float w_z = 1.f;
    float w_u = 1e-4f;
    float w_du = 0.05f;

    // Tier-2 位姿跟踪（>0 时启用；需 latent YAML 含 decoder）
    float w_xy = 0.f;
    float w_yaw = 0.f;
    int sqp_iters = 2;

    std::array<float, 4> du_max{0.f, 0.f, 0.f, 0.f};
    float throttle_du_max = 0.f;
    float rudder_du_max = 0.f;

    float osqp_eps_abs = 1e-4f;
    float osqp_eps_rel = 1e-4f;
    int osqp_max_iter = 4000;
    int osqp_verbose = 0;

    std::array<float, 4> u_min{-100.f, -35.f, -100.f, -35.f};
    std::array<float, 4> u_max{100.f, 35.f, 100.f, 35.f};

    /** v4 潜空间权重 YAML（export_v4_encode_weights.py 生成） */
    std::string latent_model = "cpp/koopman_mpc/weights/koopman_v4_latent.yaml";
    /** 闭环仿真 plant（仅 demo / simulate 使用） */
    std::string onnx_plant = "cpp/koopman_mpc/weights/koopman_rollout.onnx";
};

}  // namespace koopman_control
