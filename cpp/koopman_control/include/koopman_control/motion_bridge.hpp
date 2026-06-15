#pragma once

/**
 * @file motion_bridge.hpp
 * @brief motion.cpp 桥接：参考重采样 + OSQP 潜空间 MPC
 */

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "koopman_control/mpc_config.hpp"

namespace koopman_control {

struct MotionRefPoint {
    float x = 0.f;
    float y = 0.f;
    float psi = 0.f;
    float u = 0.f;
    float v = 0.f;
    float r = 0.f;
};

struct MotionBridgeConfig {
    float ref_dt = 1.0f;
    float ref_time_offset = 0.5f;
};

struct MotionSolveInput {
    float u = 0.f;
    float v = 0.f;
    float r = 0.f;
    std::array<float, 4> u_prev{};
    bool has_u_prev = false;
    std::vector<MotionRefPoint> ref;
};

struct MotionSolveTiming {
    double ref_resample_ms = 0.;
    double qp_solve_ms = 0.;
    double solve_step_ms = 0.;
    int osqp_iters = 0;
    int osqp_status = 0;
};

struct MotionSolveOutput {
    std::array<float, 4> control{};
    float cost = 0.f;
    int horizon = 0;
    MotionSolveTiming timing;
};

class KoopmanMotionMpc {
public:
    KoopmanMotionMpc(const std::string& latent_yaml_path, MpcConfig mpc_cfg = {},
                     MotionBridgeConfig bridge_cfg = {});
    ~KoopmanMotionMpc();

    KoopmanMotionMpc(const KoopmanMotionMpc&) = delete;
    KoopmanMotionMpc& operator=(const KoopmanMotionMpc&) = delete;

    bool solve(const MotionSolveInput& in, MotionSolveOutput& out);
    int horizon() const;
    void resetWarmStart();

private:
    std::vector<std::array<float, 6>> buildRefWindow(const MotionSolveInput& in) const;

    MotionBridgeConfig bridge_;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::vector<std::array<float, 6>> resampleMotionRefToHorizon(
    const std::vector<MotionRefPoint>& ref,
    int horizon,
    float mpc_dt,
    float ref_dt,
    float ref_time_offset);

}  // namespace koopman_control
