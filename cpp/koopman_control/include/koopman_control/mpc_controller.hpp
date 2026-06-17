#pragma once

/**
 * @file mpc_controller.hpp
 * @brief v4 潜空间 OSQP-MPC 控制器（唯一 MPC 求解路径）
 */

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "koopman_control/koopman_decoder.hpp"
#include "koopman_control/koopman_encode.hpp"
#include "koopman_control/koopman_latent_model.hpp"
#include "koopman_control/latent_mpc_qp.hpp"
#include "koopman_control/mpc_config.hpp"

// KOOPMAN_ENABLE_ONNX=0 时编译「仅 MPC（OSQP）」版本：不依赖 ONNX Runtime，
// 不提供 ONNX plant 构造与 simulate()。默认 1（含 ONNX，供闭环仿真 / demo）。
#ifndef KOOPMAN_ENABLE_ONNX
#define KOOPMAN_ENABLE_ONNX 1
#endif
#if KOOPMAN_ENABLE_ONNX
#include "koopman_control/koopman_onnx_model.hpp"
#endif

namespace koopman_control {

struct MpcTrajectory {
    std::vector<float> t;
    std::vector<std::array<float, 6>> state;
    std::vector<std::array<float, 4>> control;
    std::vector<std::array<float, 6>> ref_state;
    std::vector<float> cost_history;
};

struct MpcSolveTiming {
    double qp_setup_ms{0.};
    double qp_solve_ms{0.};
    int osqp_iters{0};
    int osqp_status{0};
};

struct TrackingMetrics {
    float xy_rmse_m{0.f};
    float xy_max_m{0.f};
    float yaw_rmse_deg{0.f};
    float final_xy_err_m{0.f};
};

class KoopmanMpcController {
public:
    KoopmanMpcController(std::string latent_yaml_path, MpcConfig cfg,
                       LatentMpcQpConfig qp_cfg = {});

#if KOOPMAN_ENABLE_ONNX
    /** 闭环仿真需 ONNX 作为被控对象（plant）；MPC 优化仅用 OSQP + 潜空间矩阵 */
    KoopmanMpcController(std::string latent_yaml_path, std::string onnx_plant_path, MpcConfig cfg,
                       LatentMpcQpConfig qp_cfg = {});
#endif

    std::pair<std::array<float, 4>, float> solveStep(
        const std::array<float, 6>& state0,
        const std::vector<std::array<float, 6>>& ref_window,
        const std::array<float, 4>* u_prev_applied = nullptr,
        MpcSolveTiming* timing = nullptr);

#if KOOPMAN_ENABLE_ONNX
    MpcTrajectory simulate(const std::array<float, 6>& state0,
                           const std::vector<std::array<float, 6>>& ref_traj,
                           const std::vector<std::array<float, 4>>* ref_ctrl,
                           int max_steps);
#endif

    int horizon() const { return cfg_.horizon; }
    const MpcConfig& config() const { return cfg_; }
    void resetWarmStart();

private:
    MpcConfig cfg_;
    LatentMpcQpConfig qp_cfg_;
    KoopmanLatentModel model_;
    KoopmanEncoder encoder_;
    KoopmanDecoder decoder_;
    LatentMpcQpSolver solver_;
#if KOOPMAN_ENABLE_ONNX
    std::unique_ptr<KoopmanOnnxModel> plant_;
#endif

    std::vector<float> u_warm_tilde_;
    bool has_warm_{false};

    bool poseTrackingEnabled() const;
    std::array<float, 3> normalizeDyn(const std::array<float, 3>& dyn) const;
    std::vector<float> buildRefLatentStack(const std::vector<std::array<float, 6>>& ref_window) const;
    std::vector<float> buildRefPoseStack(const std::vector<std::array<float, 6>>& ref_window) const;
};

TrackingMetrics computeMetrics(const MpcTrajectory& traj);

}  // namespace koopman_control
