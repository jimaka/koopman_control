#pragma once

#include <vector>

#include "koopman_torch_model.hpp"
#include "mpc_config.hpp"

namespace koopman_mpc {

struct MpcTrajectory {
    std::vector<float> t;
    std::vector<std::array<float, 6>> state;
    std::vector<std::array<float, 4>> control;
    std::vector<std::array<float, 6>> ref_state;
    std::vector<float> cost_history;
};

struct TrackingMetrics {
    float xy_rmse_m = 0.f;
    float xy_max_m = 0.f;
    float yaw_rmse_deg = 0.f;
    float final_xy_err_m = 0.f;
};

class KoopmanMpcController {
public:
    KoopmanMpcController(KoopmanTorchModel model, MpcConfig cfg);

    std::pair<std::array<float, 4>, float> solveStep(
        const std::array<float, 6>& state0,
        const std::vector<std::array<float, 6>>& ref_window);

    MpcTrajectory simulate(const std::array<float, 6>& state0,
                             const std::vector<std::array<float, 6>>& ref_traj,
                             const std::vector<std::array<float, 4>>* ref_ctrl,
                             int max_steps);

private:
    torch::Tensor mpcCost(const torch::Tensor& state0,
                          const torch::Tensor& ref,
                          const torch::Tensor& u_flat,
                          const torch::Tensor& u_prev) const;

    torch::Tensor clampU(torch::Tensor u) const;

    KoopmanTorchModel model_;
    MpcConfig cfg_;
    torch::Tensor u_warm_;
    bool has_warm_{false};
};

TrackingMetrics computeMetrics(const MpcTrajectory& traj);

}  // namespace koopman_mpc
