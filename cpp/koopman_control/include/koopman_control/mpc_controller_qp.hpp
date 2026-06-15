#pragma once

/**
 * @file mpc_controller_qp.hpp
 * @brief v4 潜空间 Tier-1 QP-MPC 控制器（与 KoopmanMpcController API 对齐）。
 */

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "koopman_control/koopman_encode.hpp"
#include "koopman_control/koopman_latent_model.hpp"
#include "koopman_control/latent_mpc_qp.hpp"
#include "koopman_control/mpc_config.hpp"

namespace koopman_control {

class KoopmanQpMpcController {
public:
    KoopmanQpMpcController(std::string latent_yaml_path, MpcConfig cfg, LatentMpcQpConfig qp_cfg = {});

    int horizon() const { return cfg_.horizon; }
    const MpcConfig& config() const { return cfg_; }

    std::pair<std::array<float, 4>, float> solveStep(
        const std::array<float, 6>& state0,
        const std::vector<std::array<float, 6>>& ref_window,
        const std::array<float, 4>* u_prev_applied = nullptr);

    void resetWarmStart();

private:
    MpcConfig cfg_;
    LatentMpcQpConfig qp_cfg_;
    KoopmanLatentModel model_;
    KoopmanEncoder encoder_;
    LatentMpcQpSolver solver_;

    std::vector<float> u_warm_tilde_;
    bool has_warm_{false};

    std::array<float, 3> normalizeDyn(const std::array<float, 3>& dyn) const;
    std::vector<float> buildRefLatentStack(const std::vector<std::array<float, 6>>& ref_window) const;
};

}  // namespace koopman_control
