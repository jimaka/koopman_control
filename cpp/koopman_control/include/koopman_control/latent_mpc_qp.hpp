#pragma once

/**
 * @file latent_mpc_qp.hpp
 * @brief Tier-1 潜空间 condensed QP（OSQP 求解）。
 */

#include <array>
#include <vector>

#include "koopman_control/koopman_latent_model.hpp"
#include "koopman_control/mpc_config.hpp"
#include "koopman_control/pose_linearize.hpp"

namespace koopman_control {

struct LatentMpcQpConfig {
    float w_z{1.f};
    float w_u{1e-4f};
    float w_du{0.05f};
    float osqp_eps_abs{1e-4f};
    float osqp_eps_rel{1e-4f};
    int osqp_max_iter{4000};
    int osqp_verbose{0};
};

struct LatentMpcQpSolution {
    std::vector<float> u_tilde_stack;
    float cost{0.f};
    int osqp_status{0};
    int osqp_iters{0};
};

/** 潜空间盒约束 + 变化率约束 QP，OSQP 求解 */
class LatentMpcQpSolver {
public:
    LatentMpcQpSolver(const KoopmanLatentModel& model, MpcConfig mpc_cfg, LatentMpcQpConfig qp_cfg);

    LatentMpcQpSolution solve(const std::vector<float>& z0,
                               const std::vector<float>& z_ref_stack,
                               const std::array<float, 4>& u_prev_phys,
                               const std::vector<float>* u_init_tilde_stack = nullptr,
                               const PoseLinearization* pose = nullptr) const;

    int numDecisionVars() const { return n_ * nu_; }

private:
    const KoopmanLatentModel& model_;
    MpcConfig mpc_cfg_;
    LatentMpcQpConfig qp_cfg_;

    int n_{20};
    int nz_{48};
    int nu_{4};
    int hold_{1};
    int n_opt_{20};

    mutable detail::Matrix H_;
    mutable bool mats_ready_{false};

    void ensureMats() const;
    void buildHessian() const;

    std::vector<float> expandToFull(const std::vector<float>& u_opt) const;
    float evalCost(const std::vector<float>& z0,
                   const std::vector<float>& z_ref_stack,
                   const std::vector<float>& u_tilde_full) const;
};

}  // namespace koopman_control
