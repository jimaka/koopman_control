#pragma once

/**
 * @file latent_mpc_qp.hpp
 * @brief Tier-1 潜空间 condensed QP：H,f 组装 + 投影梯度求解（盒约束）。
 */

#include <array>
#include <vector>

#include "koopman_control/koopman_latent_model.hpp"
#include "koopman_control/mpc_config.hpp"

namespace koopman_control {

struct LatentMpcQpConfig {
    float w_z{1.f};       ///< Q_z 对角权重（潜空间跟踪）
    float w_u{1e-4f};     ///< R 对角权重（归一化控制）
    float w_du{0.05f};    ///< S 对角权重（归一化控制增量）
    int max_iters{200};
    float step_size{0.05f};
    float tol{1e-5f};
};

struct LatentMpcQpSolution {
    std::vector<float> u_tilde_stack;  ///< 长度 nu * n_opt
    float cost{0.f};
    int iters{0};
};

/** 构造并求解 Tier-1 盒约束 QP（决策变量为归一化控制序列） */
class LatentMpcQpSolver {
public:
    LatentMpcQpSolver(const KoopmanLatentModel& model, MpcConfig mpc_cfg, LatentMpcQpConfig qp_cfg);

    /**
     * @param z0 当前潜状态 (48,)
     * @param z_ref_stack 参考潜状态堆叠 (48*N,)
     * @param u_prev_phys 上一步物理控制
     * @param u_init_tilde_stack 可选 warm-start（归一化）
     */
    LatentMpcQpSolution solve(const std::vector<float>& z0,
                               const std::vector<float>& z_ref_stack,
                               const std::array<float, 4>& u_prev_phys,
                               const std::vector<float>* u_init_tilde_stack = nullptr) const;

    int numDecisionVars() const { return nu_ * n_opt_; }

private:
    const KoopmanLatentModel& model_;
    MpcConfig mpc_cfg_;
    LatentMpcQpConfig qp_cfg_;

    int n_{20};
    int nz_{48};
    int nu_{4};
    int n_opt_{20};
    int hold_{1};

    mutable detail::Matrix H_;
    mutable detail::Matrix D_du_;
    mutable bool mats_ready_{false};

    void ensureMats() const;
    void buildHessian() const;

    std::vector<float> expandToFull(const std::vector<float>& u_opt) const;
    void projectBox(std::vector<float>& u_tilde_full,
                    const std::array<float, 4>& u_prev_phys) const;
    float evalCost(const std::vector<float>& z0,
                   const std::vector<float>& z_ref_stack,
                   const std::vector<float>& u_tilde_full,
                   const std::array<float, 4>& u_prev_phys) const;
    std::vector<float> gradCost(const std::vector<float>& z0,
                                const std::vector<float>& z_ref_stack,
                                const std::vector<float>& u_tilde_full,
                                const std::array<float, 4>& u_prev_phys) const;
};

}  // namespace koopman_control
