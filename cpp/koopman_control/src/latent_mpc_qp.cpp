/**
 * @file latent_mpc_qp.cpp
 */

#include "koopman_control/latent_mpc_qp.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace koopman_control {

LatentMpcQpSolver::LatentMpcQpSolver(const KoopmanLatentModel& model, MpcConfig mpc_cfg, LatentMpcQpConfig qp_cfg)
    : model_(model), mpc_cfg_(mpc_cfg), qp_cfg_(qp_cfg) {
    n_ = model.horizon();
    nz_ = model.nz();
    nu_ = model.nu();
    hold_ = std::max(1, mpc_cfg_.control_hold_steps);
    if (n_ % hold_ != 0) {
        throw std::runtime_error("horizon must be divisible by control_hold_steps");
    }
    const int n_blk = n_ / hold_;
    int opt_blk = (mpc_cfg_.opt_control_steps + hold_ - 1) / hold_;
    opt_blk = std::max(1, std::min(n_blk, opt_blk));
    n_opt_ = opt_blk * hold_;  // 细步数上实际优化的步数
}

void LatentMpcQpSolver::ensureMats() const {
    if (mats_ready_) {
        return;
    }
    buildHessian();
    mats_ready_ = true;
}

void LatentMpcQpSolver::buildHessian() const {
    const int n = n_;
    const int nz = nz_;
    const int nu = nu_;

    const detail::Matrix& Theta = model_.Theta();
    const detail::Matrix ThetaT = detail::Matrix::transpose(Theta);

    detail::Matrix Q(nz * n, nz * n, 0.f);
    for (int i = 0; i < nz * n; ++i) {
        Q(i, i) = qp_cfg_.w_z;
    }

    const detail::Matrix H_theta = detail::Matrix::matmul(detail::Matrix::matmul(ThetaT, Q), Theta);

    H_ = detail::Matrix(nu * n, nu * n, 0.f);
    for (int r = 0; r < nu * n; ++r) {
        for (int c = 0; c < nu * n; ++c) {
            H_(r, c) = H_theta(r, c);
        }
    }
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < nu; ++j) {
            H_(k * nu + j, k * nu + j) += qp_cfg_.w_u;
        }
    }

    D_du_ = detail::Matrix(std::max(0, nu * (n - 1)), nu * n, 0.f);
    for (int k = 1; k < n; ++k) {
        for (int j = 0; j < nu; ++j) {
            const int row = (k - 1) * nu + j;
            D_du_(row, k * nu + j) = 1.f;
            D_du_(row, (k - 1) * nu + j) = -1.f;
        }
    }
    if (D_du_.rows() > 0) {
        const detail::Matrix D_du_T = detail::Matrix::transpose(D_du_);
        const detail::Matrix S_du = detail::Matrix::matmul(D_du_T, D_du_);
        for (int r = 0; r < nu * n; ++r) {
            for (int c = 0; c < nu * n; ++c) {
                H_(r, c) += qp_cfg_.w_du * S_du(r, c);
            }
        }
    }
}

std::vector<float> LatentMpcQpSolver::expandToFull(const std::vector<float>& u_opt) const {
    std::vector<float> full(static_cast<size_t>(n_ * nu_), 0.f);
    const int copy_steps = std::min(n_opt_, n_);
    for (int k = 0; k < copy_steps; ++k) {
        for (int j = 0; j < nu_; ++j) {
            full[static_cast<size_t>(k * nu_ + j)] = u_opt[static_cast<size_t>(k * nu_ + j)];
        }
    }
    if (n_opt_ > 0 && n_opt_ < n_) {
        for (int k = n_opt_; k < n_; ++k) {
            for (int j = 0; j < nu_; ++j) {
                full[static_cast<size_t>(k * nu_ + j)] =
                    full[static_cast<size_t>((n_opt_ - 1) * nu_ + j)];
            }
        }
    }
    if (hold_ > 1) {
        for (int k = 0; k < n_; ++k) {
            const int leader = (k / hold_) * hold_;
            for (int j = 0; j < nu_; ++j) {
                full[static_cast<size_t>(k * nu_ + j)] = full[static_cast<size_t>(leader * nu_ + j)];
            }
        }
    }
    return full;
}

void LatentMpcQpSolver::projectBox(std::vector<float>& u_tilde_full,
                                     const std::array<float, 4>& u_prev_phys) const {
    for (int k = 0; k < n_; ++k) {
        std::array<float, 4> u_phys{};
        for (int j = 0; j < nu_; ++j) {
            const float tilde = u_tilde_full[static_cast<size_t>(k * nu_ + j)];
            u_phys[static_cast<size_t>(j)] =
                tilde * model_.ctrlStd()[static_cast<size_t>(j)] + model_.ctrlMean()[static_cast<size_t>(j)];
        }
        for (int j = 0; j < nu_; ++j) {
            float lo = mpc_cfg_.u_min[static_cast<size_t>(j)];
            float hi = mpc_cfg_.u_max[static_cast<size_t>(j)];
            float prev = u_prev_phys[static_cast<size_t>(j)];
            if (k > 0) {
                const float prev_tilde = u_tilde_full[static_cast<size_t>((k - 1) * nu_ + j)];
                prev = prev_tilde * model_.ctrlStd()[static_cast<size_t>(j)] +
                       model_.ctrlMean()[static_cast<size_t>(j)];
            }
            float du_max = mpc_cfg_.du_max[static_cast<size_t>(j)];
            if (du_max <= 0.f) {
                du_max = (j == 0 || j == 2) ? mpc_cfg_.throttle_du_max : mpc_cfg_.rudder_du_max;
            }
            if (du_max > 0.f) {
                lo = std::max(lo, prev - du_max);
                hi = std::min(hi, prev + du_max);
            }
            u_phys[static_cast<size_t>(j)] = std::max(lo, std::min(hi, u_phys[static_cast<size_t>(j)]));
            u_tilde_full[static_cast<size_t>(k * nu_ + j)] =
                (u_phys[static_cast<size_t>(j)] - model_.ctrlMean()[static_cast<size_t>(j)]) /
                model_.ctrlStd()[static_cast<size_t>(j)];
        }
    }
}

float LatentMpcQpSolver::evalCost(const std::vector<float>& z0,
                                  const std::vector<float>& z_ref_stack,
                                  const std::vector<float>& u_tilde_full,
                                  const std::array<float, 4>& /*u_prev_phys*/) const {
    ensureMats();
    const std::vector<float> z_pred = model_.predictStacked(z0, u_tilde_full);
    float c = 0.f;
    for (size_t i = 0; i < z_pred.size(); ++i) {
        const float e = z_pred[i] - z_ref_stack[i];
        c += qp_cfg_.w_z * e * e;
    }
    for (int k = 0; k < n_; ++k) {
        for (int j = 0; j < nu_; ++j) {
            const float u = u_tilde_full[static_cast<size_t>(k * nu_ + j)];
            c += qp_cfg_.w_u * u * u;
        }
    }
    for (int k = 1; k < n_; ++k) {
        for (int j = 0; j < nu_; ++j) {
            const float du = u_tilde_full[static_cast<size_t>(k * nu_ + j)] -
                             u_tilde_full[static_cast<size_t>((k - 1) * nu_ + j)];
            c += qp_cfg_.w_du * du * du;
        }
    }
    return c;
}

std::vector<float> LatentMpcQpSolver::gradCost(const std::vector<float>& z0,
                                               const std::vector<float>& z_ref_stack,
                                               const std::vector<float>& u_tilde_full,
                                               const std::array<float, 4>& /*u_prev_phys*/) const {
    ensureMats();
    const std::vector<float> z_free = model_.freeResponse(z0);
    const std::vector<float> theta_u = detail::Matrix::matvec(model_.Theta(), u_tilde_full);
    std::vector<float> dz(static_cast<size_t>(nz_ * n_));
    for (size_t i = 0; i < dz.size(); ++i) {
        dz[i] = z_free[i] + theta_u[i] - z_ref_stack[i];
    }
    std::vector<float> g = detail::Matrix::matvec(detail::Matrix::transpose(model_.Theta()), dz);
    for (size_t i = 0; i < g.size(); ++i) {
        g[i] *= 2.f * qp_cfg_.w_z;
        g[i] += 2.f * qp_cfg_.w_u * u_tilde_full[i];
    }
    for (int k = 1; k < n_; ++k) {
        for (int j = 0; j < nu_; ++j) {
            const size_t idx = static_cast<size_t>(k * nu_ + j);
            const size_t idx0 = static_cast<size_t>((k - 1) * nu_ + j);
            const float du = u_tilde_full[idx] - u_tilde_full[idx0];
            g[idx] += 2.f * qp_cfg_.w_du * du;
            g[idx0] -= 2.f * qp_cfg_.w_du * du;
        }
    }
    (void)z0;
    return g;
}

LatentMpcQpSolution LatentMpcQpSolver::solve(const std::vector<float>& z0,
                                             const std::vector<float>& z_ref_stack,
                                             const std::array<float, 4>& u_prev_phys,
                                             const std::vector<float>* u_init_tilde_stack) const {
    ensureMats();
    if (static_cast<int>(z0.size()) != nz_) {
        throw std::runtime_error("z0 size mismatch");
    }
    if (static_cast<int>(z_ref_stack.size()) != nz_ * n_) {
        throw std::runtime_error("z_ref_stack size mismatch");
    }

    std::vector<float> u_full(static_cast<size_t>(n_ * nu_), 0.f);
    if (u_init_tilde_stack && static_cast<int>(u_init_tilde_stack->size()) == n_ * nu_) {
        u_full = *u_init_tilde_stack;
    } else if (u_init_tilde_stack && static_cast<int>(u_init_tilde_stack->size()) == n_opt_ * nu_) {
        u_full = expandToFull(*u_init_tilde_stack);
    }
    projectBox(u_full, u_prev_phys);
    float best_cost = evalCost(z0, z_ref_stack, u_full, u_prev_phys);
    std::vector<float> best_u = u_full;

    float step = qp_cfg_.step_size;
    int done_iters = 0;
    for (int it = 0; it < qp_cfg_.max_iters; ++it) {
        done_iters = it + 1;
        const std::vector<float> g = gradCost(z0, z_ref_stack, u_full, u_prev_phys);
        float gnorm = 0.f;
        for (float gi : g) {
            gnorm += gi * gi;
        }
        gnorm = std::sqrt(gnorm);
        if (gnorm < qp_cfg_.tol) {
            break;
        }

        std::vector<float> trial = u_full;
        bool improved = false;
        float trial_step = step;
        for (int ls = 0; ls < 12; ++ls) {
            for (size_t i = 0; i < trial.size(); ++i) {
                trial[i] = u_full[i] - trial_step * g[i];
            }
            projectBox(trial, u_prev_phys);
            const float tc = evalCost(z0, z_ref_stack, trial, u_prev_phys);
            if (tc < best_cost - 1e-8f) {
                best_cost = tc;
                best_u = trial;
                u_full = trial;
                improved = true;
                step = trial_step;
                break;
            }
            trial_step *= 0.5f;
        }
        if (!improved) {
            step = std::max(1e-4f, step * 0.5f);
        }
    }

    std::vector<float> u_decision(static_cast<size_t>(n_opt_ * nu_));
    for (int k = 0; k < n_opt_; ++k) {
        for (int j = 0; j < nu_; ++j) {
            u_decision[static_cast<size_t>(k * nu_ + j)] = best_u[static_cast<size_t>(k * nu_ + j)];
        }
    }
    return {u_decision, best_cost, done_iters};
}

}  // namespace koopman_control
