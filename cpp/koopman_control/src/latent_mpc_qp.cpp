/**
 * @file latent_mpc_qp.cpp
 * @brief OSQP 求解潜空间 condensed QP
 */

#include "koopman_control/latent_mpc_qp.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "osqp.h"

namespace koopman_control {
namespace {

constexpr c_float kOsqpInf = 1e30f;

/** 稠密对称矩阵上三角 → OSQP CSC（仅上三角含对角） */
void denseUpperTriToCsc(const detail::Matrix& sym, std::vector<c_float>& x, std::vector<c_int>& i,
                        std::vector<c_int>& p) {
    const int n = sym.rows();
    p.assign(static_cast<size_t>(n + 1), 0);
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row <= col; ++row) {
            const float v = sym(row, col);
            if (row == col || std::fabs(v) > 0.f) {
                ++p[static_cast<size_t>(col + 1)];
            }
        }
    }
    for (size_t col = 1; col < p.size(); ++col) {
        p[col] += p[col - 1];
    }
    x.resize(static_cast<size_t>(p.back()));
    i.resize(static_cast<size_t>(p.back()));
    int nnz = 0;
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row <= col; ++row) {
            const float v = sym(row, col);
            if (row == col || std::fabs(v) > 0.f) {
                i[static_cast<size_t>(nnz)] = row;
                x[static_cast<size_t>(nnz)] = static_cast<c_float>(v);
                ++nnz;
            }
        }
    }
}

/** 稠密 A → CSC（保留所有非零） */
void denseToCsc(const detail::Matrix& a, std::vector<c_float>& x, std::vector<c_int>& i,
                std::vector<c_int>& p) {
    const int m = a.rows();
    const int n = a.cols();
    p.assign(static_cast<size_t>(n + 1), 0);
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            if (std::fabs(a(row, col)) > 0.f) {
                ++p[static_cast<size_t>(col + 1)];
            }
        }
    }
    for (size_t col = 1; col < p.size(); ++col) {
        p[col] += p[col - 1];
    }
    x.resize(static_cast<size_t>(p.back()));
    i.resize(static_cast<size_t>(p.back()));
    int nnz = 0;
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            const float v = a(row, col);
            if (std::fabs(v) > 0.f) {
                i[static_cast<size_t>(nnz)] = row;
                x[static_cast<size_t>(nnz)] = static_cast<c_float>(v);
                ++nnz;
            }
        }
    }
}

float effectiveDuMax(const MpcConfig& cfg, int channel) {
    if (cfg.du_max[static_cast<size_t>(channel)] > 0.f) {
        return cfg.du_max[static_cast<size_t>(channel)];
    }
    if (channel == 0 || channel == 2) {
        return cfg.throttle_du_max > 0.f ? cfg.throttle_du_max : 0.f;
    }
    return cfg.rudder_du_max > 0.f ? cfg.rudder_du_max : 0.f;
}

}  // namespace

LatentMpcQpSolver::LatentMpcQpSolver(const KoopmanLatentModel& model, MpcConfig mpc_cfg,
                                     LatentMpcQpConfig qp_cfg)
    : model_(model), mpc_cfg_(mpc_cfg), qp_cfg_(qp_cfg) {
    // 使用配置 horizon：model 在 KoopmanMpcController 构造体中于 solver_ 之后 load，
    // 初始化时 model.horizon() 仍为默认值，不能与 mpc_cfg 不一致。
    n_ = mpc_cfg_.horizon;
    nz_ = model.nz();
    nu_ = model.nu();
    hold_ = std::max(1, mpc_cfg_.control_hold_steps);
    if (n_ % hold_ != 0) {
        throw std::runtime_error("horizon must be divisible by control_hold_steps");
    }
    const int n_blk = n_ / hold_;
    int opt_blk = (mpc_cfg_.opt_control_steps + hold_ - 1) / hold_;
    opt_blk = std::max(1, std::min(n_blk, opt_blk));
    n_opt_ = opt_blk * hold_;
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

    detail::Matrix D_du(std::max(0, nu * (n - 1)), nu * n, 0.f);
    for (int k = 1; k < n; ++k) {
        for (int j = 0; j < nu; ++j) {
            const int row = (k - 1) * nu + j;
            D_du(row, k * nu + j) = 1.f;
            D_du(row, (k - 1) * nu + j) = -1.f;
        }
    }
    if (D_du.rows() > 0) {
        const detail::Matrix D_du_T = detail::Matrix::transpose(D_du);
        const detail::Matrix S_du = detail::Matrix::matmul(D_du_T, D_du);
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

float LatentMpcQpSolver::evalCost(const std::vector<float>& z0,
                                  const std::vector<float>& z_ref_stack,
                                  const std::vector<float>& u_tilde_full) const {
    ensureMats();
    const std::vector<float> z_pred = model_.predictStacked(z0, u_tilde_full);
    float c = 0.f;
    for (size_t i = 0; i < z_pred.size(); ++i) {
        const float e = z_pred[i] - z_ref_stack[i];
        c += qp_cfg_.w_z * e * e;
    }
    for (size_t i = 0; i < u_tilde_full.size(); ++i) {
        c += qp_cfg_.w_u * u_tilde_full[i] * u_tilde_full[i];
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

LatentMpcQpSolution LatentMpcQpSolver::solve(const std::vector<float>& z0,
                                             const std::vector<float>& z_ref_stack,
                                             const std::array<float, 4>& u_prev_phys,
                                             const std::vector<float>* u_init_tilde_stack,
                                             const PoseLinearization* pose) const {
    ensureMats();
    const int nvar = n_ * nu_;
    if (static_cast<int>(z0.size()) != nz_) {
        throw std::runtime_error("z0 size mismatch");
    }
    if (static_cast<int>(z_ref_stack.size()) != nz_ * n_) {
        throw std::runtime_error("z_ref_stack size mismatch");
    }

    const std::vector<float> z_free = model_.freeResponse(z0);
    std::vector<float> err(static_cast<size_t>(nz_ * n_));
    for (size_t i = 0; i < err.size(); ++i) {
        err[i] = z_free[i] - z_ref_stack[i];
    }
    std::vector<float> q = detail::Matrix::matvec(detail::Matrix::transpose(model_.Theta()), err);
    for (float& qi : q) {
        qi *= 2.f * qp_cfg_.w_z;
    }

    detail::Matrix P = detail::Matrix(nvar, nvar, 0.f);
    for (int r = 0; r < nvar; ++r) {
        for (int c = 0; c < nvar; ++c) {
            P(r, c) = 2.f * H_(r, c);
        }
    }

    // Tier-2：叠加位姿软约束项 P += 2 PhiᵀQpPhi, q += 2 PhiᵀQp b
    if (pose != nullptr && pose->valid && pose->Phi.rows() == 3 * n_ &&
        pose->Phi.cols() == nvar) {
        const detail::Matrix& Phi = pose->Phi;
        const int rows = Phi.rows();
        detail::Matrix scaled(rows, nvar, 0.f);
        for (int r = 0; r < rows; ++r) {
            const float w = pose->wq[static_cast<size_t>(r)];
            for (int c = 0; c < nvar; ++c) {
                scaled(r, c) = w * Phi(r, c);
            }
        }
        const detail::Matrix PhiT = detail::Matrix::transpose(Phi);
        const detail::Matrix Ppose = detail::Matrix::matmul(PhiT, scaled);
        for (int r = 0; r < nvar; ++r) {
            for (int c = 0; c < nvar; ++c) {
                P(r, c) += 2.f * Ppose(r, c);
            }
        }
        std::vector<float> wb(static_cast<size_t>(rows));
        for (int r = 0; r < rows; ++r) {
            wb[static_cast<size_t>(r)] = pose->wq[static_cast<size_t>(r)] * pose->b[static_cast<size_t>(r)];
        }
        const std::vector<float> qpose = detail::Matrix::matvec(PhiT, wb);
        for (int i = 0; i < nvar; ++i) {
            q[static_cast<size_t>(i)] += 2.f * qpose[static_cast<size_t>(i)];
        }
    }

    const int n_rate = nu_ * n_;
    const int n_cons = nvar + n_rate;
    detail::Matrix A(n_cons, nvar, 0.f);
    std::vector<c_float> l(static_cast<size_t>(n_cons), 0.f);
    std::vector<c_float> u(static_cast<size_t>(n_cons), 0.f);

    for (int i = 0; i < nvar; ++i) {
        A(i, i) = 1.f;
        const int ch = i % nu_;
        const float sigma = model_.ctrlStd()[static_cast<size_t>(ch)];
        const float mu = model_.ctrlMean()[static_cast<size_t>(ch)];
        l[static_cast<size_t>(i)] =
            (mpc_cfg_.u_min[static_cast<size_t>(ch)] - mu) / sigma;
        u[static_cast<size_t>(i)] =
            (mpc_cfg_.u_max[static_cast<size_t>(ch)] - mu) / sigma;
    }

    int row = nvar;
    for (int k = 0; k < n_; ++k) {
        for (int j = 0; j < nu_; ++j) {
            const int idx = k * nu_ + j;
            const float sigma = model_.ctrlStd()[static_cast<size_t>(j)];
            const float mu = model_.ctrlMean()[static_cast<size_t>(j)];
            const float du_max = effectiveDuMax(mpc_cfg_, j);
            if (du_max <= 0.f || sigma <= 0.f) {
                l[static_cast<size_t>(row)] = -kOsqpInf;
                u[static_cast<size_t>(row)] = kOsqpInf;
                A(row, idx) = 1.f;
                ++row;
                continue;
            }
            const float du_tilde = du_max / sigma;
            if (k == 0) {
                const float prev_tilde =
                    (u_prev_phys[static_cast<size_t>(j)] - mu) / sigma;
                A(row, idx) = 1.f;
                l[static_cast<size_t>(row)] = prev_tilde - du_tilde;
                u[static_cast<size_t>(row)] = prev_tilde + du_tilde;
            } else {
                A(row, idx) = 1.f;
                A(row, (k - 1) * nu_ + j) = -1.f;
                l[static_cast<size_t>(row)] = -du_tilde;
                u[static_cast<size_t>(row)] = du_tilde;
            }
            ++row;
        }
    }

    std::vector<c_float> Px, Ax, q_osqp(static_cast<size_t>(nvar));
    std::vector<c_int> Pi, Pp, Ai, Ap;
    denseUpperTriToCsc(P, Px, Pi, Pp);
    denseToCsc(A, Ax, Ai, Ap);
    for (int i = 0; i < nvar; ++i) {
        q_osqp[static_cast<size_t>(i)] = static_cast<c_float>(q[static_cast<size_t>(i)]);
    }

    OSQPSettings settings;
    osqp_set_default_settings(&settings);
    settings.eps_abs = qp_cfg_.osqp_eps_abs;
    settings.eps_rel = qp_cfg_.osqp_eps_rel;
    settings.max_iter = qp_cfg_.osqp_max_iter;
    settings.verbose = qp_cfg_.osqp_verbose;
    settings.warm_start = 1;

    OSQPData data{};
    data.n = nvar;
    data.m = n_cons;
    data.P = csc_matrix(data.n, data.n, static_cast<c_int>(Px.size()), Px.data(), Pi.data(), Pp.data());
    data.q = q_osqp.data();
    data.A = csc_matrix(data.m, data.n, static_cast<c_int>(Ax.size()), Ax.data(), Ai.data(), Ap.data());
    data.l = l.data();
    data.u = u.data();

    OSQPWorkspace* work = nullptr;
    const c_int setup_status = osqp_setup(&work, &data, &settings);
    if (setup_status != 0 || work == nullptr) {
        throw std::runtime_error("osqp_setup failed");
    }

    if (u_init_tilde_stack && static_cast<int>(u_init_tilde_stack->size()) >= nvar) {
        std::vector<c_float> x_warm(static_cast<size_t>(nvar));
        for (int i = 0; i < nvar; ++i) {
            x_warm[static_cast<size_t>(i)] = static_cast<c_float>((*u_init_tilde_stack)[static_cast<size_t>(i)]);
        }
        osqp_warm_start_x(work, x_warm.data());
    }

    const c_int solve_status = osqp_solve(work);
    LatentMpcQpSolution sol;
    sol.osqp_status = static_cast<int>(work->info->status_val);
    sol.osqp_iters = static_cast<int>(work->info->iter);

    std::vector<float> u_full(static_cast<size_t>(nvar), 0.f);
    if (solve_status != 0 || work->solution == nullptr || work->solution->x == nullptr) {
        const std::string status = work->info != nullptr ? work->info->status : "unknown";
        osqp_cleanup(work);
        throw std::runtime_error("osqp_solve failed: " + status);
    }
    for (int i = 0; i < nvar; ++i) {
        u_full[static_cast<size_t>(i)] = static_cast<float>(work->solution->x[i]);
    }
    osqp_cleanup(work);

    if (hold_ > 1 || n_opt_ < n_) {
        u_full = expandToFull(u_full);
    }
    sol.u_tilde_stack = u_full;
    sol.cost = evalCost(z0, z_ref_stack, u_full);
    return sol;
}

}  // namespace koopman_control
