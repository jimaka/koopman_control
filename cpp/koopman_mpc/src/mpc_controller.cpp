#include "mpc_controller.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace koopman_mpc {

namespace {

std::array<float, 6> statesRow(const std::vector<float>& states, int row) {
    std::array<float, 6> s{};
    const int off = row * 6;
    for (int j = 0; j < 6; ++j) {
        s[j] = states[off + j];
    }
    return s;
}

float wrapYaw(float a, float b) {
    const float d = a - b;
    return std::atan2(std::sin(d), std::cos(d));
}

}  // namespace

KoopmanMpcController::KoopmanMpcController(KoopmanOnnxModel model, MpcConfig cfg)
    : model_(std::move(model)), cfg_(cfg) {
    const int onnx_h = model_.horizon();
    if (cfg_.horizon != onnx_h) {
        throw std::runtime_error(
            "MPC horizon must equal ONNX traced horizon (" + std::to_string(onnx_h) +
            "), got " + std::to_string(cfg_.horizon));
    }
    if (cfg_.opt_control_steps <= 0 || cfg_.opt_control_steps > cfg_.horizon) {
        throw std::runtime_error("opt_control_steps must be in [1, horizon]");
    }
    u_warm_.assign(cfg_.horizon * 4, 0.f);
}

void KoopmanMpcController::fillHoldTail(std::vector<float>& u_flat) const {
    const int H = cfg_.horizon;
    const int opt_steps = cfg_.opt_control_steps;
    if (opt_steps >= H) {
        return;
    }
    const int hold = opt_steps - 1;
    for (int i = opt_steps; i < H; ++i) {
        for (int j = 0; j < 4; ++j) {
            u_flat[i * 4 + j] = u_flat[hold * 4 + j];
        }
    }
}

void KoopmanMpcController::clampUFlat(std::vector<float>& u_flat) const {
    const int opt_steps = cfg_.opt_control_steps;
    for (int i = 0; i < opt_steps; ++i) {
        for (int j = 0; j < 4; ++j) {
            float& v = u_flat[i * 4 + j];
            v = std::max(cfg_.u_min[j], std::min(cfg_.u_max[j], v));
        }
    }
    fillHoldTail(u_flat);
}

float KoopmanMpcController::mpcCost(const std::array<float, 6>& state0,
                                    const std::vector<std::array<float, 6>>& ref,
                                    const std::vector<float>& u_flat,
                                    const std::array<float, 4>& u_prev) const {
    const int H = cfg_.horizon;
    auto states = model_.rollout(state0, u_flat, cfg_.dt);

    float c = 0.f;
    for (int k = 0; k <= H; ++k) {
        const auto s = statesRow(states, k);
        const auto& r = ref[k];
        const float dx = s[0] - r[0];
        const float dy = s[1] - r[1];
        c += cfg_.w_xy * (dx * dx + dy * dy);
        const float dyaw = wrapYaw(s[2], r[2]);
        c += cfg_.w_yaw * dyaw * dyaw;
        for (int j = 0; j < 3; ++j) {
            const float dv = s[3 + j] - r[3 + j];
            c += cfg_.w_vel * dv * dv;
        }
    }
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < 4; ++j) {
            const float u = u_flat[i * 4 + j];
            c += cfg_.w_u * u * u;
        }
    }
    for (int i = 0; i < H; ++i) {
        std::array<float, 4> up = u_prev;
        if (i > 0) {
            for (int j = 0; j < 4; ++j) {
                up[j] = u_flat[(i - 1) * 4 + j];
            }
        }
        for (int j = 0; j < 4; ++j) {
            const float du = u_flat[i * 4 + j] - up[j];
            c += cfg_.w_du * du * du;
        }
    }
    return c;
}

std::vector<float> KoopmanMpcController::numericGrad(
    const std::array<float, 6>& state0,
    const std::vector<std::array<float, 6>>& ref,
    std::vector<float> u_flat,
    const std::array<float, 4>& u_prev) const {
    clampUFlat(u_flat);
    const float f0 = mpcCost(state0, ref, u_flat, u_prev);
    const float eps = 1e-3f;
    const size_t n_opt = static_cast<size_t>(cfg_.opt_control_steps * 4);
    std::vector<float> grad(u_flat.size(), 0.f);
    for (size_t i = 0; i < n_opt; ++i) {
        auto up = u_flat;
        up[i] += eps;
        clampUFlat(up);
        const float fp = mpcCost(state0, ref, up, u_prev);
        grad[i] = (fp - f0) / eps;
    }
    return grad;
}

std::pair<std::array<float, 4>, float> KoopmanMpcController::solveStep(
    const std::array<float, 6>& state0,
    const std::vector<std::array<float, 6>>& ref_window) {
    const int H = cfg_.horizon;
    const size_t n_opt = static_cast<size_t>(cfg_.opt_control_steps * 4);
    std::vector<float> u(H * 4);
    if (has_warm_) {
        for (int i = 0; i < H - 1; ++i) {
            for (int j = 0; j < 4; ++j) {
                u[i * 4 + j] = u_warm_[(i + 1) * 4 + j];
            }
        }
        for (int j = 0; j < 4; ++j) {
            u[(H - 1) * 4 + j] = u_warm_[(H - 1) * 4 + j];
        }
    }
    clampUFlat(u);

    std::array<float, 4> u_prev{};
    for (int j = 0; j < 4; ++j) {
        u_prev[j] = u[j];
    }

    std::vector<std::array<float, 6>> ref(H + 1);
    for (int k = 0; k <= H; ++k) {
        ref[k] = ref_window[k];
    }

    float best_cost = 1e30f;
    std::vector<float> best_u = u;

    std::vector<float> m(u.size(), 0.f);
    std::vector<float> v(u.size(), 0.f);
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps_adam = 1e-8f;

    for (int it = 0; it < cfg_.opt_iters; ++it) {
        clampUFlat(u);
        const float cost = mpcCost(state0, ref, u, u_prev);
        if (cost < best_cost) {
            best_cost = cost;
            best_u = u;
        }
        auto grad = numericGrad(state0, ref, u, u_prev);
        const float t = static_cast<float>(it + 1);
        const float bc1 = 1.f - std::pow(beta1, t);
        const float bc2 = 1.f - std::pow(beta2, t);
        for (size_t i = 0; i < n_opt; ++i) {
            m[i] = beta1 * m[i] + (1.f - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1.f - beta2) * grad[i] * grad[i];
            const float m_hat = m[i] / bc1;
            const float v_hat = v[i] / bc2;
            u[i] -= cfg_.opt_lr * m_hat / (std::sqrt(v_hat) + eps_adam);
        }
    }

    clampUFlat(best_u);
    u_warm_ = best_u;
    has_warm_ = true;

    std::array<float, 4> u0_out{};
    for (int j = 0; j < 4; ++j) {
        u0_out[j] = best_u[j];
    }
    return {u0_out, best_cost};
}

MpcTrajectory KoopmanMpcController::simulate(
    const std::array<float, 6>& state0,
    const std::vector<std::array<float, 6>>& ref_traj,
    const std::vector<std::array<float, 4>>* ref_ctrl,
    int max_steps) {
    MpcTrajectory traj;
    const int T = static_cast<int>(ref_traj.size());
    const int n_sim = std::min(max_steps, T - 1);
    const int H = cfg_.horizon;

    traj.state.resize(n_sim + 1);
    traj.control.resize(n_sim);
    traj.ref_state.assign(ref_traj.begin(), ref_traj.begin() + n_sim + 1);
    traj.t.resize(n_sim + 1);

    traj.state[0] = state0;
    traj.t[0] = 0.f;

    if (ref_ctrl && static_cast<int>(ref_ctrl->size()) >= H) {
        u_warm_.assign(H * 4, 0.f);
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < 4; ++j) {
                u_warm_[i * 4 + j] = (*ref_ctrl)[i][j];
            }
        }
        fillHoldTail(u_warm_);
        has_warm_ = true;
    }

    std::array<float, 6> cur = state0;
    for (int t = 0; t < n_sim; ++t) {
        traj.t[t + 1] = (t + 1) * cfg_.dt;

        std::vector<std::array<float, 6>> ref_win;
        ref_win.reserve(H + 1);
        for (int k = 0; k <= H; ++k) {
            const int idx = std::min(t + k, T - 1);
            ref_win.push_back(ref_traj[idx]);
        }

        auto [u_opt, c] = solveStep(cur, ref_win);
        traj.control[t] = u_opt;
        traj.cost_history.push_back(c);

        std::vector<float> u_roll(H * 4, 0.f);
        for (int j = 0; j < 4; ++j) {
            u_roll[j] = u_opt[j];
        }
        fillHoldTail(u_roll);
        auto next = model_.rollout(cur, u_roll, cfg_.dt);
        cur = statesRow(next, 1);
        traj.state[t + 1] = cur;
    }
    return traj;
}

TrackingMetrics computeMetrics(const MpcTrajectory& traj) {
    TrackingMetrics m;
    const size_t n = std::min(traj.state.size(), traj.ref_state.size());
    if (n == 0) {
        return m;
    }
    double sum_xy2 = 0.0;
    double max_xy = 0.0;
    double sum_yaw2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const float dx = traj.state[i][0] - traj.ref_state[i][0];
        const float dy = traj.state[i][1] - traj.ref_state[i][1];
        const float xy = std::sqrt(dx * dx + dy * dy);
        sum_xy2 += xy * xy;
        max_xy = std::max(max_xy, static_cast<double>(xy));
        const float dyaw = wrapYaw(traj.state[i][2], traj.ref_state[i][2]);
        sum_yaw2 += dyaw * dyaw;
    }
    m.xy_rmse_m = static_cast<float>(std::sqrt(sum_xy2 / n));
    m.xy_max_m = static_cast<float>(max_xy);
    m.yaw_rmse_deg =
        static_cast<float>(std::sqrt(sum_yaw2 / n) * 180.0 / 3.141592653589793);
    const float fdx = traj.state[n - 1][0] - traj.ref_state[n - 1][0];
    const float fdy = traj.state[n - 1][1] - traj.ref_state[n - 1][1];
    m.final_xy_err_m = std::sqrt(fdx * fdx + fdy * fdy);
    return m;
}

}  // namespace koopman_mpc
