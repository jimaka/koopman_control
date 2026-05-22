/**
 * @file mpc_controller.cpp
 * @brief Koopman MPC：代价函数、数值梯度、Adam 优化与闭环仿真
 */

#include "koopman_control/mpc_controller.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace koopman_control {

namespace {

/** 从 ONNX 输出 flat 向量取第 row 行状态 */
std::array<float, 6> statesRow(const std::vector<float>& states, int row) {
    std::array<float, 6> s{};
    const int off = row * 6;
    for (int j = 0; j < 6; ++j) {
        s[j] = states[off + j];
    }
    return s;
}

/** 航向误差 wrap 到 [-pi, pi] */
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
    if (cfg_.control_hold_steps <= 0) {
        cfg_.control_hold_steps = 1;
    }
    if (cfg_.horizon % cfg_.control_hold_steps != 0) {
        throw std::runtime_error("horizon must be divisible by control_hold_steps");
    }
    if (cfg_.opt_control_steps <= 0 || cfg_.opt_control_steps > cfg_.horizon) {
        throw std::runtime_error("opt_control_steps must be in [1, horizon]");
    }
    u_warm_.assign(cfg_.horizon * 4, 0.f);
}

int KoopmanMpcController::controlHoldSteps() const {
    return std::max(1, cfg_.control_hold_steps);
}

int KoopmanMpcController::numControlBlocks() const {
    return cfg_.horizon / controlHoldSteps();
}

int KoopmanMpcController::optControlBlocks() const {
    const int hold = controlHoldSteps();
    const int n_blk = numControlBlocks();
    const int opt_blk = (cfg_.opt_control_steps + hold - 1) / hold;
    return std::max(1, std::min(n_blk, opt_blk));
}

void KoopmanMpcController::expandBlocksToFlat(const std::vector<float>& u_blocks,
                                                std::vector<float>& u_flat) const {
    const int hold = controlHoldSteps();
    const int H = cfg_.horizon;
    const int n_blk = numControlBlocks();
    if (static_cast<int>(u_blocks.size()) != n_blk * 4) {
        throw std::runtime_error("u_blocks size must be numControlBlocks*4");
    }
    if (static_cast<int>(u_flat.size()) != H * 4) {
        u_flat.assign(H * 4, 0.f);
    }
    for (int i = 0; i < H; ++i) {
        const int b = i / hold;
        for (int j = 0; j < 4; ++j) {
            u_flat[i * 4 + j] = u_blocks[b * 4 + j];
        }
    }
}

void KoopmanMpcController::extractBlocksFromFlat(const std::vector<float>& u_flat,
                                                   std::vector<float>& u_blocks) const {
    const int hold = controlHoldSteps();
    const int n_blk = numControlBlocks();
    u_blocks.assign(n_blk * 4, 0.f);
    for (int b = 0; b < n_blk; ++b) {
        const int step = b * hold;
        for (int j = 0; j < 4; ++j) {
            u_blocks[b * 4 + j] = u_flat[step * 4 + j];
        }
    }
}

void KoopmanMpcController::enforceBlockingOnFlat(std::vector<float>& u_flat) const {
    const int hold = controlHoldSteps();
    if (hold <= 1) {
        return;
    }
    const int H = cfg_.horizon;
    for (int i = 0; i < H; ++i) {
        const int leader = (i / hold) * hold;
        for (int j = 0; j < 4; ++j) {
            u_flat[i * 4 + j] = u_flat[leader * 4 + j];
        }
    }
}

void KoopmanMpcController::fillHoldBlocks(std::vector<float>& u_blocks) const {
    const int n_blk = numControlBlocks();
    const int opt_blk = optControlBlocks();
    if (opt_blk >= n_blk) {
        return;
    }
    const int hold_idx = opt_blk - 1;
    for (int b = opt_blk; b < n_blk; ++b) {
        for (int j = 0; j < 4; ++j) {
            u_blocks[b * 4 + j] = u_blocks[hold_idx * 4 + j];
        }
    }
}

void KoopmanMpcController::clampBlocks(std::vector<float>& u_blocks,
                                           const std::array<float, 4>& u_prev_step0) const {
    const int opt_blk = optControlBlocks();
    for (int b = 0; b < opt_blk; ++b) {
        std::array<float, 4> prev = u_prev_step0;
        if (b > 0) {
            for (int j = 0; j < 4; ++j) {
                prev[j] = u_blocks[(b - 1) * 4 + j];
            }
        }
        for (int j = 0; j < 4; ++j) {
            float& v = u_blocks[b * 4 + j];
            v = std::max(cfg_.u_min[j], std::min(cfg_.u_max[j], v));
            const float du_max = effectiveDuMax(j);
            if (du_max > 0.f) {
                v = std::max(prev[j] - du_max, std::min(prev[j] + du_max, v));
            }
        }
    }
}

void KoopmanMpcController::finalizeBlocks(std::vector<float>& u_blocks,
                                              const std::array<float, 4>& u_prev_step0,
                                              std::vector<float>& u_flat) const {
    clampBlocks(u_blocks, u_prev_step0);
    fillHoldBlocks(u_blocks);
    expandBlocksToFlat(u_blocks, u_flat);
}

void KoopmanMpcController::resetWarmStart() {
    u_warm_.assign(cfg_.horizon * 4, 0.f);
    has_warm_ = false;
    u_applied_.fill(0.f);
    has_applied_ = false;
}

float KoopmanMpcController::effectiveDuMax(int channel) const {
    if (channel < 0 || channel >= 4) {
        return 0.f;
    }
    if (cfg_.du_max[channel] > 0.f) {
        return cfg_.du_max[channel];
    }
    if (channel == 0 || channel == 2) {
        return cfg_.throttle_du_max > 0.f ? cfg_.throttle_du_max : 0.f;
    }
    return cfg_.rudder_du_max > 0.f ? cfg_.rudder_du_max : 0.f;
}

float KoopmanMpcController::duWeight(int channel) const {
    if (channel == 0 || channel == 2) {
        return cfg_.w_du_throttle >= 0.f ? cfg_.w_du_throttle : cfg_.w_du;
    }
    return cfg_.w_du_rudder >= 0.f ? cfg_.w_du_rudder : cfg_.w_du;
}

float KoopmanMpcController::mpcCost(const std::array<float, 6>& state0,
                                    const std::vector<std::array<float, 6>>& ref,
                                    const std::vector<float>& u_flat,
                                    const std::array<float, 4>& u_prev) const {
    const int H = cfg_.horizon;
    const auto t_rollout = std::chrono::high_resolution_clock::now();
    auto states = model_.rollout(state0, u_flat, cfg_.dt);
    ++step_rollout_count_;
    step_inference_ms_ += std::chrono::duration<double, std::milli>(
                              std::chrono::high_resolution_clock::now() - t_rollout)
                              .count();

    float c = 0.f;
    // 跟踪误差：位置 / 航向 / 速度
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
    // 控制幅值惩罚
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < 4; ++j) {
            c += cfg_.w_u * u_flat[i * 4 + j] * u_flat[i * 4 + j];
        }
    }
    // 控制增量惩罚（首步相对 u_prev）
    for (int i = 0; i < H; ++i) {
        std::array<float, 4> up = u_prev;
        if (i > 0) {
            for (int j = 0; j < 4; ++j) {
                up[j] = u_flat[(i - 1) * 4 + j];
            }
        }
        for (int j = 0; j < 4; ++j) {
            const float du = u_flat[i * 4 + j] - up[j];
            c += duWeight(j) * du * du;
        }
    }
    return c;
}

std::vector<float> KoopmanMpcController::numericGrad(
    const std::array<float, 6>& state0,
    const std::vector<std::array<float, 6>>& ref,
    std::vector<float> u_blocks,
    const std::array<float, 4>& u_prev) const {
    std::vector<float> u_flat(cfg_.horizon * 4);
    finalizeBlocks(u_blocks, u_prev, u_flat);
    const float f0 = mpcCost(state0, ref, u_flat, u_prev);
    const float eps = 1e-3f;
    const size_t n_opt = static_cast<size_t>(optControlBlocks() * 4);
    std::vector<float> grad(u_blocks.size(), 0.f);
    for (size_t i = 0; i < n_opt; ++i) {
        auto up = u_blocks;
        up[i] += eps;
        finalizeBlocks(up, u_prev, u_flat);
        grad[i] = (mpcCost(state0, ref, u_flat, u_prev) - f0) / eps;
    }
    return grad;
}

std::pair<std::array<float, 4>, float> KoopmanMpcController::solveStep(
    const std::array<float, 6>& state0,
    const std::vector<std::array<float, 6>>& ref_window,
    const std::array<float, 4>* u_prev_applied,
    MpcSolveTiming* timing) {
    const auto t_solve_start = std::chrono::high_resolution_clock::now();
    step_inference_ms_ = 0.;
    step_rollout_count_ = 0;
    const int H = cfg_.horizon;
    const int n_blk = numControlBlocks();
    const size_t n_opt = static_cast<size_t>(optControlBlocks() * 4);
    std::vector<float> u_flat(H * 4);
    std::vector<float> blocks(n_blk * 4, 0.f);

    std::array<float, 4> u_prev{};
    if (u_prev_applied != nullptr) {
        u_prev = *u_prev_applied;
    } else if (has_applied_) {
        u_prev = u_applied_;
    }

    // warm-start：上一步解左移一位（receding horizon）
    if (has_warm_) {
        for (int i = 0; i < H - 1; ++i) {
            for (int j = 0; j < 4; ++j) {
                u_flat[i * 4 + j] = u_warm_[(i + 1) * 4 + j];
            }
        }
        for (int j = 0; j < 4; ++j) {
            u_flat[(H - 1) * 4 + j] = u_warm_[(H - 1) * 4 + j];
        }
        enforceBlockingOnFlat(u_flat);
        extractBlocksFromFlat(u_flat, blocks);
    }
    finalizeBlocks(blocks, u_prev, u_flat);

    std::vector<std::array<float, 6>> ref(H + 1);
    for (int k = 0; k <= H; ++k) {
        ref[k] = ref_window[k];
    }

    float best_cost = 1e30f;
    std::vector<float> best_blocks = blocks;
    std::vector<float> m(blocks.size(), 0.f);
    std::vector<float> v(blocks.size(), 0.f);
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps_adam = 1e-8f;

    int opt_iters_done = 0;
    for (int it = 0; it < cfg_.opt_iters; ++it) {
        opt_iters_done = it + 1;
        finalizeBlocks(blocks, u_prev, u_flat);
        const float cost = mpcCost(state0, ref, u_flat, u_prev);
        if (cost < best_cost) {
            best_cost = cost;
            best_blocks = blocks;
        }
        auto grad = numericGrad(state0, ref, blocks, u_prev);
        const float t = static_cast<float>(it + 1);
        const float bc1 = 1.f - std::pow(beta1, t);
        const float bc2 = 1.f - std::pow(beta2, t);
        for (size_t i = 0; i < n_opt; ++i) {
            m[i] = beta1 * m[i] + (1.f - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1.f - beta2) * grad[i] * grad[i];
            blocks[i] -= cfg_.opt_lr * (m[i] / bc1) / (std::sqrt(v[i] / bc2) + eps_adam);
        }
    }

    finalizeBlocks(best_blocks, u_prev, u_flat);
    u_warm_ = u_flat;
    has_warm_ = true;

    std::array<float, 4> u0_out{};
    for (int j = 0; j < 4; ++j) {
        u0_out[j] = u_flat[j];
    }
    u_applied_ = u0_out;
    has_applied_ = true;

    const double solve_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::high_resolution_clock::now() -
                                t_solve_start)
                                .count();
    const double inference_ms = step_inference_ms_;
    const double opt_ms = std::max(0., solve_ms - inference_ms);
    if (timing != nullptr) {
        timing->inference_ms = inference_ms;
        timing->opt_ms = opt_ms;
        timing->opt_iters_cfg = cfg_.opt_iters;
        timing->opt_iters_done = opt_iters_done;
        timing->rollout_count = step_rollout_count_;
    }
    printf("Koopman solveStep: total=%.3f ms | inference=%.3f | mpc_opt=%.3f | "
           "mpc_iters=%d/%d rollouts=%d (H=%d, opt_blocks=%d)\n",
           solve_ms, inference_ms, opt_ms, opt_iters_done, cfg_.opt_iters,
           step_rollout_count_, cfg_.horizon, optControlBlocks());
    return {u0_out, best_cost};
}

MpcTrajectory KoopmanMpcController::simulate(const std::array<float, 6>& state0,
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
        enforceBlockingOnFlat(u_warm_);
        has_warm_ = true;
    }

    std::array<float, 6> cur = state0;
    for (int t = 0; t < n_sim; ++t) {
        traj.t[t + 1] = (t + 1) * cfg_.dt;
        // 截取长度 H+1 的参考窗口；末端不足则 hold 最后一点
        std::vector<std::array<float, 6>> ref_win;
        ref_win.reserve(H + 1);
        for (int k = 0; k <= H; ++k) {
            ref_win.push_back(ref_traj[std::min(t + k, T - 1)]);
        }
        auto [u_opt, c] = solveStep(cur, ref_win, nullptr);
        traj.control[t] = u_opt;
        traj.cost_history.push_back(c);

        cur = statesRow(model_.rollout(cur, u_warm_, cfg_.dt), 1);
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

}  // namespace koopman_control
