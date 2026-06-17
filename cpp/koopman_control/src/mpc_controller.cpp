/**
 * @file mpc_controller.cpp
 * @brief OSQP 潜空间 MPC（Tier-1）
 */

#include "koopman_control/mpc_controller.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>

namespace koopman_control {
namespace {

#if KOOPMAN_ENABLE_ONNX
std::array<float, 6> rolloutOneStepOnnx(const KoopmanOnnxModel& plant, const std::array<float, 6>& state0,
                                        const std::array<float, 4>& u0, float dt) {
    // ONNX 图固定 H 步 rollout，要求 u_seq 长度为 H*4；单步推进只取 states[1]，
    // 其余步用 u0 填充（state[1] 仅依赖 u_seq[0]，填充值不影响结果）。
    const int H = plant.horizon();
    std::vector<float> u_flat(static_cast<size_t>(H * 4));
    for (int k = 0; k < H; ++k) {
        for (int j = 0; j < 4; ++j) {
            u_flat[static_cast<size_t>(k * 4 + j)] = u0[static_cast<size_t>(j)];
        }
    }
    const auto states = plant.rollout(state0, u_flat, dt);
    std::array<float, 6> out{};
    for (int j = 0; j < 6; ++j) {
        out[static_cast<size_t>(j)] = states[static_cast<size_t>(6 + j)];
    }
    return out;
}
#endif

}  // namespace

KoopmanMpcController::KoopmanMpcController(std::string latent_yaml_path, MpcConfig cfg,
                                           LatentMpcQpConfig qp_cfg)
    : cfg_(cfg), qp_cfg_(qp_cfg), solver_(model_, cfg_, qp_cfg_) {
    model_.loadFromYaml(latent_yaml_path, cfg_.horizon);
    model_.precomputePredictionMatrices();
    if (model_.horizon() != cfg_.horizon) {
        throw std::runtime_error("latent model horizon mismatch after YAML load");
    }
    encoder_.loadFromYaml(latent_yaml_path);
    decoder_.loadFromYaml(latent_yaml_path);  // 旧 YAML 无 decoder 时静默禁用 Tier-2
}

bool KoopmanMpcController::poseTrackingEnabled() const {
    return (cfg_.w_xy > 0.f || cfg_.w_yaw > 0.f) && decoder_.loaded();
}

#if KOOPMAN_ENABLE_ONNX
KoopmanMpcController::KoopmanMpcController(std::string latent_yaml_path, std::string onnx_plant_path,
                                           MpcConfig cfg, LatentMpcQpConfig qp_cfg)
    : KoopmanMpcController(std::move(latent_yaml_path), cfg, qp_cfg) {
    plant_ = std::make_unique<KoopmanOnnxModel>(onnx_plant_path);
    if (plant_->horizon() != cfg_.horizon) {
        throw std::runtime_error("ONNX plant horizon must match MPC horizon");
    }
}
#endif

std::array<float, 3> KoopmanMpcController::normalizeDyn(const std::array<float, 3>& dyn) const {
    std::array<float, 3> out{};
    for (int i = 0; i < 3; ++i) {
        out[static_cast<size_t>(i)] =
            (dyn[static_cast<size_t>(i)] - model_.dynMean()[static_cast<size_t>(i)]) /
            model_.dynStd()[static_cast<size_t>(i)];
    }
    return out;
}

std::vector<float> KoopmanMpcController::buildRefLatentStack(
    const std::vector<std::array<float, 6>>& ref_window) const {
    const int n = model_.horizon();
    const int nz = model_.nz();
    std::vector<float> stack(static_cast<size_t>(nz * n), 0.f);
    for (int k = 0; k < n; ++k) {
        const size_t idx = static_cast<size_t>(std::min(k, static_cast<int>(ref_window.size()) - 1));
        const std::array<float, 3> dyn = {ref_window[idx][3], ref_window[idx][4], ref_window[idx][5]};
        const std::vector<float> z = encoder_.encode(normalizeDyn(dyn));
        for (int i = 0; i < nz; ++i) {
            stack[static_cast<size_t>(k * nz + i)] = z[static_cast<size_t>(i)];
        }
    }
    return stack;
}

std::vector<float> KoopmanMpcController::buildRefPoseStack(
    const std::vector<std::array<float, 6>>& ref_window) const {
    const int n = model_.horizon();
    std::vector<float> stack(static_cast<size_t>(3 * n), 0.f);
    for (int k = 1; k <= n; ++k) {
        const size_t idx = static_cast<size_t>(std::min(k, static_cast<int>(ref_window.size()) - 1));
        const int r = (k - 1) * 3;
        stack[static_cast<size_t>(r + 0)] = ref_window[idx][0];
        stack[static_cast<size_t>(r + 1)] = ref_window[idx][1];
        stack[static_cast<size_t>(r + 2)] = ref_window[idx][2];
    }
    return stack;
}

std::pair<std::array<float, 4>, float> KoopmanMpcController::solveStep(
    const std::array<float, 6>& state0,
    const std::vector<std::array<float, 6>>& ref_window,
    const std::array<float, 4>* u_prev_applied,
    MpcSolveTiming* timing) {
    if (static_cast<int>(ref_window.size()) < model_.horizon() + 1) {
        throw std::runtime_error("ref_window too short");
    }

    const auto t0 = std::chrono::high_resolution_clock::now();
    const std::array<float, 3> dyn = {state0[3], state0[4], state0[5]};
    const std::vector<float> z0 = encoder_.encode(normalizeDyn(dyn));
    const std::vector<float> z_ref = buildRefLatentStack(ref_window);

    std::array<float, 4> u_prev{};
    if (u_prev_applied) {
        u_prev = *u_prev_applied;
    }

    const bool pose_on = poseTrackingEnabled();
    const std::array<float, 3> pose0 = {state0[0], state0[1], state0[2]};
    const std::vector<float> pose_ref = pose_on ? buildRefPoseStack(ref_window) : std::vector<float>{};

    // 标称控制序列（SQP 线性化工作点）：优先使用 warm start
    const int nvar = model_.horizon() * model_.nu();
    std::vector<float> U = has_warm_ && static_cast<int>(u_warm_tilde_.size()) == nvar
                              ? u_warm_tilde_
                              : std::vector<float>(static_cast<size_t>(nvar), 0.f);

    const int iters = pose_on ? std::max(1, cfg_.sqp_iters) : 1;
    LatentMpcQpSolution sol;
    for (int it = 0; it < iters; ++it) {
        PoseLinearization pl;
        const PoseLinearization* plp = nullptr;
        if (pose_on) {
            pl = buildPoseLinearization(model_, decoder_, z0, pose0, U, pose_ref, cfg_.dt,
                                        cfg_.w_xy, cfg_.w_yaw);
            plp = pl.valid ? &pl : nullptr;
        }
        sol = solver_.solve(z0, z_ref, u_prev, &U, plp);
        U = sol.u_tilde_stack;
    }
    const auto t1 = std::chrono::high_resolution_clock::now();

    u_warm_tilde_ = sol.u_tilde_stack;
    has_warm_ = true;

    std::array<float, 4> u0_phys{};
    if (static_cast<int>(sol.u_tilde_stack.size()) >= model_.nu()) {
        u0_phys = model_.denormalizeControl(
            {sol.u_tilde_stack[0], sol.u_tilde_stack[1], sol.u_tilde_stack[2], sol.u_tilde_stack[3]});
    }

    if (timing != nullptr) {
        timing->qp_setup_ms = 0.;
        timing->qp_solve_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        timing->osqp_iters = sol.osqp_iters;
        timing->osqp_status = sol.osqp_status;
    }

    return {u0_phys, sol.cost};
}

#if KOOPMAN_ENABLE_ONNX
MpcTrajectory KoopmanMpcController::simulate(const std::array<float, 6>& state0,
                                             const std::vector<std::array<float, 6>>& ref_traj,
                                             const std::vector<std::array<float, 4>>* ref_ctrl,
                                             int max_steps) {
    if (!plant_) {
        throw std::runtime_error("simulate requires ONNX plant model");
    }

    MpcTrajectory traj;
    const int T = static_cast<int>(ref_traj.size());
    const int n_sim = std::min(max_steps, T - 1);
    const int H = cfg_.horizon;
    const int ref_stride =
        std::max(1, static_cast<int>(std::lround(cfg_.dt / cfg_.data_dt)));

    traj.state.resize(static_cast<size_t>(n_sim + 1));
    traj.control.resize(static_cast<size_t>(n_sim));
    traj.ref_state.assign(ref_traj.begin(), ref_traj.begin() + n_sim + 1);
    traj.t.resize(static_cast<size_t>(n_sim + 1));
    traj.state[0] = state0;
    traj.t[0] = 0.f;

    if (ref_ctrl && static_cast<int>(ref_ctrl->size()) >= (H - 1) * ref_stride + 1) {
        u_warm_tilde_.assign(static_cast<size_t>(H * model_.nu()), 0.f);
        for (int i = 0; i < H; ++i) {
            const auto u_n = model_.normalizeControl(
                (*ref_ctrl)[static_cast<size_t>(i * ref_stride)]);
            for (int j = 0; j < model_.nu(); ++j) {
                u_warm_tilde_[static_cast<size_t>(i * model_.nu() + j)] = u_n[static_cast<size_t>(j)];
            }
        }
        has_warm_ = true;
    }

    std::array<float, 6> cur = state0;
    for (int t = 0; t < n_sim; ++t) {
        traj.t[static_cast<size_t>(t + 1)] = static_cast<float>(t + 1) * cfg_.dt;
        const int base = t * ref_stride;
        std::vector<std::array<float, 6>> ref_win;
        ref_win.reserve(static_cast<size_t>(H + 1));
        for (int k = 0; k <= H; ++k) {
            ref_win.push_back(
                ref_traj[static_cast<size_t>(std::min(base + k * ref_stride, T - 1))]);
        }
        auto [u_opt, c] = solveStep(cur, ref_win, nullptr);
        traj.control[static_cast<size_t>(t)] = u_opt;
        traj.cost_history.push_back(c);
        cur = rolloutOneStepOnnx(*plant_, cur, u_opt, cfg_.dt);
        traj.state[static_cast<size_t>(t + 1)] = cur;
    }
    return traj;
}
#endif  // KOOPMAN_ENABLE_ONNX

void KoopmanMpcController::resetWarmStart() {
    u_warm_tilde_.clear();
    has_warm_ = false;
}

TrackingMetrics computeMetrics(const MpcTrajectory& traj) {
    TrackingMetrics m;
    const size_t n = std::min(traj.state.size(), traj.ref_state.size());
    if (n == 0) {
        return m;
    }
    float xy_sse = 0.f;
    float yaw_sse = 0.f;
    float xy_max = 0.f;
    for (size_t i = 0; i < n; ++i) {
        const float dx = traj.state[i][0] - traj.ref_state[i][0];
        const float dy = traj.state[i][1] - traj.ref_state[i][1];
        const float xy = std::sqrt(dx * dx + dy * dy);
        xy_sse += xy * xy;
        xy_max = std::max(xy_max, xy);
        const float dyaw = std::atan2(std::sin(traj.state[i][2] - traj.ref_state[i][2]),
                                      std::cos(traj.state[i][2] - traj.ref_state[i][2]));
        yaw_sse += dyaw * dyaw;
    }
    m.xy_rmse_m = std::sqrt(xy_sse / static_cast<float>(n));
    m.xy_max_m = xy_max;
    m.yaw_rmse_deg = std::sqrt(yaw_sse / static_cast<float>(n)) * 180.f / 3.14159265f;
    m.final_xy_err_m =
        std::hypot(traj.state[n - 1][0] - traj.ref_state[n - 1][0],
                   traj.state[n - 1][1] - traj.ref_state[n - 1][1]);
    return m;
}

}  // namespace koopman_control
