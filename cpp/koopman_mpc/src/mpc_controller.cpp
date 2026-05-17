#include "mpc_controller.hpp"

#include <cmath>
#include <numeric>

namespace koopman_mpc {

namespace {

torch::Tensor array6ToTensor(const std::array<float, 6>& s) {
    return torch::tensor({s[0], s[1], s[2], s[3], s[4], s[5]},
                         torch::dtype(torch::kFloat32));
}

torch::Tensor refWindowToTensor(const std::vector<std::array<float, 6>>& ref) {
    const int64_t n = static_cast<int64_t>(ref.size());
    auto t = torch::zeros({n, 6}, torch::kFloat32);
    auto acc = t.accessor<float, 2>();
    for (int64_t i = 0; i < n; ++i) {
        for (int j = 0; j < 6; ++j) {
            acc[i][j] = ref[i][j];
        }
    }
    return t;
}

}  // namespace

KoopmanMpcController::KoopmanMpcController(KoopmanTorchModel model, MpcConfig cfg)
    : model_(std::move(model)), cfg_(cfg) {
    if (cfg_.horizon != KoopmanTorchModel::kTracedHorizon) {
        throw std::runtime_error(
            "MPC horizon must equal TorchScript traced horizon (" +
            std::to_string(KoopmanTorchModel::kTracedHorizon) + ")");
    }
}

torch::Tensor KoopmanMpcController::clampU(torch::Tensor u) const {
    auto umin = torch::tensor(
        {cfg_.u_min[0], cfg_.u_min[1], cfg_.u_min[2], cfg_.u_min[3]},
        torch::dtype(torch::kFloat32));
    auto umax = torch::tensor(
        {cfg_.u_max[0], cfg_.u_max[1], cfg_.u_max[2], cfg_.u_max[3]},
        torch::dtype(torch::kFloat32));
    return torch::max(torch::min(u, umax), umin);
}

torch::Tensor KoopmanMpcController::mpcCost(const torch::Tensor& state0,
                                            const torch::Tensor& ref,
                                            const torch::Tensor& u_flat,
                                            const torch::Tensor& u_prev) const {
    const int64_t H = cfg_.horizon;
    auto u_seq = u_flat.view({H, 4});
    auto traj = model_.rollout(state0, u_seq, cfg_.dt);

    auto err_xy = traj.index({torch::indexing::Slice(), torch::indexing::Slice(0, 2)}) -
                  ref.index({torch::indexing::Slice(), torch::indexing::Slice(0, 2)});
    auto dyaw = traj.index({torch::indexing::Slice(), 2}) -
                ref.index({torch::indexing::Slice(), 2});
    dyaw = torch::atan2(torch::sin(dyaw), torch::cos(dyaw));
    auto err_vel = traj.index({torch::indexing::Slice(), torch::indexing::Slice(3, 6)}) -
                   ref.index({torch::indexing::Slice(), torch::indexing::Slice(3, 6)});

    auto c = cfg_.w_xy * torch::sum(err_xy * err_xy) +
             cfg_.w_yaw * torch::sum(dyaw * dyaw) +
             cfg_.w_vel * torch::sum(err_vel * err_vel) +
             cfg_.w_u * torch::sum(u_seq * u_seq);

    auto u_shifted = torch::cat({u_prev.unsqueeze(0), u_seq.index({torch::indexing::Slice(0, -1)})}, 0);
    auto du = u_seq - u_shifted;
    c = c + cfg_.w_du * torch::sum(du * du);
    return c;
}

std::pair<std::array<float, 4>, float> KoopmanMpcController::solveStep(
    const std::array<float, 6>& state0,
    const std::vector<std::array<float, 6>>& ref_window) {
    const int64_t H = cfg_.horizon;
    auto s0 = array6ToTensor(state0);
    auto ref = refWindowToTensor(ref_window);

    torch::Tensor u0;
    if (has_warm_) {
        u0 = torch::cat({u_warm_.index({torch::indexing::Slice(1, torch::indexing::None)}),
                         u_warm_.index({torch::indexing::Slice(-1, torch::indexing::None)})},
                        0);
    } else {
        u0 = torch::zeros({H, 4}, torch::kFloat32);
    }
    u0 = clampU(u0);
    auto u_prev = u0.index({0}).detach().clone();

    auto u_param = u0.clone().set_requires_grad(true);
    torch::optim::Adam optimizer({u_param}, torch::optim::AdamOptions(cfg_.opt_lr));

    float best_cost = 1e30f;
    torch::Tensor best_u = u0.clone();

    for (int it = 0; it < cfg_.opt_iters; ++it) {
        optimizer.zero_grad();
        auto u_clamped = clampU(u_param);
        auto cost = mpcCost(s0, ref, u_clamped.reshape({-1}), u_prev);
        cost.backward();
        optimizer.step();

        const float cval = cost.item<float>();
        if (cval < best_cost) {
            best_cost = cval;
            best_u = clampU(u_param).detach().clone();
        }
    }

    u_warm_ = best_u.clone();
    has_warm_ = true;

    std::array<float, 4> u0_out{};
    auto acc = best_u.accessor<float, 2>();
    for (int j = 0; j < 4; ++j) {
        u0_out[j] = acc[0][j];
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
        u_warm_ = torch::zeros({H, 4}, torch::kFloat32);
        auto acc = u_warm_.accessor<float, 2>();
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < 4; ++j) {
                acc[i][j] = (*ref_ctrl)[i][j];
            }
        }
        has_warm_ = true;
    }

    std::array<float, 6> cur = state0;
    for (int t = 0; t < n_sim; ++t) {
        traj.t[t + 1] = (t + 1) * cfg_.dt;

        std::vector<std::array<float, 6>> ref_win;
        ref_win.reserve(H + 1);
        for (int k = 0; k <= H; ++k) {
            int idx = t + k;
            if (idx < T) {
                ref_win.push_back(ref_traj[idx]);
            } else {
                ref_win.push_back(ref_traj[T - 1]);
            }
        }

        auto [u_opt, c] = solveStep(cur, ref_win);
        traj.control[t] = u_opt;
        traj.cost_history.push_back(c);

        auto s0 = array6ToTensor(cur);
        auto u_roll = torch::zeros({KoopmanTorchModel::kTracedHorizon, 4}, torch::kFloat32);
        auto uacc = u_roll.accessor<float, 2>();
        for (int j = 0; j < 4; ++j) {
            uacc[0][j] = u_opt[j];
        }
        for (int i = 1; i < KoopmanTorchModel::kTracedHorizon; ++i) {
            for (int j = 0; j < 4; ++j) {
                uacc[i][j] = uacc[0][j];
            }
        }
        auto next = model_.rollout(s0, u_roll, cfg_.dt);
        auto nacc = next.accessor<float, 2>();
        for (int j = 0; j < 6; ++j) {
            cur[j] = nacc[1][j];
        }
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
        float dyaw = traj.state[i][2] - traj.ref_state[i][2];
        dyaw = std::atan2(std::sin(dyaw), std::cos(dyaw));
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
