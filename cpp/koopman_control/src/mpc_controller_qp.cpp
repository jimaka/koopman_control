/**
 * @file mpc_controller_qp.cpp
 */

#include "koopman_control/mpc_controller_qp.hpp"

#include <cmath>
#include <stdexcept>

namespace koopman_control {

KoopmanQpMpcController::KoopmanQpMpcController(std::string latent_yaml_path, MpcConfig cfg, LatentMpcQpConfig qp_cfg)
    : cfg_(cfg), qp_cfg_(qp_cfg), solver_(model_, cfg_, qp_cfg_) {
    model_.loadFromYaml(latent_yaml_path, cfg_.horizon);
    model_.precomputePredictionMatrices();
    encoder_.loadFromYaml(latent_yaml_path);
    if (model_.horizon() != cfg_.horizon) {
        throw std::runtime_error("model horizon != mpc horizon");
    }
}

std::array<float, 3> KoopmanQpMpcController::normalizeDyn(const std::array<float, 3>& dyn) const {
    std::array<float, 3> out{};
    for (int i = 0; i < 3; ++i) {
        out[static_cast<size_t>(i)] =
            (dyn[static_cast<size_t>(i)] - model_.dynMean()[static_cast<size_t>(i)]) /
            model_.dynStd()[static_cast<size_t>(i)];
    }
    return out;
}

std::vector<float> KoopmanQpMpcController::buildRefLatentStack(
    const std::vector<std::array<float, 6>>& ref_window) const {
    const int n = model_.horizon();
    const int nz = model_.nz();
    std::vector<float> stack(static_cast<size_t>(nz * n), 0.f);
  for (int k = 0; k < n; ++k) {
        const size_t idx = static_cast<size_t>(std::min(k, static_cast<int>(ref_window.size()) - 1));
        std::array<float, 3> dyn = {ref_window[idx][3], ref_window[idx][4], ref_window[idx][5]};
        const auto dyn_n = normalizeDyn(dyn);
        const std::vector<float> z = encoder_.encode(dyn_n);
        for (int i = 0; i < nz; ++i) {
            stack[static_cast<size_t>(k * nz + i)] = z[static_cast<size_t>(i)];
        }
    }
    return stack;
}

std::pair<std::array<float, 4>, float> KoopmanQpMpcController::solveStep(
    const std::array<float, 6>& state0,
    const std::vector<std::array<float, 6>>& ref_window,
    const std::array<float, 4>* u_prev_applied) {
    if (static_cast<int>(ref_window.size()) < model_.horizon() + 1) {
        throw std::runtime_error("ref_window too short");
    }

    const std::array<float, 3> dyn = {state0[3], state0[4], state0[5]};
    const std::vector<float> z0 = encoder_.encode(normalizeDyn(dyn));
    const std::vector<float> z_ref = buildRefLatentStack(ref_window);

    std::array<float, 4> u_prev{};
    if (u_prev_applied) {
        u_prev = *u_prev_applied;
    }

    const std::vector<float>* warm = has_warm_ ? &u_warm_tilde_ : nullptr;
    const LatentMpcQpSolution sol = solver_.solve(z0, z_ref, u_prev, warm);

    std::vector<float> u_full(static_cast<size_t>(model_.horizon() * model_.nu()), 0.f);
    for (int k = 0; k < model_.horizon(); ++k) {
        for (int j = 0; j < model_.nu(); ++j) {
            if (k < static_cast<int>(sol.u_tilde_stack.size()) / model_.nu()) {
                u_full[static_cast<size_t>(k * model_.nu() + j)] =
                    sol.u_tilde_stack[static_cast<size_t>(k * model_.nu() + j)];
            }
        }
    }
    u_warm_tilde_ = u_full;
    has_warm_ = true;

    std::array<float, 4> u0_phys = model_.denormalizeControl(
        {sol.u_tilde_stack[0], sol.u_tilde_stack[1], sol.u_tilde_stack[2], sol.u_tilde_stack[3]});
    return {u0_phys, sol.cost};
}

void KoopmanQpMpcController::resetWarmStart() {
    u_warm_tilde_.clear();
    has_warm_ = false;
}

}  // namespace koopman_control
