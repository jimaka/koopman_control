/**
 * @file benchmark_latent_sqp.cpp
 * @brief 潜空间 MPC 各阶段耗时基准（Tier-1 QP / Tier-2 线性化 / SQP 外迭代）。
 *
 * 用于核对「SQP 外迭代次数」与控制周期的实时预算关系：
 *   ./benchmark_latent_sqp <latent_yaml> [repeats]
 *
 * 输出各阶段 mean / p50 / p95 耗时（ms），以及 sqp_iters=1..4 的整轮 solveStep 耗时。
 */
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "koopman_control/koopman_decoder.hpp"
#include "koopman_control/koopman_encode.hpp"
#include "koopman_control/koopman_latent_model.hpp"
#include "koopman_control/latent_mpc_qp.hpp"
#include "koopman_control/mpc_config.hpp"
#include "koopman_control/mpc_controller.hpp"
#include "koopman_control/pose_linearize.hpp"

using namespace koopman_control;
using Clock = std::chrono::high_resolution_clock;

namespace {

struct Stat {
    double mean{0.}, p50{0.}, p95{0.};
};

Stat summarize(std::vector<double> samples) {
    Stat s;
    if (samples.empty()) {
        return s;
    }
    std::sort(samples.begin(), samples.end());
    double sum = 0.;
    for (double v : samples) {
        sum += v;
    }
    s.mean = sum / static_cast<double>(samples.size());
    s.p50 = samples[samples.size() / 2];
    s.p95 = samples[static_cast<size_t>(0.95 * static_cast<double>(samples.size() - 1))];
    return s;
}

void report(const char* name, const Stat& s) {
    printf("  %-38s mean=%7.3f ms  p50=%7.3f ms  p95=%7.3f ms\n", name, s.mean, s.p50, s.p95);
}

double elapsedMs(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
    const std::string yaml = argc > 1 ? argv[1] : "cpp/koopman_mpc/weights/koopman_v4_latent.yaml";
    const int repeats = argc > 2 ? std::atoi(argv[2]) : 200;

    YAML::Node root = YAML::LoadFile(yaml);
    const int N = root["horizon_default"] ? root["horizon_default"].as<int>() : 10;
    const float dt = root["dt"] ? root["dt"].as<float>() : 4.0f;

    MpcConfig cfg;
    cfg.horizon = N;
    cfg.dt = dt;
    cfg.opt_control_steps = N;
    cfg.control_hold_steps = 1;
    cfg.throttle_du_max = 15.f;
    cfg.rudder_du_max = 3.5f;
    cfg.latent_model = yaml;

    KoopmanLatentModel model;
    model.loadFromYaml(yaml, N);
    model.precomputePredictionMatrices();
    KoopmanEncoder enc;
    enc.loadFromYaml(yaml);
    KoopmanDecoder dec;
    dec.loadFromYaml(yaml);
    if (!dec.loaded()) {
        printf("[FAIL] decoder missing in %s (Tier-2 unavailable)\n", yaml.c_str());
        return 1;
    }

    const int nz = model.nz();
    const int nu = model.nu();
    const int nvar = N * nu;
    printf("=== latent SQP benchmark ===\n");
    printf("  yaml=%s\n  nz=%d nu=%d N=%d dt=%.2fs nvar=%d repeats=%d\n", yaml.c_str(), nz, nu, N, dt,
           nvar, repeats);

    // 名义工作点：巡航速度 + 直线参考
    const std::array<float, 3> dyn_n = {0.5f, 0.f, 0.f};
    const std::vector<float> z0 = enc.encode(dyn_n);
    const std::vector<float> z_ref(static_cast<size_t>(nz * N), 0.f);
    std::vector<float> pose_ref(static_cast<size_t>(3 * N), 0.f);
    for (int m = 1; m <= N; ++m) {
        pose_ref[static_cast<size_t>((m - 1) * 3 + 0)] = 3.0f * dt * static_cast<float>(m);
    }
    const std::array<float, 3> pose0 = {0.f, 0.f, 0.f};
    const std::array<float, 4> u_prev = {40.f, 0.f, 40.f, 0.f};
    const std::vector<float> U0(static_cast<size_t>(nvar), 0.f);

    // --- 阶段耗时 ---
    std::vector<double> t_encode, t_lin, t_qp1, t_qp2;
    LatentMpcQpConfig qp{};
    qp.w_z = cfg.w_z;
    qp.w_u = cfg.w_u;
    qp.w_du = cfg.w_du;
    LatentMpcQpSolver solver(model, cfg, qp);
    // 预热（首次 solve 构建并缓存 Hessian）
    solver.solve(z0, z_ref, u_prev, &U0, nullptr);

    for (int i = 0; i < repeats; ++i) {
        auto t0 = Clock::now();
        const std::vector<float> z = enc.encode(dyn_n);
        t_encode.push_back(elapsedMs(t0));

        t0 = Clock::now();
        const PoseLinearization pl =
            buildPoseLinearization(model, dec, z0, pose0, U0, pose_ref, dt, 1.f, 50.f);
        t_lin.push_back(elapsedMs(t0));

        t0 = Clock::now();
        solver.solve(z0, z_ref, u_prev, &U0, nullptr);
        t_qp1.push_back(elapsedMs(t0));

        t0 = Clock::now();
        solver.solve(z0, z_ref, u_prev, &U0, &pl);
        t_qp2.push_back(elapsedMs(t0));
        (void)z;
    }

    printf("\n-- 单阶段 --\n");
    report("encoder encode (dyn -> z)", summarize(t_encode));
    report("buildPoseLinearization (Phi, b)", summarize(t_lin));
    report("QP solve Tier-1 (osqp_setup+solve)", summarize(t_qp1));
    report("QP solve Tier-2 (+PhiWPhi)", summarize(t_qp2));

    // --- 整轮 solveStep（含 SQP 外迭代）---
    printf("\n-- solveStep（含参考 encode + SQP 外迭代）--\n");
    std::vector<std::array<float, 6>> ref_window(static_cast<size_t>(N + 1));
    for (int k = 0; k <= N; ++k) {
        ref_window[static_cast<size_t>(k)] = {3.0f * dt * static_cast<float>(k), 0.f, 0.f,
                                              3.0f, 0.f, 0.f};
    }
    const std::array<float, 6> state0 = {0.f, 0.f, 0.f, 3.0f, 0.f, 0.f};

    for (const bool tier2 : {false, true}) {
        for (const int iters : {1, 2, 4, 8}) {
            if (!tier2 && iters > 1) {
                continue;  // Tier-1 与 sqp_iters 无关（代价精确二次）
            }
            MpcConfig c = cfg;
            c.w_xy = tier2 ? 1.f : 0.f;
            c.w_yaw = tier2 ? 50.f : 0.f;
            c.sqp_iters = iters;
            KoopmanMpcController ctrl(yaml, c, qp);
            ctrl.solveStep(state0, ref_window, &u_prev);  // 预热
            std::vector<double> samples;
            for (int i = 0; i < repeats; ++i) {
                MpcSolveTiming tm;
                auto t0 = Clock::now();
                ctrl.solveStep(state0, ref_window, &u_prev, &tm);
                samples.push_back(elapsedMs(t0));
            }
            char label[96];
            snprintf(label, sizeof(label), "%s sqp_iters=%d", tier2 ? "Tier-2" : "Tier-1", iters);
            report(label, summarize(samples));
        }
    }

    printf("\n  控制周期 dt=%.2f s = %.0f ms —— 以上耗时占比可直接读出实时余量。\n", dt,
           static_cast<double>(dt) * 1000.0);
    return 0;
}
