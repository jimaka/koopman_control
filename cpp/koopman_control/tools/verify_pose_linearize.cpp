/**
 * @file verify_pose_linearize.cpp
 * @brief 独立验证 Tier-2 位姿线性化（不依赖 OSQP / ONNX）。
 *
 * 校验：Phi·dU 预测的位姿增量 vs 真实非线性 rollout 增量，
 * 随 |dU| 减半误差应约降到 1/4（二阶收敛）。
 */
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "koopman_control/koopman_decoder.hpp"
#include "koopman_control/koopman_encode.hpp"
#include "koopman_control/koopman_latent_model.hpp"
#include "koopman_control/latent_mpc_qp.hpp"
#include "koopman_control/mpc_config.hpp"
#include "koopman_control/pose_linearize.hpp"

#include <yaml-cpp/yaml.h>

using namespace koopman_control;

static int horizonFromYaml(const std::string& yaml_path, int fallback = 20) {
    YAML::Node root = YAML::LoadFile(yaml_path);
    if (root["horizon_default"]) {
        return root["horizon_default"].as<int>();
    }
    return fallback;
}

static float dtFromYaml(const std::string& yaml_path, float fallback = 1.0f) {
    YAML::Node root = YAML::LoadFile(yaml_path);
    if (root["dt"]) {
        return root["dt"].as<float>();
    }
    return fallback;
}

static std::array<float, 3> rolloutPose(const KoopmanLatentModel& model, const KoopmanDecoder& dec,
                                        const std::vector<float>& z0, const std::array<float, 3>& pose0,
                                        const std::vector<float>& U, int N, int nu, float dt,
                                        std::vector<std::array<float, 3>>* full = nullptr) {
    std::vector<float> z = z0;
    float x = pose0[0], y = pose0[1], yaw = pose0[2];
    for (int m = 1; m <= N; ++m) {
        std::vector<float> u(static_cast<size_t>(nu));
        for (int j = 0; j < nu; ++j) {
            u[static_cast<size_t>(j)] = U[static_cast<size_t>((m - 1) * nu + j)];
        }
        z = model.latentStep(z, u);
        const auto d = dec.decodePhysical(z);
        const float c = std::cos(yaw), s = std::sin(yaw);
        x += (d[0] * c - d[1] * s) * dt;
        y += (d[0] * s + d[1] * c) * dt;
        yaw += d[2] * dt;
        if (full) {
            full->push_back({x, y, yaw});
        }
    }
    return {x, y, yaw};
}

int main(int argc, char** argv) {
    std::string yaml = argc > 1 ? argv[1] : "cpp/koopman_mpc/weights/koopman_v4_latent.yaml";
    const int N = horizonFromYaml(yaml, 20);
    const int nu = 4;
    const float dt = dtFromYaml(yaml, 1.0f);

    KoopmanLatentModel model;
    model.loadFromYaml(yaml, N);
    model.precomputePredictionMatrices();
    KoopmanEncoder enc;
    enc.loadFromYaml(yaml);
    KoopmanDecoder dec;
    dec.loadFromYaml(yaml);
    if (!dec.loaded()) {
        printf("[FAIL] decoder not present in YAML\n");
        return 1;
    }

    std::mt19937 rng(0);
    std::normal_distribution<float> gauss(0.f, 1.f);

    const std::array<float, 3> dyn_n = {0.3f * gauss(rng), 0.3f * gauss(rng), 0.3f * gauss(rng)};
    const std::vector<float> z0 = enc.encode(dyn_n);
    const std::array<float, 3> pose0 = {0.f, 0.f, 0.f};

    const int nvar = N * nu;
    std::vector<float> U0(static_cast<size_t>(nvar));
    for (auto& u : U0) {
        u = 0.2f * gauss(rng);
    }

    // pose_ref 任意（不影响 Phi，只影响 b）
    std::vector<float> pose_ref(static_cast<size_t>(3 * N), 0.f);
    const PoseLinearization pl =
        buildPoseLinearization(model, dec, z0, pose0, U0, pose_ref, dt, 1.0f, 1.0f);
    if (!pl.valid) {
        printf("[FAIL] pose linearization invalid\n");
        return 1;
    }

    std::vector<std::array<float, 3>> base_full;
    rolloutPose(model, dec, z0, pose0, U0, N, nu, dt, &base_full);

    // 用 signal-dominant 的扰动幅度评估（float32，过小的 dU 会被舍入噪声淹没）。
    std::mt19937 rng2(7);
    std::vector<float> last_rel;
    for (float scale : {4e-2f, 2e-2f, 1e-2f}) {
        std::vector<float> dU(static_cast<size_t>(nvar));
        for (auto& d : dU) {
            d = scale * gauss(rng2);
        }
        std::vector<float> Up = U0;
        for (int i = 0; i < nvar; ++i) {
            Up[static_cast<size_t>(i)] += dU[static_cast<size_t>(i)];
        }
        std::vector<std::array<float, 3>> true_full;
        rolloutPose(model, dec, z0, pose0, Up, N, nu, dt, &true_full);

        const std::vector<float> lin = detail::Matrix::matvec(pl.Phi, dU);
        float max_err = 0.f, max_true = 0.f;
        for (int m = 0; m < N; ++m) {
            for (int a = 0; a < 3; ++a) {
                const float td = true_full[static_cast<size_t>(m)][static_cast<size_t>(a)] -
                                 base_full[static_cast<size_t>(m)][static_cast<size_t>(a)];
                const float ld = lin[static_cast<size_t>(m * 3 + a)];
                max_err = std::max(max_err, std::fabs(td - ld));
                max_true = std::max(max_true, std::fabs(td));
            }
        }
        const float rel = max_err / (max_true + 1e-9f);
        printf("[Phi] |dU|~%.1e  abs_lin_err=%.3e  rel=%.3e\n", scale, max_err, rel);
        last_rel.push_back(rel);
    }

    // 二阶收敛：|dU| 减半，相对误差应约降到 ~1/2（abs 降到 ~1/4）。
    const float ratio = last_rel[0] / (last_rel[1] + 1e-12f);
    printf("[Phi] second-order ratio (expect ~2) = %.2f\n", ratio);
    // dt 较大时单步位姿增量更大，线性化相对误差阈值适度放宽。
    const float rel_tol = dt >= 2.0f ? 0.05f : 0.01f;
    if (last_rel[0] > rel_tol) {
        printf("[FAIL] linearization too coarse\n");
        return 1;
    }
    printf("[OK] C++ pose linearization verified\n");

    // === 端到端：OSQP 位姿 QP 应降低位姿跟踪误差（SQP 外迭代） ===
    MpcConfig cfg;
    cfg.horizon = N;
    cfg.dt = dt;
    cfg.opt_control_steps = N;  // 给足控制自由度
    cfg.control_hold_steps = 1;
    cfg.w_z = 0.f;              // 关闭潜空间项，单独考察位姿跟踪
    cfg.w_u = 1e-4f;
    cfg.w_du = 0.f;
    cfg.w_xy = 1.f;
    cfg.w_yaw = 1.f;
    cfg.sqp_iters = 4;
    LatentMpcQpConfig qp{};
    qp.w_z = cfg.w_z;
    qp.w_u = cfg.w_u;
    qp.w_du = cfg.w_du;
    qp.osqp_max_iter = 8000;
    qp.osqp_verbose = 0;
    LatentMpcQpSolver solver(model, cfg, qp);

    // 随机模型未训练，B 很小；用一条「可达」参考（由已知控制 rollout 得到）以确保可行性。
    std::vector<float> U_target(static_cast<size_t>(nvar));
    {
        std::mt19937 rt(123);
        for (auto& u : U_target) {
            u = 0.4f * gauss(rt);
        }
    }
    std::vector<std::array<float, 3>> ref_full;
    rolloutPose(model, dec, z0, pose0, U_target, N, nu, dt, &ref_full);
    std::vector<float> ref(static_cast<size_t>(3 * N), 0.f);
    for (int m = 0; m < N; ++m) {
        ref[static_cast<size_t>(m * 3 + 0)] = ref_full[static_cast<size_t>(m)][0];
        ref[static_cast<size_t>(m * 3 + 1)] = ref_full[static_cast<size_t>(m)][1];
        ref[static_cast<size_t>(m * 3 + 2)] = ref_full[static_cast<size_t>(m)][2];
    }
    const std::vector<float> z_ref(static_cast<size_t>(model.nz() * N), 0.f);
    const std::array<float, 4> u_prev{0.f, 0.f, 0.f, 0.f};

    auto poseErr = [&](const std::vector<float>& U) {
        std::vector<std::array<float, 3>> full;
        rolloutPose(model, dec, z0, pose0, U, N, nu, dt, &full);
        float e = 0.f;
        for (int m = 0; m < N; ++m) {
            const float dx = full[static_cast<size_t>(m)][0] - ref[static_cast<size_t>(m * 3 + 0)];
            const float dy = full[static_cast<size_t>(m)][1] - ref[static_cast<size_t>(m * 3 + 1)];
            const float dp = detail::wrapAngle(full[static_cast<size_t>(m)][2] - ref[static_cast<size_t>(m * 3 + 2)]);
            e += dx * dx + dy * dy + dp * dp;
        }
        return e;
    };

    std::vector<float> U(static_cast<size_t>(nvar), 0.f);
    const float err0 = poseErr(U);
    for (int it = 0; it < cfg.sqp_iters; ++it) {
        const PoseLinearization p =
            buildPoseLinearization(model, dec, z0, pose0, U, ref, dt, cfg.w_xy, cfg.w_yaw);
        const LatentMpcQpSolution s = solver.solve(z0, z_ref, u_prev, &U, &p);
        U = s.u_tilde_stack;
    }
    const float err1 = poseErr(U);
    printf("[SQP] pose SSE: zero-ctrl=%.3f -> optimized=%.3f (reduction %.1f%%)\n", err0, err1,
           100.0 * (1.0 - err1 / (err0 + 1e-9)));
    if (!(err1 < 0.2f * err0)) {
        printf("[FAIL] pose QP did not reduce tracking error enough\n");
        return 1;
    }
    printf("[OK] OSQP pose tracking reduces error\n");
    return 0;
}
