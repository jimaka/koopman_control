#include <yaml-cpp/yaml.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "koopman_control/koopman_onnx_model.hpp"
#include "koopman_control/mpc_config_loader.hpp"
#include "koopman_control/mpc_controller.hpp"

namespace fs = std::filesystem;
using koopman_control::KoopmanMpcController;
using koopman_control::MpcConfig;
using koopman_control::computeMetrics;
using koopman_control::latentQpConfigFromMpc;
using koopman_control::loadMpcConfigFromYaml;
using koopman_control::syncHorizonWithOnnx;

static std::string resolvePath(const std::string& rel) {
    if (fs::exists(rel)) {
        return fs::absolute(rel).string();
    }
    const fs::path from_cpp = fs::path("../../../") / rel;
    if (fs::exists(from_cpp)) {
        return fs::absolute(from_cpp).string();
    }
    return rel;
}

int main(int argc, char** argv) {
    std::string ref_json = "cpp/koopman_mpc/weights/cpp_test_ref.json";
    std::string config_path = "cpp/koopman_control/config/mpc_config.yaml";
    int steps = 40;
    bool smoketest = false;

    MpcConfig cfg;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--smoketest") {
            smoketest = true;
        } else if (arg == "--ref" && i + 1 < argc) {
            ref_json = argv[++i];
        } else if (arg == "--steps" && i + 1 < argc) {
            steps = std::stoi(argv[++i]);
        } else if (arg == "--horizon" && i + 1 < argc) {
            cfg.horizon = std::stoi(argv[++i]);
        } else if (arg == "--opt_control_steps" && i + 1 < argc) {
            cfg.opt_control_steps = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: koopman_mpc_cpp [--config mpc_config.yaml] [--ref ref.json]\n"
                      << "       [--steps N] [--horizon H] [--opt_control_steps N] [--smoketest]\n";
            return 0;
        }
    }

    config_path = resolvePath(config_path);
    cfg = loadMpcConfigFromYaml(config_path, cfg);
    ref_json = resolvePath(ref_json);
    cfg.latent_model = resolvePath(cfg.latent_model);
    cfg.onnx_plant = resolvePath(cfg.onnx_plant);

    if (smoketest) {
        cfg.opt_control_steps = std::min(cfg.opt_control_steps, 20);
        steps = 10;
    }

    try {
        KoopmanOnnxModel plant(cfg.onnx_plant);
        cfg = syncHorizonWithOnnx(cfg, plant.horizon());
        KoopmanMpcController mpc(cfg.latent_model, cfg.onnx_plant, cfg, latentQpConfigFromMpc(cfg));

        std::ifstream in(ref_json);
        if (!in) {
            std::cerr << "Cannot open ref file: " << ref_json << "\n";
            return 1;
        }

        std::vector<std::array<float, 6>> ref_state;
        std::vector<std::array<float, 4>> ref_ctrl;
        std::string tag;
        while (in >> tag) {
            if (tag == "state") {
                std::array<float, 6> row{};
                for (int j = 0; j < 6; ++j) {
                    in >> row[j];
                }
                ref_state.push_back(row);
            } else if (tag == "ctrl") {
                std::array<float, 4> row{};
                for (int j = 0; j < 4; ++j) {
                    in >> row[j];
                }
                ref_ctrl.push_back(row);
            }
        }
        if (ref_state.size() < 3) {
            std::cerr << "Ref trajectory too short in " << ref_json << "\n";
            return 1;
        }

        auto traj = mpc.simulate(ref_state[0], ref_state, &ref_ctrl, steps);
        auto metrics = computeMetrics(traj);

        std::cout << "=== MPC TRACKING (C++ / OSQP) ===\n";
        std::cout << "  xy_rmse_m: " << metrics.xy_rmse_m << "\n";
        std::cout << "  xy_max_m: " << metrics.xy_max_m << "\n";
        std::cout << "  yaw_rmse_deg: " << metrics.yaw_rmse_deg << "\n";
        std::cout << "  final_xy_err_m: " << metrics.final_xy_err_m << "\n";
        std::cout << "  steps: " << steps << " horizon: " << cfg.horizon
                  << " opt_control_steps: " << cfg.opt_control_steps << "\n";
        std::cout << "=================================\n";

        if (smoketest && metrics.xy_rmse_m > 8.f) {
            std::cerr << "[smoketest] FAIL xy_rmse too large\n";
            return 1;
        }
        if (smoketest) {
            std::cout << "[smoketest] OK\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
