#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "koopman_torch_model.hpp"
#include "mpc_controller.hpp"

namespace fs = std::filesystem;

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
    std::string weights_dir = "cpp/koopman_mpc/weights";
    std::string ref_json = "cpp/koopman_mpc/weights/cpp_test_ref.json";
    int steps = 40;
    int horizon = 20;
    int opt_iters = 25;
    bool smoketest = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--smoketest") {
            smoketest = true;
        } else if (arg == "--weights" && i + 1 < argc) {
            weights_dir = argv[++i];
        } else if (arg == "--ref" && i + 1 < argc) {
            ref_json = argv[++i];
        } else if (arg == "--steps" && i + 1 < argc) {
            steps = std::stoi(argv[++i]);
        } else if (arg == "--horizon" && i + 1 < argc) {
            horizon = std::stoi(argv[++i]);
        } else if (arg == "--opt_iters" && i + 1 < argc) {
            opt_iters = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: koopman_mpc_cpp [--weights DIR] [--ref cpp_test_ref.json]\n"
                << "       [--steps N] [--horizon H] [--opt_iters N] [--smoketest]\n";
            return 0;
        }
    }

    const std::string ts_path = resolvePath(weights_dir + "/koopman_rollout.pt");
    ref_json = resolvePath(ref_json);

    try {
        koopman_mpc::KoopmanTorchModel model(ts_path);
        koopman_mpc::MpcConfig cfg;
        cfg.horizon = horizon;
        cfg.opt_iters = opt_iters;
        if (smoketest) {
            cfg.horizon = koopman_mpc::KoopmanTorchModel::kTracedHorizon;
            cfg.opt_iters = 12;
            steps = 20;
        }

        koopman_mpc::KoopmanMpcController mpc(std::move(model), cfg);

        // 读取 export_cpp_test_ref.py 写的参考航迹（纯文本 CSV 风格）
        std::ifstream in(ref_json);
        if (!in) {
            std::cerr << "Cannot open ref file: " << ref_json << "\n"
                      << "Run: python3 cpp/koopman_mpc/scripts/export_cpp_test_ref.py\n";
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
        auto metrics = koopman_mpc::computeMetrics(traj);

        std::cout << "=== MPC TRACKING (C++) ===\n";
        std::cout << "  xy_rmse_m: " << metrics.xy_rmse_m << "\n";
        std::cout << "  xy_max_m: " << metrics.xy_max_m << "\n";
        std::cout << "  yaw_rmse_deg: " << metrics.yaw_rmse_deg << "\n";
        std::cout << "  final_xy_err_m: " << metrics.final_xy_err_m << "\n";
        std::cout << "  steps: " << steps << " horizon: " << cfg.horizon << "\n";
        std::cout << "========================\n";

        if (smoketest && metrics.xy_rmse_m > 5.f) {
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
