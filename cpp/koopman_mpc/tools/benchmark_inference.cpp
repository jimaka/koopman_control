/** ONNX rollout 与 MPC solveStep 推理耗时 benchmark */
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "koopman_control/koopman_onnx_model.hpp"
#include "koopman_control/mpc_config_loader.hpp"
#include "koopman_control/mpc_controller.hpp"

namespace {

struct LatencyStats {
    double mean_ms = 0.;
    double median_ms = 0.;
    double min_ms = 0.;
    double max_ms = 0.;
    double p95_ms = 0.;
};

LatencyStats summarize(std::vector<double> lat_ms) {
    if (lat_ms.empty()) {
        return {};
    }
    std::sort(lat_ms.begin(), lat_ms.end());
    LatencyStats s;
    s.min_ms = lat_ms.front();
    s.max_ms = lat_ms.back();
    s.mean_ms = std::accumulate(lat_ms.begin(), lat_ms.end(), 0.0) / lat_ms.size();
    s.median_ms = lat_ms[lat_ms.size() / 2];
    const size_t p95_idx = static_cast<size_t>(std::ceil(0.95 * lat_ms.size())) - 1;
    s.p95_ms = lat_ms[std::min(p95_idx, lat_ms.size() - 1)];
    return s;
}

void printStats(const char* label, const LatencyStats& s, int warmup, int iters) {
    std::cout << "=== " << label << " ===\n";
    std::cout << "  warmup/iters: " << warmup << "/" << iters << "\n";
    std::cout << "  mean_ms  : " << s.mean_ms << "\n";
    std::cout << "  median_ms: " << s.median_ms << "\n";
    std::cout << "  min_ms   : " << s.min_ms << "\n";
    std::cout << "  max_ms   : " << s.max_ms << "\n";
    std::cout << "  p95_ms   : " << s.p95_ms << "\n";
    if (s.mean_ms > 0.) {
        std::cout << "  fps      : " << (1000.0 / s.mean_ms) << "\n";
    }
}

bool loadRolloutInputs(const std::string& txt_path,
                       std::array<float, 6>& state0,
                       std::vector<float>& u_flat,
                       float& dt) {
    std::ifstream in(txt_path);
    if (!in) {
        return false;
    }
    for (int i = 0; i < 6; ++i) {
        in >> state0[i];
    }
    int H = 0;
    in >> H;
    u_flat.resize(H * 4);
    for (int i = 0; i < H * 4; ++i) {
        in >> u_flat[i];
    }
    dt = 1.0f;
    return true;
}

std::vector<std::array<float, 6>> makeRefWindow(int horizon) {
    std::vector<std::array<float, 6>> ref(static_cast<size_t>(horizon + 1));
    for (int k = 0; k <= horizon; ++k) {
        ref[k] = {static_cast<float>(k), 0.f, 0.f, 2.f, 0.f, 0.f};
    }
    return ref;
}

}  // namespace

int main(int argc, char** argv) {
    std::string onnx = "cpp/koopman_mpc/weights/koopman_rollout.onnx";
    std::string config = "cpp/koopman_mpc/src/mpc_config.yaml";
    std::string check_txt = "cpp/koopman_mpc/weights/rollout_check.npz";
    int warmup = 50;
    int iters = 1000;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--onnx" && i + 1 < argc) {
            onnx = argv[++i];
        } else if (arg == "--config" && i + 1 < argc) {
            config = argv[++i];
        } else if (arg == "--check" && i + 1 < argc) {
            check_txt = argv[++i];
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup = std::stoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            iters = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: benchmark_inference [--onnx PATH] [--config PATH]\n"
                      << "       [--check rollout_check.npz] [--warmup N] [--iters N]\n";
            return 0;
        }
    }

    try {
        std::array<float, 6> state0{};
        std::vector<float> u_flat;
        float dt = 1.0f;
        if (!loadRolloutInputs(check_txt + ".txt", state0, u_flat, dt)) {
            std::cerr << "Missing " << check_txt << ".txt\n";
            return 1;
        }

        koopman_control::KoopmanOnnxModel model(onnx);
        const int H = model.horizon();
        std::cout << "ONNX: " << onnx << "  horizon=" << H << "  dt=" << dt << "\n";

        std::vector<double> rollout_lat;
        rollout_lat.reserve(static_cast<size_t>(warmup + iters));
        for (int i = 0; i < warmup + iters; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            (void)model.rollout(state0, u_flat, dt);
            const double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (i >= warmup) {
                rollout_lat.push_back(ms);
            }
        }
        printStats("ONNX rollout (single forward)", summarize(rollout_lat), warmup, iters);

        koopman_control::MpcConfig cfg;
        cfg = koopman_control::loadMpcConfigFromYaml(config, cfg);
        cfg = koopman_control::syncHorizonWithOnnx(cfg, H);
        koopman_control::KoopmanMpcController mpc(std::move(model), cfg);

        const auto ref_window = makeRefWindow(H);
        std::array<float, 4> u_prev{0.f, 0.f, 0.f, 0.f};

        std::vector<double> mpc_lat;
        mpc_lat.reserve(static_cast<size_t>(warmup + iters));
        for (int i = 0; i < warmup + iters; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            (void)mpc.solveStep(state0, ref_window, &u_prev);
            const double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (i >= warmup) {
                mpc_lat.push_back(ms);
            }
        }

        std::cout << "MPC config: opt_iters=" << cfg.opt_iters
                  << " opt_control_steps=" << cfg.opt_control_steps
                  << " control_hold_steps=" << cfg.control_hold_steps << "\n";
        printStats("MPC solveStep (full optimization)", summarize(mpc_lat), warmup, iters);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
