#include <yaml-cpp/yaml.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "koopman_onnx_model.hpp"
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
  std::string config_path = "";
  int steps = 40;
  bool smoketest = false;

  koopman_mpc::MpcConfig cfg;

  // 1. 初次遍历：查找并加载 --config 参数
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--config" && i + 1 < argc) {
      config_path = argv[i + 1];
      break;
    }
  }

  // 若提供了 YAML 配置文件，则解析并覆盖默认值
  if (!config_path.empty()) {
    try {
      YAML::Node yaml_node = YAML::LoadFile(resolvePath(config_path));
      if (yaml_node["horizon"]) cfg.horizon = yaml_node["horizon"].as<int>();
      if (yaml_node["opt_control_steps"])
        cfg.opt_control_steps = yaml_node["opt_control_steps"].as<int>();
      if (yaml_node["dt"]) cfg.dt = yaml_node["dt"].as<float>();
      if (yaml_node["w_xy"]) cfg.w_xy = yaml_node["w_xy"].as<float>();
      if (yaml_node["w_yaw"]) cfg.w_yaw = yaml_node["w_yaw"].as<float>();
      if (yaml_node["w_vel"]) cfg.w_vel = yaml_node["w_vel"].as<float>();
      if (yaml_node["w_u"]) cfg.w_u = yaml_node["w_u"].as<float>();
      if (yaml_node["w_du"]) cfg.w_du = yaml_node["w_du"].as<float>();
      if (yaml_node["opt_iters"])
        cfg.opt_iters = yaml_node["opt_iters"].as<int>();
      if (yaml_node["opt_lr"]) cfg.opt_lr = yaml_node["opt_lr"].as<float>();
      if (yaml_node["u_min"] && yaml_node["u_min"].IsSequence() &&
          yaml_node["u_min"].size() == 4) {
        for (int j = 0; j < 4; ++j)
          cfg.u_min[j] = yaml_node["u_min"][j].as<float>();
      }
      if (yaml_node["u_max"] && yaml_node["u_max"].IsSequence() &&
          yaml_node["u_max"].size() == 4) {
        for (int j = 0; j < 4; ++j)
          cfg.u_max[j] = yaml_node["u_max"][j].as<float>();
      }

      if (yaml_node["steps"]) steps = yaml_node["steps"].as<int>();
      if (yaml_node["weights_dir"])
        weights_dir = yaml_node["weights_dir"].as<std::string>();
      if (yaml_node["ref_json"])
        ref_json = yaml_node["ref_json"].as<std::string>();
    } catch (const std::exception& e) {
      std::cerr << "Error loading YAML config '" << config_path
                << "': " << e.what() << "\n";
      return 1;
    }
  }

  // 2. 二次遍历：允许通过命令行参数最终覆盖 YAML 配置或默认值
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--smoketest") {
      smoketest = true;
    } else if (arg == "--config" && i + 1 < argc) {
      ++i;  // 已在第一阶段处理过，直接跳过
    } else if (arg == "--weights" && i + 1 < argc) {
      weights_dir = argv[++i];
    } else if (arg == "--ref" && i + 1 < argc) {
      ref_json = argv[++i];
    } else if (arg == "--steps" && i + 1 < argc) {
      steps = std::stoi(argv[++i]);
    } else if (arg == "--horizon" && i + 1 < argc) {
      cfg.horizon = std::stoi(argv[++i]);
    } else if (arg == "--opt_control_steps" && i + 1 < argc) {
      cfg.opt_control_steps = std::stoi(argv[++i]);
    } else if (arg == "--opt_iters" && i + 1 < argc) {
      cfg.opt_iters = std::stoi(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: koopman_mpc_cpp [--config config.yaml] [--weights DIR] "
             "[--ref cpp_test_ref.json]\n"
          << "       [--steps N] [--horizon H] [--opt_control_steps N] [--opt_iters N] [--smoketest]\n";
      return 0;
    }
  }

  const std::string onnx_path =
      resolvePath(weights_dir + "/koopman_rollout.onnx");
  ref_json = resolvePath(ref_json);

  try {
    koopman_mpc::KoopmanOnnxModel model(onnx_path);
    const int onnx_horizon = model.horizon();
    if (cfg.horizon != onnx_horizon) {
      std::cout << "[config] horizon " << cfg.horizon << " -> " << onnx_horizon
                << " (from ONNX)\n";
      cfg.horizon = onnx_horizon;
    }
    if (smoketest) {
      cfg.opt_iters = 5;
      cfg.opt_control_steps = std::min(cfg.opt_control_steps, 20);
      steps = 10;
    }

    koopman_mpc::KoopmanMpcController mpc(std::move(model), cfg);

    std::ifstream in(ref_json);
    if (!in) {
      std::cerr
          << "Cannot open ref file: " << ref_json << "\n"
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

    std::cout << "=== MPC TRACKING (C++ / ONNX) ===\n";
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
