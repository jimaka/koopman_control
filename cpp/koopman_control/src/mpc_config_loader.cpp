#include "koopman_control/mpc_config_loader.hpp"

#include <yaml-cpp/yaml.h>

#include <stdexcept>

namespace koopman_control {

MpcConfig syncHorizonWithOnnx(MpcConfig cfg, int onnx_horizon) {
    cfg.horizon = onnx_horizon;
    if (cfg.opt_control_steps > cfg.horizon) {
        cfg.opt_control_steps = cfg.horizon;
    }
    return cfg;
}

MpcConfig loadMpcConfigFromYaml(const std::string& yaml_path, MpcConfig cfg) {
    YAML::Node node = YAML::LoadFile(yaml_path);
    if (node["horizon"]) {
        cfg.horizon = node["horizon"].as<int>();
    }
    if (node["opt_control_steps"]) {
        cfg.opt_control_steps = node["opt_control_steps"].as<int>();
    }
    if (node["dt"]) {
        cfg.dt = node["dt"].as<float>();
    }
    if (node["w_xy"]) {
        cfg.w_xy = node["w_xy"].as<float>();
    }
    if (node["w_yaw"]) {
        cfg.w_yaw = node["w_yaw"].as<float>();
    }
    if (node["w_vel"]) {
        cfg.w_vel = node["w_vel"].as<float>();
    }
    if (node["w_u"]) {
        cfg.w_u = node["w_u"].as<float>();
    }
    if (node["w_du"]) {
        cfg.w_du = node["w_du"].as<float>();
    }
    if (node["opt_iters"]) {
        cfg.opt_iters = node["opt_iters"].as<int>();
    }
    if (node["opt_lr"]) {
        cfg.opt_lr = node["opt_lr"].as<float>();
    }
    if (node["u_min"] && node["u_min"].IsSequence() && node["u_min"].size() == 4) {
        for (int j = 0; j < 4; ++j) {
            cfg.u_min[j] = node["u_min"][j].as<float>();
        }
    }
    if (node["u_max"] && node["u_max"].IsSequence() && node["u_max"].size() == 4) {
        for (int j = 0; j < 4; ++j) {
            cfg.u_max[j] = node["u_max"][j].as<float>();
        }
    }
    return cfg;
}

}  // namespace koopman_control
