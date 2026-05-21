/**
 * @file mpc_config_loader.cpp
 * @brief YAML 配置加载与 ONNX horizon 同步
 */

#include "koopman_control/mpc_config_loader.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>
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
    if (node["control_hold_steps"]) {
        cfg.control_hold_steps = node["control_hold_steps"].as<int>();
    }
    if (node["control_period"]) {
        const float period = node["control_period"].as<float>();
        if (period > 0.f && cfg.dt > 0.f) {
            cfg.control_hold_steps =
                std::max(1, static_cast<int>(std::lround(period / cfg.dt)));
        }
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
    if (node["w_du_throttle"]) {
        cfg.w_du_throttle = node["w_du_throttle"].as<float>();
    }
    if (node["w_du_rudder"]) {
        cfg.w_du_rudder = node["w_du_rudder"].as<float>();
    }
    if (node["throttle_du_max"]) {
        cfg.throttle_du_max = node["throttle_du_max"].as<float>();
    }
    if (node["rudder_du_max"]) {
        cfg.rudder_du_max = node["rudder_du_max"].as<float>();
    }
    if (node["du_max"] && node["du_max"].IsSequence() && node["du_max"].size() == 4) {
        for (int j = 0; j < 4; ++j) {
            cfg.du_max[j] = node["du_max"][j].as<float>();
        }
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
