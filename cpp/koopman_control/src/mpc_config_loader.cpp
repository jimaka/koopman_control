/**
 * @file mpc_config_loader.cpp
 */

#include "koopman_control/mpc_config_loader.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>

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
    if (node["data_dt"]) {
        cfg.data_dt = node["data_dt"].as<float>();
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
    if (node["w_z"]) {
        cfg.w_z = node["w_z"].as<float>();
    }
    if (node["w_u"]) {
        cfg.w_u = node["w_u"].as<float>();
    }
    if (node["w_du"]) {
        cfg.w_du = node["w_du"].as<float>();
    }
    if (node["w_xy"]) {
        cfg.w_xy = node["w_xy"].as<float>();
    }
    if (node["w_yaw"]) {
        cfg.w_yaw = node["w_yaw"].as<float>();
    }
    if (node["sqp_iters"]) {
        cfg.sqp_iters = node["sqp_iters"].as<int>();
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
    if (node["osqp_eps_abs"]) {
        cfg.osqp_eps_abs = node["osqp_eps_abs"].as<float>();
    }
    if (node["osqp_eps_rel"]) {
        cfg.osqp_eps_rel = node["osqp_eps_rel"].as<float>();
    }
    if (node["osqp_max_iter"]) {
        cfg.osqp_max_iter = node["osqp_max_iter"].as<int>();
    }
    if (node["osqp_verbose"]) {
        cfg.osqp_verbose = node["osqp_verbose"].as<int>();
    }
    if (node["latent_model"]) {
        cfg.latent_model = node["latent_model"].as<std::string>();
    }
    if (node["onnx_plant"]) {
        cfg.onnx_plant = node["onnx_plant"].as<std::string>();
    }
    if (node["onnx_path"]) {
        cfg.onnx_plant = node["onnx_path"].as<std::string>();
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

LatentMpcQpConfig latentQpConfigFromMpc(const MpcConfig& cfg) {
    LatentMpcQpConfig qp{};
    qp.w_z = cfg.w_z;
    qp.w_u = cfg.w_u;
    qp.w_du = cfg.w_du;
    qp.osqp_eps_abs = cfg.osqp_eps_abs;
    qp.osqp_eps_rel = cfg.osqp_eps_rel;
    qp.osqp_max_iter = cfg.osqp_max_iter;
    qp.osqp_verbose = cfg.osqp_verbose;
    return qp;
}

}  // namespace koopman_control
