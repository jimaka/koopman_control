#pragma once

/**
 * @file mpc_config_loader.hpp
 * @brief 从 YAML 加载 MPC 配置，并与 ONNX horizon 对齐
 */

#include <string>

#include "koopman_control/latent_mpc_qp.hpp"
#include "koopman_control/mpc_config.hpp"

namespace koopman_control {

MpcConfig loadMpcConfigFromYaml(const std::string& yaml_path, MpcConfig cfg = {});
MpcConfig syncHorizonWithOnnx(MpcConfig cfg, int onnx_horizon);
LatentMpcQpConfig latentQpConfigFromMpc(const MpcConfig& cfg);

}  // namespace koopman_control
