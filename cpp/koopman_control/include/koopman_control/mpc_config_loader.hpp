#pragma once

#include <string>

#include "koopman_control/mpc_config.hpp"

namespace koopman_control {

/** 从 YAML 加载 MPC 参数；缺失字段保留 cfg 中已有默认值 */
MpcConfig loadMpcConfigFromYaml(const std::string& yaml_path, MpcConfig cfg = {});

/** 根据 ONNX 模型 horizon 校正 cfg.horizon */
MpcConfig syncHorizonWithOnnx(MpcConfig cfg, int onnx_horizon);

}  // namespace koopman_control
