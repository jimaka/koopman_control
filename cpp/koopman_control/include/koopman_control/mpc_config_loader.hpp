#pragma once

/**
 * @file mpc_config_loader.hpp
 * @brief 从 YAML 加载 MPC 配置，并与 ONNX horizon 对齐
 */

#include <string>

#include "koopman_control/mpc_config.hpp"

namespace koopman_control {

/** 从 YAML 加载 MPC 参数；文件中缺失的字段保留 cfg 已有默认值 */
MpcConfig loadMpcConfigFromYaml(const std::string& yaml_path, MpcConfig cfg = {});

/** 用 ONNX 实际 horizon 覆盖 cfg.horizon，并裁剪 opt_control_steps */
MpcConfig syncHorizonWithOnnx(MpcConfig cfg, int onnx_horizon);

}  // namespace koopman_control
