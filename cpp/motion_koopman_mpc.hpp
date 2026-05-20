#pragma once

/**
 * @file motion_koopman_mpc.hpp
 * @brief 供 motion.cpp 调用的 Koopman MPC 封装（在 ship_control 中定义 USE_KOOPMAN_MPC=1 启用）。
 */

#ifdef USE_KOOPMAN_MPC

#include <memory>
#include <string>

#include "koopman_control/motion_bridge.hpp"

namespace elane {
namespace control {

/** 将 motion 侧 mpc_state 风格目标转为 Koopman 参考点（避免 motion.cpp 直接依赖模板细节） */
struct MotionMpcTargetView {
    float x = 0.f;
    float y = 0.f;
    float psi = 0.f;
    float u = 0.f;
    float v = 0.f;
};

class MotionKoopmanMpcHelper {
public:
    void init(const std::string& yaml_path, const std::string& onnx_path, float mpc_during);

    bool solveStep(float u, float v, float r, const std::vector<MotionMpcTargetView>& targets,
                   koopman_control::MotionSolveOutput& out);

    int horizon() const;
    void resetWarmStart();

private:
    std::unique_ptr<koopman_control::KoopmanMotionMpc> solver_;
};

}  // namespace control
}  // namespace elane

#endif  // USE_KOOPMAN_MPC
