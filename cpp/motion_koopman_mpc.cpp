#ifdef USE_KOOPMAN_MPC

#include "motion_koopman_mpc.hpp"

#include "koopman_control/mpc_config_loader.hpp"

namespace elane {
namespace control {

void MotionKoopmanMpcHelper::init(const std::string& yaml_path, const std::string& onnx_path,
                                  float mpc_during) {
    koopman_control::MpcConfig cfg = koopman_control::loadMpcConfigFromYaml(yaml_path);
    koopman_control::MotionBridgeConfig bridge;
    bridge.ref_dt = mpc_during;
    bridge.ref_time_offset = 0.5f;
    solver_ = std::make_unique<koopman_control::KoopmanMotionMpc>(onnx_path, cfg, bridge);
}

bool MotionKoopmanMpcHelper::solveStep(float u, float v, float r,
                                       const std::vector<MotionMpcTargetView>& targets,
                                       koopman_control::MotionSolveOutput& out) {
    if (!solver_) {
        return false;
    }
    koopman_control::MotionSolveInput in;
    in.u = u;
    in.v = v;
    in.r = r;
    in.ref.reserve(targets.size());
    for (const auto& t : targets) {
        koopman_control::MotionRefPoint p;
        p.x = t.x;
        p.y = t.y;
        p.psi = t.psi;
        p.u = t.u;
        p.v = t.v;
        p.r = 0.f;
        in.ref.push_back(p);
    }
    return solver_->solve(in, out);
}

int MotionKoopmanMpcHelper::horizon() const {
    return solver_ ? solver_->horizon() : 0;
}

void MotionKoopmanMpcHelper::resetWarmStart() {
    if (solver_) {
        solver_->resetWarmStart();
    }
}

}  // namespace control
}  // namespace elane

#endif  // USE_KOOPMAN_MPC
