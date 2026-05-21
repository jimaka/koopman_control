/**
 * @file motion_bridge.cpp
 * @brief motion.cpp 桥接层：参考轨迹重采样 + 单步 MPC 求解
 *
 * motion 侧 PointChange() 通常只生成 mpc_steps 个参考点（如 20~40），
 * 而 Koopman ONNX 需要 H+1 个参考（H=200）。本文件按时间轴线性插值对齐。
 */

#include "koopman_control/motion_bridge.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

#include "koopman_control/mpc_config_loader.hpp"
#include "koopman_control/mpc_controller.hpp"

namespace koopman_control {

namespace {

/** 在 motion 参考序列上按时间 t_query 线性插值 */
MotionRefPoint interpolateRef(const std::vector<MotionRefPoint>& ref, float t_query,
                              float ref_dt, float ref_time_offset) {
    if (ref.empty()) {
        return {};
    }
    if (ref.size() == 1) {
        return ref.front();
    }

    const float t0 = ref_time_offset;
    const float t_last = t0 + static_cast<float>(ref.size() - 1) * ref_dt;
    if (t_query <= t0) {
        return ref.front();
    }
    if (t_query >= t_last) {
        return ref.back();
    }

    const float idx_f = (t_query - t0) / ref_dt;
    const int i0 = static_cast<int>(std::floor(idx_f));
    const int i1 = std::min(i0 + 1, static_cast<int>(ref.size()) - 1);
    const float alpha = idx_f - static_cast<float>(i0);

    MotionRefPoint out;
    const auto& a = ref[i0];
    const auto& b = ref[i1];
    out.x = a.x + alpha * (b.x - a.x);
    out.y = a.y + alpha * (b.y - a.y);
    out.psi = a.psi + alpha * (b.psi - a.psi);
    out.u = a.u + alpha * (b.u - a.u);
    out.v = a.v + alpha * (b.v - a.v);
    out.r = a.r + alpha * (b.r - a.r);
    return out;
}

}  // namespace

std::vector<std::array<float, 6>> resampleMotionRefToHorizon(
    const std::vector<MotionRefPoint>& ref,
    int horizon,
    float mpc_dt,
    float ref_dt,
    float ref_time_offset) {
    if (horizon <= 0) {
        throw std::invalid_argument("horizon must be positive");
    }
    if (ref.empty()) {
        throw std::invalid_argument("motion ref must not be empty");
    }

    std::vector<std::array<float, 6>> out(static_cast<size_t>(horizon + 1));
    for (int k = 0; k <= horizon; ++k) {
        const float t_k = static_cast<float>(k) * mpc_dt;
        const MotionRefPoint p = interpolateRef(ref, t_k, ref_dt, ref_time_offset);
        out[static_cast<size_t>(k)] = {p.x, p.y, p.psi, p.u, p.v, p.r};
    }
    return out;
}

/** KoopmanMotionMpc 内部实现（隐藏 ONNX 与 Controller 细节） */
class KoopmanMotionMpc::Impl {
public:
    Impl(const std::string& onnx_path, MpcConfig mpc_cfg, MotionBridgeConfig bridge_cfg)
        : bridge_(bridge_cfg) {
        KoopmanOnnxModel model(onnx_path);
        mpc_cfg = syncHorizonWithOnnx(mpc_cfg, model.horizon());
        controller_ = std::make_unique<KoopmanMpcController>(std::move(model), mpc_cfg);
    }

    MotionBridgeConfig bridge_;
    std::unique_ptr<KoopmanMpcController> controller_;
};

KoopmanMotionMpc::KoopmanMotionMpc(const std::string& onnx_path, MpcConfig mpc_cfg,
                                   MotionBridgeConfig bridge_cfg)
    : bridge_(bridge_cfg), impl_(std::make_unique<Impl>(onnx_path, mpc_cfg, bridge_cfg)) {}

KoopmanMotionMpc::~KoopmanMotionMpc() = default;

bool KoopmanMotionMpc::solve(const MotionSolveInput& in, MotionSolveOutput& out) {
    if (!impl_ || !impl_->controller_) {
        return false;
    }
    if (in.ref.empty()) {
        return false;
    }

    const auto ref_window = buildRefWindow(in);
    // motion 约定：船体坐标系原点为当前位置，故 x=y=yaw=0
    std::array<float, 6> state0{0.f, 0.f, 0.f, in.u, in.v, in.r};
    const std::array<float, 4>* u_prev_ptr = in.has_u_prev ? &in.u_prev : nullptr;
    auto [u_opt, cost] = impl_->controller_->solveStep(state0, ref_window, u_prev_ptr);
    out.control = u_opt;
    out.cost = cost;
    out.horizon = impl_->controller_->horizon();
    return true;
}

int KoopmanMotionMpc::horizon() const {
    return impl_ && impl_->controller_ ? impl_->controller_->horizon() : 0;
}

void KoopmanMotionMpc::resetWarmStart() {
    if (impl_ && impl_->controller_) {
        impl_->controller_->resetWarmStart();
    }
}

std::vector<std::array<float, 6>> KoopmanMotionMpc::buildRefWindow(
    const MotionSolveInput& in) const {
    const int H = horizon();
    return resampleMotionRefToHorizon(in.ref, H, impl_->controller_->config().dt, bridge_.ref_dt,
                                      bridge_.ref_time_offset);
}

}  // namespace koopman_control
