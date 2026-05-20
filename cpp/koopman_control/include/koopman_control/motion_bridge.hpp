#pragma once

/**
 * @file motion_bridge.hpp
 * @brief 供 motion.cpp（Elane ship_control 节点）调用的 Koopman MPC 桥接层。
 *
 * 不依赖 ROS；仅使用标准库与 koopman_control 库本身。
 * motion.cpp 将 PointChange() 生成的船体坐标参考点传入本接口即可。
 */

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "koopman_control/mpc_config.hpp"

namespace koopman_control {

/** 与 motion.cpp / ship_control::mpc_state 对齐的单点参考（船体坐标系） */
struct MotionRefPoint {
    float x = 0.f;
    float y = 0.f;
    float psi = 0.f;  /** 航向 yaw，rad */
    float u = 0.f;    /** 纵向速度 */
    float v = 0.f;    /** 横向速度 */
    float r = 0.f;    /** 艏摇角速度，缺省可填 0 */
};

/** motion 侧参考点时间轴参数（对应 PointChange 中 mpc_during） */
struct MotionBridgeConfig {
    /** motion 参考点之间的时间间隔，秒；通常等于 mpc_during */
    float ref_dt = 0.1f;
    /** PointChange 采样时间偏置，秒；原代码为 i*mpc_during+0.5 中的 0.5 */
    float ref_time_offset = 0.5f;
};

struct MotionSolveInput {
    /** 船体坐标系下当前状态：位置取原点，即 motion 中 ini_state 前 3 维为 0 */
    float u = 0.f;
    float v = 0.f;
    float r = 0.f;
    /** PointChange / mpc_states_gl.targets 填写的参考序列（长度 >= 1） */
    std::vector<MotionRefPoint> ref;
};

struct MotionSolveOutput {
    std::array<float, 4> control{};
    float cost = 0.f;
    int horizon = 0;
};

/**
 * @brief 面向 motion.cpp 的单步 MPC 求解器（加载 ONNX + 内部 KoopmanMpcController）。
 */
class KoopmanMotionMpc {
public:
    KoopmanMotionMpc(const std::string& onnx_path, MpcConfig mpc_cfg = {},
                     MotionBridgeConfig bridge_cfg = {});
    ~KoopmanMotionMpc();

    KoopmanMotionMpc(const KoopmanMotionMpc&) = delete;
    KoopmanMotionMpc& operator=(const KoopmanMotionMpc&) = delete;

    bool solve(const MotionSolveInput& in, MotionSolveOutput& out);

    int horizon() const;
    void resetWarmStart();

private:
    std::vector<std::array<float, 6>> buildRefWindow(const MotionSolveInput& in) const;

    MotionBridgeConfig bridge_;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/** 将 MotionRefPoint 序列重采样为 Koopman MPC 所需的 H+1 个 6 维参考状态 */
std::vector<std::array<float, 6>> resampleMotionRefToHorizon(
    const std::vector<MotionRefPoint>& ref,
    int horizon,
    float mpc_dt,
    float ref_dt,
    float ref_time_offset);

}  // namespace koopman_control
