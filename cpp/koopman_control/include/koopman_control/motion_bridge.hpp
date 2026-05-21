#pragma once

/**
 * @file motion_bridge.hpp
 * @brief 供 motion.cpp（Elane ship_control 节点）调用的 Koopman MPC 桥接层
 *
 * 设计原则：
 * - 不依赖 ROS，仅标准库 + koopman_control
 * - 输入 motion 侧 PointChange() 生成的船体坐标参考点
 * - 内部重采样到 ONNX horizon 并调用 KoopmanMpcController
 */

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "koopman_control/mpc_config.hpp"

namespace koopman_control {

/** 单点参考状态（船体坐标系，与 ship_control::mpc_state 字段对应） */
struct MotionRefPoint {
    float x = 0.f;    ///< 纵向位置 (m)
    float y = 0.f;    ///< 横向位置 (m)
    float psi = 0.f;  ///< 航向 yaw (rad)
    float u = 0.f;    ///< 纵向速度 (m/s)
    float v = 0.f;    ///< 横向速度 (m/s)
    float r = 0.f;    ///< 艏摇角速度 (rad/s)，缺省可填 0
};

/** motion 参考轨迹时间轴参数（对应 PointChange 中 mpc_during） */
struct MotionBridgeConfig {
    float ref_dt = 0.1f;           ///< 相邻参考点间隔 (s)，通常 = mpc_during
    float ref_time_offset = 0.5f;  ///< 首点采样偏置 (s)，对应 i*mpc_during+0.5
};

/** 单步 MPC 求解输入 */
struct MotionSolveInput {
    float u = 0.f;  ///< 当前纵向速度
    float v = 0.f;  ///< 当前横向速度
    float r = 0.f;  ///< 当前艏摇角速度
    /** 上一步实际下发的 4 维控制；has_u_prev=true 时用于变化速率约束 */
    std::array<float, 4> u_prev{};
    bool has_u_prev = false;
    /** PointChange / mpc_states_gl.targets 填写的参考序列（长度 >= 1） */
    std::vector<MotionRefPoint> ref;
};

/** 单步 MPC 求解输出 */
struct MotionSolveOutput {
    std::array<float, 4> control{};  ///< 4 维 Koopman 控制量 u0
    float cost = 0.f;                ///< 优化代价
    int horizon = 0;                 ///< 当前 ONNX horizon
};

/**
 * @brief 面向 motion.cpp 的单步 MPC 求解器
 *
 * 构造时加载 ONNX 并创建内部 KoopmanMpcController。
 */
class KoopmanMotionMpc {
public:
    KoopmanMotionMpc(const std::string& onnx_path, MpcConfig mpc_cfg = {},
                     MotionBridgeConfig bridge_cfg = {});
    ~KoopmanMotionMpc();

    KoopmanMotionMpc(const KoopmanMotionMpc&) = delete;
    KoopmanMotionMpc& operator=(const KoopmanMotionMpc&) = delete;

    /** 单步求解；成功返回 true 并写入 out */
    bool solve(const MotionSolveInput& in, MotionSolveOutput& out);

    int horizon() const;
    void resetWarmStart();

private:
    std::vector<std::array<float, 6>> buildRefWindow(const MotionSolveInput& in) const;

    MotionBridgeConfig bridge_;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief 将 motion 稀疏参考重采样为 H+1 个 6 维状态
 * @param mpc_dt   MPC/ONNX 时间步（cfg.dt）
 * @param ref_dt   motion 参考点间隔（bridge.ref_dt）
 */
std::vector<std::array<float, 6>> resampleMotionRefToHorizon(
    const std::vector<MotionRefPoint>& ref,
    int horizon,
    float mpc_dt,
    float ref_dt,
    float ref_time_offset);

}  // namespace koopman_control
