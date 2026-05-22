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
    float ref_dt = 1.0f;           ///< 相邻参考点间隔 (s)，运行时通常 = mpc_during
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

/** 单步 MPC 求解耗时（毫秒）与迭代统计 */
struct MotionSolveTiming {
    double ref_resample_ms = 0.;  ///< 参考轨迹重采样
    double inference_ms = 0.;     ///< ONNX rollout 累计
    double mpc_opt_ms = 0.;       ///< Adam 优化 + 数值梯度（不含推理）
    double solve_step_ms = 0.;    ///< ref_resample + inference + mpc_opt
    int mpc_opt_iters_cfg = 0;    ///< 配置的最大 Adam 迭代次数
    int mpc_opt_iters_done = 0;   ///< 本步实际执行的 Adam 迭代次数
    int mpc_rollout_count = 0;    ///< 本步 ONNX rollout 调用次数
};

/** 单步 MPC 求解输出 */
struct MotionSolveOutput {
    std::array<float, 4> control{};  ///< 4 维 Koopman 控制量 u0
    float cost = 0.f;                ///< 优化代价
    int horizon = 0;                 ///< 当前 ONNX horizon
    MotionSolveTiming timing;        ///< 本步求解各阶段耗时
};

/**
 * @brief 面向 motion.cpp 的单步 MPC 求解器
 *
 * 构造时加载 ONNX 并创建内部 KoopmanMpcController。
 */
class KoopmanOnnxModel;

class KoopmanMotionMpc {
public:
    KoopmanMotionMpc(const std::string& onnx_path, MpcConfig mpc_cfg = {},
                     MotionBridgeConfig bridge_cfg = {});
    /** 使用已加载的 ONNX 模型（避免在 ROS 初始化后重复加载） */
    KoopmanMotionMpc(KoopmanOnnxModel model, MpcConfig mpc_cfg,
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
